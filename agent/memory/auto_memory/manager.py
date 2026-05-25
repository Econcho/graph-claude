from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agent.memory.auto_memory.extractor import (
    AutoMemoryExtractor,
    AutoMemoryExtractorInput,
)
from agent.memory.auto_memory.recall import AutoMemoryRecall
from agent.memory.auto_memory.store import AutoMemoryRecord, AutoMemoryStore
from agent.memory.session_memory.serializer import (
    latest_assistant_has_no_tool_call,
    message_id,
)
from agent.observe.observer import RuntimeObserver


@dataclass(frozen=True)
class AutoMemoryConfig:
    max_recalled_items: int = 3
    recent_messages_window: int = 12
    max_extract_failures: int = 3


@dataclass(frozen=True)
class AutoMemoryExtractDecision:
    should_extract: bool
    reason: str
    latest_assistant_has_no_tool_call: bool
    has_new_messages: bool
    current_run_is_main_agent: bool


class AutoMemoryManager:
    def __init__(
        self,
        llm_client,
        *,
        config: AutoMemoryConfig | None = None,
        observer: RuntimeObserver | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or AutoMemoryConfig()
        self.observer = observer

    def recall(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("auto_memory_enabled", True):
            self._emit(
                "auto_memory_recall_empty",
                {"reason": "disabled"},
                output_node="auto_memory",
                input_node="context",
            )
            return {}

        store = self._make_store(state)
        base_meta = {
            "auto_memory_ref": store.ref,
            "auto_memory_index_loaded": store.exists(),
        }
        self._emit(
            "auto_memory_recall_start",
            {"auto_memory_ref": store.ref},
            output_node="context",
            input_node="auto_memory",
        )

        try:
            recall = AutoMemoryRecall(
                store,
                max_results=self.config.max_recalled_items,
            )
            items = recall.recall(state)
        except Exception as error:
            self._emit(
                "auto_memory_recall_empty",
                {"reason": "recall_error", "error": str(error)},
                output_node="auto_memory",
                input_node="context",
            )
            return {
                **base_meta,
                "auto_memory_recalled_items": [],
                "auto_memory_last_error": str(error),
            }

        surfaced = list(state.get("auto_memory_already_surfaced") or [])
        for item in items:
            memory_id = item.get("id")
            if memory_id and memory_id not in surfaced:
                surfaced.append(memory_id)

        event_type = "auto_memory_recall_selected" if items else "auto_memory_recall_empty"
        self._emit(
            event_type,
            {
                "selected_count": len(items),
                "selected_ids": [item.get("id") for item in items],
                "auto_memory_ref": store.ref,
            },
            output_node="auto_memory",
            input_node="context",
        )

        return {
            **base_meta,
            "auto_memory_recalled_items": items,
            "auto_memory_already_surfaced": surfaced,
            "auto_memory_last_error": None,
        }

    def maybe_extract(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("auto_memory_enabled", True):
            decision = self._disabled_decision(state)
            self._emit_extract_check(decision)
            return {}

        store = self._make_store(state)
        base_meta = {
            "auto_memory_ref": store.ref,
            "auto_memory_index_loaded": store.exists(),
        }
        decision = self._decide_extract(state)
        self._emit_extract_check(decision)

        if not decision.should_extract:
            return {
                **base_meta,
                "auto_memory_last_extract_status": "skipped",
                "auto_memory_last_error": None,
            }

        extract_run_id = f"auto_memory_{uuid4().hex[:8]}"
        started_at = _utc_now()
        self._emit(
            "auto_memory_extract_start",
            {
                "extract_run_id": extract_run_id,
                "auto_memory_ref": store.ref,
            },
            output_node="finalize",
            input_node="auto_memory",
        )

        try:
            store.ensure()
            manifest = [self._record_manifest(record) for record in store.list_memories()]
            extractor = AutoMemoryExtractor(self.llm_client)
            result = extractor.extract(
                AutoMemoryExtractorInput(
                    messages=list(state.get("messages", [])),
                    final_answer=str(state.get("final_answer") or ""),
                    memory_index=store.read_index_or_template(),
                    memory_manifest=manifest,
                    runtime_metadata={
                        "workspace_root": state.get("workspace_root"),
                        "session_id": state.get("session_id"),
                        "permission_mode": state.get("permission_mode"),
                        "message_count": len(state.get("messages", [])),
                    },
                    recent_messages_window=self.config.recent_messages_window,
                )
            )

            if result.status != "success":
                raise RuntimeError(result.error or "Auto memory extraction failed.")

            self._emit(
                "auto_memory_extract_proposal",
                {
                    "extract_run_id": extract_run_id,
                    "proposal_count": len(result.proposals),
                    "proposals": result.proposals,
                },
                output_node="auto_memory",
                input_node="auto_memory_store",
            )

            written = self._write_proposals(store, result.proposals)
            finished_at = _utc_now()
            messages = list(state.get("messages", []))
            last_index = len(messages) - 1
            last_id = message_id(messages[last_index], last_index) if messages else None

            if not written:
                self._emit(
                    "auto_memory_extract_empty",
                    {
                        "extract_run_id": extract_run_id,
                        "proposal_count": len(result.proposals),
                    },
                    output_node="auto_memory",
                    input_node="finalize",
                )
                status = "empty"
            else:
                self._emit(
                    "auto_memory_index_update",
                    {
                        "extract_run_id": extract_run_id,
                        "written_count": len(written),
                        "auto_memory_ref": store.ref,
                    },
                    output_node="auto_memory_store",
                    input_node="auto_memory",
                )
                status = "success"

            if self.observer and hasattr(self.observer, "on_memory_updated"):
                self.observer.on_memory_updated(
                    memory_type="auto",
                    status=status,
                    ref=store.ref,
                    reason=decision.reason,
                )

            return {
                **base_meta,
                "auto_memory_index_loaded": True,
                "auto_memory_last_extract_at": finished_at,
                "auto_memory_last_extract_message_id": last_id,
                "auto_memory_extract_in_progress": False,
                "auto_memory_extract_failures": 0,
                "auto_memory_last_error": None,
                "auto_memory_last_extract_run_id": extract_run_id,
                "auto_memory_last_extract_status": status,
            }

        except Exception as error:
            failure_count = int(state.get("auto_memory_extract_failures") or 0) + 1
            self._emit(
                "auto_memory_extract_failure",
                {
                    "extract_run_id": extract_run_id,
                    "error": str(error),
                    "failure_count": failure_count,
                    "auto_memory_ref": store.ref,
                },
                output_node="auto_memory",
                input_node="finalize",
            )
            if self.observer and hasattr(self.observer, "on_memory_updated"):
                self.observer.on_memory_updated(
                    memory_type="auto",
                    status="failed",
                    ref=store.ref,
                    reason=decision.reason,
                )
            return {
                **base_meta,
                "auto_memory_extract_in_progress": False,
                "auto_memory_extract_failures": failure_count,
                "auto_memory_last_error": str(error),
                "auto_memory_last_extract_run_id": extract_run_id,
                "auto_memory_last_extract_status": "failed",
            }

    def _write_proposals(
        self,
        store: AutoMemoryStore,
        proposals: list[dict[str, Any]],
    ) -> list[AutoMemoryRecord]:
        written: list[AutoMemoryRecord] = []
        existing_by_name = {
            record.name.lower(): record for record in store.list_memories()
        }

        for proposal in proposals:
            action = proposal.get("action")
            if action == "none":
                continue

            try:
                if action == "update":
                    target_id = str(proposal.get("target_memory_id") or "").strip()
                    if not target_id:
                        continue
                    record = store.update_memory(target_id, proposal)
                elif action == "create":
                    existing = existing_by_name.get(str(proposal.get("name", "")).lower())
                    if existing:
                        record = store.update_memory(existing.id, proposal)
                    else:
                        record = store.create_memory(
                            {
                                **proposal,
                                "source": "auto_memory_extractor",
                            }
                        )
                    existing_by_name[record.name.lower()] = record
                else:
                    continue
            except Exception as error:
                self._emit(
                    "auto_memory_write_failure",
                    {
                        "proposal": proposal,
                        "error": str(error),
                    },
                    output_node="auto_memory_store",
                    input_node="auto_memory",
                )
                continue

            self._emit(
                "auto_memory_write_success",
                {
                    "action": action,
                    "memory": self._record_manifest(record),
                    "auto_memory_ref": store.ref,
                },
                output_node="auto_memory_store",
                input_node="auto_memory",
            )
            written.append(record)

        return written

    def _decide_extract(self, state: dict[str, Any]) -> AutoMemoryExtractDecision:
        messages = list(state.get("messages", []))
        latest_no_tool_call = latest_assistant_has_no_tool_call(
            state.get("assistant_message")
        )
        current_run_is_main_agent = state.get("agent_id") not in {
            "session_memory_fork_agent",
            "auto_memory_extractor",
        }
        last_id = None
        if messages:
            last_id = message_id(messages[-1], len(messages) - 1)
        has_new_messages = bool(last_id) and (
            last_id != state.get("auto_memory_last_extract_message_id")
        )
        failures = int(state.get("auto_memory_extract_failures") or 0)

        should_extract = (
            latest_no_tool_call
            and current_run_is_main_agent
            and has_new_messages
            and not state.get("auto_memory_extract_in_progress", False)
            and failures < self.config.max_extract_failures
            and not state.get("auto_memory_main_agent_wrote_memory_this_turn", False)
        )

        reason = "conditions_met" if should_extract else "conditions_not_met"
        if not latest_no_tool_call:
            reason = "latest_assistant_has_tool_call"
        elif not current_run_is_main_agent:
            reason = "not_main_agent"
        elif not has_new_messages:
            reason = "no_new_messages"
        elif state.get("auto_memory_extract_in_progress", False):
            reason = "extract_in_progress"
        elif failures >= self.config.max_extract_failures:
            reason = "max_failures_reached"
        elif state.get("auto_memory_main_agent_wrote_memory_this_turn", False):
            reason = "main_agent_wrote_memory_this_turn"

        return AutoMemoryExtractDecision(
            should_extract=should_extract,
            reason=reason,
            latest_assistant_has_no_tool_call=latest_no_tool_call,
            has_new_messages=has_new_messages,
            current_run_is_main_agent=current_run_is_main_agent,
        )

    def _disabled_decision(self, state: dict[str, Any]) -> AutoMemoryExtractDecision:
        messages = list(state.get("messages", []))
        return AutoMemoryExtractDecision(
            should_extract=False,
            reason="disabled",
            latest_assistant_has_no_tool_call=latest_assistant_has_no_tool_call(
                state.get("assistant_message")
            ),
            has_new_messages=bool(messages),
            current_run_is_main_agent=True,
        )

    def _emit_extract_check(self, decision: AutoMemoryExtractDecision) -> None:
        self._emit(
            "auto_memory_extract_check",
            asdict(decision),
            output_node="finalize",
            input_node="auto_memory",
        )

    def _make_store(self, state: dict[str, Any]) -> AutoMemoryStore:
        return AutoMemoryStore(
            workspace_root=state.get("workspace_root") or state.get("workspace") or ".",
        )

    def _record_manifest(self, record: AutoMemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "type": record.type,
            "name": record.name,
            "description": record.description,
            "updated_at": record.updated_at,
        }

    def _emit(
        self,
        event_type: str,
        content: dict[str, Any],
        *,
        output_node: str,
        input_node: str,
    ) -> None:
        if not self.observer:
            return
        self.observer.emit(
            event_type,
            output_node=output_node,
            input_node=input_node,
            content=content,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
