from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agent.memory.session_memory.fork_agent import (
    SessionMemoryForkAgent,
    SessionMemoryForkInput,
)
from agent.memory.session_memory.serializer import (
    count_tool_calls,
    estimate_message_tokens,
    latest_assistant_has_no_tool_call,
    message_id,
    serialize_messages,
)
from agent.memory.session_memory.store import SessionMemoryStore
from agent.observe.observer import RuntimeObserver


@dataclass(frozen=True)
class SessionMemoryConfig:
    init_token_threshold: int = 10_000
    update_token_delta_threshold: int = 5_000
    tool_call_delta_threshold: int = 3
    max_update_failures: int = 3
    recent_messages_window: int = 12
    max_tool_result_preview_chars: int = 2000
    max_fork_turns: int = 3


@dataclass(frozen=True)
class SessionMemoryDecision:
    should_update: bool
    reason: str
    current_estimated_tokens: int
    token_delta: int
    current_tool_call_count: int
    tool_call_delta: int
    latest_assistant_has_no_tool_call: bool


class SessionMemoryManager:
    def __init__(
        self,
        llm_client,
        *,
        config: SessionMemoryConfig | None = None,
        observer: RuntimeObserver | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or SessionMemoryConfig()
        self.observer = observer

    def maybe_update(self, state: dict[str, Any]) -> dict[str, Any]:
        if not state.get("session_memory_enabled", True):
            decision = self._make_disabled_decision(state)
            self._emit_check(decision)
            return {}

        store = self._make_store(state)
        base_meta = {
            "session_memory_ref": store.ref,
            "session_memory_exists": store.exists(),
        }
        decision = self._decide(state)
        self._emit_check(decision)

        if not decision.should_update:
            return {
                **base_meta,
                "session_memory_last_update_reason": decision.reason,
                "session_memory_last_fork_agent_status": "skipped",
            }

        update_run_id = f"session_memory_{uuid4().hex[:8]}"
        started_at = _utc_now()
        self._emit(
            "session_memory_fork_agent_start",
            {
                "update_run_id": update_run_id,
                "message_range": self._message_range(state),
                "session_memory_ref": store.ref,
            },
        )

        try:
            old_memory = store.read_or_template()
            fork_input = self._build_fork_input(state, old_memory)
            fork_agent = SessionMemoryForkAgent(
                self.llm_client,
                store,
                max_turns=self.config.max_fork_turns,
            )
            result = fork_agent.run(fork_input)
            finished_at = _utc_now()

            self._emit(
                "session_memory_fork_agent_finish",
                {
                    "update_run_id": update_run_id,
                    "status": result.status,
                    "turns": result.turns,
                    "error": result.error,
                },
            )

            validation = store.validate(store.read_or_template())
            self._emit(
                "session_memory_file_validate",
                {
                    "valid": validation.valid,
                    "errors": validation.errors,
                    "session_memory_ref": store.ref,
                },
            )

            if result.status != "success":
                raise RuntimeError(result.error or "Session memory fork agent failed.")

            if not validation.valid:
                raise RuntimeError("; ".join(validation.errors))

            messages = list(state.get("messages", []))
            last_index = len(messages) - 1
            patch = {
                **base_meta,
                "session_memory_exists": True,
                "session_memory_last_summarized_message_id": (
                    message_id(messages[last_index], last_index) if messages else None
                ),
                "session_memory_last_summarized_message_index": last_index,
                "session_memory_last_summarized_token_count": (
                    decision.current_estimated_tokens
                ),
                "session_memory_last_summarized_tool_call_count": (
                    decision.current_tool_call_count
                ),
                "session_memory_updated_at": finished_at,
                "session_memory_update_in_progress": False,
                "session_memory_update_failures": 0,
                "session_memory_last_update_reason": decision.reason,
                "session_memory_last_error": None,
                "session_memory_update_run_id": update_run_id,
                "session_memory_update_started_at": started_at,
                "session_memory_update_finished_at": finished_at,
                "session_memory_last_fork_agent_status": "success",
            }
            self._emit(
                "session_memory_update_success",
                {
                    "updated_at": finished_at,
                    "summarized_message_index": last_index,
                    "output_chars": result.content_chars,
                    "session_memory_ref": store.ref,
                },
            )
            if self.observer and hasattr(self.observer, "on_memory_updated"):
                self.observer.on_memory_updated(
                    memory_type="session",
                    status="success",
                    ref=store.ref,
                    reason=decision.reason,
                )
            return patch

        except Exception as error:
            finished_at = _utc_now()
            failure_count = int(state.get("session_memory_update_failures") or 0) + 1
            self._emit(
                "session_memory_update_failure",
                {
                    "error": str(error),
                    "failure_count": failure_count,
                    "session_memory_ref": store.ref,
                },
            )
            if self.observer and hasattr(self.observer, "on_memory_updated"):
                self.observer.on_memory_updated(
                    memory_type="session",
                    status="failed",
                    ref=store.ref,
                    reason=decision.reason,
                )
            return {
                **base_meta,
                "session_memory_update_in_progress": False,
                "session_memory_update_failures": failure_count,
                "session_memory_last_update_reason": decision.reason,
                "session_memory_last_error": str(error),
                "session_memory_update_run_id": update_run_id,
                "session_memory_update_started_at": started_at,
                "session_memory_update_finished_at": finished_at,
                "session_memory_last_fork_agent_status": "failed",
            }

    def _decide(self, state: dict[str, Any]) -> SessionMemoryDecision:
        messages = list(state.get("messages", []))
        current_tokens = estimate_message_tokens(messages)
        current_tool_calls = count_tool_calls(messages)
        last_tokens = int(state.get("session_memory_last_summarized_token_count") or 0)
        last_tool_calls = int(
            state.get("session_memory_last_summarized_tool_call_count") or 0
        )
        token_delta = current_tokens - last_tokens
        tool_call_delta = current_tool_calls - last_tool_calls
        latest_no_tool_call = latest_assistant_has_no_tool_call(
            state.get("assistant_message")
        )
        failures = int(state.get("session_memory_update_failures") or 0)

        should_update = (
            current_tokens >= self.config.init_token_threshold
            and token_delta >= self.config.update_token_delta_threshold
            and (
                tool_call_delta >= self.config.tool_call_delta_threshold
                or latest_no_tool_call
            )
            and not state.get("session_memory_update_in_progress", False)
            and failures < self.config.max_update_failures
        )

        reason = "thresholds_met" if should_update else "thresholds_not_met"
        if state.get("session_memory_update_in_progress", False):
            reason = "update_in_progress"
        elif failures >= self.config.max_update_failures:
            reason = "max_failures_reached"
        elif current_tokens < self.config.init_token_threshold:
            reason = "below_init_token_threshold"
        elif token_delta < self.config.update_token_delta_threshold:
            reason = "below_token_delta_threshold"
        elif (
            tool_call_delta < self.config.tool_call_delta_threshold
            and not latest_no_tool_call
        ):
            reason = "below_tool_call_delta_and_not_natural_break"

        return SessionMemoryDecision(
            should_update=should_update,
            reason=reason,
            current_estimated_tokens=current_tokens,
            token_delta=token_delta,
            current_tool_call_count=current_tool_calls,
            tool_call_delta=tool_call_delta,
            latest_assistant_has_no_tool_call=latest_no_tool_call,
        )

    def _build_fork_input(
        self,
        state: dict[str, Any],
        old_session_memory: str,
    ) -> SessionMemoryForkInput:
        messages = list(state.get("messages", []))
        last_index = state.get("session_memory_last_summarized_message_index")
        start_index = int(last_index) + 1 if isinstance(last_index, int) else 0
        new_messages = messages[start_index:]
        recent_start = max(0, len(messages) - self.config.recent_messages_window)
        recent_messages = messages[recent_start:]

        return SessionMemoryForkInput(
            old_session_memory=old_session_memory,
            new_messages_since_last_summary=serialize_messages(
                new_messages,
                start_index=start_index,
                max_tool_result_preview_chars=self.config.max_tool_result_preview_chars,
            ),
            recent_messages_window=serialize_messages(
                recent_messages,
                start_index=recent_start,
                max_tool_result_preview_chars=self.config.max_tool_result_preview_chars,
            ),
            runtime_metadata={
                "session_id": state.get("session_id"),
                "workspace_root": state.get("workspace_root"),
                "permission_mode": state.get("permission_mode"),
                "message_count": len(messages),
            },
        )

    def _make_store(self, state: dict[str, Any]) -> SessionMemoryStore:
        return SessionMemoryStore(
            workspace_root=state.get("workspace_root") or state.get("workspace") or ".",
            session_id=state.get("session_id"),
        )

    def _message_range(self, state: dict[str, Any]) -> dict[str, int | None]:
        messages = list(state.get("messages", []))
        last_index = state.get("session_memory_last_summarized_message_index")
        start_index = int(last_index) + 1 if isinstance(last_index, int) else 0
        end_index = len(messages) - 1 if messages else None
        return {
            "start": start_index,
            "end": end_index,
        }

    def _make_disabled_decision(self, state: dict[str, Any]) -> SessionMemoryDecision:
        messages = list(state.get("messages", []))
        return SessionMemoryDecision(
            should_update=False,
            reason="disabled",
            current_estimated_tokens=estimate_message_tokens(messages),
            token_delta=0,
            current_tool_call_count=count_tool_calls(messages),
            tool_call_delta=0,
            latest_assistant_has_no_tool_call=latest_assistant_has_no_tool_call(
                state.get("assistant_message")
            ),
        )

    def _emit_check(self, decision: SessionMemoryDecision) -> None:
        self._emit(
            "session_memory_check",
            {
                "current_estimated_tokens": decision.current_estimated_tokens,
                "init_threshold": self.config.init_token_threshold,
                "token_delta": decision.token_delta,
                "update_token_delta_threshold": (
                    self.config.update_token_delta_threshold
                ),
                "tool_call_delta": decision.tool_call_delta,
                "tool_call_delta_threshold": self.config.tool_call_delta_threshold,
                "latest_assistant_has_no_tool_call": (
                    decision.latest_assistant_has_no_tool_call
                ),
                "should_update": decision.should_update,
                "reason": decision.reason,
            },
        )

    def _emit(self, event_type: str, content: dict[str, Any]) -> None:
        if not self.observer:
            return
        self.observer.emit(
            event_type,
            output_node="session_memory",
            input_node="trace",
            content=content,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
