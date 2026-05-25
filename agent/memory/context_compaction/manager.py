from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.memory.context_compaction.full_compact import FullCompactAgent
from agent.memory.context_compaction.microcompact import microcompact_messages
from agent.memory.context_compaction.session_compact import session_compact_messages
from agent.memory.context_compaction.token_counter import estimate_message_tokens
from agent.observe.observer import RuntimeObserver


FORK_AGENT_IDS = {"session_memory_fork_agent", "auto_memory_extractor"}


@dataclass(frozen=True)
class ContextCompactionConfig:
    enabled: bool = True
    microcompact_enabled: bool = True
    session_compact_enabled: bool = True
    microcompact_keep_recent_tool_results: int = 2
    session_compact_trigger_tokens: int = 80_000
    session_compact_recent_keep_max_tokens: int = 40_000
    session_compact_recent_keep_min_messages: int = 5
    full_compact_enabled: bool = True
    full_compact_trigger_tokens: int = 80_000
    full_compact_recent_keep_max_tokens: int = 20_000
    full_compact_max_output_chars: int = 80_000
    max_failures: int = 3


@dataclass(frozen=True)
class ContextCompactionResult:
    messages: list[Any]
    meta: dict[str, Any]
    patch: dict[str, Any]


class ContextCompactionManager:
    def __init__(
        self,
        llm_client=None,
        *,
        config: ContextCompactionConfig | None = None,
        observer: RuntimeObserver | None = None,
    ):
        self.llm_client = llm_client
        self.config = config or ContextCompactionConfig()
        self.observer = observer

    def compact(self, state: dict[str, Any]) -> ContextCompactionResult:
        original_messages = list(state.get("messages", []))
        original_tokens = estimate_message_tokens(original_messages)
        base_meta = {
            "enabled": bool(state.get("context_compaction_enabled", True)),
            "status": "skipped",
            "reason": "not_run",
            "original_tokens": original_tokens,
            "after_microcompact_tokens": original_tokens,
            "final_tokens": original_tokens,
            "original_message_count": len(original_messages),
            "final_message_count": len(original_messages),
            "microcompacted_tool_result_count": 0,
            "session_memory_ref": None,
            "full_compact_status": None,
        }

        self._emit(
            "context_compaction_check",
            {
                "tokens_before": original_tokens,
                "messages_before": len(original_messages),
                "trigger_tokens": self.config.session_compact_trigger_tokens,
            },
        )

        if not self.config.enabled or not state.get("context_compaction_enabled", True):
            return self._result(original_messages, base_meta | {"reason": "disabled"})

        if state.get("agent_id") in FORK_AGENT_IDS:
            return self._result(original_messages, base_meta | {"reason": "fork_agent"})

        if int(state.get("context_compaction_failures") or 0) >= self.config.max_failures:
            return self._result(
                original_messages,
                base_meta | {"reason": "max_failures_reached"},
            )

        try:
            messages = original_messages
            meta = dict(base_meta)

            if self.config.microcompact_enabled:
                micro = microcompact_messages(
                    messages,
                    keep_recent_tool_results=(
                        self.config.microcompact_keep_recent_tool_results
                    ),
                )
                messages = micro.messages
                after_micro_tokens = estimate_message_tokens(messages)
                meta.update(
                    {
                        "status": (
                            "microcompacted"
                            if micro.compacted_tool_result_count
                            else "unchanged"
                        ),
                        "reason": "microcompact_only",
                        "after_microcompact_tokens": after_micro_tokens,
                        "final_tokens": after_micro_tokens,
                        "final_message_count": len(messages),
                        "microcompacted_tool_result_count": (
                            micro.compacted_tool_result_count
                        ),
                    }
                )
                if micro.compacted_tool_result_count:
                    self._emit(
                        "microcompact_applied",
                        {
                            "tokens_before": original_tokens,
                            "tokens_after": after_micro_tokens,
                            "compacted_tool_result_count": (
                                micro.compacted_tool_result_count
                            ),
                        },
                    )

            if (
                not self.config.session_compact_enabled
                or meta["after_microcompact_tokens"]
                < self.config.session_compact_trigger_tokens
            ):
                reason = (
                    "session_compact_disabled"
                    if not self.config.session_compact_enabled
                    else "below_session_compact_threshold"
                )
                meta["reason"] = reason
                self._emit(
                    "session_compact_skipped",
                    {
                        "reason": reason,
                        "tokens_after_microcompact": meta[
                            "after_microcompact_tokens"
                        ],
                        "trigger_tokens": self.config.session_compact_trigger_tokens,
                    },
                )
                return self._maybe_full_compact(
                    messages=messages,
                    meta=meta,
                    original_messages=original_messages,
                    original_tokens=original_tokens,
                    state=state,
                    reason=reason,
                )

            session_result = session_compact_messages(
                messages,
                workspace_root=state.get("workspace_root") or state.get("workspace") or ".",
                session_id=state.get("session_id"),
                recent_keep_max_tokens=(
                    self.config.session_compact_recent_keep_max_tokens
                ),
                recent_keep_min_messages=(
                    self.config.session_compact_recent_keep_min_messages
                ),
            )
            if not session_result.compacted:
                meta.update(
                    {
                        "reason": session_result.reason,
                        "session_memory_ref": session_result.session_memory_ref,
                    }
                )
                self._emit(
                    "session_compact_skipped",
                    {
                        "reason": session_result.reason,
                        "tokens_after_microcompact": meta[
                            "after_microcompact_tokens"
                        ],
                        "session_memory_ref": session_result.session_memory_ref,
                    },
                )
                return self._maybe_full_compact(
                    messages=messages,
                    meta=meta,
                    original_messages=original_messages,
                    original_tokens=original_tokens,
                    state=state,
                    reason=session_result.reason,
                )

            final_tokens = estimate_message_tokens(session_result.messages)
            meta.update(
                {
                    "status": "session_compacted",
                    "reason": session_result.reason,
                    "final_tokens": final_tokens,
                    "final_message_count": len(session_result.messages),
                    "session_memory_ref": session_result.session_memory_ref,
                }
            )
            self._emit(
                "session_compact_applied",
                {
                    "tokens_before": original_tokens,
                    "tokens_after": final_tokens,
                    "messages_before": len(original_messages),
                    "messages_after": len(session_result.messages),
                    "reason": session_result.reason,
                    "session_memory_ref": session_result.session_memory_ref,
                },
            )
            if final_tokens >= self.config.full_compact_trigger_tokens:
                return self._maybe_full_compact(
                    messages=session_result.messages,
                    meta=meta,
                    original_messages=original_messages,
                    original_tokens=original_tokens,
                    state=state,
                    reason="session_compact_still_over_threshold",
                )

            return self._result(session_result.messages, meta)

        except Exception as error:
            failures = int(state.get("context_compaction_failures") or 0) + 1
            failure_meta = base_meta | {
                "status": "failed",
                "reason": "exception",
                "error": str(error),
            }
            self._emit(
                "context_compaction_failure",
                {
                    "error": str(error),
                    "failure_count": failures,
                    "tokens_before": original_tokens,
                },
            )
            return ContextCompactionResult(
                messages=original_messages,
                meta=failure_meta,
                patch={
                    "context_compaction_last_status": "failed",
                    "context_compaction_last_reason": "exception",
                    "context_compaction_original_tokens": original_tokens,
                    "context_compaction_after_microcompact_tokens": original_tokens,
                    "context_compaction_final_tokens": original_tokens,
                    "context_compaction_original_message_count": len(original_messages),
                    "context_compaction_final_message_count": len(original_messages),
                    "context_compaction_failures": failures,
                    "context_compaction_last_error": str(error),
                },
            )

    def _maybe_full_compact(
        self,
        *,
        messages: list[Any],
        meta: dict[str, Any],
        original_messages: list[Any],
        original_tokens: int,
        state: dict[str, Any],
        reason: str,
    ) -> ContextCompactionResult:
        current_tokens = estimate_message_tokens(messages)
        if (
            not self.config.full_compact_enabled
            or self.llm_client is None
            or current_tokens < self.config.full_compact_trigger_tokens
        ):
            skip_reason = (
                "full_compact_disabled"
                if not self.config.full_compact_enabled
                else "missing_llm_client"
                if self.llm_client is None
                else "below_full_compact_threshold"
            )
            self._emit(
                "full_compact_skipped",
                {
                    "reason": skip_reason,
                    "tokens_before": current_tokens,
                    "trigger_tokens": self.config.full_compact_trigger_tokens,
                    "previous_reason": reason,
                },
            )
            return self._result(messages, meta)

        self._emit(
            "full_compact_start",
            {
                "tokens_before": current_tokens,
                "messages_before": len(messages),
                "reason": reason,
            },
        )
        agent = FullCompactAgent(
            self.llm_client,
            recent_keep_max_tokens=self.config.full_compact_recent_keep_max_tokens,
            max_output_chars=self.config.full_compact_max_output_chars,
        )
        result = agent.compact(messages)
        if not result.compacted:
            failures = int(state.get("context_compaction_failures") or 0) + 1
            meta.update(
                {
                    "full_compact_status": result.status,
                    "full_compact_error": result.error,
                }
            )
            self._emit(
                "full_compact_failure",
                {
                    "reason": result.reason,
                    "error": result.error,
                    "failure_count": failures,
                    "tokens_before": current_tokens,
                },
            )
            compact_result = self._result(messages, meta)
            compact_result.patch["context_compaction_failures"] = failures
            compact_result.patch["context_compaction_last_error"] = result.error
            return compact_result

        final_tokens = estimate_message_tokens(result.messages)
        meta.update(
            {
                "status": result.status,
                "reason": result.reason,
                "final_tokens": final_tokens,
                "final_message_count": len(result.messages),
                "full_compact_status": result.status,
            }
        )
        self._emit(
            "full_compact_success",
            {
                "tokens_before": original_tokens,
                "tokens_after": final_tokens,
                "messages_before": len(original_messages),
                "messages_after": len(result.messages),
                "reason": result.reason,
                "summary_chars": result.summary_chars,
            },
        )
        return self._result(result.messages, meta)

    def _result(
        self,
        messages: list[Any],
        meta: dict[str, Any],
    ) -> ContextCompactionResult:
        return ContextCompactionResult(
            messages=messages,
            meta=meta,
            patch={
                "context_compaction_last_status": meta["status"],
                "context_compaction_last_reason": meta["reason"],
                "context_compaction_original_tokens": meta["original_tokens"],
                "context_compaction_after_microcompact_tokens": meta[
                    "after_microcompact_tokens"
                ],
                "context_compaction_final_tokens": meta["final_tokens"],
                "context_compaction_original_message_count": meta[
                    "original_message_count"
                ],
                "context_compaction_final_message_count": meta[
                    "final_message_count"
                ],
                "context_compaction_last_error": None,
            },
        )

    def _emit(self, event_type: str, content: dict[str, Any]) -> None:
        if not self.observer:
            return
        self.observer.emit(
            event_type,
            output_node="context_compaction",
            input_node="context",
            content=content,
        )
