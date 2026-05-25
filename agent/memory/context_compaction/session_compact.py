from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from agent.memory.context_compaction.token_counter import estimate_message_tokens
from agent.memory.session_memory.store import (
    SESSION_MEMORY_TEMPLATE,
    SessionMemoryStore,
)


SESSION_COMPACT_PREFIX = """This conversation was compacted.

Session Memory:
{session_memory}

Recent messages are preserved verbatim below.
Continue from the preserved recent messages without asking the user to repeat context.
"""


@dataclass(frozen=True)
class SessionCompactResult:
    compacted: bool
    messages: list[Any]
    reason: str
    session_memory_ref: str | None


def session_compact_messages(
    messages: list[Any],
    *,
    workspace_root: str,
    session_id: str | None,
    recent_keep_max_tokens: int = 40_000,
    recent_keep_min_messages: int = 5,
) -> SessionCompactResult:
    store = SessionMemoryStore(
        workspace_root=workspace_root,
        session_id=session_id,
    )
    if not store.exists():
        return SessionCompactResult(
            compacted=False,
            messages=list(messages),
            reason="missing_session_memory",
            session_memory_ref=store.ref,
        )

    session_memory = store.read_or_template().strip()
    if not session_memory or session_memory == SESSION_MEMORY_TEMPLATE.strip():
        return SessionCompactResult(
            compacted=False,
            messages=list(messages),
            reason="empty_session_memory",
            session_memory_ref=store.ref,
        )

    keep_start = _calculate_recent_keep_start(
        messages,
        recent_keep_max_tokens=recent_keep_max_tokens,
        recent_keep_min_messages=recent_keep_min_messages,
    )
    keep_start = _adjust_start_to_preserve_tool_pairs(messages, keep_start)

    recent_messages = list(messages[keep_start:])
    summary_message = HumanMessage(
        content=SESSION_COMPACT_PREFIX.format(session_memory=session_memory)
    )
    return SessionCompactResult(
        compacted=True,
        messages=[summary_message, *recent_messages],
        reason="threshold_exceeded",
        session_memory_ref=store.ref,
    )


def _calculate_recent_keep_start(
    messages: list[Any],
    *,
    recent_keep_max_tokens: int,
    recent_keep_min_messages: int,
) -> int:
    if len(messages) <= recent_keep_min_messages:
        return 0

    start = max(0, len(messages) - recent_keep_min_messages)
    while start > 0:
        candidate = list(messages[start - 1 :])
        if estimate_message_tokens(candidate) > recent_keep_max_tokens:
            break
        start -= 1

    return start


def _adjust_start_to_preserve_tool_pairs(messages: list[Any], start: int) -> int:
    if start <= 0:
        return 0

    needed_tool_call_ids = _tool_result_ids(messages[start:])
    available_tool_call_ids = _assistant_tool_call_ids(messages[start:])
    missing = needed_tool_call_ids - available_tool_call_ids

    if not missing:
        return start

    adjusted = start
    for index in range(start - 1, -1, -1):
        found = _assistant_tool_call_ids([messages[index]])
        if found & missing:
            adjusted = index
            missing -= found
            if not missing:
                break

    return adjusted


def _tool_result_ids(messages: list[Any]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        tool_call_id = _message_attr(message, "tool_call_id")
        if _message_role(message) == "tool" and tool_call_id:
            ids.add(str(tool_call_id))
    return ids


def _assistant_tool_call_ids(messages: list[Any]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        for call in _extract_tool_calls(message):
            call_id = call.get("id") or call.get("tool_call_id")
            if call_id:
                ids.add(str(call_id))
    return ids


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        return message.get("tool_calls", []) or []

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return tool_calls

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        return additional_kwargs.get("tool_calls", []) or []

    return []


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "unknown")

    msg_type = getattr(message, "type", None)
    if msg_type:
        return str(msg_type)

    return type(message).__name__


def _message_attr(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)
