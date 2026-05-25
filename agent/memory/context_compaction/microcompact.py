from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import ToolMessage


CLEARED_TOOL_RESULT_CONTENT = "[Old tool result content cleared by microcompact]"


@dataclass(frozen=True)
class MicrocompactResult:
    messages: list[Any]
    compacted_tool_result_count: int


def microcompact_messages(
    messages: list[Any],
    *,
    keep_recent_tool_results: int = 2,
) -> MicrocompactResult:
    tool_indexes = [
        index for index, message in enumerate(messages) if _is_tool_message(message)
    ]
    if not tool_indexes:
        return MicrocompactResult(
            messages=list(messages),
            compacted_tool_result_count=0,
        )

    keep_count = max(0, keep_recent_tool_results)
    keep = set(tool_indexes[-keep_count:]) if keep_count else set()
    compacted: list[Any] = []
    compacted_count = 0

    for index, message in enumerate(messages):
        if index in keep or not _is_tool_message(message):
            compacted.append(message)
            continue

        compacted.append(_replace_tool_content(message, CLEARED_TOOL_RESULT_CONTENT))
        compacted_count += 1

    return MicrocompactResult(
        messages=compacted,
        compacted_tool_result_count=compacted_count,
    )


def _is_tool_message(message: Any) -> bool:
    if isinstance(message, dict):
        role = message.get("role") or message.get("type")
        return role == "tool"

    return getattr(message, "type", None) == "tool"


def _replace_tool_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated

    if isinstance(message, ToolMessage):
        kwargs = {
            "content": content,
            "tool_call_id": getattr(message, "tool_call_id", ""),
            "name": getattr(message, "name", None),
        }
        status = getattr(message, "status", None)
        if status:
            kwargs["status"] = status
        return ToolMessage(**kwargs)

    if hasattr(message, "copy"):
        try:
            return message.copy(update={"content": content})
        except Exception:
            pass

    return message
