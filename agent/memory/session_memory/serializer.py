from __future__ import annotations

import json
from typing import Any


def serialize_messages(
    messages: list[Any],
    *,
    start_index: int = 0,
    max_tool_result_preview_chars: int = 2000,
) -> str:
    chunks: list[str] = []
    for offset, message in enumerate(messages):
        index = start_index + offset
        chunks.append(
            serialize_message(
                message,
                index=index,
                max_tool_result_preview_chars=max_tool_result_preview_chars,
            )
        )
    return "\n\n".join(chunks)


def serialize_message(
    message: Any,
    *,
    index: int,
    max_tool_result_preview_chars: int = 2000,
) -> str:
    role = _message_role(message)
    message_id = _message_id(message)
    content = _message_content(message)
    tool_calls = _extract_tool_calls(message)

    header = f"[message_index={index} role={role}"
    if message_id:
        header += f" id={message_id}"
    header += "]"

    lines = [header]
    if role == "tool":
        lines.append("result_preview:")
        lines.append(_truncate(content, max_tool_result_preview_chars))
        lines.append(f"original_chars: {len(content)}")
        lines.append(f"truncated: {len(content) > max_tool_result_preview_chars}")
        tool_call_id = _message_attr(message, "tool_call_id")
        name = _message_attr(message, "name")
        if tool_call_id:
            lines.append(f"tool_call_id: {tool_call_id}")
        if name:
            lines.append(f"tool_name: {name}")
        return "\n".join(lines)

    if content:
        lines.append(content)

    if tool_calls:
        lines.append("tool_calls:")
        for call in tool_calls:
            call_id = call.get("id") or call.get("tool_call_id") or ""
            name = call.get("name") or ""
            tool_input = call.get("args") or call.get("input") or {}
            lines.append(
                f"- id={call_id} name={name} input={json.dumps(tool_input, ensure_ascii=False)}"
            )

    return "\n".join(lines)


def estimate_message_tokens(messages: list[Any]) -> int:
    serialized = serialize_messages(messages)
    return max(1, len(serialized) // 4) if serialized else 0


def count_tool_calls(messages: list[Any]) -> int:
    return sum(len(_extract_tool_calls(message)) for message in messages)


def latest_assistant_has_no_tool_call(message: Any) -> bool:
    if _message_role(message) not in {"assistant", "ai", "AIMessage"}:
        return False
    return not _extract_tool_calls(message)


def message_id(message: Any, index: int) -> str:
    return _message_id(message) or f"message_index:{index}"


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    if message is None:
        return []

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


def _message_id(message: Any) -> str | None:
    if isinstance(message, dict):
        value = message.get("id")
    else:
        value = getattr(message, "id", None)
    return str(value) if value else None


def _message_content(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content
    return str(content)


def _message_attr(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"
