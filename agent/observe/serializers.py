from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


MAX_TEXT_CHARS = 2000


def truncate_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"


def to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, str):
        return truncate_text(obj)

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]

    if hasattr(obj, "content"):
        return {
            "type": type(obj).__name__,
            "content": truncate_text(str(getattr(obj, "content", ""))),
            "tool_calls": to_jsonable(getattr(obj, "tool_calls", None)),
            "additional_kwargs": to_jsonable(getattr(obj, "additional_kwargs", None)),
        }

    return repr(obj)


def summarize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "role": message.get("role"),
            "content": to_jsonable(message.get("content")),
            "tool_calls": to_jsonable(message.get("tool_calls")),
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
        }

    return to_jsonable(message)


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    messages = state.get("messages", [])
    pending_tool_calls = state.get("pending_tool_calls", [])
    tool_results = state.get("tool_results", [])

    last_message = None
    if messages:
        last_message = summarize_message(messages[-1])

    return {
        "message_count": len(messages),
        "last_message": last_message,
        "pending_tool_calls": to_jsonable(pending_tool_calls),
        "pending_tool_call_count": len(pending_tool_calls),
        "current_tool_call": to_jsonable(state.get("current_tool_call")),
        "current_tool_result": to_jsonable(state.get("current_tool_result")),
        "tool_result_count": len(tool_results),
        "final_answer": to_jsonable(state.get("final_answer")),
        "error": to_jsonable(state.get("error")),
    }


def summarize_llm_request(request: dict[str, Any]) -> dict[str, Any]:
    messages = request.get("messages", [])
    tools = request.get("tools", [])

    return {
        "message_count": len(messages),
        "last_message": summarize_message(messages[-1]) if messages else None,
        "tool_names": [
            tool.get("name")
            for tool in tools
            if isinstance(tool, dict)
        ],
        "tool_count": len(tools),
        "system_prompt": to_jsonable(request.get("system_prompt")),
    }


def summarize_llm_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return {
            "role": response.get("role"),
            "content": to_jsonable(response.get("content")),
            "tool_calls": to_jsonable(response.get("tool_calls")),
        }

    return to_jsonable(response)
