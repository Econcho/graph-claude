from __future__ import annotations

from typing import Any

from agent.graph.state import AgentState
from agent.observe.observer import RuntimeObserver
from agent.tools.adapters.langgraph_adapter import internal_tool_result_to_langgraph_message
from agent.tools.base import ToolCall, ToolResult


def _dict_to_tool_call(raw: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=raw["id"],
        name=raw["name"],
        input=raw.get("args") or raw.get("input") or {},
    )


def _dict_to_tool_result(raw: dict[str, Any]) -> ToolResult:
    return ToolResult(
        ok=bool(raw.get("ok")),
        content=raw.get("content", ""),
        data=raw.get("data"),
        error=raw.get("error"),
    )


def make_tool_result_node(observer: RuntimeObserver | None = None):
    def tool_result_node(state: AgentState) -> dict[str, Any]:
        raw_call = state.get("current_tool_call")
        raw_result = state.get("current_tool_result")

        if not raw_call or not raw_result:
            return {}

        tool_call = _dict_to_tool_call(raw_call)
        tool_result = _dict_to_tool_result(raw_result)
        tool_message = internal_tool_result_to_langgraph_message(tool_call, tool_result)

        if observer:
            observer.on_tool_result_appended(
                tool_call=tool_call,
                tool_result=tool_result,
                tool_message=tool_message,
            )

        previous_results = state.get("tool_results", [])
        touched_paths = list(state.get("skill_touched_paths") or [])
        data = raw_result.get("data") if isinstance(raw_result, dict) else None
        if isinstance(data, dict) and isinstance(data.get("path"), str):
            touched_paths.append(data["path"])

        return {
            "messages": [tool_message],
            "tool_results": previous_results + [raw_result],
            "skill_touched_paths": touched_paths,
            "current_tool_call": None,
            "current_tool_result": None,
        }

    return tool_result_node


tool_result_node = make_tool_result_node()
