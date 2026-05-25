from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from agent.graph.state import AgentState
from agent.observe.observer import RuntimeObserver
from agent.tools.adapters.langgraph_adapter import state_to_tool_context
from agent.tools.executor import ToolExecutor


def _tool_result_to_dict(result: Any) -> dict[str, Any]:
    if is_dataclass(result):
        return asdict(result)

    if isinstance(result, dict):
        return result

    return {
        "ok": False,
        "content": f"Invalid tool result type: {type(result).__name__}",
        "data": None,
        "error": "invalid_tool_result",
    }


def make_tool_executor_node(
    tool_executor: ToolExecutor,
    observer: RuntimeObserver | None = None,
):
    def tool_executor_node(state: AgentState) -> dict[str, Any]:
        raw_call = state.get("current_tool_call")
        if raw_call is None:
            return {
                "current_tool_result": None,
            }

        ctx = state_to_tool_context(state)

        if observer:
            observer.on_tool_call_start(raw_call)

        try:
            result = tool_executor.execute(raw_call, ctx)
        except Exception as error:
            if observer:
                observer.on_tool_call_error(raw_call, error)
            raise

        if observer:
            observer.on_tool_call_end(raw_call, result)

        return {
            "current_tool_result": _tool_result_to_dict(result),
        }

    return tool_executor_node
