from __future__ import annotations

from typing import Any

from agent.graph.state import AgentState


def extract_tool_calls(message: Any) -> list[dict[str, Any]]:
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


def tool_orchestrator_node(state: AgentState) -> dict[str, Any]:
    """Select the next raw tool call to execute.

    This node only schedules tool calls. It must not look up tools, validate
    input, check permissions, run hooks, execute tools, or append tool results.

    Future CC-style orchestration can split pending calls into concurrent and
    sequential batches here based on tool.is_concurrency_safe(input). The v0
    strategy is strict FIFO.
    """
    pending = list(state.get("pending_tool_calls") or [])

    if not pending:
        pending = extract_tool_calls(state.get("assistant_message"))

    if not pending:
        return {
            "current_tool_call": None,
            "pending_tool_calls": [],
        }

    return {
        "current_tool_call": pending[0],
        "pending_tool_calls": pending[1:],
        "current_tool_result": None,
    }
