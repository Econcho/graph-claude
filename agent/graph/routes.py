from __future__ import annotations

from typing import Literal

from agent.graph.state import AgentState
from agent.graph.nodes.tool_orchestrator_node import extract_tool_calls


def route_after_llm(state: AgentState) -> Literal["tool_orchestrator", "finalize"]:
    if state.get("pending_tool_calls") or extract_tool_calls(state.get("assistant_message")):
        return "tool_orchestrator"
    return "finalize"


def route_after_tool_result(state: AgentState) -> Literal["tool_orchestrator", "context"]:
    if state.get("pending_tool_calls"):
        return "tool_orchestrator"
    return "context"
