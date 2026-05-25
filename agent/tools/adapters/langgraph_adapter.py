from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from agent.tools.base import ToolCall, ToolContext, ToolResult


def langgraph_tool_call_to_internal(raw_call: dict[str, Any]) -> ToolCall:
    """Convert a LangChain/LangGraph tool call into the internal tool call."""
    return ToolCall(
        id=raw_call.get("id") or raw_call.get("tool_call_id") or "",
        name=raw_call["name"],
        input=raw_call.get("args") or raw_call.get("input") or {},
    )


def internal_tool_result_to_langgraph_message(
    tool_call: ToolCall,
    result: ToolResult,
) -> ToolMessage:
    """Convert an internal tool result into a LangGraph transcript message."""
    return ToolMessage(
        content=result.content,
        tool_call_id=tool_call.id,
        name=tool_call.name,
        status="success" if result.ok else "error",
    )


def state_to_tool_context(state: dict[str, Any]) -> ToolContext:
    """Convert AgentState into the internal tool runtime context."""
    workspace_root = state.get("workspace_root") or state.get("workspace")
    return ToolContext(
        workspace_root=Path(workspace_root),
        permission_mode=state.get("permission_mode", "default"),
        session_id=state.get("session_id"),
        agent_id=state.get("agent_id"),
        skill_call_depth=int(state.get("skill_call_depth") or 0),
        active_skill_name=state.get("active_skill_name"),
        team_name=state.get("team_name"),
        agent_name=state.get("agent_name"),
        subagent_depth=int(state.get("subagent_depth") or 0),
    )


def internal_model_spec_to_langchain_tool(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal model spec into an OpenAI-compatible tool schema."""
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        },
    }


def internal_model_specs_to_langchain_tools(
    specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [internal_model_spec_to_langchain_tool(spec) for spec in specs]
