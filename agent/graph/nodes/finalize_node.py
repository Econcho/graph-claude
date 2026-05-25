from __future__ import annotations

from typing import Any

from agent.graph.state import AgentState
from agent.memory.auto_memory import AutoMemoryManager, stop_auto_memory_hook


def _message_content(message: Any) -> str:
    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    return str(content)


def finalize_node(state: AgentState) -> dict[str, Any]:
    return {"final_answer": _message_content(state.get("assistant_message"))}


def make_finalize_node(
    auto_memory_manager: AutoMemoryManager | None = None,
):
    def finalize_node_with_auto_memory(state: AgentState) -> dict[str, Any]:
        final_answer = _message_content(state.get("assistant_message"))
        updated_state = {
            **state,
            "final_answer": final_answer,
        }
        auto_memory_patch = stop_auto_memory_hook(
            updated_state,
            auto_memory_manager,
        )
        return {
            "final_answer": final_answer,
            **auto_memory_patch,
        }

    return finalize_node_with_auto_memory
