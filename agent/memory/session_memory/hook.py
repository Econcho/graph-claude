from __future__ import annotations

from typing import Any

from agent.memory.session_memory.manager import SessionMemoryManager


def post_llm_session_memory_hook(
    state: dict[str, Any],
    manager: SessionMemoryManager | None,
) -> dict[str, Any]:
    if manager is None:
        return {}

    try:
        return manager.maybe_update(state)
    except Exception as error:
        return {
            "session_memory_update_in_progress": False,
            "session_memory_last_error": str(error),
            "session_memory_last_fork_agent_status": "failed",
        }
