from __future__ import annotations

from typing import Any

from agent.memory.auto_memory.manager import AutoMemoryManager


def stop_auto_memory_hook(
    state: dict[str, Any],
    manager: AutoMemoryManager | None,
) -> dict[str, Any]:
    if manager is None:
        return {}

    try:
        return manager.maybe_extract(state)
    except Exception as error:
        return {
            "auto_memory_extract_in_progress": False,
            "auto_memory_last_error": str(error),
            "auto_memory_last_extract_status": "failed",
        }
