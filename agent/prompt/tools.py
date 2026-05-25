from __future__ import annotations

from typing import Any

from agent.tools.adapters.langgraph_adapter import state_to_tool_context
from agent.tools.registry import ToolRegistry


class ToolSpecProvider:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def build(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        ctx = state_to_tool_context(state)
        context_snapshot = state.get("context_snapshot", {})
        skill_summaries = context_snapshot.get(
            "skill_summaries",
            state.get("skill_summaries", []),
        )
        conditional_skill_summaries = context_snapshot.get(
            "conditional_skill_summaries",
            state.get("conditional_skill_summaries", []),
        )
        return _filter_skill_tool_specs(
            self.registry.get_model_tool_specs(ctx),
            has_skills=bool(skill_summaries or conditional_skill_summaries),
            active_skill_name=state.get("active_skill_name"),
        )


def _filter_skill_tool_specs(
    tool_specs: list[dict[str, Any]],
    *,
    has_skills: bool,
    active_skill_name: str | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for spec in tool_specs:
        name = spec.get("name")
        if name == "use_skill" and not has_skills:
            continue
        if name == "shell_command" and active_skill_name is None:
            continue
        filtered.append(spec)
    return filtered
