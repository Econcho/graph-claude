from __future__ import annotations

from typing import Any

from agent.prompt.analyzer import with_token_estimate
from agent.prompt.models import PromptSection


AUTO_MEMORY_PROMPT_HEADER = """Relevant long-term memory:

These memories are point-in-time observations from prior runs. Use them as
context, but verify current workspace facts with tools before relying on them.
"""

COORDINATOR_PROMPT = """Coordinator mode is active.

Break complex work into delegated tasks, use the agent tool for independent work,
track results, and synthesize the final answer. Do not do all worker tasks
yourself when delegation would reduce context or parallelize safely.
"""


class DynamicPromptSectionBuilder:
    def build(self, state: dict[str, Any]) -> list[PromptSection]:
        sections: list[PromptSection] = []
        context_snapshot = state.get("context_snapshot", {})

        memory_section = self._auto_memory_section(
            context_snapshot.get("relevant_memories", [])
        )
        if memory_section:
            sections.append(memory_section)

        skill_section = self._skills_section(
            context_snapshot.get("skill_summaries", state.get("skill_summaries", [])),
            context_snapshot.get(
                "conditional_skill_summaries",
                state.get("conditional_skill_summaries", []),
            ),
        )
        if skill_section:
            sections.append(skill_section)

        coordinator_section = self._coordinator_section(state)
        if coordinator_section:
            sections.append(coordinator_section)

        notifications_section = self._multi_agent_notifications_section(
            context_snapshot.get(
                "multi_agent_notifications",
                state.get("multi_agent_notifications", []),
            ),
            context_snapshot.get(
                "multi_agent_mailbox_messages",
                state.get("multi_agent_mailbox_messages", []),
            ),
        )
        if notifications_section:
            sections.append(notifications_section)

        return sections

    def _auto_memory_section(
        self,
        recalled_items: list[dict[str, Any]],
    ) -> PromptSection | None:
        if not recalled_items:
            return None

        lines = [AUTO_MEMORY_PROMPT_HEADER.rstrip()]
        for item in recalled_items:
            lines.extend(
                [
                    "",
                    f"- id: {item.get('id')}",
                    f"  type: {item.get('type')}",
                    f"  name: {item.get('name')}",
                    f"  updated_at: {item.get('updated_at')}",
                    f"  description: {item.get('description')}",
                    f"  content: {item.get('content')}",
                    f"  how_to_apply: {item.get('how_to_apply')}",
                ]
            )
        return _dynamic_section("auto_memory.recalled", "\n".join(lines))

    def _skills_section(
        self,
        skill_summaries: list[dict[str, Any]],
        conditional_skill_summaries: list[dict[str, Any]],
    ) -> PromptSection | None:
        if not skill_summaries and not conditional_skill_summaries:
            return None

        lines: list[str] = []
        if skill_summaries:
            lines.append("Available skills:")
            for item in skill_summaries:
                when = item.get("when_to_use")
                suffix = f" When to use: {when}" if when else ""
                lines.append(
                    f"- {item.get('name')}: {item.get('description')}{suffix}"
                )

        if conditional_skill_summaries:
            if lines:
                lines.append("")
            lines.append("Relevant conditional skills for recently touched files:")
            for item in conditional_skill_summaries:
                when = item.get("when_to_use")
                suffix = f" When to use: {when}" if when else ""
                lines.append(
                    f"- {item.get('name')}: {item.get('description')}{suffix}"
                )

        return _dynamic_section("skills.available", "\n".join(lines))

    def _coordinator_section(self, state: dict[str, Any]) -> PromptSection | None:
        if not state.get("coordinator_mode"):
            return None
        return _dynamic_section("multi_agent.coordinator", COORDINATOR_PROMPT)

    def _multi_agent_notifications_section(
        self,
        notifications: list[str],
        mailbox_messages: list[dict[str, Any]],
    ) -> PromptSection | None:
        if not notifications and not mailbox_messages:
            return None

        lines = ["Multi-agent updates:"]
        for notification in notifications:
            lines.append("")
            lines.append(str(notification))
        if mailbox_messages:
            lines.append("")
            lines.append("Mailbox messages:")
            for message in mailbox_messages:
                lines.append(
                    f"- from {message.get('from')}: {message.get('message')}"
                )
        return _dynamic_section("multi_agent.notifications", "\n".join(lines))


def _dynamic_section(name: str, content: str) -> PromptSection:
    return with_token_estimate(
        PromptSection(
            name=name,
            content=content.strip(),
            source="dynamic",
            cacheable=False,
            cache_break=True,
        )
    )
