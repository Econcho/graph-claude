from __future__ import annotations

from agent.prompt.analyzer import with_token_estimate
from agent.prompt.models import PromptSection


class EffectiveSystemPromptResolver:
    def resolve(
        self,
        *,
        default_sections: list[PromptSection],
        dynamic_sections: list[PromptSection],
        override_system_prompt: str | None = None,
        agent_system_prompt: str | None = None,
        custom_system_prompt: str | None = None,
        append_system_prompt: str | None = None,
    ) -> list[PromptSection]:
        sections: list[PromptSection]

        if _present(override_system_prompt):
            sections = [
                _override_section(
                    "override_system_prompt",
                    override_system_prompt or "",
                    source="override",
                )
            ]
        elif _present(agent_system_prompt):
            sections = [
                _override_section(
                    "agent_system_prompt",
                    agent_system_prompt or "",
                    source="agent",
                )
            ]
        elif _present(custom_system_prompt):
            sections = [
                _override_section(
                    "custom_system_prompt",
                    custom_system_prompt or "",
                    source="custom",
                )
            ]
        else:
            sections = [*default_sections, *dynamic_sections]

        if _present(append_system_prompt):
            sections.append(
                _override_section(
                    "append_system_prompt",
                    append_system_prompt or "",
                    source="append",
                )
            )

        return sections


def _override_section(name: str, content: str, *, source: str) -> PromptSection:
    return with_token_estimate(
        PromptSection(
            name=name,
            content=content.strip(),
            source=source,
            cacheable=False,
            cache_break=True,
        )
    )


def _present(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())
