from __future__ import annotations

from agent.prompt.analyzer import with_token_estimate
from agent.prompt.models import PromptSection


SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "<SYSTEM_PROMPT_DYNAMIC_BOUNDARY>"


DEFAULT_SYSTEM_PROMPT = """You are a coding agent.

You can inspect and modify files through tools.
When you need file content, use read_file.
When you need to create or update a file, use write_file.
Do not invent file contents you have not read.
Do not save or rely on Auto Memory for code patterns, architecture, file paths,
project structure, git history, recent changes, debugging steps, CLAUDE.md
content, or temporary task details; inspect the workspace or git history instead.
If the user asks to save a PR list or activity summary, ask what is surprising
or non-obvious and only preserve that part as memory.
If a user task matches an available skill, call use_skill with the skill name
before attempting the task yourself. Do not guess or reconstruct the full skill
instructions from the summary.
"""


class DefaultSystemPromptBuilder:
    def build(self) -> list[PromptSection]:
        return [
            self._section(
                "intro",
                "You are a coding agent.",
            ),
            self._section(
                "system",
                (
                    "You can inspect and modify files through tools.\n"
                    "Do not invent file contents you have not read."
                ),
            ),
            self._section(
                "doing_tasks",
                (
                    "When you need file content, use read_file.\n"
                    "When you need to create or update a file, use write_file."
                ),
            ),
            self._section(
                "actions",
                (
                    "If a user task matches an available skill, call use_skill "
                    "with the skill name before attempting the task yourself."
                ),
            ),
            self._section(
                "using_tools",
                (
                    "Do not guess or reconstruct full skill instructions from "
                    "the summary. Use tools to verify workspace facts."
                ),
            ),
            self._section(
                "tone_style",
                (
                    "Answer directly and keep implementation explanations focused "
                    "on actionable engineering details."
                ),
            ),
            self._section(
                "output_efficiency",
                (
                    "Avoid unnecessary verbosity. Preserve important constraints, "
                    "paths, commands, and verification results."
                ),
            ),
            with_token_estimate(
                PromptSection(
                    name="dynamic_boundary",
                    content=SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
                    source="default",
                    cacheable=True,
                    cache_break=True,
                )
            ),
            self._section(
                "auto_memory_policy",
                (
                    "Do not save or rely on Auto Memory for code patterns, "
                    "architecture, file paths, project structure, git history, "
                    "recent changes, debugging steps, CLAUDE.md content, or "
                    "temporary task details; inspect the workspace or git history "
                    "instead.\n"
                    "If the user asks to save a PR list or activity summary, ask "
                    "what is surprising or non-obvious and only preserve that part "
                    "as memory."
                ),
                cacheable=False,
            ),
        ]

    def _section(
        self,
        name: str,
        content: str,
        *,
        cacheable: bool = True,
        cache_break: bool = False,
    ) -> PromptSection:
        return with_token_estimate(
            PromptSection(
                name=name,
                content=content.strip(),
                source="default",
                cacheable=cacheable,
                cache_break=cache_break,
            )
        )
