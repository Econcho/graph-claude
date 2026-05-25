from __future__ import annotations


class SpecializedPromptRegistry:
    """Registry marker for task-specific prompts.

    Full compact, session memory, and auto memory extraction already own their
    prompts in their modules. This registry keeps the boundary explicit so those
    prompts do not get folded into the main system prompt.
    """

    prompt_names = {
        "full_compact",
        "session_memory_update",
        "auto_memory_extract",
        "skill_execution",
    }

    def has(self, name: str) -> bool:
        return name in self.prompt_names
