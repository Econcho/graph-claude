from __future__ import annotations

from typing import Any

from agent.memory.session_memory.store import SessionMemoryStore
from agent.tools.base import ToolContext, ToolResult


class EditSessionMemoryTool:
    name = "edit_session_memory"
    description = "Replace the current session memory markdown."

    input_schema = {
        "type": "object",
        "properties": {
            "updated_session_memory_markdown": {
                "type": "string",
                "description": "Complete updated Session Memory markdown.",
            }
        },
        "required": ["updated_session_memory_markdown"],
    }

    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, store: SessionMemoryStore):
        self.store = store

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        markdown = tool_input.get("updated_session_memory_markdown")
        if not isinstance(markdown, str) or not markdown.strip():
            raise ValueError("updated_session_memory_markdown must be non-empty")

        validation = self.store.validate(markdown)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

    def to_model_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> ToolResult:
        markdown = tool_input["updated_session_memory_markdown"]
        self.store.write(markdown)
        return ToolResult(
            ok=True,
            content="Session Memory updated.",
            data={
                "session_memory_ref": self.store.ref,
                "content_chars": len(markdown),
            },
        )
