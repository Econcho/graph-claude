from __future__ import annotations

from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path


class WriteFileTool:
    name = "write_file"
    description = "Write UTF-8 text content to a file inside the workspace."

    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Content to write.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite the file if it already exists.",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    aliases: list[str] = []

    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        path = tool_input.get("path")
        content = tool_input.get("content")
        overwrite = tool_input.get("overwrite", False)

        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")

        if not isinstance(content, str):
            raise ValueError("content must be a string")

        if not isinstance(overwrite, bool):
            raise ValueError("overwrite must be a boolean")

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
        target = resolve_workspace_path(
            ctx.workspace_root,
            tool_input["path"],
        )

        overwrite = tool_input.get("overwrite", False)

        if target.exists() and not overwrite:
            return ToolResult(
                ok=False,
                content=(
                    f"File already exists: {tool_input['path']}. "
                    f"Set overwrite=true to replace it."
                ),
                error="file_exists",
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(tool_input["content"], encoding="utf-8")

        return ToolResult(
            ok=True,
            content=f"File written: {tool_input['path']}",
            data={
                "path": str(target),
            },
        )
