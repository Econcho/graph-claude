from __future__ import annotations

from typing import Any

from ..base import ToolContext, ToolResult
from ..path_utils import resolve_workspace_path


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file inside the workspace."

    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path inside the workspace.",
            }
        },
        "required": ["path"],
    }

    aliases: list[str] = []

    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(
        self,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> None:
        path = tool_input.get("path")

        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")

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

        if not target.exists():
            return ToolResult(
                ok=False,
                content=f"File not found: {tool_input['path']}",
                error="file_not_found",
            )

        if not target.is_file():
            return ToolResult(
                ok=False,
                content=f"Path is not a file: {tool_input['path']}",
                error="not_a_file",
            )

        content = target.read_text(encoding="utf-8")

        return ToolResult(
            ok=True,
            content=content,
            data={
                "path": str(target),
            },
        )