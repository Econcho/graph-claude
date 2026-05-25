from __future__ import annotations

from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path

from .common import is_excluded


class ListDirTool:
    name = "list_dir"
    description = "List files and directories inside the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "default": "."},
            "recursive": {"type": "boolean", "default": False},
            "max_entries": {"type": "integer", "default": 200},
        },
    }
    aliases: list[str] = ["ls"]
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        path = tool_input.get("path", ".")
        recursive = tool_input.get("recursive", False)
        max_entries = tool_input.get("max_entries", 200)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(recursive, bool):
            raise ValueError("recursive must be a boolean")
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_workspace_path(ctx.workspace_root, tool_input.get("path", "."))
        if not target.exists():
            return ToolResult(False, f"Directory not found: {tool_input.get('path', '.')}", error="directory_not_found")
        if not target.is_dir():
            return ToolResult(False, f"Path is not a directory: {tool_input.get('path', '.')}", error="not_a_directory")

        recursive = bool(tool_input.get("recursive", False))
        max_entries = int(tool_input.get("max_entries", 200))
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries = []
        truncated = False
        for path in iterator:
            rel = path.relative_to(ctx.workspace_root.resolve())
            if is_excluded(rel):
                continue
            entries.append(
                {
                    "path": rel.as_posix(),
                    "type": "directory" if path.is_dir() else "file",
                    "size": path.stat().st_size if path.is_file() else None,
                }
            )
            if len(entries) >= max_entries:
                truncated = True
                break

        lines = [f"{item['type']}\t{item['path']}" for item in entries]
        if truncated:
            lines.append(f"...[truncated after {max_entries} entries]")
        return ToolResult(True, "\n".join(lines), data={"entries": entries, "truncated": truncated})
