from __future__ import annotations

from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path


class ReadManyFilesTool:
    name = "read_many_files"
    description = "Read multiple UTF-8 text files from the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}},
            "max_files": {"type": "integer", "default": 20},
            "max_chars_per_file": {"type": "integer", "default": 8000},
        },
        "required": ["paths"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        paths = tool_input.get("paths")
        max_files = tool_input.get("max_files", 20)
        max_chars = tool_input.get("max_chars_per_file", 8000)
        if not isinstance(paths, list) or not paths:
            raise ValueError("paths must be a non-empty list")
        if not all(isinstance(path, str) and path.strip() for path in paths):
            raise ValueError("paths must contain non-empty strings")
        if not isinstance(max_files, int) or max_files <= 0:
            raise ValueError("max_files must be a positive integer")
        if not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars_per_file must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        max_files = int(tool_input.get("max_files", 20))
        max_chars = int(tool_input.get("max_chars_per_file", 8000))
        records = []
        for raw_path in tool_input["paths"][:max_files]:
            record: dict[str, Any] = {"path": raw_path, "ok": False}
            try:
                target = resolve_workspace_path(ctx.workspace_root, raw_path)
                if not target.exists():
                    record["error"] = "file_not_found"
                elif not target.is_file():
                    record["error"] = "not_a_file"
                else:
                    content = target.read_text(encoding="utf-8")
                    record.update(
                        {
                            "ok": True,
                            "content": content[:max_chars],
                            "truncated": len(content) > max_chars,
                        }
                    )
            except Exception as error:
                record["error"] = str(error)
            records.append(record)
        content = "\n\n".join(
            f"## {item['path']}\n{item.get('content', item.get('error', ''))}"
            for item in records
        )
        return ToolResult(True, content, data={"files": records})
