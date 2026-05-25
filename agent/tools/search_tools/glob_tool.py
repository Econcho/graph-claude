from __future__ import annotations

from typing import Any

from agent.tools.base import ToolContext, ToolResult

from .common import is_excluded, validate_glob_pattern


class GlobTool:
    name = "glob"
    description = "Find workspace files by a relative glob pattern."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "max_results": {"type": "integer", "default": 200},
        },
        "required": ["pattern"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        pattern = tool_input.get("pattern")
        max_results = tool_input.get("max_results", 200)
        if not isinstance(pattern, str):
            raise ValueError("pattern must be a string")
        validate_glob_pattern(pattern)
        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.workspace_root.resolve()
        pattern = tool_input["pattern"]
        max_results = int(tool_input.get("max_results", 200))
        matches = []
        truncated = False
        for path in root.glob(pattern):
            try:
                rel = path.resolve().relative_to(root)
            except ValueError:
                continue
            if is_excluded(rel):
                continue
            matches.append(rel.as_posix())
            if len(matches) >= max_results:
                truncated = True
                break
        content = "\n".join(matches)
        if truncated:
            content += f"\n...[truncated after {max_results} results]"
        return ToolResult(True, content, data={"matches": matches, "truncated": truncated})
