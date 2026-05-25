from __future__ import annotations

import re
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path

from .common import is_probably_binary, iter_workspace_files


MAX_GREP_FILE_SIZE = 1_000_000


class GrepTool:
    name = "grep"
    description = "Search text files in the workspace and return matching lines."
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
            "case_sensitive": {"type": "boolean", "default": False},
            "max_matches": {"type": "integer", "default": 100},
        },
        "required": ["pattern"],
    }
    aliases: list[str] = ["search_text"]
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        pattern = tool_input.get("pattern")
        path = tool_input.get("path", ".")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("pattern must be a non-empty string")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(tool_input.get("case_sensitive", False), bool):
            raise ValueError("case_sensitive must be a boolean")
        max_matches = tool_input.get("max_matches", 100)
        if not isinstance(max_matches, int) or max_matches <= 0:
            raise ValueError("max_matches must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.workspace_root.resolve()
        start = resolve_workspace_path(root, tool_input.get("path", "."))
        if not start.exists():
            return ToolResult(False, f"Path not found: {tool_input.get('path', '.')}", error="path_not_found")

        flags = 0 if tool_input.get("case_sensitive", False) else re.IGNORECASE
        try:
            regex = re.compile(tool_input["pattern"], flags)
        except re.error as error:
            return ToolResult(False, f"Invalid regex: {error}", error="invalid_regex")

        files = [start] if start.is_file() else list(iter_workspace_files(root, start))
        max_matches = int(tool_input.get("max_matches", 100))
        matches = []
        for file_path in files:
            if file_path.stat().st_size > MAX_GREP_FILE_SIZE or is_probably_binary(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                rel = file_path.relative_to(root).as_posix()
                preview = line.strip()
                matches.append({"path": rel, "line": line_number, "preview": preview[:500]})
                if len(matches) >= max_matches:
                    return _result(matches, truncated=True)
        return _result(matches, truncated=False)


def _result(matches: list[dict[str, Any]], *, truncated: bool) -> ToolResult:
    content = "\n".join(f"{m['path']}:{m['line']}: {m['preview']}" for m in matches)
    if truncated:
        content += f"\n...[truncated after {len(matches)} matches]"
    return ToolResult(True, content, data={"matches": matches, "truncated": truncated})
