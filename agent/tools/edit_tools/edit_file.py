from __future__ import annotations

import difflib
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path


class EditFileTool:
    name = "edit_file"
    description = "Apply exact text replacements to an existing UTF-8 file in the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["old", "new"],
                },
            },
        },
        "required": ["path", "edits"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        path = tool_input.get("path")
        edits = tool_input.get("edits")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits must be a non-empty list")
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise ValueError(f"edits[{index}] must be an object")
            if not isinstance(edit.get("old"), str) or edit.get("old") == "":
                raise ValueError(f"edits[{index}].old must be a non-empty string")
            if not isinstance(edit.get("new"), str):
                raise ValueError(f"edits[{index}].new must be a string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_workspace_path(ctx.workspace_root, tool_input["path"])
        if not target.exists():
            return ToolResult(False, f"File not found: {tool_input['path']}", error="file_not_found")
        if not target.is_file():
            return ToolResult(False, f"Path is not a file: {tool_input['path']}", error="not_a_file")

        original = target.read_text(encoding="utf-8")
        updated = original
        for index, edit in enumerate(tool_input["edits"]):
            old = edit["old"]
            count = updated.count(old)
            if count == 0:
                return ToolResult(False, f"Edit {index} did not match: old text not found.", error="old_text_not_found")
            if count > 1:
                return ToolResult(False, f"Edit {index} matched {count} times; old text must be unique.", error="old_text_not_unique")
            updated = updated.replace(old, edit["new"], 1)

        if updated == original:
            return ToolResult(True, "No changes made.", data={"path": str(target), "changed": False})

        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{tool_input['path']}",
                tofile=f"b/{tool_input['path']}",
            )
        )
        target.write_text(updated, encoding="utf-8")
        return ToolResult(
            True,
            diff,
            data={"path": str(target), "changed": True, "edit_count": len(tool_input["edits"])},
        )
