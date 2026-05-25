from __future__ import annotations

from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path


class MkdirTool:
    name = "mkdir"
    description = "Create a directory inside the workspace."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_path(tool_input, "path")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_workspace_path(ctx.workspace_root, tool_input["path"])
        target.mkdir(parents=True, exist_ok=True)
        return ToolResult(True, f"Directory created: {tool_input['path']}", data={"path": str(target)})


class MoveFileTool:
    name = "move_file"
    description = "Move or rename a file inside the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["source", "destination"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_path(tool_input, "source")
        _validate_path(tool_input, "destination")
        if not isinstance(tool_input.get("overwrite", False), bool):
            raise ValueError("overwrite must be a boolean")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        source = resolve_workspace_path(ctx.workspace_root, tool_input["source"])
        destination = resolve_workspace_path(ctx.workspace_root, tool_input["destination"])
        if not source.exists():
            return ToolResult(False, f"File not found: {tool_input['source']}", error="file_not_found")
        if not source.is_file():
            return ToolResult(False, f"Path is not a file: {tool_input['source']}", error="not_a_file")
        if destination.exists() and not tool_input.get("overwrite", False):
            return ToolResult(False, f"Destination exists: {tool_input['destination']}", error="destination_exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        return ToolResult(True, f"Moved {tool_input['source']} to {tool_input['destination']}", data={"path": str(destination), "source": str(source), "destination": str(destination)})


class DeleteFileTool:
    name = "delete_file"
    description = "Delete a file inside the workspace."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_path(tool_input, "path")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_workspace_path(ctx.workspace_root, tool_input["path"])
        if not target.exists():
            return ToolResult(False, f"File not found: {tool_input['path']}", error="file_not_found")
        if not target.is_file():
            return ToolResult(False, f"Path is not a file: {tool_input['path']}", error="not_a_file")
        target.unlink()
        return ToolResult(True, f"File deleted: {tool_input['path']}", data={"path": str(target)})


def _validate_path(tool_input: dict[str, Any], key: str) -> None:
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
