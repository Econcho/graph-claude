from __future__ import annotations

import subprocess
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path


DEFAULT_MAX_CHARS = 12_000


class GitStatusTool:
    name = "git_status"
    description = "Show read-only git status for the workspace."
    input_schema = {"type": "object", "properties": {}}
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        return

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        repo = _ensure_repo(ctx)
        if repo:
            return repo
        result = _git(ctx, ["status", "--short"])
        return ToolResult(result.ok, result.content or "Working tree clean.", data=result.data, error=result.error)


class GitDiffTool:
    name = "git_diff"
    description = "Show read-only git diff for the workspace or one path."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "staged": {"type": "boolean", "default": False},
            "max_chars": {"type": "integer", "default": DEFAULT_MAX_CHARS},
        },
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if "path" in tool_input and not isinstance(tool_input["path"], str):
            raise ValueError("path must be a string")
        if not isinstance(tool_input.get("staged", False), bool):
            raise ValueError("staged must be a boolean")
        if not isinstance(tool_input.get("max_chars", DEFAULT_MAX_CHARS), int):
            raise ValueError("max_chars must be an integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        repo = _ensure_repo(ctx)
        if repo:
            return repo
        args = ["diff"]
        if tool_input.get("staged", False):
            args.append("--staged")
        if tool_input.get("path"):
            path = resolve_workspace_path(ctx.workspace_root, tool_input["path"])
            args.extend(["--", path.relative_to(ctx.workspace_root.resolve()).as_posix()])
        result = _git(ctx, args)
        max_chars = int(tool_input.get("max_chars", DEFAULT_MAX_CHARS))
        return _with_truncated_content(result, max_chars)


class GitLogTool:
    name = "git_log"
    description = "Show recent git commits."
    input_schema = {"type": "object", "properties": {"max_count": {"type": "integer", "default": 10}}}
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if not isinstance(tool_input.get("max_count", 10), int) or tool_input.get("max_count", 10) <= 0:
            raise ValueError("max_count must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        repo = _ensure_repo(ctx)
        if repo:
            return repo
        count = int(tool_input.get("max_count", 10))
        return _git(ctx, ["log", f"-n{count}", "--pretty=format:%h%x09%an%x09%ad%x09%s", "--date=short"])


class GitShowTool:
    name = "git_show"
    description = "Show a git revision or object."
    input_schema = {
        "type": "object",
        "properties": {
            "rev": {"type": "string"},
            "max_chars": {"type": "integer", "default": DEFAULT_MAX_CHARS},
        },
        "required": ["rev"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if not isinstance(tool_input.get("rev"), str) or not tool_input["rev"].strip():
            raise ValueError("rev must be a non-empty string")
        if not isinstance(tool_input.get("max_chars", DEFAULT_MAX_CHARS), int):
            raise ValueError("max_chars must be an integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        repo = _ensure_repo(ctx)
        if repo:
            return repo
        result = _git(ctx, ["show", "--stat", "--patch", tool_input["rev"]])
        return _with_truncated_content(result, int(tool_input.get("max_chars", DEFAULT_MAX_CHARS)))


def _ensure_repo(ctx: ToolContext) -> ToolResult | None:
    result = _git(ctx, ["rev-parse", "--is-inside-work-tree"])
    if result.ok and result.content.strip() == "true":
        return None
    return ToolResult(False, "Workspace is not a git repository.", error="not_git_repository")


def _git(ctx: ToolContext, args: list[str]) -> ToolResult:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ctx.workspace_root), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
    except OSError as error:
        return ToolResult(False, f"git failed to start: {error}", error="git_unavailable")
    content = completed.stdout.strip()
    if completed.stderr.strip():
        content = f"{content}\nstderr:\n{completed.stderr.strip()}".strip()
    return ToolResult(
        completed.returncode == 0,
        content,
        data={"returncode": completed.returncode},
        error=None if completed.returncode == 0 else "git_failed",
    )


def _with_truncated_content(result: ToolResult, max_chars: int) -> ToolResult:
    content = result.content
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + f"\n...[truncated {len(result.content) - max_chars} chars]"
    data = dict(result.data or {})
    data["truncated"] = truncated
    return ToolResult(result.ok, content, data=data, error=result.error)
