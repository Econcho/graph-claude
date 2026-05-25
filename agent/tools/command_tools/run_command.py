from __future__ import annotations

import subprocess
import sys
from typing import Any

from agent.tools.base import ToolContext, ToolResult


MAX_STREAM_CHARS = 6_000
DEFAULT_TIMEOUT_SECONDS = 30


class RunCommandTool:
    name = "run_command"
    description = "Run a shell command in the workspace and return stdout/stderr."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "shell": {"type": "string", "enum": ["powershell", "bash"]},
            "timeout_seconds": {"type": "integer", "default": DEFAULT_TIMEOUT_SECONDS},
        },
        "required": ["command"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_command_input(tool_input)

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return run_command_payload(tool_input, ctx)


def run_command_payload(tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = tool_input["command"]
    shell = tool_input.get("shell") or _default_shell()
    timeout = int(tool_input.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    try:
        completed = subprocess.run(
            _shell_argv(shell, command),
            cwd=ctx.workspace_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(False, f"Command timed out after {timeout}s.", error="command_timeout")
    except OSError as error:
        return ToolResult(False, f"Command failed to start: {error}", error="command_start_failed")

    stdout = _truncate(completed.stdout, MAX_STREAM_CHARS)
    stderr = _truncate(completed.stderr, MAX_STREAM_CHARS)
    content = _format_output(stdout, stderr)
    return ToolResult(
        completed.returncode == 0,
        content,
        data={
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "shell": shell,
        },
        error=None if completed.returncode == 0 else "command_failed",
    )


def _validate_command_input(tool_input: dict[str, Any]) -> None:
    command = tool_input.get("command")
    shell = tool_input.get("shell")
    timeout = tool_input.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    if shell is not None and shell not in {"powershell", "bash"}:
        raise ValueError("shell must be powershell or bash")
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive integer")


def _default_shell() -> str:
    return "powershell" if sys.platform.startswith("win") else "bash"


def _shell_argv(shell: str, command: str) -> list[str]:
    if shell == "powershell":
        return ["powershell.exe", "-NoProfile", "-Command", command]
    return ["bash", "-lc", command]


def _format_output(stdout: str, stderr: str) -> str:
    parts: list[str] = []
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append("stderr:\n" + stderr.rstrip())
    return "\n".join(parts).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"
