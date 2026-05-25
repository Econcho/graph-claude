from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any

from agent.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from agent.skills.manager import SkillManager


MAX_COMMAND_OUTPUT_CHARS = 12_000
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30


class UseSkillTool:
    name = "use_skill"
    description = (
        "Load and execute a named Skill. Use this before attempting a task that "
        "matches an available skill summary."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name to load and execute.",
            },
            "args": {
                "type": "string",
                "description": "Optional free-form arguments for the skill.",
                "default": "",
            },
        },
        "required": ["name"],
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = False

    def __init__(self, manager: "SkillManager"):
        self.manager = manager

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        name = tool_input.get("name")
        args = tool_input.get("args", "")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(args, str):
            raise ValueError("args must be a string")

    def to_model_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self.manager.execute_skill(
            name=tool_input["name"],
            args=tool_input.get("args", ""),
            ctx=ctx,
        )


class ShellCommandTool:
    name = "shell_command"
    description = "Execute a shell command in the workspace and return stdout/stderr."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Command text to execute.",
            },
            "shell": {
                "type": "string",
                "description": "Optional shell: powershell or bash.",
                "enum": ["powershell", "bash"],
            },
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
        command = tool_input.get("command")
        shell = tool_input.get("shell")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        if shell is not None and shell not in {"powershell", "bash"}:
            raise ValueError("shell must be powershell or bash")

    def to_model_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = tool_input["command"]
        shell = tool_input.get("shell") or _default_shell()
        argv = _shell_argv(shell, command)

        try:
            completed = subprocess.run(
                argv,
                cwd=ctx.workspace_root,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False,
                content=f"Command timed out after {DEFAULT_COMMAND_TIMEOUT_SECONDS}s.",
                error="command_timeout",
            )
        except OSError as error:
            return ToolResult(
                ok=False,
                content=f"Command failed to start: {error}",
                error="command_start_failed",
            )

        output = _format_output(completed.stdout, completed.stderr)
        return ToolResult(
            ok=completed.returncode == 0,
            content=_truncate(output, MAX_COMMAND_OUTPUT_CHARS),
            data={
                "returncode": completed.returncode,
                "shell": shell,
            },
            error=None if completed.returncode == 0 else "command_failed",
        )


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
