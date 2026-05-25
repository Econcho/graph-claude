from __future__ import annotations

import re
from typing import Any

from agent.tools.base import ToolContext, ToolResult

from .run_command import run_command_payload


class RunTestsTool:
    name = "run_tests"
    description = "Run the project test command and return a concise test summary."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "default": "uv run pytest"},
            "shell": {"type": "string", "enum": ["powershell", "bash"]},
            "timeout_seconds": {"type": "integer", "default": 120},
        },
    }
    aliases: list[str] = []
    is_read_only = False
    is_concurrency_safe = False
    is_destructive = True

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        command = tool_input.get("command", "uv run pytest")
        timeout = tool_input.get("timeout_seconds", 120)
        shell = tool_input.get("shell")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        if shell is not None and shell not in {"powershell", "bash"}:
            raise ValueError("shell must be powershell or bash")
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive integer")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        payload = {
            "command": tool_input.get("command", "uv run pytest"),
            "shell": tool_input.get("shell"),
            "timeout_seconds": tool_input.get("timeout_seconds", 120),
        }
        result = run_command_payload(payload, ctx)
        summary = parse_pytest_summary(result.content)
        data = dict(result.data or {})
        data["summary"] = summary
        content = result.content
        if summary:
            content = f"Test summary: {summary}\n\n{content}".strip()
        return ToolResult(result.ok, content, data=data, error=result.error)


def parse_pytest_summary(output: str) -> str | None:
    patterns = [
        r"=+\s*(\d+ passed(?:, \d+ skipped)?(?:, \d+ failed)?[^=]*)\s*=+",
        r"=+\s*(\d+ failed[^=]*)\s*=+",
        r"=+\s*(no tests ran[^=]*)\s*=+",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None
