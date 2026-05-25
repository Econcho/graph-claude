from __future__ import annotations

import re
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.command_tools.run_command import run_command_payload


class ParseTestOutputTool:
    name = "parse_test_output"
    description = "Parse test output into structured diagnostics."
    input_schema = {
        "type": "object",
        "properties": {
            "output": {"type": "string"},
            "kind": {"type": "string", "enum": ["pytest", "generic"], "default": "pytest"},
        },
        "required": ["output"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if not isinstance(tool_input.get("output"), str):
            raise ValueError("output must be a string")
        if tool_input.get("kind", "pytest") not in {"pytest", "generic"}:
            raise ValueError("kind must be pytest or generic")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        diagnostics = parse_pytest_output(tool_input["output"]) if tool_input.get("kind", "pytest") == "pytest" else []
        content = "\n".join(f"{d.get('file')}:{d.get('line')}: {d.get('message')}" for d in diagnostics)
        return ToolResult(True, content or "No diagnostics parsed.", data={"diagnostics": diagnostics})


class CollectDiagnosticsTool:
    name = "collect_diagnostics"
    description = "Run a diagnostic command and parse its output."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "shell": {"type": "string", "enum": ["powershell", "bash"]},
            "timeout_seconds": {"type": "integer", "default": 120},
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
        if not isinstance(tool_input.get("command"), str) or not tool_input["command"].strip():
            raise ValueError("command must be a non-empty string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        result = run_command_payload(tool_input, ctx)
        diagnostics = parse_pytest_output(result.content)
        data = dict(result.data or {})
        data["diagnostics"] = diagnostics
        return ToolResult(result.ok, result.content, data=data, error=result.error)


def parse_pytest_output(output: str) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for line in output.splitlines():
        failed = re.match(r"FAILED\s+([^:\s]+)(?:::([^\s]+))?\s+-\s+(.+)", line)
        if failed:
            diagnostics.append(
                {
                    "file": failed.group(1),
                    "line": None,
                    "test": failed.group(2),
                    "message": failed.group(3)[:500],
                }
            )
            continue
        location = re.match(r"(.+?):(\d+):\s+(.+)", line)
        if location and "site-packages" not in location.group(1):
            diagnostics.append(
                {
                    "file": location.group(1),
                    "line": int(location.group(2)),
                    "message": location.group(3)[:500],
                }
            )
    return diagnostics
