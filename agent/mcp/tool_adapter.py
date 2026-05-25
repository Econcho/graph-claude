from __future__ import annotations

import json
from typing import Any

from agent.mcp.client import McpProtocolError
from agent.mcp.models import McpToolDefinition
from agent.tools.base import ToolContext, ToolResult


MAX_MCP_RESULT_CHARS = 12_000


class McpToolAdapter:
    aliases: list[str] = []
    is_concurrency_safe = False

    def __init__(
        self,
        *,
        definition: McpToolDefinition,
        call_tool,
    ):
        self.definition = definition
        self._call_tool = call_tool
        self.name = definition.exposed_name
        self.description = definition.description
        self.input_schema = definition.input_schema
        self.is_read_only = bool(definition.annotations.get("readOnlyHint"))
        self.is_destructive = not self.is_read_only

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if not isinstance(tool_input, dict):
            raise ValueError("MCP tool input must be an object")

    def to_model_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            raw_result = self._call_tool(
                self.definition.server_name,
                self.definition.tool_name,
                tool_input,
            )
        except McpProtocolError as error:
            return ToolResult(
                ok=False,
                content=f"MCP tool failed: {error}",
                data={
                    "server": self.definition.server_name,
                    "tool": self.definition.tool_name,
                    "code": error.code,
                    "data": error.data,
                },
                error="mcp_tool_error",
            )
        except Exception as error:
            return ToolResult(
                ok=False,
                content=f"MCP tool failed: {error}",
                data={
                    "server": self.definition.server_name,
                    "tool": self.definition.tool_name,
                },
                error="mcp_tool_error",
            )

        content = _format_mcp_result(raw_result)
        is_error = bool(raw_result.get("isError"))
        return ToolResult(
            ok=not is_error,
            content=_truncate(content, MAX_MCP_RESULT_CHARS),
            data={
                "server": self.definition.server_name,
                "tool": self.definition.tool_name,
                "raw_result": raw_result,
            },
            error="mcp_tool_error" if is_error else None,
        )


def _format_mcp_result(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        if parts:
            return "\n".join(parts)

    structured = result.get("structuredContent")
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False, indent=2)

    return json.dumps(result, ensure_ascii=False, indent=2)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"
