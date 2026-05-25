from __future__ import annotations

from typing import Callable

from .base import Tool, ToolContext


class ToolRegistry:
    """
    工具注册表。

    它不关心具体工具是否继承某个父类。
    只要求工具对象满足 Tool Protocol。
    """

    def __init__(self, tools: list[Tool]):
        self._tools = tools
        self._index: dict[str, Tool] = {}

        for tool in tools:
            self._validate_tool_shape(tool)
            self._register_index(tool.name, tool)

            for alias in tool.aliases:
                self._register_index(alias, tool)

    def _validate_tool_shape(self, tool: Tool) -> None:
        required_attrs = [
            "name",
            "description",
            "input_schema",
            "aliases",
            "is_read_only",
            "is_concurrency_safe",
            "is_destructive",
            "is_enabled",
            "validate_input",
            "to_model_spec",
            "run",
        ]

        for attr in required_attrs:
            if not hasattr(tool, attr):
                raise TypeError(f"Invalid tool: missing attribute {attr}")

        if not isinstance(tool.name, str) or not tool.name:
            raise TypeError("Invalid tool: name must be a non-empty string")

        if not isinstance(tool.description, str) or not tool.description:
            raise TypeError("Invalid tool: description must be a non-empty string")

        if not isinstance(tool.input_schema, dict):
            raise TypeError("Invalid tool: input_schema must be a dict")

        if not isinstance(tool.aliases, list):
            raise TypeError("Invalid tool: aliases must be a list")

        if not callable(tool.is_enabled):
            raise TypeError("Invalid tool: is_enabled must be callable")

        if not callable(tool.validate_input):
            raise TypeError("Invalid tool: validate_input must be callable")

        if not callable(tool.to_model_spec):
            raise TypeError("Invalid tool: to_model_spec must be callable")

        if not callable(tool.run):
            raise TypeError("Invalid tool: run must be callable")

    def _register_index(self, name: str, tool: Tool) -> None:
        if name in self._index:
            raise ValueError(f"Duplicate tool name or alias: {name}")

        self._index[name] = tool

    def get_tools(
        self,
        ctx: ToolContext,
        predicate: Callable[[Tool], bool] | None = None,
    ) -> list[Tool]:
        tools = [
            tool
            for tool in self._tools
            if tool.is_enabled(ctx)
        ]

        if predicate is not None:
            tools = [
                tool
                for tool in tools
                if predicate(tool)
            ]

        return tools

    def find_tool(
        self,
        name: str,
        ctx: ToolContext,
    ) -> Tool | None:
        tool = self._index.get(name)

        if tool is None:
            return None

        if not tool.is_enabled(ctx):
            return None

        return tool

    def get_model_tool_specs(self, ctx: ToolContext) -> list[dict]:
        return [
            tool.to_model_spec()
            for tool in self.get_tools(ctx)
        ]


def build_default_registry() -> ToolRegistry:
    from .builtin import BUILTIN_TOOLS

    return ToolRegistry(BUILTIN_TOOLS)
