from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agent.tools.base import Tool, ToolCall, ToolContext, ToolInputError, ToolResult
from agent.tools.permissions import (
    PermissionAsker,
    PermissionDecision,
    PermissionRules,
    default_cli_permission_asker,
    default_settings_path,
    evaluate_permission_rules,
    load_permission_rules,
)
from agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from agent.observe.observer import RuntimeObserver


@dataclass(frozen=True)
class HookResult:
    behavior: Literal["allow", "ask", "deny", "none"] = "none"
    reason: str | None = None
    source: str | None = None
    updated_input: dict[str, Any] | None = None
    additional_context: str | None = None


def normalize_raw_tool_call(raw: dict[str, Any] | ToolCall) -> ToolCall:
    if isinstance(raw, ToolCall):
        return raw

    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ToolInputError("tool_call.name is required")

    return ToolCall(
        id=str(raw.get("id") or raw.get("tool_call_id") or ""),
        name=name,
        input=raw.get("args") or raw.get("input") or {},
    )


def tool_result_to_dict(result: ToolResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "content": result.content,
        "data": result.data,
        "error": result.error,
    }


class ToolExecutor:
    """Execute one raw tool_call through a minimal CC-style runToolUse pipeline."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        permission_rules: PermissionRules | None = None,
        settings_path: str | Path | None = None,
        permission_asker: PermissionAsker | None = None,
        observer: "RuntimeObserver | None" = None,
    ):
        self.registry = registry
        self.permission_rules = permission_rules
        self.settings_path = Path(settings_path) if settings_path else None
        self.permission_asker = permission_asker or default_cli_permission_asker
        self.observer = observer

    def execute(
        self,
        raw_tool_call: dict[str, Any] | ToolCall,
        ctx: ToolContext,
    ) -> ToolResult:
        try:
            tool_call = normalize_raw_tool_call(raw_tool_call)
        except Exception as error:
            return ToolResult(
                ok=False,
                content=f"Invalid tool call: {error}",
                error="invalid_tool_call",
            )

        tool = self.registry.find_tool(tool_call.name, ctx)
        if tool is None:
            return ToolResult(
                ok=False,
                content=f"Unknown or unavailable tool: {tool_call.name}",
                error="tool_not_found",
            )

        try:
            self._validate_schema(tool, tool_call.input)
            tool.validate_input(tool_call.input, ctx)
        except Exception as error:
            return ToolResult(
                ok=False,
                content=f"Invalid input for tool {tool.name}: {error}",
                error="invalid_tool_input",
            )

        normalized_input = self.backfill_input(tool, tool_call.input, ctx)
        hook_results = self.run_pre_tool_use_hooks(tool, normalized_input, ctx)

        for hook_result in hook_results:
            if hook_result.updated_input is not None:
                normalized_input = hook_result.updated_input
            if hook_result.behavior == "deny":
                return ToolResult(
                    ok=False,
                    content=hook_result.reason or f"Tool {tool.name} denied by hook.",
                    error="hook_denied",
                )

        permission = self.evaluate_permission(tool, normalized_input, ctx, hook_results)
        if permission.updated_input is not None:
            normalized_input = permission.updated_input

        approved: bool | None = None
        if permission.behavior == "ask":
            approved = self.permission_asker(tool, normalized_input, permission)
            self._emit_permission_decision(tool, permission, approved)
            if not approved:
                return ToolResult(
                    ok=False,
                    content=(
                        "Permission denied by user: "
                        f"{permission.reason or f'Tool {tool.name} requires permission.'}"
                    ),
                    error="permission_denied",
                    data={
                        "permission_source": permission.source,
                    },
                )

        if permission.behavior == "deny":
            self._emit_permission_decision(tool, permission, None)
            return ToolResult(
                ok=False,
                content=permission.reason or f"Tool {tool.name} denied by permission policy.",
                error="permission_denied",
                data={
                    "permission_source": permission.source,
                },
            )

        if permission.behavior != "ask":
            self._emit_permission_decision(tool, permission, approved)

        try:
            result = tool.run(normalized_input, ctx)
        except Exception as error:
            return ToolResult(
                ok=False,
                content=f"Tool {tool.name} failed: {error}",
                error="tool_execution_error",
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                content=f"Invalid tool result type: {type(result).__name__}",
                error="invalid_tool_result",
            )

        return result

    def _validate_schema(self, tool: Tool, tool_input: dict[str, Any]) -> None:
        required = tool.input_schema.get("required", [])

        for key in required:
            if key not in tool_input:
                raise ToolInputError(f"Missing required field: {key}")

    def backfill_input(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> dict[str, Any]:
        return dict(tool_input)

    def run_pre_tool_use_hooks(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        ctx: ToolContext,
    ) -> list[HookResult]:
        return []

    def evaluate_permission(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        ctx: ToolContext,
        hook_results: list[HookResult],
    ) -> PermissionDecision:
        rules_decision = evaluate_permission_rules(
            self._load_permission_rules(ctx),
            tool,
            tool_input,
        )
        if rules_decision is not None:
            return rules_decision

        for hook_result in hook_results:
            if hook_result.behavior == "ask":
                return PermissionDecision(
                    behavior="ask",
                    reason=hook_result.reason,
                    source=hook_result.source or "hook",
                    updated_input=hook_result.updated_input,
                )

        mode = ctx.permission_mode or "default"

        if mode == "plan":
            if tool.is_read_only:
                return PermissionDecision("allow", source="permission_mode:plan")
            return PermissionDecision(
                "deny",
                reason=f"Tool {tool.name} is not allowed in plan mode.",
                source="permission_mode:plan",
            )

        if mode == "accept_edits":
            if tool.is_read_only or tool.name in {
                "write_file",
                "edit_file",
                "mkdir",
                "move_file",
                "delete_file",
            }:
                return PermissionDecision("allow", source="permission_mode:accept_edits")
            return PermissionDecision(
                "ask",
                reason=f"Tool {tool.name} requires permission.",
                source="permission_mode:accept_edits",
            )

        if tool.is_destructive:
            return PermissionDecision(
                "ask",
                reason=f"Tool {tool.name} is destructive and requires permission.",
                source="permission_mode:default",
            )

        return PermissionDecision("allow", source="permission_mode:default")

    def _load_permission_rules(self, ctx: ToolContext) -> PermissionRules:
        if self.permission_rules is not None:
            return self.permission_rules

        settings_path = self.settings_path or default_settings_path()
        return load_permission_rules(settings_path)

    def _emit_permission_decision(
        self,
        tool: Tool,
        decision: PermissionDecision,
        approved: bool | None,
    ) -> None:
        if self.observer is None or not hasattr(self.observer, "on_tool_permission_decision"):
            return
        self.observer.on_tool_permission_decision(
            tool_name=tool.name,
            behavior=decision.behavior,
            source=decision.source,
            reason=decision.reason,
            approved=approved,
            is_read_only=tool.is_read_only,
            is_destructive=tool.is_destructive,
        )
