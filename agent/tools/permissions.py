from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Literal

from agent.tools.base import Tool, ToolContext


PermissionBehavior = Literal["allow", "ask", "deny"]
PermissionAsker = Callable[[Tool, dict[str, Any], "PermissionDecision"], bool]


def default_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.json"


@dataclass(frozen=True)
class PermissionRule:
    behavior: PermissionBehavior
    tool_name: str
    pattern: str | None = None
    raw: str = ""

    def matches(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        if not fnmatch(tool_name, self.tool_name):
            return False

        if self.pattern is None:
            return True

        candidate_keys = ("path", "source", "destination")
        for key in candidate_keys:
            path = tool_input.get(key)
            if not isinstance(path, str):
                continue
            normalized_path = path.replace("\\", "/")
            if fnmatch(normalized_path, self.pattern):
                return True
        paths = tool_input.get("paths")
        if isinstance(paths, list):
            for path in paths:
                if isinstance(path, str) and fnmatch(path.replace("\\", "/"), self.pattern):
                    return True
        return False


@dataclass(frozen=True)
class PermissionRules:
    allow: list[PermissionRule] = field(default_factory=list)
    ask: list[PermissionRule] = field(default_factory=list)
    deny: list[PermissionRule] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "PermissionRules":
        return cls()


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str | None = None
    source: str | None = None
    updated_input: dict[str, Any] | None = None


def parse_permission_rule(raw_rule: Any, behavior: PermissionBehavior) -> PermissionRule:
    if not isinstance(raw_rule, str):
        raise ValueError("permission rule must be a string")

    rule = raw_rule.strip()
    if not rule:
        raise ValueError("permission rule must not be empty")

    if "(" not in rule and ")" not in rule:
        return PermissionRule(
            behavior=behavior,
            tool_name=rule,
            raw=rule,
        )

    if not rule.endswith(")") or rule.count("(") != 1:
        raise ValueError(f"invalid permission rule: {rule}")

    tool_name, pattern = rule[:-1].split("(", 1)
    tool_name = tool_name.strip()
    pattern = pattern.strip().replace("\\", "/")

    if not tool_name or not pattern:
        raise ValueError(f"invalid permission rule: {rule}")

    return PermissionRule(
        behavior=behavior,
        tool_name=tool_name,
        pattern=pattern,
        raw=rule,
    )


def load_permission_rules(settings_path: str | Path) -> PermissionRules:
    path = Path(settings_path)
    if not path.exists():
        return PermissionRules.empty()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        return PermissionRules(warnings=[f"Failed to read {path}: {error}"])

    permissions = data.get("permissions", {}) if isinstance(data, dict) else {}
    if not isinstance(permissions, dict):
        return PermissionRules.empty()

    warnings: list[str] = []
    parsed: dict[PermissionBehavior, list[PermissionRule]] = {
        "allow": [],
        "ask": [],
        "deny": [],
    }

    for behavior in ("allow", "ask", "deny"):
        rules = permissions.get(behavior, [])
        if not isinstance(rules, list):
            warnings.append(f"permissions.{behavior} must be a list")
            continue

        for raw_rule in rules:
            try:
                parsed[behavior].append(parse_permission_rule(raw_rule, behavior))
            except ValueError as error:
                warnings.append(str(error))

    return PermissionRules(
        allow=parsed["allow"],
        ask=parsed["ask"],
        deny=parsed["deny"],
        warnings=warnings,
    )


def evaluate_permission_rules(
    rules: PermissionRules,
    tool: Tool,
    tool_input: dict[str, Any],
) -> PermissionDecision | None:
    for behavior, rule_list in (
        ("deny", rules.deny),
        ("ask", rules.ask),
        ("allow", rules.allow),
    ):
        for rule in rule_list:
            if rule.matches(tool.name, tool_input):
                return PermissionDecision(
                    behavior=behavior,
                    reason=f"Matched permission rule: {rule.raw}",
                    source=f"rule:{behavior}:{rule.raw}",
                )

    return None


def default_cli_permission_asker(
    tool: Tool,
    tool_input: dict[str, Any],
    decision: PermissionDecision,
) -> bool:
    print()
    print("Permission required")
    print()
    print("Tool:")
    print(f"  {tool.name}")
    print()
    print("Input:")
    for key, value in tool_input.items():
        print(f"  {key}: {_format_cli_value(value)}")
    print()
    print("Reason:")
    print(f"  {decision.reason or f'Tool {tool.name} requires permission.'}")
    print()

    if not sys.stdin.isatty():
        return False

    answer = input("Approve? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def _format_cli_value(value: Any, limit: int = 500) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)

    if len(text) <= limit:
        return text

    omitted = len(text) - limit
    return f"{text[:limit]}...[truncated {omitted} chars]"
