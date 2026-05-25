from __future__ import annotations

import re
from typing import Any

from agent.tools.base import ToolContext, ToolResult
from agent.tools.path_utils import resolve_workspace_path
from agent.tools.search_tools.common import is_probably_binary, iter_workspace_files


SUPPORTED_SUFFIXES = {".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".md"}


class OutlineFileTool:
    name = "outline_file"
    description = "Return a heuristic outline of functions/classes/headings in a source file."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        if not isinstance(tool_input.get("path"), str) or not tool_input["path"].strip():
            raise ValueError("path must be a non-empty string")

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_workspace_path(ctx.workspace_root, tool_input["path"])
        if not target.exists() or not target.is_file():
            return ToolResult(False, f"File not found: {tool_input['path']}", error="file_not_found")
        symbols = outline_path(ctx.workspace_root.resolve(), target)
        content = "\n".join(f"{s['line']}: {s['kind']} {s['name']}" for s in symbols)
        return ToolResult(True, content, data={"symbols": symbols, "confidence": "heuristic"})


class FindDefinitionTool:
    name = "find_definition"
    description = "Find heuristic definition candidates for a symbol."
    input_schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "path": {"type": "string", "default": "."}},
        "required": ["symbol"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_symbol_input(tool_input)

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        matches = find_symbol(ctx, tool_input["symbol"], tool_input.get("path", "."), definitions_only=True)
        return _matches_result(matches)


class FindReferencesTool:
    name = "find_references"
    description = "Find heuristic reference candidates for a symbol."
    input_schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "path": {"type": "string", "default": "."}},
        "required": ["symbol"],
    }
    aliases: list[str] = []
    is_read_only = True
    is_concurrency_safe = True
    is_destructive = False

    def is_enabled(self, ctx: ToolContext) -> bool:
        return True

    def validate_input(self, tool_input: dict[str, Any], ctx: ToolContext) -> None:
        _validate_symbol_input(tool_input)

    def to_model_spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
        matches = find_symbol(ctx, tool_input["symbol"], tool_input.get("path", "."), definitions_only=False)
        return _matches_result(matches)


def outline_path(root, path) -> list[dict[str, Any]]:
    if path.suffix not in SUPPORTED_SUFFIXES or is_probably_binary(path):
        return []
    symbols = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        kind = None
        name = None
        if path.suffix in {".py", ".pyi"}:
            match = re.match(r"(async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
            if match:
                kind = "function" if "def" in match.group(1) else "class"
                name = match.group(2)
        elif path.suffix in {".js", ".jsx", ".ts", ".tsx"}:
            match = re.match(r"(export\s+)?(async\s+)?(function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)", stripped)
            if match:
                kind = match.group(3)
                name = match.group(4)
            else:
                match = re.match(r"(export\s+)?(const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", stripped)
                if match:
                    kind = "variable"
                    name = match.group(3)
        elif path.suffix == ".md":
            match = re.match(r"(#+)\s+(.+)", stripped)
            if match:
                kind = f"heading{len(match.group(1))}"
                name = match.group(2)
        if kind and name:
            symbols.append({"path": path.relative_to(root).as_posix(), "line": line_number, "kind": kind, "name": name})
    return symbols


def find_symbol(ctx: ToolContext, symbol: str, start_path: str, *, definitions_only: bool) -> list[dict[str, Any]]:
    root = ctx.workspace_root.resolve()
    start = resolve_workspace_path(root, start_path)
    files = [start] if start.is_file() else list(iter_workspace_files(root, start))
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    matches = []
    for file_path in files:
        if file_path.suffix not in SUPPORTED_SUFFIXES or is_probably_binary(file_path):
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        definition_lines = {item["line"] for item in outline_path(root, file_path) if item["name"] == symbol}
        for line_number, line in enumerate(lines, start=1):
            if not pattern.search(line):
                continue
            if definitions_only and line_number not in definition_lines:
                continue
            matches.append({"path": file_path.relative_to(root).as_posix(), "line": line_number, "preview": line.strip()[:500], "confidence": "heuristic"})
            if len(matches) >= 100:
                return matches
    return matches


def _validate_symbol_input(tool_input: dict[str, Any]) -> None:
    if not isinstance(tool_input.get("symbol"), str) or not tool_input["symbol"].strip():
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(tool_input.get("path", "."), str) or not tool_input.get("path", ".").strip():
        raise ValueError("path must be a non-empty string")


def _matches_result(matches: list[dict[str, Any]]) -> ToolResult:
    content = "\n".join(f"{m['path']}:{m['line']}: {m['preview']}" for m in matches)
    return ToolResult(True, content, data={"matches": matches, "confidence": "heuristic"})
