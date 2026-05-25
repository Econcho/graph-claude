from __future__ import annotations

import re


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_mcp_name(name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", name.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def build_mcp_tool_name(server_name: str, tool_name: str) -> str:
    return f"mcp__{sanitize_mcp_name(server_name)}__{sanitize_mcp_name(tool_name)}"
