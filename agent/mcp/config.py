from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.mcp.models import McpConfig, McpServerConfig, McpTransportType


DEFAULT_IDE_ALLOWED_TOOLS = [
    "mcp__ide__getDiagnostics",
    "mcp__ide__getOpenEditorFiles",
]


def default_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.json"


def load_mcp_config(settings_path: str | Path | None = None) -> McpConfig:
    path = Path(settings_path) if settings_path else default_settings_path()
    if not path.exists():
        return McpConfig(enabled=False)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return McpConfig(enabled=False)

    raw_mcp = data.get("mcp", {}) if isinstance(data, dict) else {}
    if not isinstance(raw_mcp, dict):
        return McpConfig(enabled=False)

    raw_servers = raw_mcp.get("servers", {})
    servers: dict[str, McpServerConfig] = {}
    if isinstance(raw_servers, dict):
        for name, raw_server in raw_servers.items():
            server = _parse_server_config(str(name), raw_server)
            if server is not None:
                servers[server.name] = server

    return McpConfig(
        enabled=bool(raw_mcp.get("enabled", True)),
        description_max_chars=int(raw_mcp.get("description_max_chars", 2048)),
        request_timeout_seconds=float(raw_mcp.get("request_timeout_seconds", 30)),
        auth_cache_ttl_seconds=int(raw_mcp.get("auth_cache_ttl_seconds", 900)),
        local_connection_batch_size=int(raw_mcp.get("local_connection_batch_size", 3)),
        remote_connection_batch_size=int(raw_mcp.get("remote_connection_batch_size", 20)),
        ide_allowed_tools=_string_list(
            raw_mcp.get("ide_allowed_tools", DEFAULT_IDE_ALLOWED_TOOLS)
        ),
        servers=servers,
    )


def _parse_server_config(name: str, raw: Any) -> McpServerConfig | None:
    if not isinstance(raw, dict):
        return None

    transport = str(raw.get("type", "stdio"))
    if transport == "streamable-http":
        transport = "streamable_http"

    if transport not in {"stdio", "streamable_http", "http", "sse", "ws", "ws-ide"}:
        transport = "stdio"

    return McpServerConfig(
        name=name,
        enabled=bool(raw.get("enabled", True)),
        type=transport,  # type: ignore[arg-type]
        command=_optional_string(raw.get("command")),
        args=_string_list(raw.get("args", [])),
        env=_string_dict(raw.get("env", {}), expand_env=True),
        url=_optional_string(raw.get("url")),
        headers=_string_dict(raw.get("headers", {}), expand_env=True),
    )


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _string_dict(value: Any, *, expand_env: bool) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, str] = {}
    for key, raw_value in value.items():
        text = str(raw_value)
        if expand_env and text.startswith("$") and len(text) > 1:
            text = os.getenv(text[1:], "")
        result[str(key)] = text
    return result
