from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


McpTransportType = Literal["stdio", "streamable_http", "http", "sse", "ws", "ws-ide"]


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    enabled: bool
    type: McpTransportType
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class McpConfig:
    enabled: bool = True
    description_max_chars: int = 2048
    request_timeout_seconds: float = 30
    auth_cache_ttl_seconds: int = 900
    local_connection_batch_size: int = 3
    remote_connection_batch_size: int = 20
    ide_allowed_tools: list[str] = field(default_factory=list)
    servers: dict[str, McpServerConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class McpToolDefinition:
    server_name: str
    tool_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)
