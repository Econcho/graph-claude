from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agent.mcp.auth_cache import McpAuthCache
from agent.mcp.client import McpConnectionError, McpProtocolError, StdioMcpClient
from agent.mcp.config import load_mcp_config
from agent.mcp.models import McpConfig, McpServerConfig, McpToolDefinition
from agent.mcp.names import build_mcp_tool_name
from agent.mcp.tool_adapter import McpToolAdapter
from agent.observe.observer import RuntimeObserver
from agent.tools.base import Tool


class McpManager:
    def __init__(
        self,
        *,
        config: McpConfig | None = None,
        settings_path: str | Path | None = None,
        observer: RuntimeObserver | None = None,
        workspace_root: str | Path | None = None,
        auth_cache: McpAuthCache | None = None,
    ):
        self.config = config or load_mcp_config(settings_path)
        self.observer = observer
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.auth_cache = auth_cache or McpAuthCache(
            ttl_seconds=self.config.auth_cache_ttl_seconds
        )
        self._clients: dict[str, StdioMcpClient] = {}
        self._client_fingerprints: dict[str, str] = {}

    def discover_tools(self, *, skip_names: set[str] | None = None) -> list[Tool]:
        skip_names = skip_names or set()
        self._emit(
            "mcp_config_loaded",
            output_node="settings",
            input_node="mcp",
            content={
                "enabled": self.config.enabled,
                "server_count": len(self.config.servers),
            },
        )

        if not self.config.enabled:
            return []

        tools: list[Tool] = []
        seen = set(skip_names)
        for server in self.config.servers.values():
            if not server.enabled:
                continue

            try:
                definitions = self._discover_server_tools(server)
            except Exception as error:
                self._emit(
                    "mcp_server_connect_error",
                    output_node="mcp",
                    input_node="tool_registry",
                    content={
                        "server": server.name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                continue

            for definition in definitions:
                if definition.exposed_name in seen:
                    continue
                if (
                    definition.server_name == "ide"
                    and definition.exposed_name not in self.config.ide_allowed_tools
                ):
                    continue
                seen.add(definition.exposed_name)
                tools.append(
                    McpToolAdapter(
                        definition=definition,
                        call_tool=self.call_tool,
                    )
                )
        return tools

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        server = self.config.servers.get(server_name)
        if server is None:
            raise McpConnectionError(f"Unknown MCP server: {server_name}")

        self._emit(
            "mcp_tool_call_start",
            output_node="tool_executor",
            input_node="mcp",
            content={
                "server": server_name,
                "tool": tool_name,
                "arguments": arguments,
            },
        )

        try:
            client = self._ensure_client(server)
            result = client.call_tool(tool_name, arguments)
        except McpProtocolError as error:
            if _is_session_expired(error):
                self._drop_client(server.name)
                self._emit(
                    "mcp_reconnect",
                    output_node="mcp",
                    input_node="mcp",
                    content={
                        "server": server_name,
                        "reason": "session_expired",
                    },
                )
                client = self._ensure_client(server)
                result = client.call_tool(tool_name, arguments)
            else:
                self._record_auth_if_needed(server.name, error)
                self._emit_tool_error(server_name, tool_name, error)
                raise
        except Exception as error:
            self._emit_tool_error(server_name, tool_name, error)
            raise

        self._emit(
            "mcp_tool_call_end",
            output_node="mcp",
            input_node="tool_result",
            content={
                "server": server_name,
                "tool": tool_name,
                "result": result,
            },
        )
        if self.observer and hasattr(self.observer, "on_mcp_used"):
            self.observer.on_mcp_used(
                server=server_name,
                tool=tool_name,
                ok=not bool(result.get("isError")),
                result=result,
            )
        return result

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients = {}
        self._client_fingerprints = {}

    def _discover_server_tools(
        self,
        server: McpServerConfig,
    ) -> list[McpToolDefinition]:
        self._emit(
            "mcp_server_connect_start",
            output_node="mcp",
            input_node="mcp",
            content={
                "server": server.name,
                "transport": server.type,
            },
        )
        if self.auth_cache.needs_auth(server.name):
            raise McpConnectionError(f"MCP server needs auth: {server.name}")

        client = self._ensure_client(server)
        raw_tools = client.list_tools()
        definitions = [
            self._definition_from_raw(server.name, raw_tool)
            for raw_tool in raw_tools
        ]

        self._emit(
            "mcp_server_connect_end",
            output_node="mcp",
            input_node="tool_registry",
            content={
                "server": server.name,
                "transport": server.type,
                "tool_count": len(definitions),
            },
        )
        for definition in definitions:
            self._emit(
                "mcp_tool_discovered",
                output_node="mcp",
                input_node="tool_registry",
                content={
                    "server": server.name,
                    "tool": definition.tool_name,
                    "exposed_name": definition.exposed_name,
                    "read_only": bool(definition.annotations.get("readOnlyHint")),
                },
            )
        return definitions

    def _ensure_client(self, server: McpServerConfig) -> StdioMcpClient:
        fingerprint = _server_fingerprint(server)
        client = self._clients.get(server.name)
        if client is not None and self._client_fingerprints.get(server.name) == fingerprint:
            return client

        if server.type != "stdio":
            raise McpConnectionError(f"unsupported_transport:{server.type}")

        client = StdioMcpClient(
            server,
            timeout_seconds=self.config.request_timeout_seconds,
            cwd=self.workspace_root,
        )
        client.connect()
        self._clients[server.name] = client
        self._client_fingerprints[server.name] = fingerprint
        return client

    def _drop_client(self, server_name: str) -> None:
        client = self._clients.pop(server_name, None)
        self._client_fingerprints.pop(server_name, None)
        if client is not None:
            client.close()

    def _definition_from_raw(
        self,
        server_name: str,
        raw_tool: dict[str, Any],
    ) -> McpToolDefinition:
        tool_name = str(raw_tool.get("name") or "unnamed")
        description = str(raw_tool.get("description") or "")
        input_schema = raw_tool.get("inputSchema") or raw_tool.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}

        annotations = raw_tool.get("annotations", {})
        if not isinstance(annotations, dict):
            annotations = {}

        return McpToolDefinition(
            server_name=server_name,
            tool_name=tool_name,
            exposed_name=build_mcp_tool_name(server_name, tool_name),
            description=_truncate(description, self.config.description_max_chars),
            input_schema=input_schema,
            annotations=annotations,
        )

    def _record_auth_if_needed(self, server_name: str, error: Exception) -> None:
        if not _looks_like_auth_error(error):
            return
        self.auth_cache.mark_needs_auth(server_name)
        self._emit(
            "mcp_auth_cached",
            output_node="mcp",
            input_node="mcp",
            content={
                "server": server_name,
            },
        )

    def _emit_tool_error(
        self,
        server_name: str,
        tool_name: str,
        error: Exception,
    ) -> None:
        self._emit(
            "mcp_tool_call_error",
            output_node="mcp",
            input_node="tool_result",
            content={
                "server": server_name,
                "tool": tool_name,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def _emit(
        self,
        result_type: str,
        *,
        output_node: str,
        input_node: str,
        content: dict[str, Any],
    ) -> None:
        if not self.observer:
            return
        self.observer.emit(
            result_type,
            output_node=output_node,
            input_node=input_node,
            content=content,
        )


def _server_fingerprint(server: McpServerConfig) -> str:
    payload = {
        "name": server.name,
        "type": server.type,
        "command": server.command,
        "args": server.args,
        "env": server.env,
        "url": server.url,
        "headers": server.headers,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n...[truncated {omitted} chars]"


def _is_session_expired(error: McpProtocolError) -> bool:
    return error.code == -32001 or "session" in str(error).lower() and "expired" in str(error).lower()


def _looks_like_auth_error(error: Exception) -> bool:
    text = str(error).lower()
    if isinstance(error, McpProtocolError) and error.code in {401, 403}:
        return True
    return any(token in text for token in ("401", "403", "auth", "unauthorized", "forbidden"))
