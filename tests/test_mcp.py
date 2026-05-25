from __future__ import annotations

import json
import sys
from pathlib import Path

from agent.mcp import McpConfig, McpManager, McpServerConfig, build_mcp_tool_name, load_mcp_config
from agent.mcp.auth_cache import McpAuthCache
from agent.mcp.models import McpToolDefinition
from agent.mcp.tool_adapter import McpToolAdapter
from agent.prompt import PromptRuntime
from agent.tools import PermissionRules, ToolContext, ToolExecutor, ToolRegistry


def test_build_mcp_tool_name_sanitizes_names():
    assert build_mcp_tool_name("github", "get file.contents") == "mcp__github__get_file_contents"


def test_load_github_config_expands_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_test")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "mcp": {
                    "servers": {
                        "github": {
                            "enabled": True,
                            "type": "stdio",
                            "command": "docker",
                            "args": ["run"],
                            "env": {
                                "GITHUB_PERSONAL_ACCESS_TOKEN": "$GITHUB_PERSONAL_ACCESS_TOKEN"
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_mcp_config(settings)

    assert config.servers["github"].env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_test"


def test_mcp_description_is_truncated(tmp_path: Path):
    manager = McpManager(
        config=McpConfig(
            description_max_chars=10,
            servers={},
        )
    )
    definition = manager._definition_from_raw(
        "github",
        {
            "name": "long",
            "description": "x" * 20,
            "inputSchema": {"type": "object", "properties": {}},
        },
    )

    assert definition.exposed_name == "mcp__github__long"
    assert len(definition.description) > 10
    assert "truncated" in definition.description


def test_mcp_tool_adapter_maps_read_only_annotation():
    definition = McpToolDefinition(
        server_name="github",
        tool_name="get_file_contents",
        exposed_name="mcp__github__get_file_contents",
        description="Read file",
        input_schema={"type": "object", "properties": {}},
        annotations={"readOnlyHint": True},
    )

    tool = McpToolAdapter(definition=definition, call_tool=lambda *_: {})

    assert tool.is_read_only is True
    assert tool.is_destructive is False


def test_fake_stdio_mcp_server_lists_and_calls_tool(tmp_path: Path):
    script = _write_fake_mcp_server(tmp_path)
    config = McpConfig(
        request_timeout_seconds=5,
        servers={
            "fake": McpServerConfig(
                name="fake",
                enabled=True,
                type="stdio",
                command=sys.executable,
                args=[str(script)],
            )
        },
    )
    manager = McpManager(config=config, auth_cache=McpAuthCache(path=tmp_path / "auth.json"))

    try:
        tools = manager.discover_tools()
        assert [tool.name for tool in tools] == ["mcp__fake__echo"]

        result = tools[0].run({"text": "hello"}, ToolContext(workspace_root=tmp_path))
        assert result.ok is True
        assert result.content == "echo: hello"
    finally:
        manager.close()


def test_mcp_tool_specs_enter_prompt_runtime(tmp_path: Path):
    definition = McpToolDefinition(
        server_name="github",
        tool_name="get_file_contents",
        exposed_name="mcp__github__get_file_contents",
        description="Read GitHub file",
        input_schema={"type": "object", "properties": {}},
        annotations={"readOnlyHint": True},
    )
    registry = ToolRegistry(
        [McpToolAdapter(definition=definition, call_tool=lambda *_: {})]
    )

    result = PromptRuntime(registry).build_main_request(
        {
            "workspace_root": str(tmp_path),
            "messages": [],
        }
    )

    assert any(tool["name"] == "mcp__github__get_file_contents" for tool in result.tools)


def test_mcp_tool_uses_existing_permission_system(tmp_path: Path):
    definition = McpToolDefinition(
        server_name="github",
        tool_name="create_issue",
        exposed_name="mcp__github__create_issue",
        description="Create issue",
        input_schema={"type": "object", "properties": {}},
    )
    tool = McpToolAdapter(
        definition=definition,
        call_tool=lambda *_: {
            "content": [{"type": "text", "text": "created"}],
        },
    )
    executor = ToolExecutor(
        ToolRegistry([tool]),
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: False,
    )

    result = executor.execute(
        {"id": "call_1", "name": "mcp__github__create_issue", "args": {}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "permission_denied"


def test_permission_rule_wildcard_matches_mcp_tools(tmp_path: Path):
    settings = tmp_path / "settings.json"
    settings.write_text(
        '{"permissions":{"allow":["mcp__github__get_file_contents"],"ask":["mcp__github__*"]}}',
        encoding="utf-8",
    )
    definition = McpToolDefinition(
        server_name="github",
        tool_name="get_file_contents",
        exposed_name="mcp__github__get_file_contents",
        description="Read GitHub file",
        input_schema={"type": "object", "properties": {}},
    )
    tool = McpToolAdapter(
        definition=definition,
        call_tool=lambda *_: {
            "content": [{"type": "text", "text": "ok"}],
        },
    )

    result = ToolExecutor(ToolRegistry([tool]), settings_path=settings).execute(
        {"id": "call_1", "name": "mcp__github__get_file_contents", "args": {}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "permission_denied"


def test_mcp_server_connection_failure_does_not_fail_discovery(tmp_path: Path):
    config = McpConfig(
        request_timeout_seconds=1,
        servers={
            "bad": McpServerConfig(
                name="bad",
                enabled=True,
                type="stdio",
                command="missing-mcp-command",
            )
        },
    )
    manager = McpManager(config=config, auth_cache=McpAuthCache(path=tmp_path / "auth.json"))

    assert manager.discover_tools() == []


def _write_fake_mcp_server(tmp_path: Path) -> Path:
    script = tmp_path / "fake_mcp_server.py"
    script.write_text(
        r'''
from __future__ import annotations

import json
import sys


def send(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    request_id = message.get("id")
    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "0.1.0"},
            },
        })
    elif method == "notifications/initialized":
        continue
    elif method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            },
        })
    elif method == "tools/call":
        text = message.get("params", {}).get("arguments", {}).get("text", "")
        send({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": f"echo: {text}"}],
                "isError": False,
            },
        })
''',
        encoding="utf-8",
    )
    return script
