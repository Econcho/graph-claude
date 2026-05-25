from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.mcp import McpConfig, McpManager, McpServerConfig
from agent.mcp.auth_cache import McpAuthCache
from agent.tools import PermissionRules, ToolContext, ToolExecutor, ToolRegistry


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GITHUB_MCP_LIVE") != "1"
    or not os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    reason="GitHub MCP live test requires RUN_GITHUB_MCP_LIVE=1 and GITHUB_PERSONAL_ACCESS_TOKEN.",
)


def test_github_mcp_live_read_only_tool(tmp_path: Path):
    config = McpConfig(
        request_timeout_seconds=60,
        servers={
            "github": McpServerConfig(
                name="github",
                enabled=True,
                type="stdio",
                command="docker",
                args=[
                    "run",
                    "-i",
                    "--rm",
                    "-e",
                    "GITHUB_PERSONAL_ACCESS_TOKEN",
                    "-e",
                    "GITHUB_TOOLSETS",
                    "ghcr.io/github/github-mcp-server",
                ],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv(
                        "GITHUB_PERSONAL_ACCESS_TOKEN", ""
                    ),
                    "GITHUB_TOOLSETS": "repos,issues,pull_requests,users",
                },
            )
        },
    )
    manager = McpManager(config=config, auth_cache=McpAuthCache(path=tmp_path / "auth.json"))

    try:
        tools = manager.discover_tools()
        tool_names = {tool.name for tool in tools}
        assert tool_names

        candidate = _select_read_candidate(tools)
        if candidate is None:
            pytest.skip(f"No supported read-only GitHub MCP candidate found: {sorted(tool_names)}")

        tool, args = candidate
        executor = ToolExecutor(
            ToolRegistry([tool]),
            permission_rules=PermissionRules.empty(),
            permission_asker=lambda tool, tool_input, decision: True,
        )

        result = executor.execute(
            {"id": "call_live", "name": tool.name, "args": args},
            ToolContext(workspace_root=tmp_path),
        )

        assert result.ok is True
        assert result.content
    finally:
        manager.close()


def _select_read_candidate(tools):
    by_name = {tool.name: tool for tool in tools}

    get_file = by_name.get("mcp__github__get_file_contents")
    if get_file is not None:
        return get_file, {
            "owner": "github",
            "repo": "github-mcp-server",
            "path": "README.md",
        }

    search_repos = by_name.get("mcp__github__search_repositories")
    if search_repos is not None:
        return search_repos, {
            "query": "repo:github/github-mcp-server",
        }

    return None
