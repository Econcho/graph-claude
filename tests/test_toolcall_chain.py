from pathlib import Path

from agent.graph.nodes.tool_orchestrator_node import tool_orchestrator_node
from agent.tools import (
    BUILTIN_TOOLS,
    PermissionRules,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    default_settings_path,
)


def test_tool_orchestrator_handles_no_tool_calls():
    result = tool_orchestrator_node(
        {
            "assistant_message": {
                "role": "assistant",
                "content": "done",
                "tool_calls": [],
            }
        }
    )

    assert result["current_tool_call"] is None
    assert result["pending_tool_calls"] == []


def test_tool_orchestrator_selects_single_tool_call():
    call = {"id": "call_001", "name": "read_file", "args": {"path": "README.md"}}
    result = tool_orchestrator_node(
        {
            "assistant_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            }
        }
    )

    assert result["current_tool_call"] == call
    assert result["pending_tool_calls"] == []


def test_tool_orchestrator_selects_first_and_keeps_remaining():
    first = {"id": "call_001", "name": "read_file", "args": {"path": "a.txt"}}
    second = {"id": "call_002", "name": "read_file", "args": {"path": "b.txt"}}
    result = tool_orchestrator_node(
        {
            "assistant_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [first, second],
            }
        }
    )

    assert result["current_tool_call"] == first
    assert result["pending_tool_calls"] == [second]


def test_tool_orchestrator_prefers_existing_pending_tool_calls():
    pending = {"id": "call_pending", "name": "read_file", "args": {"path": "p.txt"}}
    ignored = {"id": "call_ignored", "name": "read_file", "args": {"path": "i.txt"}}
    result = tool_orchestrator_node(
        {
            "pending_tool_calls": [pending],
            "assistant_message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [ignored],
            },
        }
    )

    assert result["current_tool_call"] == pending
    assert result["pending_tool_calls"] == []


def _executor(**kwargs) -> ToolExecutor:
    return ToolExecutor(ToolRegistry(BUILTIN_TOOLS), **kwargs)


def test_tool_executor_reads_file(tmp_path: Path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    result = _executor(settings_path=tmp_path / "settings.json").execute(
        {"id": "call_001", "name": "read_file", "args": {"path": "README.md"}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is True
    assert result.content == "hello"


def test_tool_executor_returns_tool_not_found(tmp_path: Path):
    result = _executor(settings_path=tmp_path / "settings.json").execute(
        {"id": "call_001", "name": "unknown_tool", "args": {}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "tool_not_found"


def test_tool_executor_returns_invalid_tool_input(tmp_path: Path):
    result = _executor().execute(
        {"id": "call_001", "name": "read_file", "args": {}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "invalid_tool_input"


def test_tool_executor_asks_and_rejects_write_file_by_default(tmp_path: Path):
    result = _executor(
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: False,
    ).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "README.md", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "permission_denied"
    assert not (tmp_path / "README.md").exists()


def test_tool_executor_asks_and_approves_write_file_by_default(tmp_path: Path):
    result = _executor(
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: True,
    ).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "README.md", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hello"


def test_tool_executor_accepts_write_file_with_accept_edits(tmp_path: Path):
    result = _executor(permission_rules=PermissionRules.empty()).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "README.md", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path, permission_mode="accept_edits"),
    )

    assert result.ok is True
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "hello"


def test_tool_executor_denies_write_file_in_plan_mode(tmp_path: Path):
    result = _executor(
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: True,
    ).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "README.md", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path, permission_mode="plan"),
    )

    assert result.ok is False
    assert result.error == "permission_denied"
    assert not (tmp_path / "README.md").exists()


def test_permission_deny_rule_overrides_accept_edits(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        '{"permissions":{"deny":["write_file(secrets/**)"]}}',
        encoding="utf-8",
    )

    result = _executor().execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "secrets/token.txt", "content": "secret"},
        },
        ToolContext(workspace_root=tmp_path, permission_mode="accept_edits"),
    )

    assert result.ok is False
    assert result.error == "permission_denied"
    assert not (tmp_path / "secrets" / "token.txt").exists()


def test_permission_allow_rule_overrides_default_ask(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        '{"permissions":{"allow":["write_file(tmp/**)"]}}',
        encoding="utf-8",
    )

    result = _executor(
        settings_path=tmp_path / "settings.json",
        permission_asker=lambda tool, tool_input, decision: False,
    ).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "tmp/a.txt", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is True
    assert (tmp_path / "tmp" / "a.txt").read_text(encoding="utf-8") == "hello"


def test_permission_ask_rule_overrides_accept_edits(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        '{"permissions":{"ask":["write_file(docs/**)"]}}',
        encoding="utf-8",
    )

    result = _executor(
        settings_path=tmp_path / "settings.json",
        permission_asker=lambda tool, tool_input, decision: False,
    ).execute(
        {
            "id": "call_001",
            "name": "write_file",
            "args": {"path": "docs/a.txt", "content": "hello"},
        },
        ToolContext(workspace_root=tmp_path, permission_mode="accept_edits"),
    )

    assert result.ok is False
    assert result.error == "permission_denied"
    assert not (tmp_path / "docs" / "a.txt").exists()


def test_permission_deny_has_priority_over_allow(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        '{"permissions":{"allow":["read_file"],"deny":["read_file(.env)"]}}',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("secret", encoding="utf-8")

    result = _executor(settings_path=tmp_path / "settings.json").execute(
        {"id": "call_001", "name": "read_file", "args": {"path": ".env"}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "permission_denied"


def test_tool_executor_reads_default_agent_settings(tmp_path: Path):
    assert default_settings_path().exists()
    (tmp_path / ".env").write_text("secret", encoding="utf-8")

    result = _executor().execute(
        {"id": "call_001", "name": "read_file", "args": {"path": ".env"}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "permission_denied"


def test_tool_executor_accepts_explicit_empty_permission_rules(tmp_path: Path):
    result = _executor(permission_rules=PermissionRules.empty()).execute(
        {"id": "call_001", "name": "read_file", "args": {"path": "missing.md"}},
        ToolContext(workspace_root=tmp_path),
    )

    assert result.ok is False
    assert result.error == "file_not_found"
