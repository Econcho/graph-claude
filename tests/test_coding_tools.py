from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.tools import BUILTIN_TOOLS, PermissionRules, ToolContext, ToolExecutor, ToolRegistry


def _executor(**kwargs) -> ToolExecutor:
    return ToolExecutor(ToolRegistry(BUILTIN_TOOLS), **kwargs)


def _ctx(tmp_path: Path, *, mode: str = "default") -> ToolContext:
    return ToolContext(workspace_root=tmp_path, permission_mode=mode)


def test_list_dir_glob_and_grep(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "hidden.py").write_text("hidden", encoding="utf-8")
    executor = _executor(permission_rules=PermissionRules.empty())

    listed = executor.execute(
        {"id": "1", "name": "list_dir", "args": {"path": ".", "recursive": True}},
        _ctx(tmp_path),
    )
    assert listed.ok
    assert "src/app.py" in listed.content
    assert ".venv" not in listed.content

    globbed = executor.execute(
        {"id": "2", "name": "glob", "args": {"pattern": "src/*.py"}},
        _ctx(tmp_path),
    )
    assert globbed.ok
    assert globbed.data["matches"] == ["src/app.py"]

    grepped = executor.execute(
        {"id": "3", "name": "grep", "args": {"pattern": "hello", "path": "src"}},
        _ctx(tmp_path),
    )
    assert grepped.ok
    assert grepped.data["matches"][0]["path"] == "src/app.py"


def test_edit_file_exact_replacement_and_duplicate_guard(tmp_path: Path):
    target = tmp_path / "README.md"
    target.write_text("hello\nworld\n", encoding="utf-8")
    executor = _executor(permission_rules=PermissionRules.empty())

    result = executor.execute(
        {
            "id": "1",
            "name": "edit_file",
            "args": {"path": "README.md", "edits": [{"old": "world", "new": "agent"}]},
        },
        _ctx(tmp_path, mode="accept_edits"),
    )
    assert result.ok
    assert "+agent" in result.content
    assert target.read_text(encoding="utf-8") == "hello\nagent\n"

    target.write_text("same same", encoding="utf-8")
    duplicate = executor.execute(
        {
            "id": "2",
            "name": "edit_file",
            "args": {"path": "README.md", "edits": [{"old": "same", "new": "x"}]},
        },
        _ctx(tmp_path, mode="accept_edits"),
    )
    assert duplicate.ok is False
    assert duplicate.error == "old_text_not_unique"


def test_run_command_requires_permission(tmp_path: Path):
    executor = _executor(
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: False,
    )
    result = executor.execute(
        {
            "id": "1",
            "name": "run_command",
            "args": {"command": f"{sys.executable} -c \"print('hello')\""},
        },
        _ctx(tmp_path),
    )
    assert result.ok is False
    assert result.error == "permission_denied"


def test_read_many_file_ops_and_permissions(tmp_path: Path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    executor = _executor(permission_rules=PermissionRules.empty())

    many = executor.execute(
        {"id": "1", "name": "read_many_files", "args": {"paths": ["a.txt", "missing.txt"]}},
        _ctx(tmp_path),
    )
    assert many.ok
    assert many.data["files"][0]["ok"] is True
    assert many.data["files"][1]["error"] == "file_not_found"

    mkdir = executor.execute(
        {"id": "2", "name": "mkdir", "args": {"path": "nested"}},
        _ctx(tmp_path, mode="accept_edits"),
    )
    assert mkdir.ok
    assert (tmp_path / "nested").is_dir()

    moved = executor.execute(
        {"id": "3", "name": "move_file", "args": {"source": "a.txt", "destination": "nested/a.txt"}},
        _ctx(tmp_path, mode="accept_edits"),
    )
    assert moved.ok
    assert (tmp_path / "nested" / "a.txt").exists()

    deleted = executor.execute(
        {"id": "4", "name": "delete_file", "args": {"path": "b.txt"}},
        _ctx(tmp_path, mode="accept_edits"),
    )
    assert deleted.ok
    assert not (tmp_path / "b.txt").exists()


def test_parse_test_output_and_collect_diagnostics_permission(tmp_path: Path):
    executor = _executor(permission_rules=PermissionRules.empty())
    parsed = executor.execute(
        {
            "id": "1",
            "name": "parse_test_output",
            "args": {"output": "FAILED tests/test_a.py::test_x - AssertionError: bad"},
        },
        _ctx(tmp_path),
    )
    assert parsed.ok
    assert parsed.data["diagnostics"][0]["file"] == "tests/test_a.py"

    denied = _executor(
        permission_rules=PermissionRules.empty(),
        permission_asker=lambda tool, tool_input, decision: False,
    ).execute(
        {"id": "2", "name": "collect_diagnostics", "args": {"command": "pytest"}},
        _ctx(tmp_path),
    )
    assert denied.ok is False
    assert denied.error == "permission_denied"


def test_symbol_tools(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "math_utils.py").write_text(
        "class Calculator:\n    pass\n\ndef add(a, b):\n    return a + b\n\nx = add(1, 2)\n",
        encoding="utf-8",
    )
    executor = _executor(permission_rules=PermissionRules.empty())

    outline = executor.execute(
        {"id": "1", "name": "outline_file", "args": {"path": "src/math_utils.py"}},
        _ctx(tmp_path),
    )
    assert outline.ok
    assert any(item["name"] == "add" for item in outline.data["symbols"])

    definition = executor.execute(
        {"id": "2", "name": "find_definition", "args": {"symbol": "add", "path": "src"}},
        _ctx(tmp_path),
    )
    assert definition.ok
    assert definition.data["matches"][0]["line"] == 4

    references = executor.execute(
        {"id": "3", "name": "find_references", "args": {"symbol": "add", "path": "src"}},
        _ctx(tmp_path),
    )
    assert references.ok
    assert len(references.data["matches"]) >= 2


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_tools_in_repo_and_non_repo(tmp_path: Path):
    executor = _executor(permission_rules=PermissionRules.empty())
    non_repo = executor.execute({"id": "0", "name": "git_status", "args": {}}, _ctx(tmp_path))
    assert non_repo.ok is False
    assert non_repo.error == "not_git_repository"

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    status = executor.execute({"id": "1", "name": "git_status", "args": {}}, _ctx(tmp_path))
    assert status.ok
    assert "README.md" in status.content

    diff = executor.execute({"id": "2", "name": "git_diff", "args": {}}, _ctx(tmp_path))
    assert diff.ok
    assert isinstance(diff.content, str)
