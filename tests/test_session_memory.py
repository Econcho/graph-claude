from pathlib import Path

from agent.graph.nodes.llm_node import make_llm_node
from agent.memory.session_memory import SessionMemoryConfig, SessionMemoryManager
from agent.memory.session_memory.store import SessionMemoryStore


UPDATED_MEMORY = """# Session Memory

## Current State

The user asked for a summary and the assistant answered.

## Task

Maintain a rolling summary for the current session.

## Important Files

None.

## Decisions

Session Memory is enabled for this test.

## Errors and Fixes

None.

## Pending Work

None.

## Worklog

- Recorded the latest assistant response.
"""


class FakeSessionMemoryLLM:
    def __init__(self, markdown: str = UPDATED_MEMORY):
        self.markdown = markdown
        self.calls = 0

    def invoke(self, messages, tools, system_prompt=None):
        self.calls += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_memory",
                    "name": "edit_session_memory",
                    "args": {
                        "updated_session_memory_markdown": self.markdown,
                    },
                }
            ],
        }


class FakeNoToolLLM:
    def invoke(self, messages, tools, system_prompt=None):
        return {
            "role": "assistant",
            "content": "I did not call the memory tool.",
            "tool_calls": [],
        }


def _trigger_config() -> SessionMemoryConfig:
    return SessionMemoryConfig(
        init_token_threshold=1,
        update_token_delta_threshold=1,
        tool_call_delta_threshold=99,
    )


def test_session_memory_store_validates_and_writes(tmp_path: Path):
    store = SessionMemoryStore(workspace_root=tmp_path, session_id="test/session")

    store.write(UPDATED_MEMORY)

    assert store.exists()
    assert store.read_or_template() == UPDATED_MEMORY
    assert store.path.name == "test_session.md"


def test_session_memory_manager_skips_below_threshold(tmp_path: Path):
    manager = SessionMemoryManager(
        FakeSessionMemoryLLM(),
        config=SessionMemoryConfig(
            init_token_threshold=10_000,
            update_token_delta_threshold=5_000,
        ),
    )

    patch = manager.maybe_update(
        {
            "workspace_root": str(tmp_path),
            "session_id": "skip",
            "session_memory_enabled": True,
            "messages": [{"role": "user", "content": "short"}],
            "assistant_message": {"role": "assistant", "content": "done"},
        }
    )

    assert patch["session_memory_last_fork_agent_status"] == "skipped"
    assert patch["session_memory_last_update_reason"] == "below_init_token_threshold"
    assert not (tmp_path / ".agent").exists()


def test_session_memory_manager_updates_file_and_meta(tmp_path: Path):
    fake_llm = FakeSessionMemoryLLM()
    manager = SessionMemoryManager(
        fake_llm,
        config=_trigger_config(),
    )

    patch = manager.maybe_update(
        {
            "workspace_root": str(tmp_path),
            "session_id": "memory-success",
            "session_memory_enabled": True,
            "messages": [
                {"role": "user", "content": "Please summarize the work."},
                {"role": "assistant", "content": "I summarized it."},
            ],
            "assistant_message": {"role": "assistant", "content": "I summarized it."},
        }
    )

    memory_path = tmp_path / ".agent" / "session_memory" / "memory-success.md"
    assert fake_llm.calls == 1
    assert memory_path.read_text(encoding="utf-8") == UPDATED_MEMORY
    assert patch["session_memory_last_fork_agent_status"] == "success"
    assert patch["session_memory_update_failures"] == 0
    assert patch["session_memory_last_summarized_message_index"] == 1
    assert patch["session_memory_ref"] == str(memory_path)


def test_session_memory_manager_failure_returns_meta_without_raising(tmp_path: Path):
    manager = SessionMemoryManager(
        FakeNoToolLLM(),
        config=_trigger_config(),
    )

    patch = manager.maybe_update(
        {
            "workspace_root": str(tmp_path),
            "session_id": "memory-failure",
            "session_memory_enabled": True,
            "messages": [
                {"role": "user", "content": "Please summarize the work."},
                {"role": "assistant", "content": "I summarized it."},
            ],
            "assistant_message": {"role": "assistant", "content": "I summarized it."},
        }
    )

    assert patch["session_memory_last_fork_agent_status"] == "failed"
    assert patch["session_memory_update_failures"] == 1
    assert "did not call edit_session_memory" in patch["session_memory_last_error"]


def test_llm_node_runs_post_llm_session_memory_hook(tmp_path: Path):
    class FakeMainAndMemoryLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages, tools, system_prompt=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "role": "assistant",
                    "content": "Main response.",
                    "tool_calls": [],
                }

            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_memory",
                        "name": "edit_session_memory",
                        "args": {
                            "updated_session_memory_markdown": UPDATED_MEMORY,
                        },
                    }
                ],
            }

    llm = FakeMainAndMemoryLLM()
    manager = SessionMemoryManager(llm, config=_trigger_config())
    node = make_llm_node(llm, session_memory_manager=manager)

    output = node(
        {
            "workspace_root": str(tmp_path),
            "session_id": "llm-hook",
            "session_memory_enabled": True,
            "messages": [{"role": "user", "content": "hello"}],
            "llm_request": {
                "system_prompt": "You are a test agent.",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
            },
        }
    )

    assert output["assistant_message"]["content"] == "Main response."
    assert output["session_memory_last_fork_agent_status"] == "success"
    assert llm.calls == 2
