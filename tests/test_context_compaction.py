from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph.nodes.context_node import make_context_node
from agent.graph.nodes.prompt_node import make_prompt_node
from agent.memory.context_compaction import (
    CLEARED_TOOL_RESULT_CONTENT,
    ContextCompactionConfig,
    ContextCompactionManager,
    FullCompactAgent,
    format_compact_summary,
    microcompact_messages,
    session_compact_messages,
)
from agent.memory.session_memory.store import SessionMemoryStore
from agent.tools import ToolRegistry


SESSION_MEMORY = """# Session Memory

## Current State

The task is in progress and previous work has been summarized.

## Task

Continue the current coding task.

## Important Files

None.

## Decisions

Use context compaction when the transcript grows.

## Errors and Fixes

None.

## Pending Work

Continue from recent messages.

## Worklog

- Created a session memory summary.
"""


class FakeFullCompactLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages, tools=None, system_prompt=None):
        self.calls += 1
        return self.response


def test_microcompact_clears_old_tool_results():
    messages = [
        HumanMessage(content="start"),
        ToolMessage(content="old one", tool_call_id="call_1", name="read_file"),
        ToolMessage(content="old two", tool_call_id="call_2", name="read_file"),
        ToolMessage(content="keep one", tool_call_id="call_3", name="read_file"),
        ToolMessage(content="keep two", tool_call_id="call_4", name="read_file"),
    ]

    result = microcompact_messages(messages, keep_recent_tool_results=2)

    assert result.compacted_tool_result_count == 2
    assert result.messages[1].content == CLEARED_TOOL_RESULT_CONTENT
    assert result.messages[2].content == CLEARED_TOOL_RESULT_CONTENT
    assert result.messages[3].content == "keep one"
    assert result.messages[4].content == "keep two"


def test_microcompact_preserves_tool_metadata():
    message = ToolMessage(
        content="large output",
        tool_call_id="call_meta",
        name="read_file",
        status="success",
    )

    result = microcompact_messages([message], keep_recent_tool_results=0)
    compacted = result.messages[0]

    assert compacted.content == CLEARED_TOOL_RESULT_CONTENT
    assert compacted.tool_call_id == "call_meta"
    assert compacted.name == "read_file"
    assert compacted.status == "success"


def test_session_compact_skips_without_session_memory(tmp_path: Path):
    messages = [HumanMessage(content="hello")]

    result = session_compact_messages(
        messages,
        workspace_root=str(tmp_path),
        session_id="missing",
    )

    assert result.compacted is False
    assert result.messages == messages
    assert result.reason == "missing_session_memory"


def test_session_compact_uses_session_memory_and_recent_messages(tmp_path: Path):
    SessionMemoryStore(workspace_root=tmp_path, session_id="compact").write(
        SESSION_MEMORY
    )
    messages = [
        HumanMessage(content=f"old message {index} " + ("x" * 200))
        for index in range(8)
    ]

    result = session_compact_messages(
        messages,
        workspace_root=str(tmp_path),
        session_id="compact",
        recent_keep_max_tokens=40,
        recent_keep_min_messages=2,
    )

    assert result.compacted is True
    assert isinstance(result.messages[0], HumanMessage)
    assert "This conversation was compacted." in result.messages[0].content
    assert "Session Memory:" in result.messages[0].content
    assert "Use context compaction" in result.messages[0].content
    assert result.messages[-1].content == messages[-1].content
    assert len(result.messages) < len(messages) + 1


def test_prompt_uses_context_snapshot_llm_messages():
    prompt = make_prompt_node(ToolRegistry([]))
    raw_messages = [HumanMessage(content="raw")]
    llm_messages = [HumanMessage(content="compacted")]

    output = prompt(
        {
            "workspace_root": ".",
            "messages": raw_messages,
            "context_snapshot": {"llm_messages": llm_messages},
        }
    )

    assert output["llm_request"]["messages"] == llm_messages


def test_context_node_runs_microcompact_before_session_compact(tmp_path: Path):
    SessionMemoryStore(workspace_root=tmp_path, session_id="context").write(
        SESSION_MEMORY
    )
    manager = ContextCompactionManager(
        config=ContextCompactionConfig(
            session_compact_trigger_tokens=1,
            session_compact_recent_keep_max_tokens=200,
            session_compact_recent_keep_min_messages=2,
        )
    )
    node = make_context_node(context_compaction_manager=manager)
    messages = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"path": "a.txt"},
                    "id": "call_a",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="A" * 1000, tool_call_id="call_a", name="read_file"),
        ToolMessage(content="B" * 1000, tool_call_id="call_b", name="read_file"),
        HumanMessage(content="recent"),
    ]

    output = node(
        {
            "workspace_root": str(tmp_path),
            "session_id": "context",
            "messages": messages,
            "context_compaction_enabled": True,
        }
    )

    snapshot = output["context_snapshot"]

    assert snapshot["messages"] == messages
    assert snapshot["llm_messages"] != messages
    assert snapshot["compaction"]["status"] == "session_compacted"
    assert output["context_compaction_last_status"] == "session_compacted"
    assert snapshot["llm_messages"][0].type == "human"
    assert "This conversation was compacted." in snapshot["llm_messages"][0].content


def test_full_compact_runs_when_session_memory_missing(tmp_path: Path):
    llm = FakeFullCompactLLM(
        {
            "role": "assistant",
            "content": (
                "<analysis>scratch</analysis>"
                "<summary>Continue by editing README.md.</summary>"
            ),
            "tool_calls": [],
        }
    )
    manager = ContextCompactionManager(
        llm,
        config=ContextCompactionConfig(
            session_compact_trigger_tokens=1,
            full_compact_trigger_tokens=1,
            full_compact_recent_keep_max_tokens=40,
        ),
    )
    messages = [
        HumanMessage(content=f"old message {index} " + ("x" * 200))
        for index in range(8)
    ]

    result = manager.compact(
        {
            "workspace_root": str(tmp_path),
            "session_id": "missing",
            "messages": messages,
            "context_compaction_enabled": True,
        }
    )

    assert llm.calls == 1
    assert result.meta["status"] == "full_compacted"
    assert result.messages[0].type == "human"
    assert "Continue by editing README.md." in result.messages[0].content
    assert "scratch" not in result.messages[0].content
    assert len(result.messages) < len(messages) + 1


def test_full_compact_strips_analysis_and_keeps_summary():
    formatted = format_compact_summary(
        "<analysis>private notes</analysis><summary>Useful summary</summary>"
    )

    assert formatted == "Useful summary"


def test_full_compact_accepts_unstructured_summary():
    formatted = format_compact_summary("plain summary")

    assert formatted == "plain summary"


def test_full_compact_rejects_tool_calls():
    llm = FakeFullCompactLLM(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "name": "read_file",
                    "args": {"path": "README.md"},
                }
            ],
        }
    )
    agent = FullCompactAgent(llm)
    messages = [HumanMessage(content="hello")]

    result = agent.compact(messages)

    assert result.compacted is False
    assert result.reason == "tool_call_returned"
    assert result.messages == messages


def test_context_manager_falls_back_when_full_compact_fails(tmp_path: Path):
    class FailingLLM:
        def invoke(self, messages, tools=None, system_prompt=None):
            raise RuntimeError("compact failed")

    manager = ContextCompactionManager(
        FailingLLM(),
        config=ContextCompactionConfig(
            session_compact_trigger_tokens=1,
            full_compact_trigger_tokens=1,
        ),
    )
    messages = [HumanMessage(content="x" * 1000)]

    result = manager.compact(
        {
            "workspace_root": str(tmp_path),
            "session_id": "missing",
            "messages": messages,
            "context_compaction_enabled": True,
        }
    )

    assert result.messages == messages
    assert result.patch["context_compaction_failures"] == 1
    assert result.patch["context_compaction_last_error"] == "compact failed"
