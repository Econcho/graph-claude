import json
from pathlib import Path

from agent.runtime import AgentRunner


class FakeLLMNoTool:
    def invoke(self, messages, tools, system_prompt=None):
        return {
            "role": "assistant",
            "content": "I am a coding agent.",
            "tool_calls": [],
        }


class FakeLLMReadFile:
    def __init__(self):
        self.count = 0

    def invoke(self, messages, tools, system_prompt=None):
        self.count += 1
        if self.count == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "name": "read_file",
                        "args": {"path": "README.md"},
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "README.md says this is a test project.",
            "tool_calls": [],
        }


def _run_dir(observe_dir: Path) -> Path:
    runs = list(observe_dir.glob("run_*"))
    assert len(runs) == 1
    return runs[0]


def _read_events(observe_dir: Path) -> list[dict]:
    trace = _run_dir(observe_dir) / "trace.jsonl"
    return [
        json.loads(line)
        for line in trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_mermaid(observe_dir: Path) -> str:
    return (_run_dir(observe_dir) / "trace.mmd").read_text(encoding="utf-8")


def _read_summary(observe_dir: Path) -> str:
    return (_run_dir(observe_dir) / "trace_summary.md").read_text(encoding="utf-8")


def test_runtime_observe_core_profile_records_no_tool_run(tmp_path: Path):
    observe_dir = tmp_path / ".runs"
    runner = AgentRunner(
        llm_client=FakeLLMNoTool(),
        observe=True,
        observe_dir=str(observe_dir),
    )

    result = runner.run(
        {
            "user_input": "introduce yourself",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "observe-no-tool"}},
    )

    assert result["final_answer"]
    events = _read_events(observe_dir)
    result_types = [event["result_type"] for event in events]
    assert set(events[0]) == {"id", "result_type", "output_node", "input_node", "content"}
    assert "run_input" in result_types
    assert "context_ready" in result_types
    assert "prompt_ready" in result_types
    assert "llm_response" in result_types
    assert "final_output" in result_types
    assert "node_input" not in result_types
    assert "node_output" not in result_types
    assert "llm_start" not in result_types
    assert "llm_end" not in result_types

    llm_response = next(event for event in events if event["result_type"] == "llm_response")
    assert llm_response["content"]["next_node"] == "finalize"
    assert llm_response["content"]["tool_call_count"] == 0

    mermaid = _read_mermaid(observe_dir)
    assert "flowchart TD" in mermaid
    assert "context_ready" in mermaid
    assert "prompt_ready" in mermaid
    assert "llm_response" in mermaid
    assert "node_graph" not in mermaid
    assert "node_input" not in mermaid

    summary = _read_summary(observe_dir)
    assert "# Runtime Observe Summary" in summary
    assert "## Core Timeline" in summary
    assert "## Tool Calls" in summary
    assert "## Context / Memory" in summary
    assert "## Extensions Used" in summary


def test_runtime_observe_core_profile_records_tool_run(tmp_path: Path):
    (tmp_path / "README.md").write_text("This is a test project.", encoding="utf-8")
    observe_dir = tmp_path / ".runs"
    runner = AgentRunner(
        llm_client=FakeLLMReadFile(),
        observe=True,
        observe_dir=str(observe_dir),
    )

    result = runner.run(
        {
            "user_input": "read README.md and summarize it",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "observe-tool"}},
    )

    assert result["final_answer"]
    assert result["tool_results"]
    events = _read_events(observe_dir)
    result_types = [event["result_type"] for event in events]
    assert "tool_selected" in result_types
    assert "tool_permission_decision" in result_types
    assert "tool_executed" in result_types
    assert "tool_result_returned" in result_types
    assert "tool_call_start" not in result_types
    assert "tool_call_end" not in result_types
    assert "tool_result_appended" not in result_types

    permission = next(
        event for event in events if event["result_type"] == "tool_permission_decision"
    )
    assert permission["content"]["tool_name"] == "read_file"
    assert permission["content"]["behavior"] == "allow"
    assert permission["content"]["is_read_only"] is True

    mermaid = _read_mermaid(observe_dir)
    assert "tool_selected: read_file" in mermaid
    assert "tool_permission_decision: allow" in mermaid
    assert "tool_executed: read_file" in mermaid
    assert "node_graph" not in mermaid

    summary = _read_summary(observe_dir)
    assert "| order | tool | permission | ok | result |" in summary
    assert "read_file" in summary


def test_runtime_observe_debug_profile_can_enable_node_hooks(tmp_path: Path):
    observe_dir = tmp_path / ".runs"
    config_path = tmp_path / "observe.json"
    config_path.write_text(
        json.dumps({"profile": "debug", "hooks": {"node_input": True, "node_output": True}}),
        encoding="utf-8",
    )
    runner = AgentRunner(
        llm_client=FakeLLMNoTool(),
        observe=True,
        observe_dir=str(observe_dir),
        observe_config=str(config_path),
    )

    runner.run(
        {
            "user_input": "hello",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "observe-debug"}},
    )

    result_types = [event["result_type"] for event in _read_events(observe_dir)]
    assert "node_input" in result_types
    assert "node_output" in result_types
    assert "run_input" in result_types


def test_runtime_observe_config_overrides_core_hook(tmp_path: Path):
    observe_dir = tmp_path / ".runs"
    config_path = tmp_path / "observe.json"
    config_path.write_text(
        json.dumps({"profile": "core", "hooks": {"prompt_ready": False}}),
        encoding="utf-8",
    )
    runner = AgentRunner(
        llm_client=FakeLLMNoTool(),
        observe=True,
        observe_dir=str(observe_dir),
        observe_config=str(config_path),
    )

    runner.run(
        {
            "user_input": "hello",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "observe-config"}},
    )

    result_types = [event["result_type"] for event in _read_events(observe_dir)]
    assert "prompt_ready" not in result_types
    assert "context_ready" in result_types
