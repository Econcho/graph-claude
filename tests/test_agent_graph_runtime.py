from pathlib import Path

from agent.graph import build_graph


class FakeLLMNoTool:
    def invoke(self, messages, tools, system_prompt=None):
        return {
            "role": "assistant",
            "content": "我是一个 coding agent。",
            "tool_calls": [],
        }


class FakeLLMReadFile:
    def __init__(self):
        self.count = 0
        self.seen_tool_result = False
        self.seen_tools = []

    def invoke(self, messages, tools, system_prompt=None):
        self.count += 1
        self.seen_tools = tools
        self.seen_tool_result = any(
            getattr(message, "type", None) == "tool"
            or (isinstance(message, dict) and message.get("role") == "tool")
            for message in messages
        )

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
            "content": "README.md says: This is a test project.",
            "tool_calls": [],
        }


class FakeLLMMultipleTools:
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
                        "args": {"path": "a.txt"},
                    },
                    {
                        "id": "call_002",
                        "name": "read_file",
                        "args": {"path": "b.txt"},
                    },
                ],
            }

        return {
            "role": "assistant",
            "content": "Read both files.",
            "tool_calls": [],
        }


def test_graph_finalizes_without_tool_call(tmp_path: Path):
    graph = build_graph(FakeLLMNoTool())
    result = graph.invoke(
        {
            "user_input": "介绍一下你自己",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "no-tool"}},
    )

    assert result["final_answer"] == "我是一个 coding agent。"
    assert not result.get("pending_tool_calls")


def test_graph_executes_read_file_and_returns_tool_result(tmp_path: Path):
    (tmp_path / "README.md").write_text("This is a test project.", encoding="utf-8")

    llm = FakeLLMReadFile()
    graph = build_graph(llm)
    result = graph.invoke(
        {
            "user_input": "读取 README.md 并总结",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "read-file"}},
    )

    assert result["final_answer"] == "README.md says: This is a test project."
    assert result["tool_results"]
    assert result["tool_results"][0]["ok"] is True
    assert result["tool_results"][0]["content"] == "This is a test project."
    assert llm.seen_tool_result is True
    tool_names = {tool["name"] for tool in llm.seen_tools}
    assert {"read_file", "write_file"}.issubset(tool_names)


def test_graph_consumes_multiple_pending_tool_calls(tmp_path: Path):
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    graph = build_graph(FakeLLMMultipleTools())
    result = graph.invoke(
        {
            "user_input": "读取 a.txt 和 b.txt",
            "workspace_root": str(tmp_path),
            "session_id": "test-session",
            "agent_id": None,
        },
        config={"configurable": {"thread_id": "multi-tool"}},
    )

    assert result["final_answer"] == "Read both files."
    assert len(result["tool_results"]) == 2
    assert [item["content"] for item in result["tool_results"]] == ["A", "B"]
    assert result["pending_tool_calls"] == []
