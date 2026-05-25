from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import build_graph


class ScriptedModel:
    def __init__(self):
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        tool_messages = [message for message in messages if message.type == "tool"]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "src/math_utils.py"},
                        "id": "call_read",
                        "type": "tool_call",
                    }
                ],
            )
        if len(tool_messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {
                            "path": "SUMMARY.md",
                            "content": (
                                "# math_utils\n\n"
                                "- add(a, b): returns the sum of two values.\n"
                                "- multiply(a, b): returns the product of two values.\n"
                            ),
                        },
                        "id": "call_write",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Created SUMMARY.md with add and multiply summaries.")


def test_agent_reads_and_writes_summary(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n\n"
        "def multiply(a, b):\n    return a * b\n",
        encoding="utf-8",
    )

    model = ScriptedModel()
    graph = build_graph(model=model)
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Summarize src/math_utils.py into SUMMARY.md")],
            "workspace": str(tmp_path),
            "permission_mode": "accept_edits",
            "memory_text": "",
            "final_answer": None,
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )

    summary = (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")
    assert "add" in summary
    assert "multiply" in summary
    assert result["final_answer"] == "Created SUMMARY.md with add and multiply summaries."
    assert any(message.type == "tool" for message in result["messages"])
    assert model.bound_tools is not None
    assert model.bound_tools[0]["type"] == "function"
    assert model.bound_tools[0]["function"]["name"] == "read_file"
