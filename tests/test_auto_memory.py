from pathlib import Path

from langchain_core.messages import AIMessage

from agent.graph.nodes.context_node import make_context_node
from agent.graph.nodes.finalize_node import make_finalize_node
from agent.graph.nodes.prompt_node import make_prompt_node
from agent.memory.auto_memory.extractor import (
    AutoMemoryExtractor,
    AutoMemoryExtractorInput,
)
from agent.memory.auto_memory import AutoMemoryConfig, AutoMemoryManager, AutoMemoryStore
from agent.tools import ToolRegistry


class FakeAutoMemoryLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def invoke(self, messages, tools=None, system_prompt=None):
        self.calls += 1
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [],
        }


def test_auto_memory_store_creates_index_and_item(tmp_path: Path):
    store = AutoMemoryStore(workspace_root=tmp_path)

    record = store.create_memory(
        {
            "action": "create",
            "type": "user",
            "name": "Chinese responses",
            "description": "The user prefers Chinese explanations.",
            "content": "When explaining implementation, answer in Chinese.",
            "why_it_matters": "It matches the user's preferred working language.",
            "how_to_apply": "Use Chinese for implementation explanations.",
        }
    )

    assert store.exists()
    assert (tmp_path / ".agent" / "auto_memory" / "items" / f"{record.id}.md").exists()
    assert "Chinese responses" in store.read_index_or_template()
    assert store.get_memory(record.id).content.startswith("When explaining")


def test_context_recalls_memory_and_prompt_injects_it(tmp_path: Path):
    store = AutoMemoryStore(workspace_root=tmp_path)
    store.create_memory(
        {
            "action": "create",
            "type": "project",
            "name": "LangGraph agent",
            "description": "This project is a LangGraph coding agent.",
            "content": "The graph uses context, prompt, llm, tools, and finalize nodes.",
            "why_it_matters": "Future work should preserve the graph structure.",
            "how_to_apply": "Recall this when modifying LangGraph nodes.",
        }
    )

    manager = AutoMemoryManager(
        FakeAutoMemoryLLM('{"memory_proposals": []}'),
        config=AutoMemoryConfig(max_recalled_items=3),
    )
    context = make_context_node(auto_memory_manager=manager)
    prompt = make_prompt_node(ToolRegistry([]))

    context_output = context(
        {
            "workspace_root": str(tmp_path),
            "session_id": "recall",
            "messages": [
                {
                    "role": "user",
                    "content": "How should I change the LangGraph nodes?",
                }
            ],
            "auto_memory_enabled": True,
        }
    )
    prompt_output = prompt(
        {
            "workspace_root": str(tmp_path),
            "messages": [],
            "context_snapshot": context_output["context_snapshot"],
        }
    )

    assert len(context_output["auto_memory_recalled_items"]) == 1
    assert "Relevant long-term memory" in prompt_output["system_prompt"]
    assert "LangGraph agent" in prompt_output["system_prompt"]


def test_finalize_extracts_and_writes_auto_memory(tmp_path: Path):
    llm = FakeAutoMemoryLLM(
        """
        {
          "memory_proposals": [
            {
              "action": "create",
              "type": "feedback",
              "name": "Use concise Chinese",
              "description": "The user wants concise Chinese explanations.",
              "content": "使用中文，并保持实现说明直接简洁。",
              "why_it_matters": "It matches the user's communication preference.",
              "how_to_apply": "Answer agent implementation notes in concise Chinese.",
              "confidence": 0.9
            }
          ]
        }
        """
    )
    manager = AutoMemoryManager(llm)
    finalize = make_finalize_node(auto_memory_manager=manager)

    output = finalize(
        {
            "workspace_root": str(tmp_path),
            "session_id": "extract",
            "messages": [
                {"role": "user", "content": "以后用中文简洁说明。"},
                {"role": "assistant", "content": "好的。"},
            ],
            "assistant_message": {"role": "assistant", "content": "好的。"},
            "auto_memory_enabled": True,
        }
    )

    store = AutoMemoryStore(workspace_root=tmp_path)
    records = store.list_memories()

    assert output["final_answer"] == "好的。"
    assert output["auto_memory_last_extract_status"] == "success"
    assert llm.calls == 1
    assert len(records) == 1
    assert records[0].type == "feedback"
    assert "中文" in records[0].content


def test_finalize_extracts_after_langchain_ai_message(tmp_path: Path):
    llm = FakeAutoMemoryLLM(
        """
        {
          "memory_proposals": [
            {
              "action": "create",
              "type": "user",
              "name": "Completion summary preference",
              "description": "The user prefers a summary after task completion.",
              "content": "Provide a summary after the task is completed.",
              "why_it_matters": "Future runs should close tasks with a concise summary.",
              "how_to_apply": "After finishing requested work, summarize what changed.",
              "confidence": 0.9
            }
          ]
        }
        """
    )
    manager = AutoMemoryManager(llm)
    finalize = make_finalize_node(auto_memory_manager=manager)

    output = finalize(
        {
            "workspace_root": str(tmp_path),
            "session_id": "extract-ai-message",
            "messages": [
                {"role": "user", "content": "I prefer completion summaries."},
                AIMessage(content="Done."),
            ],
            "assistant_message": AIMessage(content="Done."),
            "auto_memory_enabled": True,
        }
    )

    records = AutoMemoryStore(workspace_root=tmp_path).list_memories()

    assert output["auto_memory_last_extract_status"] == "success"
    assert len(records) == 1
    assert records[0].name == "Completion summary preference"


def test_auto_memory_extractor_prompt_contains_disallowed_memory_rules(tmp_path: Path):
    llm = FakeAutoMemoryLLM('{"memory_proposals": []}')
    extractor = AutoMemoryExtractor(llm)

    prompt = extractor._build_user_prompt(
        AutoMemoryExtractorInput(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Read src/math_utils.py and write SUMMARY.md. "
                        "I prefer Provide a summary after the task is completed."
                    ),
                },
                {"role": "assistant", "content": "Task completed."},
            ],
            final_answer="Task completed.",
            memory_index="# Auto Memory\n",
            memory_manifest=[],
            runtime_metadata={"workspace_root": str(tmp_path)},
        )
    )

    assert "Do not store code patterns" in prompt
    assert "Do not store git history" in prompt
    assert "Do not duplicate CLAUDE.md content" in prompt
    assert "surprising or non-obvious" in prompt
