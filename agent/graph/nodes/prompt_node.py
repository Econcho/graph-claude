from __future__ import annotations

from typing import Any

from agent.graph.state import AgentState
from agent.observe.observer import RuntimeObserver
from agent.prompt import DEFAULT_SYSTEM_PROMPT, PromptRuntime
from agent.tools.registry import ToolRegistry


def make_prompt_node(
    registry: ToolRegistry,
    observer: RuntimeObserver | None = None,
):
    runtime = PromptRuntime(registry, observer=observer)

    def prompt_node(state: AgentState) -> dict[str, Any]:
        result = runtime.build_main_request(state)
        llm_request = {
            "system_prompt": result.system_prompt,
            "messages": result.messages,
            "tools": result.tools,
        }

        return {
            "system_prompt": result.system_prompt,
            "tool_specs": result.tools,
            "llm_request": llm_request,
            "prompt_snapshot": result.snapshot,
            "prompt_snapshot_ref": result.snapshot_ref,
            "prompt_section_stats": result.snapshot.get("section_stats", []),
            "prompt_runtime_managed": True,
        }

    return prompt_node
