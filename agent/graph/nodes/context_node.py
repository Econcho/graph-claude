from __future__ import annotations

from typing import Any

from agent.graph.state import AgentState
from agent.memory.context_compaction import ContextCompactionManager
from agent.memory.auto_memory import AutoMemoryManager
from agent.multi_agent import MultiAgentManager
from agent.skills import SkillManager


def context_node(state: AgentState) -> dict[str, Any]:
    context_snapshot = {
        "messages": state.get("messages", []),
        "workspace_root": state.get("workspace_root"),
        "session_id": state.get("session_id"),
        "agent_id": state.get("agent_id"),
    }
    return {"context_snapshot": context_snapshot}


def make_context_node(
    auto_memory_manager: AutoMemoryManager | None = None,
    context_compaction_manager: ContextCompactionManager | None = None,
    skill_manager: SkillManager | None = None,
    multi_agent_manager: MultiAgentManager | None = None,
):
    def context_node_with_memory(state: AgentState) -> dict[str, Any]:
        multi_agent_patch = {}
        if multi_agent_manager is not None:
            notifications = multi_agent_manager.consume_notifications()
            mailbox_messages = multi_agent_manager.read_mailbox(_state_to_tool_context(state))
            multi_agent_patch = {
                "multi_agent_notifications": [
                    *list(state.get("multi_agent_notifications") or []),
                    *notifications,
                ],
            }
            if mailbox_messages:
                multi_agent_patch["multi_agent_mailbox_messages"] = mailbox_messages

        auto_memory_patch = {}
        if auto_memory_manager is not None:
            auto_memory_patch = auto_memory_manager.recall(state)

        skill_patch = {}
        if skill_manager is not None:
            skill_patch = skill_manager.prepare_context(
                {
                    **state,
                    **auto_memory_patch,
                }
            )

        compaction_patch = {}
        llm_messages = state.get("messages", [])
        compaction_meta = {
            "status": "disabled",
            "reason": "manager_not_configured",
        }
        if context_compaction_manager is not None:
            compaction_result = context_compaction_manager.compact(
                {
                    **state,
                    **auto_memory_patch,
                    **skill_patch,
                }
            )
            llm_messages = compaction_result.messages
            compaction_meta = compaction_result.meta
            compaction_patch = compaction_result.patch

        context_snapshot = {
            "messages": state.get("messages", []),
            "llm_messages": llm_messages,
            "workspace_root": state.get("workspace_root"),
            "session_id": state.get("session_id"),
            "agent_id": state.get("agent_id"),
            "compaction": compaction_meta,
            "relevant_memories": auto_memory_patch.get(
                "auto_memory_recalled_items",
                state.get("auto_memory_recalled_items", []),
            ),
            "skill_summaries": skill_patch.get(
                "skill_summaries",
                state.get("skill_summaries", []),
            ),
            "conditional_skill_summaries": skill_patch.get(
                "conditional_skill_summaries",
                state.get("conditional_skill_summaries", []),
            ),
            "multi_agent_notifications": multi_agent_patch.get(
                "multi_agent_notifications",
                state.get("multi_agent_notifications", []),
            ),
            "multi_agent_mailbox_messages": multi_agent_patch.get(
                "multi_agent_mailbox_messages",
                [],
            ),
        }

        return {
            "context_snapshot": context_snapshot,
            **multi_agent_patch,
            **auto_memory_patch,
            **skill_patch,
            **compaction_patch,
        }

    return context_node_with_memory


def _state_to_tool_context(state: AgentState):
    from agent.tools.adapters.langgraph_adapter import state_to_tool_context

    return state_to_tool_context(state)
