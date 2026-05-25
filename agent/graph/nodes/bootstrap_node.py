from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from agent.graph.state import AgentState
from agent.multi_agent.settings import load_multi_agent_config


def bootstrap_node(state: AgentState) -> dict[str, Any]:
    multi_agent_config = load_multi_agent_config()
    user_input = state.get("user_input", "")
    workspace_root = state.get("workspace_root") or state.get("workspace", "")
    messages = state.get("messages")

    if not user_input and not messages:
        return {
            "error": {
                "type": "invalid_input",
                "message": "user_input or messages is required.",
            }
        }

    if not workspace_root:
        return {
            "error": {
                "type": "invalid_input",
                "message": "workspace_root is required.",
            }
        }

    workspace = Path(workspace_root).resolve()
    if not workspace.exists() or not workspace.is_dir():
        return {
            "error": {
                "type": "invalid_workspace",
                "message": f"workspace_root does not exist or is not a directory: {workspace}",
            }
        }

    if not messages:
        messages = [HumanMessage(content=user_input)]

    return {
        "workspace_root": str(workspace),
        "messages": messages,
        "pending_tool_calls": [],
        "tool_results": [],
        "current_tool_call": None,
        "current_tool_result": None,
        "final_answer": None,
        "error": None,
        "session_memory_enabled": state.get("session_memory_enabled", True),
        "session_memory_ref": state.get("session_memory_ref"),
        "session_memory_exists": bool(state.get("session_memory_exists", False)),
        "session_memory_last_summarized_message_id": state.get(
            "session_memory_last_summarized_message_id"
        ),
        "session_memory_last_summarized_message_index": state.get(
            "session_memory_last_summarized_message_index"
        ),
        "session_memory_last_summarized_token_count": int(
            state.get("session_memory_last_summarized_token_count") or 0
        ),
        "session_memory_last_summarized_tool_call_count": int(
            state.get("session_memory_last_summarized_tool_call_count") or 0
        ),
        "session_memory_updated_at": state.get("session_memory_updated_at"),
        "session_memory_update_in_progress": bool(
            state.get("session_memory_update_in_progress", False)
        ),
        "session_memory_update_failures": int(
            state.get("session_memory_update_failures") or 0
        ),
        "session_memory_last_update_reason": state.get(
            "session_memory_last_update_reason"
        ),
        "session_memory_last_error": state.get("session_memory_last_error"),
        "session_memory_update_run_id": state.get("session_memory_update_run_id"),
        "session_memory_update_started_at": state.get(
            "session_memory_update_started_at"
        ),
        "session_memory_update_finished_at": state.get(
            "session_memory_update_finished_at"
        ),
        "session_memory_last_fork_agent_status": state.get(
            "session_memory_last_fork_agent_status"
        ),
        "auto_memory_enabled": state.get("auto_memory_enabled", False),
        "auto_memory_ref": state.get("auto_memory_ref"),
        "auto_memory_index_loaded": bool(
            state.get("auto_memory_index_loaded", False)
        ),
        "auto_memory_recalled_items": list(
            state.get("auto_memory_recalled_items") or []
        ),
        "auto_memory_already_surfaced": list(
            state.get("auto_memory_already_surfaced") or []
        ),
        "auto_memory_last_extract_at": state.get("auto_memory_last_extract_at"),
        "auto_memory_last_extract_message_id": state.get(
            "auto_memory_last_extract_message_id"
        ),
        "auto_memory_extract_in_progress": bool(
            state.get("auto_memory_extract_in_progress", False)
        ),
        "auto_memory_extract_failures": int(
            state.get("auto_memory_extract_failures") or 0
        ),
        "auto_memory_last_error": state.get("auto_memory_last_error"),
        "auto_memory_main_agent_wrote_memory_this_turn": bool(
            state.get("auto_memory_main_agent_wrote_memory_this_turn", False)
        ),
        "auto_memory_last_extract_run_id": state.get(
            "auto_memory_last_extract_run_id"
        ),
        "auto_memory_last_extract_status": state.get(
            "auto_memory_last_extract_status"
        ),
        "context_compaction_enabled": state.get("context_compaction_enabled", True),
        "context_compaction_last_status": state.get(
            "context_compaction_last_status"
        ),
        "context_compaction_last_reason": state.get(
            "context_compaction_last_reason"
        ),
        "context_compaction_original_tokens": int(
            state.get("context_compaction_original_tokens") or 0
        ),
        "context_compaction_after_microcompact_tokens": int(
            state.get("context_compaction_after_microcompact_tokens") or 0
        ),
        "context_compaction_final_tokens": int(
            state.get("context_compaction_final_tokens") or 0
        ),
        "context_compaction_original_message_count": int(
            state.get("context_compaction_original_message_count") or 0
        ),
        "context_compaction_final_message_count": int(
            state.get("context_compaction_final_message_count") or 0
        ),
        "context_compaction_failures": int(
            state.get("context_compaction_failures") or 0
        ),
        "context_compaction_last_error": state.get("context_compaction_last_error"),
        "skills_enabled": state.get("skills_enabled", True),
        "skill_summaries": list(state.get("skill_summaries") or []),
        "conditional_skill_summaries": list(
            state.get("conditional_skill_summaries") or []
        ),
        "skill_touched_paths": list(state.get("skill_touched_paths") or []),
        "skill_call_depth": int(state.get("skill_call_depth") or 0),
        "active_skill_name": state.get("active_skill_name"),
        "multi_agent_enabled": state.get(
            "multi_agent_enabled",
            multi_agent_config.enabled,
        ),
        "coordinator_mode": bool(
            state.get("coordinator_mode", multi_agent_config.coordinator_enabled)
        ),
        "team_name": state.get("team_name"),
        "agent_name": state.get("agent_name"),
        "subagent_depth": int(state.get("subagent_depth") or 0),
        "multi_agent_notifications": list(
            state.get("multi_agent_notifications") or []
        ),
        "multi_agent_mailbox_messages": list(
            state.get("multi_agent_mailbox_messages") or []
        ),
        "override_system_prompt": state.get("override_system_prompt"),
        "custom_system_prompt": state.get("custom_system_prompt"),
        "append_system_prompt": state.get("append_system_prompt"),
        "agent_system_prompt": state.get("agent_system_prompt"),
        "prompt_snapshot": state.get("prompt_snapshot") or {},
        "prompt_snapshot_ref": state.get("prompt_snapshot_ref"),
        "prompt_section_stats": list(state.get("prompt_section_stats") or []),
        "prompt_runtime_managed": bool(state.get("prompt_runtime_managed", False)),
    }
