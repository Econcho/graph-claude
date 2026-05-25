from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # User/runtime input.
    user_input: str
    workspace_root: str
    workspace: str
    permission_mode: str
    session_id: str
    agent_id: str | None
    memory_text: str

    # Transcript.
    messages: Annotated[list[Any], add_messages]

    # Context/prompt.
    context_snapshot: dict[str, Any]
    system_prompt: str
    tool_specs: list[dict[str, Any]]
    llm_request: dict[str, Any]
    override_system_prompt: str | None
    custom_system_prompt: str | None
    append_system_prompt: str | None
    agent_system_prompt: str | None
    prompt_snapshot: dict[str, Any]
    prompt_snapshot_ref: str | None
    prompt_section_stats: list[dict[str, Any]]
    prompt_runtime_managed: bool
    skills_enabled: bool
    skill_summaries: list[dict[str, Any]]
    conditional_skill_summaries: list[dict[str, Any]]
    skill_touched_paths: list[str]
    skill_call_depth: int
    active_skill_name: str | None
    multi_agent_enabled: bool
    coordinator_mode: bool
    team_name: str | None
    agent_name: str | None
    subagent_depth: int
    multi_agent_notifications: list[str]
    multi_agent_mailbox_messages: list[dict[str, Any]]

    # LLM output.
    assistant_message: Any
    pending_tool_calls: list[dict[str, Any]]

    # Tool runtime.
    current_tool_call: dict[str, Any] | None
    current_tool_result: dict[str, Any] | None
    tool_results: list[dict[str, Any]]

    # Session memory meta. Full memory content is stored outside GraphState.
    session_memory_enabled: bool
    session_memory_ref: str | None
    session_memory_exists: bool
    session_memory_last_summarized_message_id: str | None
    session_memory_last_summarized_message_index: int | None
    session_memory_last_summarized_token_count: int
    session_memory_last_summarized_tool_call_count: int
    session_memory_updated_at: str | None
    session_memory_update_in_progress: bool
    session_memory_update_failures: int
    session_memory_last_update_reason: str | None
    session_memory_last_error: str | None
    session_memory_update_run_id: str | None
    session_memory_update_started_at: str | None
    session_memory_update_finished_at: str | None
    session_memory_last_fork_agent_status: str | None

    # Auto memory meta. Full long-term memories are stored outside GraphState.
    auto_memory_enabled: bool
    auto_memory_ref: str | None
    auto_memory_index_loaded: bool
    auto_memory_recalled_items: list[dict[str, Any]]
    auto_memory_already_surfaced: list[str]
    auto_memory_last_extract_at: str | None
    auto_memory_last_extract_message_id: str | None
    auto_memory_extract_in_progress: bool
    auto_memory_extract_failures: int
    auto_memory_last_error: str | None
    auto_memory_main_agent_wrote_memory_this_turn: bool
    auto_memory_last_extract_run_id: str | None
    auto_memory_last_extract_status: str | None

    # Context compaction meta. Compaction affects LLM request messages only.
    context_compaction_enabled: bool
    context_compaction_last_status: str | None
    context_compaction_last_reason: str | None
    context_compaction_original_tokens: int
    context_compaction_after_microcompact_tokens: int
    context_compaction_final_tokens: int
    context_compaction_original_message_count: int
    context_compaction_final_message_count: int
    context_compaction_failures: int
    context_compaction_last_error: str | None

    # Final/error.
    final_answer: str | None
    error: dict[str, Any] | None
