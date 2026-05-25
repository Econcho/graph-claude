from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


DEBUG_HOOKS = {
    "run_start": False,
    "run_end": False,
    "run_error": True,
    "node_input": True,
    "node_output": True,
    "node_error": True,
    "route_decision": True,
    "llm_start": False,
    "llm_end": False,
    "llm_error": True,
    "tool_call_start": False,
    "tool_call_end": False,
    "tool_call_error": True,
    "tool_result_appended": False,
    "session_memory_check": True,
    "session_memory_fork_agent_start": True,
    "session_memory_fork_agent_finish": True,
    "session_memory_file_validate": True,
    "session_memory_update_success": True,
    "session_memory_update_failure": True,
    "auto_memory_recall_start": True,
    "auto_memory_recall_selected": True,
    "auto_memory_recall_empty": True,
    "auto_memory_extract_check": True,
    "auto_memory_extract_start": True,
    "auto_memory_extract_proposal": True,
    "auto_memory_extract_empty": True,
    "auto_memory_write_success": True,
    "auto_memory_write_failure": True,
    "auto_memory_index_update": True,
    "auto_memory_extract_failure": True,
    "context_compaction_check": True,
    "microcompact_applied": True,
    "session_compact_skipped": True,
    "session_compact_applied": True,
    "full_compact_start": True,
    "full_compact_success": True,
    "full_compact_failure": True,
    "full_compact_skipped": True,
    "context_compaction_failure": True,
    "skill_discovery_loaded": True,
    "skill_prompt_injected": True,
    "skill_call_start": True,
    "skill_call_end": True,
    "skill_call_error": True,
    "skill_prompt_shell_start": True,
    "skill_prompt_shell_end": True,
    "prompt_runtime_start": True,
    "prompt_section_built": False,
    "prompt_effective_resolved": True,
    "prompt_dump_written": True,
    "prompt_runtime_error": True,
    "mcp_config_loaded": True,
    "mcp_server_connect_start": True,
    "mcp_server_connect_end": True,
    "mcp_server_connect_error": True,
    "mcp_tool_discovered": True,
    "mcp_tool_call_start": True,
    "mcp_tool_call_end": True,
    "mcp_tool_call_error": True,
    "mcp_auth_cached": True,
    "mcp_reconnect": True,
    "agent_spawn_start": True,
    "agent_spawn_end": True,
    "agent_spawn_error": True,
    "agent_task_status_changed": True,
    "agent_message_sent": True,
    "team_created": True,
    "team_task_created": True,
    "team_task_updated": True,
    "mailbox_message_written": True,
    "mailbox_message_read": True,
}

CORE_HOOKS = {
    **{name: False for name in DEBUG_HOOKS},
    "run_input": True,
    "context_ready": True,
    "prompt_ready": True,
    "llm_response": True,
    "tool_selected": True,
    "tool_permission_decision": True,
    "tool_executed": True,
    "tool_result_returned": True,
    "memory_updated": True,
    "memory_recalled": True,
    "context_compacted": True,
    "skill_used": True,
    "mcp_used": True,
    "agent_delegated": True,
    "final_output": True,
    "runtime_error": True,
    "node_error": True,
    "llm_error": True,
    "tool_call_error": True,
    "prompt_runtime_error": True,
    "mcp_server_connect_error": True,
    "mcp_tool_call_error": True,
    "skill_call_error": True,
    "agent_spawn_error": True,
}

PROFILES = {
    "core": CORE_HOOKS,
    "debug": {
        **DEBUG_HOOKS,
        "run_input": True,
        "context_ready": True,
        "prompt_ready": True,
        "llm_response": True,
        "tool_selected": True,
        "tool_permission_decision": True,
        "tool_executed": True,
        "tool_result_returned": True,
        "memory_updated": True,
        "memory_recalled": True,
        "context_compacted": True,
        "skill_used": True,
        "mcp_used": True,
        "agent_delegated": True,
        "final_output": True,
        "runtime_error": True,
    },
}


@dataclass(frozen=True)
class ObserveConfig:
    mode: str = "dataflow"
    profile: str = "core"
    hooks: dict[str, bool] = field(default_factory=lambda: dict(CORE_HOOKS))

    def is_enabled(self, hook_name: str) -> bool:
        return self.hooks.get(hook_name, False)


def default_config_path() -> Path:
    return Path(__file__).with_name("config.json")


def load_observe_config(config_path: str | Path | None = None) -> ObserveConfig:
    path = Path(config_path) if config_path else default_config_path()
    profile = "core"
    hooks = dict(PROFILES[profile])
    mode = "dataflow"

    if not path.exists():
        return ObserveConfig(mode=mode, profile=profile, hooks=hooks)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("mode"), str):
        mode = data["mode"]
    if isinstance(data.get("profile"), str) and data["profile"] in PROFILES:
        profile = data["profile"]
        hooks = dict(PROFILES[profile])

    configured_hooks = data.get("hooks", {})
    if isinstance(configured_hooks, dict):
        hooks.update({str(key): bool(value) for key, value in configured_hooks.items()})
        if "node_start" in configured_hooks and "node_input" not in configured_hooks:
            hooks["node_input"] = bool(configured_hooks["node_start"])
        if "node_end" in configured_hooks and "node_output" not in configured_hooks:
            hooks["node_output"] = bool(configured_hooks["node_end"])

    return ObserveConfig(mode=mode, profile=profile, hooks=hooks)


DEFAULT_HOOKS = CORE_HOOKS
