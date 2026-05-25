from __future__ import annotations

from typing import Any

from agent.observe.config import ObserveConfig
from agent.observe.events import make_event
from agent.observe.serializers import (
    summarize_llm_request,
    summarize_llm_response,
    summarize_message,
    to_jsonable,
    truncate_text,
)
from agent.observe.sinks import EventSink


class RuntimeObserver:
    def __init__(
        self,
        *,
        run_id: str,
        sink: EventSink,
        enabled: bool = True,
        trace_path: str | None = None,
        config: ObserveConfig | None = None,
    ):
        self.run_id = run_id
        self.sink = sink
        self.enabled = enabled
        self.trace_path = trace_path
        self.config = config or ObserveConfig()

    def emit(
        self,
        result_type: str,
        *,
        output_node: str,
        input_node: str,
        content: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self.config.is_enabled(result_type):
            return

        try:
            event = make_event(
                result_type=result_type,
                output_node=output_node,
                input_node=input_node,
                content=to_jsonable(content or {}),
            )
            self.sink.emit(event)
        except Exception:
            return

    def on_run_start(self, initial_state: dict[str, Any]) -> None:
        self.emit(
            "run_input",
            output_node="user",
            input_node="bootstrap",
            content={
                "user_input_preview": truncate_text(
                    str(initial_state.get("user_input") or ""), 500
                ),
                "workspace_root": initial_state.get("workspace_root")
                or initial_state.get("workspace"),
                "session_id": initial_state.get("session_id"),
                "permission_mode": initial_state.get("permission_mode", "default"),
                "enabled_features": {
                    "auto_memory": bool(initial_state.get("auto_memory_enabled", True)),
                    "session_memory": bool(
                        initial_state.get("session_memory_enabled", True)
                    ),
                    "context_compaction": bool(
                        initial_state.get("context_compaction_enabled", True)
                    ),
                    "skills": bool(initial_state.get("skills_enabled", True)),
                    "multi_agent": bool(initial_state.get("multi_agent_enabled", True)),
                },
            },
        )
        self.emit(
            "run_start",
            output_node="runtime",
            input_node="bootstrap",
            content={
                "run_id": self.run_id,
                "initial_state": initial_state,
            },
        )

    def on_run_end(self, final_state: dict[str, Any]) -> None:
        self.emit(
            "final_output",
            output_node="finalize",
            input_node="user",
            content={
                "final_answer_preview": truncate_text(
                    str(final_state.get("final_answer") or ""), 1000
                ),
                "message_count": len(final_state.get("messages", []) or []),
                "tool_result_count": len(final_state.get("tool_results", []) or []),
            },
        )
        self.emit(
            "run_end",
            output_node="finalize",
            input_node="runtime",
            content={
                "run_id": self.run_id,
                "final_state": final_state,
            },
        )

    def on_run_error(
        self,
        error: Exception,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            "runtime_error",
            output_node="runtime",
            input_node="user",
            content={
                "run_id": self.run_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        self.emit(
            "run_error",
            output_node="graph",
            input_node="runtime",
            content={
                "run_id": self.run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "state": state or {},
            },
        )

    def on_node_input(self, node_name: str, state: dict[str, Any]) -> None:
        self.emit(
            "node_input",
            output_node="graph",
            input_node=node_name,
            content={
                "node": node_name,
                "state": state,
            },
        )

    def on_node_output(self, node_name: str, output: dict[str, Any]) -> None:
        self.emit(
            "node_output",
            output_node=node_name,
            input_node="graph",
            content={
                "node": node_name,
                "output": output,
            },
        )
        if node_name == "context":
            self.on_context_ready(output.get("context_snapshot", {}))
        elif node_name == "prompt":
            self.on_prompt_ready(output)
        elif node_name == "tool_orchestrator":
            self.on_tool_selected(output)

    def on_node_start(self, node_name: str, state: dict[str, Any]) -> None:
        self.on_node_input(node_name, state)

    def on_node_end(self, node_name: str, output: dict[str, Any]) -> None:
        self.on_node_output(node_name, output)

    def on_node_error(
        self,
        node_name: str,
        error: Exception,
        state: dict[str, Any],
    ) -> None:
        self.emit(
            "node_error",
            output_node=node_name,
            input_node="runtime",
            content={
                "node": node_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "state": state,
            },
        )

    def on_route_decision(
        self,
        route_name: str,
        decision: str,
        state: dict[str, Any],
    ) -> None:
        self.emit(
            "route_decision",
            output_node=route_name,
            input_node=decision,
            content={
                "route": route_name,
                "decision": decision,
                "state": state,
            },
        )

    def on_llm_start(self, request: dict[str, Any]) -> None:
        self.emit(
            "llm_start",
            output_node="prompt",
            input_node="llm",
            content={
                "request": summarize_llm_request(request),
            },
        )

    def on_llm_end(self, response: Any, tool_calls: list[dict[str, Any]]) -> None:
        next_node = "tool_orchestrator" if tool_calls else "finalize"
        self.emit(
            "llm_response",
            output_node="llm",
            input_node=next_node,
            content={
                "content_preview": _message_content_preview(response),
                "tool_call_count": len(tool_calls),
                "tool_calls": [_summarize_tool_call(call) for call in tool_calls],
                "next_node": next_node,
            },
        )
        self.emit(
            "llm_end",
            output_node="llm",
            input_node=next_node,
            content={
                "response": summarize_llm_response(response),
                "tool_calls": to_jsonable(tool_calls),
            },
        )

    def on_llm_error(
        self,
        error: Exception,
        request: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            "llm_error",
            output_node="llm",
            input_node="runtime",
            content={
                "error_type": type(error).__name__,
                "error": str(error),
                "request": summarize_llm_request(request or {}),
            },
        )

    def on_tool_call_start(self, tool_call: Any) -> None:
        self.emit(
            "tool_call_start",
            output_node="tool_orchestrator",
            input_node="tool_executor",
            content={
                "tool_call": to_jsonable(tool_call),
            },
        )

    def on_tool_call_end(self, tool_call: Any, result: Any) -> None:
        self.emit(
            "tool_executed",
            output_node="tool_executor",
            input_node="tool_result",
            content={
                "tool_name": _tool_call_name(tool_call),
                "ok": _result_ok(result),
                "error": _result_error(result),
                "content_preview": _result_content_preview(result),
                "data_summary": _summarize_value(_result_data(result)),
            },
        )
        self.emit(
            "tool_call_end",
            output_node="tool_executor",
            input_node="tool_result",
            content={
                "tool_call": to_jsonable(tool_call),
                "result": to_jsonable(result),
            },
        )

    def on_tool_call_error(self, tool_call: Any, error: Exception) -> None:
        self.emit(
            "tool_call_error",
            output_node="tool_executor",
            input_node="runtime",
            content={
                "tool_call": to_jsonable(tool_call),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def on_tool_result_appended(
        self,
        tool_call: Any,
        tool_result: Any,
        tool_message: Any,
    ) -> None:
        self.emit(
            "tool_result_returned",
            output_node="tool_result",
            input_node="context",
            content={
                "tool_name": _tool_call_name(tool_call),
                "tool_call_id": _tool_call_id(tool_call),
                "message_role": "tool",
                "returned_to": "messages",
            },
        )
        self.emit(
            "tool_result_appended",
            output_node="tool_result",
            input_node="messages",
            content={
                "tool_call": to_jsonable(tool_call),
                "tool_result": to_jsonable(tool_result),
                "tool_message": to_jsonable(tool_message),
            },
        )

    def on_skill_discovery_loaded(
        self,
        *,
        workspace_root: str,
        skill_count: int,
        warnings: list[str],
    ) -> None:
        self.emit(
            "skill_discovery_loaded",
            output_node="skills",
            input_node="context",
            content={
                "workspace_root": workspace_root,
                "skill_count": skill_count,
                "warnings": warnings,
            },
        )

    def on_skill_prompt_injected(
        self,
        *,
        skill_count: int,
        conditional_skill_count: int,
    ) -> None:
        self.emit(
            "skill_prompt_injected",
            output_node="skills",
            input_node="prompt",
            content={
                "skill_count": skill_count,
                "conditional_skill_count": conditional_skill_count,
            },
        )

    def on_skill_call_start(
        self,
        skill_name: str,
        source: str,
        context: str,
    ) -> None:
        self.emit(
            "skill_call_start",
            output_node="tool_executor",
            input_node="skills",
            content={
                "skill": skill_name,
                "source": source,
                "context": context,
            },
        )

    def on_skill_call_end(self, skill_name: str, result: Any) -> None:
        self.emit(
            "skill_used",
            output_node="skills",
            input_node="tool_result",
            content={
                "skill_name": skill_name,
                "context": (
                    _result_data(result).get("context")
                    if isinstance(_result_data(result), dict)
                    else None
                ),
                "status": "ok" if _result_ok(result) else "failed",
                "result_preview": _result_content_preview(result),
            },
        )
        self.emit(
            "skill_call_end",
            output_node="skills",
            input_node="tool_result",
            content={
                "skill": skill_name,
                "result": to_jsonable(result),
            },
        )

    def on_context_ready(self, context_snapshot: dict[str, Any]) -> None:
        if not isinstance(context_snapshot, dict):
            context_snapshot = {}
        messages = context_snapshot.get("messages", []) or []
        llm_messages = context_snapshot.get("llm_messages", []) or []
        compaction = context_snapshot.get("compaction", {}) or {}
        memories = context_snapshot.get("relevant_memories", []) or []
        skills = context_snapshot.get("skill_summaries", []) or []
        conditional_skills = (
            context_snapshot.get("conditional_skill_summaries", []) or []
        )
        notifications = context_snapshot.get("multi_agent_notifications", []) or []

        self.emit(
            "context_ready",
            output_node="context",
            input_node="prompt",
            content={
                "message_count": len(messages),
                "llm_message_count": len(llm_messages),
                "original_tokens": compaction.get("original_tokens"),
                "final_tokens": compaction.get("final_tokens"),
                "compaction_status": compaction.get("status"),
                "recalled_memory_count": len(memories),
                "skill_count": len(skills),
                "conditional_skill_count": len(conditional_skills),
                "multi_agent_notification_count": len(notifications),
            },
        )

        if memories:
            self.emit(
                "memory_recalled",
                output_node="auto_memory",
                input_node="context",
                content={
                    "count": len(memories),
                    "memory_ids": [item.get("id") for item in memories],
                    "memory_names": [item.get("name") for item in memories],
                },
            )

        if compaction.get("status") not in {None, "skipped", "unchanged"}:
            self.emit(
                "context_compacted",
                output_node="context_compaction",
                input_node="context",
                content={
                    "strategy": _compaction_strategy(compaction),
                    "tokens_before": compaction.get("original_tokens"),
                    "tokens_after": compaction.get("final_tokens"),
                    "messages_before": compaction.get("original_message_count"),
                    "messages_after": compaction.get("final_message_count"),
                    "reason": compaction.get("reason"),
                },
            )

    def on_prompt_ready(self, prompt_output: dict[str, Any]) -> None:
        tools = prompt_output.get("tool_specs", []) or []
        self.emit(
            "prompt_ready",
            output_node="prompt",
            input_node="llm",
            content={
                "section_count": len(prompt_output.get("prompt_section_stats", []) or []),
                "tool_count": len(tools),
                "tool_groups": _tool_groups(tools),
                "prompt_snapshot_ref": prompt_output.get("prompt_snapshot_ref"),
            },
        )

    def on_tool_selected(self, output: dict[str, Any]) -> None:
        tool_call = output.get("current_tool_call")
        if not tool_call:
            return
        self.emit(
            "tool_selected",
            output_node="tool_orchestrator",
            input_node="tool_executor",
            content={
                "tool_name": _tool_call_name(tool_call),
                "tool_call_id": _tool_call_id(tool_call),
                "args_summary": _summarize_value(_tool_call_args(tool_call)),
                "remaining_tool_call_count": len(output.get("pending_tool_calls", []) or []),
            },
        )

    def on_tool_permission_decision(
        self,
        *,
        tool_name: str,
        behavior: str,
        source: str | None,
        reason: str | None,
        approved: bool | None,
        is_read_only: bool,
        is_destructive: bool,
    ) -> None:
        self.emit(
            "tool_permission_decision",
            output_node="tool_executor",
            input_node="tool_executor",
            content={
                "tool_name": tool_name,
                "behavior": behavior,
                "source": source,
                "reason": reason,
                "approved": approved,
                "is_read_only": is_read_only,
                "is_destructive": is_destructive,
            },
        )

    def on_memory_updated(
        self,
        *,
        memory_type: str,
        status: str,
        ref: str | None,
        reason: str | None = None,
    ) -> None:
        self.emit(
            "memory_updated",
            output_node=f"{memory_type}_memory",
            input_node="context" if memory_type == "session" else "finalize",
            content={
                "type": memory_type,
                "status": status,
                "ref": ref,
                "reason": reason,
            },
        )

    def on_mcp_used(
        self,
        *,
        server: str,
        tool: str,
        ok: bool,
        result: Any,
    ) -> None:
        self.emit(
            "mcp_used",
            output_node="mcp",
            input_node="tool_result",
            content={
                "server": server,
                "tool": tool,
                "ok": ok,
                "result_preview": truncate_text(str(to_jsonable(result)), 1000),
            },
        )

    def on_agent_delegated(
        self,
        *,
        task_id: str | None,
        agent_id: str | None,
        mode: str,
        status: str,
        final_answer: str | None = None,
    ) -> None:
        self.emit(
            "agent_delegated",
            output_node="multi_agent",
            input_node="tool_result",
            content={
                "task_id": task_id,
                "agent_id": agent_id,
                "mode": mode,
                "status": status,
                "final_answer_preview": truncate_text(final_answer or "", 1000),
            },
        )

    def on_skill_call_error(self, skill_name: str, error: Exception) -> None:
        self.emit(
            "skill_call_error",
            output_node="skills",
            input_node="tool_result",
            content={
                "skill": skill_name,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def on_skill_prompt_shell_start(self, skill_name: str, command: str) -> None:
        self.emit(
            "skill_prompt_shell_start",
            output_node="skills",
            input_node="shell_command",
            content={
                "skill": skill_name,
                "command": command,
            },
        )

    def on_skill_prompt_shell_end(
        self,
        skill_name: str,
        command: str,
        result: Any,
    ) -> None:
        self.emit(
            "skill_prompt_shell_end",
            output_node="shell_command",
            input_node="skills",
            content={
                "skill": skill_name,
                "command": command,
                "result": to_jsonable(result),
            },
        )


def _get_attr_or_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _message_content_preview(message: Any) -> str:
    if isinstance(message, dict):
        return truncate_text(str(message.get("content") or ""), 1000)
    return truncate_text(str(getattr(message, "content", "") or ""), 1000)


def _tool_call_name(tool_call: Any) -> str | None:
    return _get_attr_or_key(tool_call, "name")


def _tool_call_id(tool_call: Any) -> str | None:
    return _get_attr_or_key(tool_call, "id") or _get_attr_or_key(tool_call, "tool_call_id")


def _tool_call_args(tool_call: Any) -> Any:
    if isinstance(tool_call, dict):
        return tool_call.get("args") or tool_call.get("input") or {}
    return getattr(tool_call, "input", {})


def _summarize_tool_call(tool_call: Any) -> dict[str, Any]:
    return {
        "name": _tool_call_name(tool_call),
        "args_summary": _summarize_value(_tool_call_args(tool_call)),
    }


def _result_ok(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("ok"))
    return bool(getattr(result, "ok", False))


def _result_error(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("error")
    return getattr(result, "error", None)


def _result_data(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("data")
    return getattr(result, "data", None)


def _result_content_preview(result: Any) -> str:
    if isinstance(result, dict):
        content = result.get("content", "")
    else:
        content = getattr(result, "content", "")
    return truncate_text(str(content or ""), 1000)


def _summarize_value(value: Any) -> Any:
    jsonable = to_jsonable(value)
    if isinstance(jsonable, str):
        return truncate_text(jsonable, 500)
    if isinstance(jsonable, dict):
        summary: dict[str, Any] = {}
        for key, item in jsonable.items():
            if isinstance(item, str):
                summary[key] = truncate_text(item, 300)
            elif isinstance(item, (int, float, bool)) or item is None:
                summary[key] = item
            elif isinstance(item, list):
                summary[key] = f"list[{len(item)}]"
            elif isinstance(item, dict):
                summary[key] = f"dict[{len(item)}]"
            else:
                summary[key] = repr(item)
        return summary
    if isinstance(jsonable, list):
        return f"list[{len(jsonable)}]"
    return jsonable


def _tool_groups(tools: list[dict[str, Any]]) -> dict[str, int]:
    groups = {"builtin": 0, "skill": 0, "mcp": 0, "multi_agent": 0}
    multi_agent_names = {
        "agent",
        "agent_poll",
        "send_message",
        "team_create",
        "task_create",
        "task_list",
        "task_update",
        "task_stop",
    }
    for spec in tools:
        name = str(spec.get("name") or "")
        if name.startswith("mcp__"):
            groups["mcp"] += 1
        elif name in {"use_skill", "shell_command"}:
            groups["skill"] += 1
        elif name in multi_agent_names:
            groups["multi_agent"] += 1
        else:
            groups["builtin"] += 1
    return groups


def _compaction_strategy(compaction: dict[str, Any]) -> str:
    status = str(compaction.get("status") or "")
    if "full" in status or compaction.get("full_compact_status"):
        return "full"
    if "session" in status:
        return "session"
    if "micro" in status:
        return "microcompact"
    return "skipped"
