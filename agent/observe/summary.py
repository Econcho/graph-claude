from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.observe.mermaid import read_trace_events
from agent.observe.serializers import truncate_text


def write_summary_from_trace(
    trace_path: str | Path,
    output_path: str | Path | None = None,
) -> Path | None:
    trace = Path(trace_path)
    if not trace.exists():
        return None

    events = read_trace_events(trace)
    target = Path(output_path) if output_path else trace.with_name("trace_summary.md")
    target.write_text(events_to_summary(events), encoding="utf-8")
    return target


def events_to_summary(events: list[dict[str, Any]]) -> str:
    run_input = _first(events, "run_input")
    final_output = _last(events, "final_output")
    timeline = [_timeline_line(event) for event in events if _is_core_event(event)]
    tool_calls = [event for event in events if event.get("result_type") == "tool_selected"]
    permissions = [event for event in events if event.get("result_type") == "tool_permission_decision"]
    tool_results = [event for event in events if event.get("result_type") == "tool_executed"]
    context_events = [event for event in events if event.get("result_type") in {"context_ready", "context_compacted", "memory_recalled", "memory_updated"}]
    extensions = [event for event in events if event.get("result_type") in {"skill_used", "mcp_used", "agent_delegated"}]

    lines = ["# Runtime Observe Summary", ""]
    lines.extend(_task_section(run_input, final_output))
    lines.extend(["", "## Core Timeline"])
    if timeline:
        for index, item in enumerate(timeline, start=1):
            lines.append(f"{index}. {item}")
    else:
        lines.append("No core timeline events recorded.")

    lines.extend(["", "## Tool Calls"])
    lines.extend(_tool_table(tool_calls, permissions, tool_results))

    lines.extend(["", "## Context / Memory"])
    if context_events:
        for event in context_events:
            lines.append(f"- {_timeline_line(event)}")
    else:
        lines.append("- No context or memory events recorded.")

    lines.extend(["", "## Extensions Used"])
    if extensions:
        for event in extensions:
            lines.append(f"- {_timeline_line(event)}")
    else:
        lines.append("- Skill: not used")
        lines.append("- MCP: not used")
        lines.append("- Multi-Agent: not used")

    lines.append("")
    return "\n".join(lines)


def _task_section(run_input: dict[str, Any] | None, final_output: dict[str, Any] | None) -> list[str]:
    content = run_input.get("content", {}) if run_input else {}
    final = final_output.get("content", {}) if final_output else {}
    return [
        "## Task",
        f"- Input: {content.get('user_input_preview') or ''}",
        f"- Workspace: {content.get('workspace_root') or ''}",
        f"- Session: {content.get('session_id') or ''}",
        f"- Permission mode: {content.get('permission_mode') or ''}",
        f"- Final answer: {final.get('final_answer_preview') or ''}",
    ]


def _tool_table(
    tool_calls: list[dict[str, Any]],
    permissions: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[str]:
    if not tool_calls and not tool_results:
        return ["No tool calls recorded."]

    lines = [
        "| order | tool | permission | ok | result |",
        "| --- | --- | --- | --- | --- |",
    ]
    total = max(len(tool_calls), len(tool_results))
    for index in range(total):
        call = tool_calls[index].get("content", {}) if index < len(tool_calls) else {}
        result = tool_results[index].get("content", {}) if index < len(tool_results) else {}
        permission = _matching_permission(call.get("tool_name"), permissions)
        lines.append(
            "| {order} | {tool} | {permission} | {ok} | {result} |".format(
                order=index + 1,
                tool=call.get("tool_name") or result.get("tool_name") or "",
                permission=permission,
                ok=result.get("ok", ""),
                result=_escape_table(str(result.get("content_preview") or "")),
            )
        )
    return lines


def _matching_permission(tool_name: str | None, permissions: list[dict[str, Any]]) -> str:
    for event in permissions:
        content = event.get("content", {})
        if content.get("tool_name") == tool_name:
            approved = content.get("approved")
            suffix = "" if approved is None else f", approved={approved}"
            return f"{content.get('behavior')} ({content.get('source')}{suffix})"
    return ""


def _timeline_line(event: dict[str, Any]) -> str:
    result_type = event.get("result_type")
    content = event.get("content", {})
    if result_type == "llm_response":
        calls = content.get("tool_calls") or []
        suffix = f" -> tool_call({calls[0].get('name')})" if calls else " -> final"
        return f"llm_response{suffix}"
    if result_type == "tool_permission_decision":
        return f"tool_permission_decision -> {content.get('tool_name')} {content.get('behavior')} source={content.get('source')}"
    if result_type == "tool_executed":
        return f"tool_executed -> {content.get('tool_name')} ok={content.get('ok')}"
    if result_type == "context_ready":
        return "context_ready -> messages={messages}, llm_messages={llm_messages}, tokens={before}->{after}".format(
            messages=content.get("message_count"),
            llm_messages=content.get("llm_message_count"),
            before=content.get("original_tokens"),
            after=content.get("final_tokens"),
        )
    if result_type == "prompt_ready":
        return f"prompt_ready -> sections={content.get('section_count')}, tools={content.get('tool_count')}"
    if result_type == "final_output":
        return "final_output -> " + truncate_text(str(content.get("final_answer_preview") or ""), 120)
    return f"{result_type} -> {truncate_text(json.dumps(content, ensure_ascii=False), 180)}"


def _is_core_event(event: dict[str, Any]) -> bool:
    return event.get("result_type") in {
        "run_input",
        "context_ready",
        "prompt_ready",
        "llm_response",
        "tool_selected",
        "tool_permission_decision",
        "tool_executed",
        "tool_result_returned",
        "context_compacted",
        "memory_recalled",
        "memory_updated",
        "skill_used",
        "mcp_used",
        "agent_delegated",
        "final_output",
        "runtime_error",
    }


def _first(events: list[dict[str, Any]], result_type: str) -> dict[str, Any] | None:
    return next((event for event in events if event.get("result_type") == result_type), None)


def _last(events: list[dict[str, Any]], result_type: str) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("result_type") == result_type:
            return event
    return None


def _escape_table(text: str) -> str:
    return truncate_text(text.replace("|", "\\|").replace("\n", " "), 120)
