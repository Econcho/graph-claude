from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _safe_id(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if not safe:
        safe = "node"
    if safe[0].isdigit():
        safe = f"n_{safe}"
    return safe


def _escape_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def read_trace_events(trace_path: str | Path) -> list[dict[str, Any]]:
    path = Path(trace_path)
    if not path.exists():
        return []

    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def events_to_mermaid(events: list[dict[str, Any]]) -> str:
    lines = ["flowchart TD"]
    declared_nodes: set[str] = set()
    declared_edges: set[tuple[str, str, str]] = set()

    def declare_node(name: str) -> str:
        node_id = f"node_{_safe_id(name)}"
        if node_id not in declared_nodes:
            lines.append(f'    {node_id}(["{_escape_label(name)}"])')
            declared_nodes.add(node_id)
        return node_id

    semantic_events = [
        event
        for event in events
        if event.get("result_type") != "node_input"
        and event.get("output_node")
        and event.get("input_node")
    ]

    for event in semantic_events:
        previous_node = str(event.get("output_node"))
        next_node = str(event.get("input_node"))
        if _is_hidden_node(previous_node) or _is_hidden_node(next_node):
            continue

        previous_node_id = declare_node(previous_node)
        next_node_id = declare_node(next_node)
        label = _edge_label(event)
        edge = (previous_node_id, next_node_id, label)
        if edge in declared_edges:
            continue

        lines.append(f'    {previous_node_id} -->|"{_escape_label(label)}"| {next_node_id}')
        declared_edges.add(edge)

    return "\n".join(lines) + "\n"


def write_mermaid_from_trace(
    trace_path: str | Path,
    output_path: str | Path | None = None,
) -> Path | None:
    trace = Path(trace_path)
    if not trace.exists():
        return None

    target = Path(output_path) if output_path else trace.with_name("trace.mmd")
    mermaid = events_to_mermaid(read_trace_events(trace))
    target.write_text(mermaid, encoding="utf-8")
    return target


def _is_hidden_node(node: str) -> bool:
    return node in {"graph", "runtime", "settings", "trace", "messages"}


def _edge_label(event: dict[str, Any]) -> str:
    result_type = str(event.get("result_type") or "")
    content = event.get("content", {})
    if not isinstance(content, dict):
        return result_type
    if result_type == "llm_response":
        calls = content.get("tool_calls") or []
        if calls and isinstance(calls, list) and isinstance(calls[0], dict):
            return f"{result_type}: {calls[0].get('name')}"
    if result_type == "tool_permission_decision":
        return f"{result_type}: {content.get('behavior')}"
    if result_type in {"tool_selected", "tool_executed"}:
        return f"{result_type}: {content.get('tool_name')}"
    if result_type == "context_compacted":
        return f"{result_type}: {content.get('strategy')}"
    return result_type
