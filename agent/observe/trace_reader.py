from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

from agent.observe.mermaid import read_trace_events


def find_observation(
    trace_path: str | Path,
    result_id: str,
) -> dict[str, Any]:
    events = read_trace_events(trace_path)
    matches = [
        event
        for event in events
        if str(event.get("id", "")).startswith(result_id)
    ]

    if not matches:
        raise ValueError(f"Observation not found: {result_id}")

    if len(matches) > 1:
        matched_ids = ", ".join(str(event.get("id")) for event in matches)
        raise ValueError(f"Observation id is ambiguous: {result_id}. Matches: {matched_ids}")

    return matches[0]


def print_observation(
    trace_path: str | Path,
    result_id: str,
    *,
    file: TextIO | None = None,
) -> dict[str, Any]:
    target = file or sys.stdout
    event = find_observation(trace_path, result_id)

    print("=" * 80, file=target)
    print("Observation", file=target)
    print("=" * 80, file=target)
    print(f"id          : {event.get('id')}", file=target)
    print(f"result_type : {event.get('result_type')}", file=target)
    print(f"output_node : {event.get('output_node')}", file=target)
    print(f"input_node  : {event.get('input_node')}", file=target)
    print("-" * 80, file=target)
    print("content:", file=target)
    print(
        json.dumps(
            event.get("content", {}),
            ensure_ascii=False,
            indent=2,
        ),
        file=target,
    )
    print("=" * 80, file=target)

    return event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print one observation from a trace.jsonl file.",
    )
    parser.add_argument("trace_path", help="Path to trace.jsonl.")
    parser.add_argument("result_id", help="Full observation id or unique id prefix.")
    args = parser.parse_args()

    try:
        print_observation(args.trace_path, args.result_id)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
