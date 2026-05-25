import json
from io import StringIO
from pathlib import Path

import pytest

from agent.observe.trace_reader import find_observation, print_observation


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def test_find_observation_accepts_unique_prefix(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "id": "abcdef01-0000-0000-0000-000000000000",
                "result_type": "node_input",
                "output_node": "graph",
                "input_node": "llm",
                "content": {"state": {"messages": []}},
            }
        ],
    )

    event = find_observation(trace, "abcdef01")

    assert event["input_node"] == "llm"


def test_print_observation_formats_content(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "id": "abcdef01-0000-0000-0000-000000000000",
                "result_type": "node_output",
                "output_node": "llm",
                "input_node": "graph",
                "content": {
                    "node": "llm",
                    "output": {
                        "messages": [
                            {
                                "type": "AIMessage",
                                "content": "hello",
                            }
                        ]
                    },
                },
            }
        ],
    )
    buffer = StringIO()

    event = print_observation(trace, "abcdef01", file=buffer)

    text = buffer.getvalue()
    assert event["result_type"] == "node_output"
    assert "Observation" in text
    assert "result_type : node_output" in text
    assert '"messages": [' in text
    assert '"content": "hello"' in text


def test_find_observation_rejects_ambiguous_prefix(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            {
                "id": "abcdef01-0000-0000-0000-000000000000",
                "result_type": "node_input",
                "output_node": "graph",
                "input_node": "llm",
                "content": {},
            },
            {
                "id": "abcdef02-0000-0000-0000-000000000000",
                "result_type": "node_output",
                "output_node": "llm",
                "input_node": "graph",
                "content": {},
            },
        ],
    )

    with pytest.raises(ValueError, match="ambiguous"):
        find_observation(trace, "abcdef")
