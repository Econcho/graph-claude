from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from agent.observe.config import load_observe_config
from agent.observe.observer import RuntimeObserver
from agent.observe.sinks import JsonlEventSink, NullEventSink


def create_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    return f"run_{timestamp}_{suffix}"


def create_observer(
    *,
    base_dir: str | Path = ".runs",
    run_id: str | None = None,
    enabled: bool = True,
    config_path: str | Path | None = None,
) -> RuntimeObserver:
    actual_run_id = run_id or create_run_id()
    config = load_observe_config(config_path)

    if not enabled:
        return RuntimeObserver(
            run_id=actual_run_id,
            sink=NullEventSink(),
            enabled=False,
            trace_path=None,
            config=config,
        )

    run_dir = Path(base_dir) / actual_run_id
    trace_path = run_dir / "trace.jsonl"
    sink = JsonlEventSink(trace_path)

    return RuntimeObserver(
        run_id=actual_run_id,
        sink=sink,
        enabled=True,
        trace_path=str(trace_path),
        config=config,
    )
