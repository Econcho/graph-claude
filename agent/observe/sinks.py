from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from agent.observe.events import RuntimeEvent
from agent.observe.serializers import to_jsonable


class EventSink(Protocol):
    def emit(self, event: RuntimeEvent) -> None:
        ...


class JsonlEventSink:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: RuntimeEvent) -> None:
        record = to_jsonable(asdict(event))

        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


class NullEventSink:
    def emit(self, event: RuntimeEvent) -> None:
        return
