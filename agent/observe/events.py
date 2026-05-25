from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RuntimeEvent:
    id: str
    result_type: str
    output_node: str
    input_node: str
    content: dict[str, Any] = field(default_factory=dict)


def make_event(
    *,
    result_type: str,
    output_node: str,
    input_node: str,
    content: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        id=str(uuid4()),
        result_type=result_type,
        output_node=output_node,
        input_node=input_node,
        content=content or {},
    )
