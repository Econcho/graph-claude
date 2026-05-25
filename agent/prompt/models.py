from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptSection:
    name: str
    content: str
    source: str
    cacheable: bool = True
    cache_break: bool = False
    token_estimate: int = 0

    def to_snapshot(self, *, include_content: bool = True) -> dict[str, Any]:
        snapshot = {
            "name": self.name,
            "source": self.source,
            "cacheable": self.cacheable,
            "cache_break": self.cache_break,
            "char_count": len(self.content),
            "token_estimate": self.token_estimate,
        }
        if include_content:
            snapshot["content"] = self.content
        return snapshot


@dataclass(frozen=True)
class PromptBuildResult:
    system_prompt: str
    messages: list[Any]
    tools: list[dict[str, Any]]
    sections: list[PromptSection]
    snapshot: dict[str, Any]
    snapshot_ref: str | None = None
