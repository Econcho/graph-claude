from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any

from agent.memory.auto_memory.store import AutoMemoryRecord, AutoMemoryStore


class AutoMemoryRecall:
    def __init__(
        self,
        store: AutoMemoryStore,
        *,
        max_results: int = 3,
    ):
        self.store = store
        self.max_results = max_results

    def recall(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        records = self.store.list_memories()
        if not records:
            return []

        query = _build_query(state)
        tokens = _tokenize(query)
        already_surfaced = set(state.get("auto_memory_already_surfaced") or [])
        scored: list[tuple[int, AutoMemoryRecord]] = []

        for record in records:
            if record.id in already_surfaced:
                continue
            score = _score(record, tokens)
            if score > 0:
                scored.append((score, record))

        scored.sort(key=lambda item: (-item[0], item[1].updated_at, item[1].id))
        return [_record_to_prompt_dict(record) for _, record in scored[: self.max_results]]


def _record_to_prompt_dict(record: AutoMemoryRecord) -> dict[str, Any]:
    data = asdict(record)
    data.pop("path", None)
    return data


def _build_query(state: dict[str, Any]) -> str:
    chunks: list[str] = []
    user_input = state.get("user_input")
    if user_input:
        chunks.append(str(user_input))

    messages = list(state.get("messages", []))[-6:]
    for message in messages:
        chunks.append(_message_content(message))

    return "\n".join(chunk for chunk in chunks if chunk)


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[0-9A-Za-z_\-\u4e00-\u9fff]{2,}", text)
    }


def _score(record: AutoMemoryRecord, tokens: set[str]) -> int:
    haystack = " ".join(
        [
            record.type,
            record.name,
            record.description,
            record.content,
            record.why_it_matters,
            record.how_to_apply,
        ]
    ).lower()

    score = 0
    for token in tokens:
        if token in haystack:
            score += 1

    if record.type in tokens:
        score += 2
    return score
