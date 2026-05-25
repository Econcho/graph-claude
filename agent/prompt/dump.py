from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.observe.serializers import to_jsonable
from agent.prompt.models import PromptBuildResult, PromptSection


class PromptDumpService:
    def __init__(self, *, enabled: bool = True, trace_path: str | None = None):
        self.enabled = enabled
        self.trace_path = trace_path

    def write_snapshot(
        self,
        *,
        system_prompt: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
        sections: list[PromptSection],
        section_stats: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str | None]:
        snapshot = {
            "system_prompt": system_prompt,
            "messages": to_jsonable(messages),
            "tools": to_jsonable(tools),
            "sections": [
                section.to_snapshot(include_content=True) for section in sections
            ],
            "section_stats": section_stats,
            "message_count": len(messages),
            "tool_count": len(tools),
        }

        if not self.enabled or not self.trace_path:
            return snapshot, None

        path = Path(self.trace_path).with_name("prompt_snapshot.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(to_jsonable(snapshot), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return snapshot, None

        return snapshot, str(path)
