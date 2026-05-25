from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptConfig:
    enabled: bool = True
    dump_enabled: bool = True
    include_claude_md: bool = True
    include_current_date: bool = True
    include_git_status: bool = True
    section_token_analysis: bool = True
    max_claude_md_chars: int = 20_000
    max_git_status_chars: int = 4_000


def default_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.json"


def load_prompt_config(settings_path: str | Path | None = None) -> PromptConfig:
    path = Path(settings_path) if settings_path else default_settings_path()
    if not path.exists():
        return PromptConfig()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return PromptConfig()

    raw = data.get("prompt", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return PromptConfig()

    defaults = PromptConfig()
    values: dict[str, Any] = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in raw:
            values[field_name] = raw[field_name]

    return PromptConfig(
        enabled=bool(values.get("enabled", defaults.enabled)),
        dump_enabled=bool(values.get("dump_enabled", defaults.dump_enabled)),
        include_claude_md=bool(
            values.get("include_claude_md", defaults.include_claude_md)
        ),
        include_current_date=bool(
            values.get("include_current_date", defaults.include_current_date)
        ),
        include_git_status=bool(
            values.get("include_git_status", defaults.include_git_status)
        ),
        section_token_analysis=bool(
            values.get("section_token_analysis", defaults.section_token_analysis)
        ),
        max_claude_md_chars=int(
            values.get("max_claude_md_chars", defaults.max_claude_md_chars)
        ),
        max_git_status_chars=int(
            values.get("max_git_status_chars", defaults.max_git_status_chars)
        ),
    )
