from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.skills.models import SkillConfig


def default_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.json"


def load_skill_config(settings_path: str | Path | None = None) -> SkillConfig:
    path = Path(settings_path) if settings_path else default_settings_path()
    if not path.exists():
        return SkillConfig()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return SkillConfig()

    raw = data.get("skills", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return SkillConfig()

    defaults = SkillConfig()
    values: dict[str, Any] = {}
    for field_name in defaults.__dataclass_fields__:
        if field_name in raw:
            values[field_name] = raw[field_name]

    return SkillConfig(
        enabled=bool(values.get("enabled", defaults.enabled)),
        scan_claude_compat_dirs=bool(
            values.get("scan_claude_compat_dirs", defaults.scan_claude_compat_dirs)
        ),
        max_candidates_per_source=int(
            values.get("max_candidates_per_source", defaults.max_candidates_per_source)
        ),
        max_loaded_per_source=int(
            values.get("max_loaded_per_source", defaults.max_loaded_per_source)
        ),
        max_prompt_skills=int(
            values.get("max_prompt_skills", defaults.max_prompt_skills)
        ),
        prompt_budget_chars=int(
            values.get("prompt_budget_chars", defaults.prompt_budget_chars)
        ),
        max_skill_bytes=int(values.get("max_skill_bytes", defaults.max_skill_bytes)),
        max_recursion_depth=int(
            values.get("max_recursion_depth", defaults.max_recursion_depth)
        ),
        prompt_shell_enabled=bool(
            values.get("prompt_shell_enabled", defaults.prompt_shell_enabled)
        ),
    )
