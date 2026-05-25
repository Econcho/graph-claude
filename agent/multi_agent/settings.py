from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.multi_agent.models import MultiAgentConfig


def default_settings_path() -> Path:
    return Path(__file__).resolve().parents[1] / "settings.json"


def load_multi_agent_config(settings_path: str | Path | None = None) -> MultiAgentConfig:
    path = Path(settings_path) if settings_path else default_settings_path()
    if not path.exists():
        return MultiAgentConfig(enabled=False)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return MultiAgentConfig(enabled=False)

    raw = data.get("multi_agent", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return MultiAgentConfig()

    defaults = MultiAgentConfig()
    return MultiAgentConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        coordinator_enabled=bool(
            raw.get("coordinator_enabled", defaults.coordinator_enabled)
        ),
        max_background_agents=int(
            raw.get("max_background_agents", defaults.max_background_agents)
        ),
        max_subagent_depth=int(
            raw.get("max_subagent_depth", defaults.max_subagent_depth)
        ),
        default_subagent_tools=_string_list(
            raw.get("default_subagent_tools", defaults.default_subagent_tools)
        ),
        default_teammate_tools=_string_list(
            raw.get("default_teammate_tools", defaults.default_teammate_tools)
        ),
        notifications_enabled=bool(
            raw.get("notifications_enabled", defaults.notifications_enabled)
        ),
        mailbox_poll_enabled=bool(
            raw.get("mailbox_poll_enabled", defaults.mailbox_poll_enabled)
        ),
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
