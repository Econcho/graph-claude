from __future__ import annotations

import json
import time
from pathlib import Path


def default_auth_cache_path() -> Path:
    return Path.home() / ".agent" / "mcp-needs-auth-cache.json"


class McpAuthCache:
    def __init__(
        self,
        *,
        path: str | Path | None = None,
        ttl_seconds: int = 900,
    ):
        self.path = Path(path) if path else default_auth_cache_path()
        self.ttl_seconds = ttl_seconds

    def mark_needs_auth(self, server_name: str) -> None:
        data = self._read()
        data[server_name] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def needs_auth(self, server_name: str) -> bool:
        data = self._read()
        timestamp = data.get(server_name)
        if not isinstance(timestamp, (int, float)):
            return False
        return (time.time() - float(timestamp)) < self.ttl_seconds

    def clear(self, server_name: str) -> None:
        data = self._read()
        if server_name not in data:
            return
        del data[server_name]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _read(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in data.items()
            if isinstance(value, (int, float))
        }
