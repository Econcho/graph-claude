from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


SESSION_MEMORY_TEMPLATE = """# Session Memory

## Current State

No session summary has been written yet.

## Task

Unknown.

## Important Files

None.

## Decisions

None.

## Errors and Fixes

None.

## Pending Work

None.

## Worklog

No work recorded yet.
"""

REQUIRED_SECTIONS = [
    "# Session Memory",
    "## Current State",
    "## Task",
    "## Important Files",
    "## Decisions",
    "## Errors and Fixes",
    "## Pending Work",
    "## Worklog",
]


@dataclass(frozen=True)
class SessionMemoryValidation:
    valid: bool
    errors: list[str]


class SessionMemoryStore:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        session_id: str | None,
        max_chars: int = 20_000,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.session_id = session_id or "default"
        self.max_chars = max_chars
        self.path = (
            self.workspace_root
            / ".agent"
            / "session_memory"
            / f"{_safe_session_id(self.session_id)}.md"
        )

    @property
    def ref(self) -> str:
        return str(self.path)

    def exists(self) -> bool:
        return self.path.exists()

    def read_or_template(self) -> str:
        if not self.path.exists():
            return SESSION_MEMORY_TEMPLATE
        return self.path.read_text(encoding="utf-8")

    def write(self, markdown: str) -> None:
        validation = self.validate(markdown)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(markdown, encoding="utf-8")

    def validate(self, markdown: str) -> SessionMemoryValidation:
        errors: list[str] = []

        if not isinstance(markdown, str) or not markdown.strip():
            errors.append("Session Memory markdown must be non-empty.")
            return SessionMemoryValidation(valid=False, errors=errors)

        text = markdown.strip()
        if text.startswith("```") and text.endswith("```"):
            errors.append("Session Memory markdown must not be wrapped in a code block.")

        if len(markdown) > self.max_chars:
            errors.append(
                f"Session Memory markdown exceeds {self.max_chars} characters."
            )

        for section in REQUIRED_SECTIONS:
            if section not in markdown:
                errors.append(f"Missing required section: {section}")

        return SessionMemoryValidation(valid=not errors, errors=errors)


def _safe_session_id(session_id: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", session_id.strip())
    return safe or "default"
