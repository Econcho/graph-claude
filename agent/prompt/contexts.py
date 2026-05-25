from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from agent.prompt.analyzer import with_token_estimate
from agent.prompt.config import PromptConfig
from agent.prompt.models import PromptSection


class UserContextProvider:
    def __init__(self, config: PromptConfig):
        self.config = config

    def build(self, state: dict) -> list[PromptSection]:
        workspace_root = state.get("workspace_root") or state.get("workspace")
        sections: list[PromptSection] = []

        if self.config.include_claude_md and workspace_root:
            claude_md = self._read_claude_md(Path(workspace_root))
            if claude_md:
                sections.append(
                    _section(
                        "user_context.claude_md",
                        claude_md,
                        source="user_context",
                        cacheable=False,
                    )
                )

        if self.config.include_current_date:
            today = datetime.now().date().isoformat()
            sections.append(
                _section(
                    "user_context.current_date",
                    f"Today's date is {today}.",
                    source="user_context",
                    cacheable=False,
                )
            )

        return sections

    def _read_claude_md(self, workspace: Path) -> str | None:
        current = workspace.resolve()
        parts: list[str] = []

        while True:
            candidate = current / "CLAUDE.md"
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8").strip()
                except Exception:
                    content = ""
                if content:
                    parts.append(f"# CLAUDE.md: {candidate}\n\n{content}")

            if current == current.parent:
                break
            current = current.parent

        if not parts:
            return None

        combined = "\n\n".join(reversed(parts))
        if len(combined) <= self.config.max_claude_md_chars:
            return combined

        omitted = len(combined) - self.config.max_claude_md_chars
        return combined[: self.config.max_claude_md_chars] + (
            f"\n...[truncated {omitted} chars]"
        )


class SystemContextProvider:
    def __init__(self, config: PromptConfig):
        self.config = config

    def build(self, state: dict) -> list[PromptSection]:
        workspace_root = state.get("workspace_root") or state.get("workspace")
        if not self.config.include_git_status or not workspace_root:
            return []

        git_status = self._git_status(Path(workspace_root))
        if not git_status:
            return []

        return [
            _section(
                "system_context.git_status",
                git_status,
                source="system_context",
                cacheable=False,
            )
        ]

    def _git_status(self, workspace: Path) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "status", "--short", "--branch"],
                cwd=workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return None

        if completed.returncode != 0:
            return None

        output = completed.stdout.strip()
        if not output:
            return None

        if len(output) <= self.config.max_git_status_chars:
            return "Git status:\n" + output

        omitted = len(output) - self.config.max_git_status_chars
        return (
            "Git status:\n"
            + output[: self.config.max_git_status_chars]
            + f"\n...[truncated {omitted} chars]"
        )


def _section(
    name: str,
    content: str,
    *,
    source: str,
    cacheable: bool,
) -> PromptSection:
    return with_token_estimate(
        PromptSection(
            name=name,
            content=content.strip(),
            source=source,
            cacheable=cacheable,
            cache_break=not cacheable,
        )
    )
