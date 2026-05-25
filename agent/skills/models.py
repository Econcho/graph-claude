from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


SkillSource = Literal["project", "user", "bundled", "mcp"]
SkillContextMode = Literal["inline", "fork"]
SkillShell = Literal["bash", "powershell"]


@dataclass(frozen=True)
class SkillConfig:
    enabled: bool = True
    scan_claude_compat_dirs: bool = True
    max_candidates_per_source: int = 300
    max_loaded_per_source: int = 200
    max_prompt_skills: int = 150
    prompt_budget_chars: int = 30_000
    max_skill_bytes: int = 262_144
    max_recursion_depth: int = 3
    prompt_shell_enabled: bool = True


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path
    base_dir: Path
    source: SkillSource
    when_to_use: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    user_invocable: bool = True
    paths: list[str] = field(default_factory=list)
    version: str | None = None
    context: SkillContextMode = "fork"
    agent: str | None = None
    shell: SkillShell | None = None


@dataclass(frozen=True)
class SkillSummary:
    name: str
    description: str
    when_to_use: str | None = None
    source: SkillSource = "project"
    paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "source": self.source,
            "paths": list(self.paths),
        }
