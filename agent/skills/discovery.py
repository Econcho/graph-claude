from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agent.skills.models import Skill, SkillConfig, SkillSource, SkillSummary
from agent.skills.parser import SkillParseError, parse_skill_file


SOURCE_PRIORITY = {
    "project": 0,
    "user": 1,
    "bundled": 2,
    "mcp": 3,
}


@dataclass(frozen=True)
class SkillDiscoveryResult:
    skills: list[Skill]
    warnings: list[str] = field(default_factory=list)

    @property
    def summaries(self) -> list[SkillSummary]:
        return [
            SkillSummary(
                name=skill.name,
                description=skill.description,
                when_to_use=skill.when_to_use,
                source=skill.source,
                paths=list(skill.paths),
            )
            for skill in self.skills
            if skill.user_invocable
        ]


class SkillDiscovery:
    def __init__(
        self,
        config: SkillConfig,
        *,
        bundled_dir: str | Path | None = None,
        user_home: str | Path | None = None,
    ):
        self.config = config
        self.bundled_dir = Path(bundled_dir) if bundled_dir else Path(__file__).with_name("bundled")
        self.user_home = Path(user_home) if user_home else Path.home()
        self._cache: dict[Path, SkillDiscoveryResult] = {}

    def discover(self, workspace_root: str | Path) -> SkillDiscoveryResult:
        workspace = Path(workspace_root).resolve()
        cached = self._cache.get(workspace)
        if cached is not None:
            return cached

        if not self.config.enabled:
            result = SkillDiscoveryResult(skills=[])
            self._cache[workspace] = result
            return result

        candidates: list[tuple[Path, SkillSource]] = []
        candidates.extend((path, "project") for path in self._project_skill_dirs(workspace))
        candidates.extend((path, "user") for path in self._user_skill_dirs())
        candidates.append((self.bundled_dir, "bundled"))

        skills, warnings = self._load_candidates(candidates)
        result = SkillDiscoveryResult(skills=skills, warnings=warnings)
        self._cache[workspace] = result
        return result

    def _project_skill_dirs(self, workspace: Path) -> list[Path]:
        dirs: list[Path] = []
        current = workspace
        home = self.user_home.resolve()

        while True:
            dirs.append(current / ".agent" / "skills")
            if self.config.scan_claude_compat_dirs:
                dirs.append(current / ".claude" / "skills")

            if current == current.parent or current == home:
                break
            current = current.parent

        return dirs

    def _user_skill_dirs(self) -> list[Path]:
        dirs = [self.user_home / ".agent" / "skills"]
        if self.config.scan_claude_compat_dirs:
            dirs.append(self.user_home / ".claude" / "skills")
        return dirs

    def _load_candidates(
        self,
        candidates: list[tuple[Path, SkillSource]],
    ) -> tuple[list[Skill], list[str]]:
        by_realpath: set[Path] = set()
        loaded_by_source = {"project": 0, "user": 0, "bundled": 0, "mcp": 0}
        candidate_count_by_source = {"project": 0, "user": 0, "bundled": 0, "mcp": 0}
        by_name: dict[str, Skill] = {}
        warnings: list[str] = []

        for base, source in candidates:
            if not base.exists() or not base.is_dir():
                continue

            for skill_file in self._iter_skill_files(base):
                candidate_count_by_source[source] += 1
                if candidate_count_by_source[source] > self.config.max_candidates_per_source:
                    break
                if loaded_by_source[source] >= self.config.max_loaded_per_source:
                    break

                realpath = skill_file.resolve()
                if realpath in by_realpath:
                    continue
                by_realpath.add(realpath)

                try:
                    skill = parse_skill_file(
                        skill_file,
                        source=source,
                        max_bytes=self.config.max_skill_bytes,
                    )
                except (OSError, UnicodeDecodeError, SkillParseError) as error:
                    warnings.append(f"{skill_file}: {error}")
                    continue

                existing = by_name.get(skill.name)
                if existing is not None:
                    if SOURCE_PRIORITY[skill.source] < SOURCE_PRIORITY[existing.source]:
                        warnings.append(
                            f"Skill {skill.name} from {skill.path} overrides {existing.path}"
                        )
                        by_name[skill.name] = skill
                    else:
                        warnings.append(
                            f"Skill {skill.name} from {skill.path} ignored; duplicate of {existing.path}"
                        )
                    continue

                by_name[skill.name] = skill
                loaded_by_source[source] += 1

        skills = sorted(
            by_name.values(),
            key=lambda skill: (SOURCE_PRIORITY[skill.source], skill.name),
        )
        return skills, warnings

    def _iter_skill_files(self, base: Path):
        for child in sorted(base.iterdir(), key=lambda path: path.name):
            if child.is_dir():
                skill_file = child / "SKILL.md"
                if skill_file.exists() and skill_file.is_file():
                    yield skill_file
