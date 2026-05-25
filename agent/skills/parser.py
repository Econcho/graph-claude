from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.skills.models import Skill, SkillContextMode, SkillShell, SkillSource


class SkillParseError(ValueError):
    pass


def parse_skill_file(
    path: Path,
    *,
    source: SkillSource,
    max_bytes: int,
) -> Skill:
    size = path.stat().st_size
    if size > max_bytes:
        raise SkillParseError(f"SKILL.md is too large: {size} bytes")

    raw = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw)
    data = _parse_frontmatter(frontmatter)

    name = _string_field(data, "name")
    description = _description_field(data)
    if not name:
        raise SkillParseError("frontmatter.name is required")
    if not description:
        raise SkillParseError("frontmatter.description is required")

    context = _string_field(data, "context") or "fork"
    if context not in {"inline", "fork"}:
        raise SkillParseError("frontmatter.context must be inline or fork")

    shell = _string_field(data, "shell")
    if shell is not None and shell not in {"bash", "powershell"}:
        raise SkillParseError("frontmatter.shell must be bash or powershell")

    return Skill(
        name=name,
        description=description,
        body=body.strip(),
        path=path,
        base_dir=path.parent,
        source=source,
        when_to_use=_string_field(data, "when_to_use"),
        allowed_tools=_list_field(data, "allowed_tools"),
        model=_string_field(data, "model"),
        effort=_string_field(data, "effort"),
        user_invocable=_bool_field(data, "user_invocable", True),
        paths=_list_field(data, "paths"),
        version=_string_field(data, "version"),
        context=context,  # type: ignore[arg-type]
        agent=_string_field(data, "agent"),
        shell=shell,  # type: ignore[arg-type]
    )


def _split_frontmatter(raw: str) -> tuple[str, str]:
    normalized = raw.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise SkillParseError("SKILL.md must start with YAML frontmatter")

    end = normalized.find("\n---\n", 4)
    if end == -1:
        raise SkillParseError("SKILL.md frontmatter must be closed by ---")

    return normalized[4:end], normalized[end + 5 :]


def _parse_frontmatter(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        stripped = line.strip()
        if stripped.startswith("- "):
            if current_key is None:
                raise SkillParseError("frontmatter list item has no key")
            result.setdefault(current_key, []).append(_parse_scalar(stripped[2:]))
            continue

        if ":" not in stripped:
            raise SkillParseError(f"invalid frontmatter line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip().replace("-", "_")
        value = value.strip()
        if not key:
            raise SkillParseError("frontmatter key must not be empty")

        if value == "":
            result[key] = []
            current_key = key
        else:
            result[key] = _parse_scalar(value)
            current_key = None

    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _string_field(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(item) for item in value).strip() or None
    return str(value).strip() or None


def _description_field(data: dict[str, Any]) -> str | None:
    value = data.get("description")
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return None
    return str(value).strip() or None


def _list_field(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _bool_field(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
