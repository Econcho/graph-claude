from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


AUTO_MEMORY_INDEX_TEMPLATE = """# Auto Memory

This file is an index of long-term memories for this workspace.

| id | type | name | description | updated_at |
| --- | --- | --- | --- | --- |
"""

ALLOWED_MEMORY_TYPES = {"user", "feedback", "project", "reference"}


@dataclass(frozen=True)
class AutoMemoryRecord:
    id: str
    type: str
    name: str
    description: str
    content: str
    why_it_matters: str
    how_to_apply: str
    created_at: str
    updated_at: str
    source: str
    path: str | None = None


class AutoMemoryStore:
    def __init__(
        self,
        *,
        workspace_root: str | Path,
        max_memory_chars: int = 20_000,
    ):
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / ".agent" / "auto_memory"
        self.index_path = self.root / "MEMORY.md"
        self.items_dir = self.root / "items"
        self.max_memory_chars = max_memory_chars

    @property
    def ref(self) -> str:
        return str(self.index_path)

    def exists(self) -> bool:
        return self.index_path.exists()

    def ensure(self) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text(AUTO_MEMORY_INDEX_TEMPLATE, encoding="utf-8")

    def read_index_or_template(self) -> str:
        if not self.index_path.exists():
            return AUTO_MEMORY_INDEX_TEMPLATE
        return self.index_path.read_text(encoding="utf-8")

    def list_memories(self) -> list[AutoMemoryRecord]:
        if not self.items_dir.exists():
            return []

        records: list[AutoMemoryRecord] = []
        for path in sorted(self.items_dir.glob("*.md")):
            try:
                records.append(self._parse_record(path))
            except Exception:
                continue
        return records

    def get_memory(self, memory_id: str) -> AutoMemoryRecord | None:
        for record in self.list_memories():
            if record.id == memory_id:
                return record
        return None

    def create_memory(self, proposal: dict[str, Any]) -> AutoMemoryRecord:
        self.ensure()
        now = _utc_now()
        record = AutoMemoryRecord(
            id=self._new_memory_id(str(proposal.get("name") or "memory")),
            type=_normalize_type(proposal.get("type")),
            name=_required_text(proposal, "name"),
            description=_required_text(proposal, "description"),
            content=_required_text(proposal, "content"),
            why_it_matters=_clean_text(proposal.get("why_it_matters")),
            how_to_apply=_clean_text(proposal.get("how_to_apply")),
            created_at=now,
            updated_at=now,
            source=_clean_text(proposal.get("source")) or "auto_memory_extractor",
        )
        self._validate_record(record)
        self._write_record(record)
        self.rebuild_index()
        return record

    def update_memory(self, memory_id: str, proposal: dict[str, Any]) -> AutoMemoryRecord:
        self.ensure()
        existing = self.get_memory(memory_id)
        if existing is None:
            raise ValueError(f"Auto memory not found: {memory_id}")

        record = AutoMemoryRecord(
            id=existing.id,
            type=_normalize_type(proposal.get("type") or existing.type),
            name=_clean_text(proposal.get("name")) or existing.name,
            description=(
                _clean_text(proposal.get("description")) or existing.description
            ),
            content=_clean_text(proposal.get("content")) or existing.content,
            why_it_matters=(
                _clean_text(proposal.get("why_it_matters"))
                or existing.why_it_matters
            ),
            how_to_apply=(
                _clean_text(proposal.get("how_to_apply")) or existing.how_to_apply
            ),
            created_at=existing.created_at,
            updated_at=_utc_now(),
            source=_clean_text(proposal.get("source")) or existing.source,
            path=existing.path,
        )
        self._validate_record(record)
        self._write_record(record)
        self.rebuild_index()
        return record

    def rebuild_index(self) -> None:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Auto Memory",
            "",
            "This file is an index of long-term memories for this workspace.",
            "",
            "| id | type | name | description | updated_at |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in self.list_memories():
            lines.append(
                "| {id} | {type} | {name} | {description} | {updated_at} |".format(
                    id=_escape_table(record.id),
                    type=_escape_table(record.type),
                    name=_escape_table(record.name),
                    description=_escape_table(record.description),
                    updated_at=_escape_table(record.updated_at),
                )
            )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_record(self, record: AutoMemoryRecord) -> None:
        path = self.items_dir / f"{record.id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_record_to_markdown(record), encoding="utf-8")

    def _parse_record(self, path: Path) -> AutoMemoryRecord:
        markdown = path.read_text(encoding="utf-8")
        sections = _parse_sections(markdown)
        meta = _parse_meta(markdown)
        return AutoMemoryRecord(
            id=meta.get("memory_id") or path.stem,
            type=_normalize_type(meta.get("type")),
            name=_parse_title(markdown) or path.stem,
            description=sections.get("description", ""),
            content=sections.get("content", ""),
            why_it_matters=sections.get("why it matters", ""),
            how_to_apply=sections.get("how to apply", ""),
            created_at=meta.get("created_at") or "",
            updated_at=meta.get("updated_at") or "",
            source=meta.get("source") or "",
            path=str(path),
        )

    def _new_memory_id(self, name: str) -> str:
        stem = re.sub(r"[^0-9A-Za-z]+", "-", name.lower()).strip("-")
        stem = stem[:40] or "memory"
        candidate = f"{stem}-{uuid4().hex[:8]}"
        while (self.items_dir / f"{candidate}.md").exists():
            candidate = f"{stem}-{uuid4().hex[:8]}"
        return candidate

    def _validate_record(self, record: AutoMemoryRecord) -> None:
        if record.type not in ALLOWED_MEMORY_TYPES:
            raise ValueError(f"Invalid auto memory type: {record.type}")
        for field in ("name", "description", "content"):
            if not getattr(record, field).strip():
                raise ValueError(f"Auto memory field is required: {field}")
        if len(record.content) > self.max_memory_chars:
            raise ValueError(
                f"Auto memory content exceeds {self.max_memory_chars} characters."
            )


def _record_to_markdown(record: AutoMemoryRecord) -> str:
    return f"""# {record.name}

Memory ID: {record.id}
Type: {record.type}
Created At: {record.created_at}
Updated At: {record.updated_at}
Source: {record.source}

## Description

{record.description}

## Content

{record.content}

## Why It Matters

{record.why_it_matters or "Not specified."}

## How To Apply

{record.how_to_apply or "Use when relevant, and verify current facts with tools."}
"""


def _parse_title(markdown: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _parse_meta(markdown: str) -> dict[str, str]:
    result: dict[str, str] = {}
    key_map = {
        "memory id": "memory_id",
        "type": "type",
        "created at": "created_at",
        "updated at": "updated_at",
        "source": "source",
    }
    for line in markdown.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = key_map.get(raw_key.strip().lower())
        if key:
            result[key] = raw_value.strip()
    return result


def _parse_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", markdown, flags=re.MULTILINE))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        sections[match.group(1).strip().lower()] = markdown[start:end].strip()
    return sections


def _normalize_type(value: Any) -> str:
    text = _clean_text(value) or "project"
    return text if text in ALLOWED_MEMORY_TYPES else "project"


def _required_text(data: dict[str, Any], key: str) -> str:
    value = _clean_text(data.get(key))
    if not value:
        raise ValueError(f"Auto memory proposal missing required field: {key}")
    return value


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
