from __future__ import annotations

from pathlib import Path


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".runs",
    "dist",
    "build",
}

TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".css",
    ".html",
    ".xml",
    ".csv",
}


def is_excluded(path: Path) -> bool:
    return any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts)


def is_probably_binary(path: Path, sample_size: int = 2048) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return True
    return b"\x00" in sample


def iter_workspace_files(root: Path, start: Path):
    for path in start.rglob("*"):
        rel = path.relative_to(root)
        if is_excluded(rel):
            continue
        if path.is_file():
            yield path


def validate_glob_pattern(pattern: str) -> None:
    if not pattern.strip():
        raise ValueError("pattern must be a non-empty string")
    parts = Path(pattern).parts
    if any(part == ".." for part in parts):
        raise ValueError("pattern must not contain '..'")
    if Path(pattern).is_absolute():
        raise ValueError("pattern must be relative to the workspace")
