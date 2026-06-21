from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ocbrain.db import EventInput
from ocbrain.ids import content_hash, stable_id
from ocbrain.text import redact_secrets, summarize_text, title_from_text

SAFE_SUFFIXES = {".md", ".txt", ".jsonl", ".log"}
OPTIONAL_SUFFIXES = {".yaml", ".yml"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "Library",
    "Caches",
}
EXCLUDED_NAME_PARTS = {
    ".env",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "token",
    "tokens",
    "keychain",
    "openclaw.json",
    "settings.json",
    ".sqlite",
    ".db",
}


@dataclass(frozen=True)
class IngestOptions:
    max_file_bytes: int = 512_000
    max_event_chars: int = 16_000
    include_yaml: bool = False


def iter_candidate_files(roots: list[Path], options: IngestOptions) -> Iterable[Path]:
    suffixes = set(SAFE_SUFFIXES)
    if options.include_yaml:
        suffixes |= OPTIONAL_SUFFIXES

    for root in roots:
        root = root.expanduser()
        if root.is_file():
            if is_safe_file(root, suffixes, options):
                yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            if is_safe_file(path, suffixes, options):
                yield path


def is_safe_file(path: Path, suffixes: set[str], options: IngestOptions) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_DIR_NAMES:
        return False
    lowered = str(path).lower()
    if any(part in lowered for part in EXCLUDED_NAME_PARTS):
        return False
    if path.suffix.lower() not in suffixes:
        return False
    try:
        if path.stat().st_size > options.max_file_bytes:
            return False
    except OSError:
        return False
    return True


def event_from_file(path: Path, options: IngestOptions) -> EventInput | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None

    redacted = redact_secrets(text)
    body = redacted[: options.max_event_chars]
    source_uri = str(path)
    digest = content_hash(redacted)
    event_id = stable_id("evt", source_uri, digest)
    title = title_from_text(redacted, path.name)
    summary = summarize_text(redacted)
    return EventInput(
        id=event_id,
        source_type=infer_source_type(path),
        source_uri=source_uri,
        content_hash=digest,
        title=title,
        summary=summary,
        body=body,
        scope=infer_scope(path),
        metadata={"size_bytes": path.stat().st_size, "suffix": path.suffix.lower()},
    )


def infer_source_type(path: Path) -> str:
    text = str(path)
    if "/sessions/" in text and path.suffix == ".jsonl":
        return "session"
    if "/memory/" in text or path.name == "MEMORY.md":
        return "memory"
    if "/memory-wiki/" in text:
        return "wiki"
    if "/task-artifacts/" in text:
        return "task-artifact"
    if "/task-status/" in text:
        return "task-status"
    if "/artifacts/" in text:
        return "artifact"
    if "/docs/" in text:
        return "doc"
    return "file"


def infer_scope(path: Path) -> str:
    text = str(path).lower()
    if "family" in text or "private" in text:
        return "private"
    if "public" in text or "site" in text:
        return "public"
    if "pelican" in text or "bountiful" in text or "ocbrain" in text:
        return "project"
    return "workspace"


def default_history_roots(workspace: Path) -> list[Path]:
    home = Path.home()
    return [
        workspace / "MEMORY.md",
        workspace / "memory",
        workspace / "artifacts",
        workspace / "task-artifacts",
        workspace / "task-status",
        workspace / "ocbrain" / "docs",
        home / ".openclaw" / "agents" / "main" / "sessions",
        home / ".openclaw" / "memory-wiki",
    ]
