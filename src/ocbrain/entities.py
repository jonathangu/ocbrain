"""Deterministic entity extraction: the things a query can name.

No model, no network. The vocabulary is operator config; the shapes below are
code.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

Vocabulary = Mapping[str, Sequence[str]]

KIND_ORDER: dict[str, int] = {
    "vocab": 0,
    "pr": 1,
    "id": 2,
    "host": 3,
    "digest": 4,
    "date": 5,
}

MIN_ENTITY_CHARS = 3

FILE_EXTENSIONS = frozenset(
    {
        "py",
        "md",
        "json",
        "jsonl",
        "yaml",
        "yml",
        "toml",
        "txt",
        "csv",
        "tsv",
        "sql",
        "sqlite",
        "sqlite3",
        "db",
        "log",
        "html",
        "htm",
        "js",
        "mjs",
        "cjs",
        "ts",
        "tsx",
        "jsx",
        "sh",
        "bash",
        "zsh",
        "ini",
        "cfg",
        "conf",
        "lock",
        "xml",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "svg",
        "pdf",
    }
)

_BOUNDARY = r"(?<![0-9A-Za-z_])"
_TRAILING = r"(?![0-9A-Za-z_])"

_PR_RE = re.compile(
    _BOUNDARY + r"(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#\d{2,6}" + _TRAILING
)
_ID_RE = re.compile(_BOUNDARY + r"(?:belief|evd|close|ret|evt)_[0-9a-f]{16}" + _TRAILING)
# A URL's hostname is matched as the bare host that follows its ``//``.
_HOST_RE = re.compile(_BOUNDARY + r"[a-z0-9-]+(?:\.[a-z0-9-]+)+" + _TRAILING, re.IGNORECASE)
_DIGEST_RE = re.compile(_BOUNDARY + r"[0-9a-f]{7,64}" + _TRAILING)
_DATE_RE = re.compile(
    _BOUNDARY + r"\d{4}-\d{2}-\d{2}" + _TRAILING
    + r"|" + _BOUNDARY + r"\d{8}(?=[-T])"
    + r"|(?<=[-T])\d{8}" + _TRAILING
)


def _vocabulary_patterns(vocabulary: Vocabulary | None) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for canonical in sorted(vocabulary or {}):
        names = {str(name) for name in (canonical, *(vocabulary[canonical] or ())) if str(name)}
        if not names:
            continue
        alternation = "|".join(
            re.escape(name) for name in sorted(names, key=lambda name: (-len(name), name))
        )
        patterns.append(
            (
                str(canonical),
                re.compile(_BOUNDARY + alternation + _TRAILING, re.IGNORECASE),
            )
        )
    return patterns


def extract_entities(
    text: str, vocabulary: Vocabulary | None = None
) -> list[tuple[str, str]]:
    """Return ``(entity, kind)`` pairs found in ``text``, deduplicated and ordered."""
    if not text:
        return []
    haystack = str(text)
    found: set[tuple[str, str]] = set()
    for canonical, pattern in _vocabulary_patterns(vocabulary):
        if pattern.search(haystack):
            found.add((canonical, "vocab"))
    for match in _PR_RE.finditer(haystack):
        found.add((match.group(0).lower(), "pr"))
    for match in _ID_RE.finditer(haystack):
        found.add((match.group(0).lower(), "id"))
    for match in _HOST_RE.finditer(haystack):
        host = match.group(0).lower()
        labels = host.split(".")
        if labels[-1] in FILE_EXTENSIONS:
            continue
        if all(len(label) <= 2 for label in labels) or all(label.isdigit() for label in labels):
            continue
        found.add((host, "host"))
    for match in _DIGEST_RE.finditer(haystack):
        found.add((match.group(0).lower(), "digest"))
    for match in _DATE_RE.finditer(haystack):
        found.add((match.group(0), "date"))
    return sorted(
        (pair for pair in found if len(pair[0]) >= MIN_ENTITY_CHARS),
        key=lambda pair: (KIND_ORDER[pair[1]], pair[0]),
    )
