from __future__ import annotations

import re

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)(\s*[:=]\s*)([^\s\"']+)"),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}"), r"\1[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED]"),
    (re.compile(r"ghp_[A-Za-z0-9_]{20,}"), "[REDACTED]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"), "[REDACTED]"),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        "[REDACTED_JWT]",
    ),
]

LEAK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    ),
    (
        "assigned_secret",
        re.compile(
            r"(?i)(api[_-]?key|secret|token|password|credential)\s*[:=]\s*(?!\[REDACTED\])[^\s\"']+"
        ),
    ),
]


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def find_probable_secret_leaks(text: str) -> list[str]:
    hits: list[str] = []
    for name, pattern in LEAK_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def claim_key(text: str, *, limit: int = 160) -> str:
    normalized = compact_whitespace(text.lower())
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized[:limit]


def title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        return stripped[:160] or fallback
    return fallback[:160]


def summarize_text(text: str, limit: int = 600) -> str:
    return compact_whitespace(text)[:limit]
