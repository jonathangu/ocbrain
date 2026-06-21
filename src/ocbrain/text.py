from __future__ import annotations

import re

SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)(\s*[:=]\s*)([^\s\"']+)"),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{16,}"), r"\1[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED]"),
]


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern, replacement in SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
