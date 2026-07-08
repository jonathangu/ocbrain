from __future__ import annotations

import math
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


# --- Prompt-injection scanners (shared: quarantine tripwires, injectable guard,
# dataset scrub). Mirrors the (name, pattern) shape of LEAK_PATTERNS above so
# find_probable_injection() reads like find_probable_secret_leaks(). ---
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_previous",
        re.compile(
            r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|preceding|all)\b[^.\n]{0,20}"
            r"\b(instruction|instructions|prompt|prompts|context|rules?|directions?)\b"
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"(?i)(\byou are now\b|\bact as\b|\bpretend to be\b|\bpretend you\b|"
            r"\bfrom now on you\b|\bnew (system )?instructions?\s*:|"
            r"^\s*system\s*:|\bdisregard your (persona|role|guidelines)\b)"
        ),
    ),
    (
        "tool_coax",
        re.compile(
            r"(?i)(\brun the following\b|\bexecute (this|the following)\b|"
            r"\b(call|invoke|use) the [\w.\-]+ tool\b|\bcurl\s+https?://|"
            r"\brm\s+-rf\b|\bsend (them|it|the) (to|contents)\b)"
        ),
    ),
    (
        "exfil_link",
        re.compile(
            r"(?i)(!\[[^\]]*\]\(https?://|"
            r"https?://[^\s)]+\?[^\s)]*=[^\s)]{6,})"
        ),
    ),
    ("base64_blob", re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")),
    (
        "invisible_chars",
        # zero-width, bidi-control, and BOM characters used to hide payloads.
        re.compile("[​-‏‪-‮⁦-⁩﻿]"),
    ),
    (
        "prompt_leak_probe",
        re.compile(
            r"(?i)(\brepeat (the )?(words|text|everything)\b[^.\n]{0,20}\babove\b|"
            r"\b(print|reveal|show|expose|leak)\b[^.\n]{0,30}"
            r"\b(system prompt|your (instructions|prompt|rules)|initial prompt)\b|"
            r"\bwhat (are|were) your (instructions|initial instructions)\b)"
        ),
    ),
]


def find_probable_injection(text: str) -> list[str]:
    """Return the names of injection patterns that fire against ``text``.

    Mirrors :func:`find_probable_secret_leaks`. Empty list == clean.
    """
    hits: list[str] = []
    for name, pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            hits.append(name)
    return hits


# Character classes that make up base64-/hex-like runs worth entropy-checking.
_ENTROPY_RUN_RE = re.compile(r"[A-Za-z0-9+/=_-]{40,}|\b[0-9a-fA-F]{40,}\b")


def _shannon_entropy(span: str) -> float:
    if not span:
        return 0.0
    counts: dict[str, int] = {}
    for ch in span:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(span)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def find_high_entropy_spans(
    text: str, *, min_length: int = 40, min_entropy: float = 3.5
) -> list[str]:
    """Return long, high-entropy base64/hex runs (e.g. dumped blobs/keys).

    A span qualifies only if it is at least ``min_length`` chars AND its
    Shannon entropy exceeds ``min_entropy`` bits/char, which keeps ordinary
    long words (low entropy) out while catching random-looking blobs.
    """
    spans: list[str] = []
    for match in _ENTROPY_RUN_RE.finditer(text):
        span = match.group(0)
        if len(span) >= min_length and _shannon_entropy(span) >= min_entropy:
            spans.append(span)
    return spans


# --- Correction / affirmation detection (shared by review signals + DPO mining,
# per spec R2: one implementation, imported by both). ---
AFFIRMATION_RE = re.compile(
    r"(?i)\b(thanks|thank you|thx|perfect|great|nice|love it|ship it|"
    r"beautiful|lgtm|awesome|exactly right|looks good|works|wonderful)\b"
)

# (pattern, weight) correction cues. Weights accumulate and cap at 1.0.
_CORRECTION_CUES: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"(?i)\b(that'?s |it'?s )?(wrong|incorrect|not right|not correct)\b"), 0.6),
    (re.compile(r"(?i)\b(that'?s not|not what i|didn'?t (ask|want)|that isn'?t)\b"), 0.5),
    (re.compile(r"(?i)\b(should( not| n'?t)? be|shouldn'?t|supposed to be|misunderstood)\b"), 0.4),
    (re.compile(r"(?i)\b(revert|undo|roll ?back|the mistake|you broke)\b"), 0.4),
    (re.compile(r"(?i)\bactually\b"), 0.3),
    (re.compile(r"(?i)\b(instead|rather)\b"), 0.3),
    (re.compile(r"(?i)\b(don'?t|do not|stop|never|no longer)\b"), 0.3),
    (re.compile(r"(?i)\b(fix|redo|try again|correct(ion)?)\b"), 0.3),
    (re.compile(r"(?i)^\s*(no,|nope\b|no\b)"), 0.3),
]


def correction_score(text: str) -> float:
    """Score how strongly ``text`` reads as a user correction, in [0, 1].

    Returns >= 0.6 for clear corrections. Pure affirmations are zeroed even if a
    weak cue happens to match (e.g. "thanks, that fixed it" is not a correction).
    """
    if not text or not text.strip():
        return 0.0
    score = 0.0
    strongest = 0.0
    for pattern, weight in _CORRECTION_CUES:
        if pattern.search(text):
            score += weight
            strongest = max(strongest, weight)
    score = min(1.0, score)
    if AFFIRMATION_RE.search(text) and strongest < 0.5:
        return 0.0
    return score


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
