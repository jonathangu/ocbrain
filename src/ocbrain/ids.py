from __future__ import annotations

import hashlib


def stable_id(prefix: str, *parts: str, length: int = 16) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return f"{prefix}_{digest.hexdigest()[:length]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
