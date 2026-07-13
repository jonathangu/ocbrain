"""Local vector codecs shared by core retrieval and optional embedding workers.

This module is deliberately network-free.  Optional companion packages may
produce vectors, while the core only needs to decode vectors already present in
an archive or compatibility database.
"""

from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Mapping
from typing import Any


def encode_embedding(vector: list[float]) -> bytes:
    """Pack a float vector into a stdlib-only ``float32`` byte blob."""
    return array("f", (float(value) for value in vector)).tobytes()


def decode_embedding(blob: bytes | memoryview | None) -> list[float]:
    """Unpack a ``float32`` blob, returning an empty list for missing data."""
    if not blob:
        return []
    values = array("f")
    values.frombytes(bytes(blob))
    return list(values)


def knowledge_text(row: sqlite3.Row | Mapping[str, Any]) -> str:
    """Compose the legacy knowledge text used by compatibility retrieval."""
    parts = [row["title"], row["subject"], row["predicate"], row["value_text"]]
    return " ".join(str(part) for part in parts if part)


__all__ = ["decode_embedding", "encode_embedding", "knowledge_text"]
