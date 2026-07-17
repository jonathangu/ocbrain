"""Rebuildable local dense-retrieval sidecar for the strict v1 core.

The semantic event/evidence ledger remains authoritative.  This module stores
only derived vectors in a separate SQLite file and talks only to a loopback
Ollama endpoint.  A missing model, server, or sidecar degrades to lexical-only
retrieval; no hosted embedding fallback exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import urllib.error
import urllib.request
from array import array
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_EMBED_MODEL = "qwen3-embedding:4b-fp16"
DEFAULT_EMBED_DIMENSIONS = 1024
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a user or agent question, retrieve the most relevant verified memory, "
    "evidence, decision, correction, or current status."
)
VECTOR_SCHEMA_VERSION = "ocbrain.vectors.v1"


class LocalEmbeddingUnavailable(RuntimeError):
    """The optional local embedding path could not be used."""


def vector_db_path(core_path: Path) -> Path:
    configured = os.environ.get("OCBRAIN_VECTOR_DB")
    if configured:
        return Path(configured).expanduser().resolve()
    return core_path.with_name(f"{core_path.stem}-vectors.sqlite")


def connection_path(conn: sqlite3.Connection) -> Path | None:
    row = conn.execute("PRAGMA database_list").fetchone()
    if row is None:
        return None
    value = str(row[2] or "")
    if not value or value == ":memory:":
        return None
    return Path(value).expanduser().resolve()


def build_vector_index(
    core_path: Path,
    *,
    output_path: Path | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Build an exact-cosine sidecar from the current serving projection."""
    core_path = core_path.expanduser().resolve()
    output_path = (output_path or vector_db_path(core_path)).expanduser().resolve()
    model = model or os.environ.get("OCBRAIN_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    endpoint = endpoint or os.environ.get("OCBRAIN_OLLAMA_URL") or DEFAULT_OLLAMA_URL
    _require_loopback(endpoint)
    model_metadata = _ollama_model_metadata(endpoint, model)
    model_digest = model_metadata.get("digest", "")
    if not model_digest or model_digest == "unknown":
        raise LocalEmbeddingUnavailable("immutable local model digest is unavailable")
    if batch_size < 1 or batch_size > 64:
        raise ValueError("batch_size must be between 1 and 64")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = sqlite3.connect(f"file:{core_path}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        source.execute("BEGIN")
        rows = list(
            source.execute(
                "SELECT belief_id, body, scope_type, scope_id, visibility, egress_policy, "
                "last_compiled_at FROM current_beliefs "
                "WHERE serve=1 AND status='current' ORDER BY belief_id"
            )
        )
        head = source.execute(
            "SELECT event_seq, event_hash FROM brain_events ORDER BY event_seq DESC LIMIT 1"
        ).fetchone()
        source.commit()
    finally:
        source.close()

    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    target = sqlite3.connect(temp_path)
    dimensions: int | None = None
    try:
        target.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE belief_vectors(
              belief_id TEXT PRIMARY KEY,
              content_hash TEXT NOT NULL,
              model TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              vector BLOB NOT NULL,
              scope_type TEXT NOT NULL,
              scope_id TEXT NOT NULL,
              visibility TEXT NOT NULL,
              egress_policy TEXT NOT NULL,
              last_compiled_at TEXT NOT NULL
            );
            CREATE INDEX idx_belief_vectors_scope
              ON belief_vectors(scope_id, egress_policy, visibility);
            """
        )
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = embed_texts(
                [_document_text(row) for row in batch],
                model=model,
                endpoint=endpoint,
                query=False,
                timeout_seconds=300,
            )
            if len(vectors) != len(batch):
                raise LocalEmbeddingUnavailable("embedding response count mismatch")
            for row, vector in zip(batch, vectors, strict=True):
                if dimensions is None:
                    dimensions = len(vector)
                if len(vector) != dimensions:
                    raise LocalEmbeddingUnavailable("embedding dimension changed within build")
                target.execute(
                    "INSERT INTO belief_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["belief_id"],
                        _sha256(str(row["body"])),
                        model,
                        dimensions,
                        _encode_vector(vector),
                        row["scope_type"],
                        row["scope_id"],
                        row["visibility"],
                        row["egress_policy"],
                        row["last_compiled_at"],
                    ),
                )
        built_at = datetime.now(UTC).isoformat(timespec="microseconds")
        metadata = {
            "schema_version": VECTOR_SCHEMA_VERSION,
            "model": model,
            "dimensions": str(dimensions or 0),
            "built_at": built_at,
            "core_path": str(core_path),
            "core_event_seq": str(head["event_seq"] if head else 0),
            "core_event_hash": str(head["event_hash"] if head else ""),
            "rows": str(len(rows)),
            "distance": "exact_cosine",
            "endpoint_class": "loopback_ollama",
            "query_instruction_sha256": _sha256(DEFAULT_QUERY_INSTRUCTION),
            "document_format": "belief_body.v1",
            "model_digest": model_digest,
            "model_quantization": model_metadata.get("quantization", "unknown"),
            "model_parameter_size": model_metadata.get("parameter_size", "unknown"),
        }
        target.executemany("INSERT INTO meta VALUES (?, ?)", metadata.items())
        target.commit()
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"vector sidecar integrity failed: {integrity}")
        target.close()
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, output_path)
        return {
            "status": "ok",
            "schema_version": VECTOR_SCHEMA_VERSION,
            "path": str(output_path),
            "model": model,
            "dimensions": dimensions or 0,
            "rows": len(rows),
            "core_event_seq": int(head["event_seq"] if head else 0),
            "core_event_hash": str(head["event_hash"] if head else ""),
            "endpoint_class": "loopback_ollama",
        }
    except Exception:
        target.close()
        temp_path.unlink(missing_ok=True)
        raise


def vector_status(core_path: Path, *, sidecar_path: Path | None = None) -> dict[str, Any]:
    core_path = core_path.expanduser().resolve()
    path = (sidecar_path or vector_db_path(core_path)).expanduser().resolve()
    if not path.is_file():
        return {"status": "missing", "healthy": False, "path": str(path)}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        meta = {str(row[0]): str(row[1]) for row in conn.execute("SELECT key, value FROM meta")}
        rows = int(conn.execute("SELECT COUNT(*) FROM belief_vectors").fetchone()[0])
        integrity = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        core = sqlite3.connect(f"file:{core_path}?mode=ro", uri=True)
        try:
            head = core.execute(
                "SELECT event_seq, event_hash FROM brain_events ORDER BY event_seq DESC LIMIT 1"
            ).fetchone()
        finally:
            core.close()
        current_seq = str(head[0] if head else 0)
        current_hash = str(head[1] if head else "")
        event_fresh = (
            meta.get("core_event_seq") == current_seq
            and meta.get("core_event_hash") == current_hash
        )
        configured_model = os.environ.get("OCBRAIN_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        configured_dimensions = int(
            os.environ.get("OCBRAIN_EMBED_DIMENSIONS") or DEFAULT_EMBED_DIMENSIONS
        )
        configured_instruction_hash = _sha256(DEFAULT_QUERY_INSTRUCTION)
        endpoint = os.environ.get("OCBRAIN_OLLAMA_URL") or DEFAULT_OLLAMA_URL
        try:
            _require_loopback(endpoint)
            installed = _ollama_model_metadata(endpoint, configured_model)
            installed_digest = installed.get("digest", "")
        except ValueError:
            installed_digest = "invalid_endpoint"
        identity_fresh = (
            meta.get("model") == configured_model
            and meta.get("dimensions") == str(configured_dimensions)
            and meta.get("query_instruction_sha256") == configured_instruction_hash
            and meta.get("model_digest", "unknown") == installed_digest
        )
        fresh = event_fresh and identity_fresh
        healthy = (
            meta.get("schema_version") == VECTOR_SCHEMA_VERSION
            and rows == int(meta.get("rows", "-1"))
            and integrity == "ok"
            and fresh
        )
        return {
            "status": "ok" if healthy else "failed",
            "healthy": healthy,
            "path": str(path),
            "rows": rows,
            "integrity": integrity,
            "fresh": fresh,
            "event_fresh": event_fresh,
            "identity_fresh": identity_fresh,
            "configured_model": configured_model,
            "configured_dimensions": configured_dimensions,
            "configured_query_instruction_sha256": configured_instruction_hash,
            "installed_model_digest": installed_digest,
            "current_core_event_seq": int(current_seq),
            "current_core_event_hash": current_hash,
            "metadata": meta,
        }
    finally:
        conn.close()


def semantic_neighbors(
    conn: sqlite3.Connection,
    query: str,
    *,
    candidate_ids: Iterable[str] | None = None,
    limit: int = 100,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return exact cosine neighbors, or an explicit lexical-fallback reason."""
    core_path = connection_path(conn)
    if core_path is None:
        return [], "core_path_unavailable"
    path = vector_db_path(core_path)
    if not path.is_file():
        return [], "vector_sidecar_missing"
    sidecar = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    sidecar.row_factory = sqlite3.Row
    try:
        meta = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM meta")}
        if meta.get("schema_version") != VECTOR_SCHEMA_VERSION:
            return [], "vector_schema_mismatch"
        core_head = conn.execute(
            "SELECT event_seq, event_hash FROM brain_events ORDER BY event_seq DESC LIMIT 1"
        ).fetchone()
        if meta.get("core_event_seq") != str(
            core_head["event_seq"] if core_head else 0
        ) or meta.get("core_event_hash") != str(core_head["event_hash"] if core_head else ""):
            return [], "vector_sidecar_stale"
        model = meta.get("model") or DEFAULT_EMBED_MODEL
        configured_model = os.environ.get("OCBRAIN_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        if model != configured_model:
            return [], "vector_model_config_mismatch"
        try:
            dimensions = int(meta.get("dimensions") or 0)
            configured_dimensions = int(
                os.environ.get("OCBRAIN_EMBED_DIMENSIONS") or DEFAULT_EMBED_DIMENSIONS
            )
        except ValueError:
            return [], "vector_dimension_metadata_invalid"
        if dimensions <= 0 or dimensions != configured_dimensions:
            return [], "vector_dimension_config_mismatch"
        if meta.get("query_instruction_sha256") != _sha256(DEFAULT_QUERY_INSTRUCTION):
            return [], "vector_query_instruction_mismatch"
        endpoint = os.environ.get("OCBRAIN_OLLAMA_URL") or DEFAULT_OLLAMA_URL
        _require_loopback(endpoint)
        installed = _ollama_model_metadata(endpoint, model)
        installed_digest = installed.get("digest", "")
        if not installed_digest or installed_digest == "unknown":
            return [], "vector_model_identity_unavailable"
        if meta.get("model_digest") != installed_digest:
            return [], "vector_model_digest_mismatch"
        query_vectors = embed_texts(
            [query],
            model=model,
            endpoint=endpoint,
            query=True,
            timeout_seconds=90,
            dimensions=dimensions,
        )
        if not query_vectors:
            return [], "empty_query_embedding"
        query_vector = query_vectors[0]
        if len(query_vector) != dimensions:
            return [], "vector_query_dimension_mismatch"
        allowed = set(candidate_ids) if candidate_ids is not None else None
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in sidecar.execute("SELECT * FROM belief_vectors ORDER BY belief_id"):
            if allowed is not None and str(row["belief_id"]) not in allowed:
                continue
            vector = _decode_vector(row["vector"])
            if len(vector) != len(query_vector):
                return [], "vector_row_dimension_mismatch"
            scored.append((_dot(query_vector, vector), row))
        scored.sort(key=lambda item: (-item[0], str(item[1]["belief_id"])))
        return [
            {
                "belief_id": str(row["belief_id"]),
                "similarity": round(score, 8),
                "content_hash": str(row["content_hash"]),
            }
            for score, row in scored[: max(limit, 1)]
        ], None
    except (OSError, sqlite3.Error, LocalEmbeddingUnavailable, ValueError) as exc:
        return [], f"local_embedding_unavailable:{type(exc).__name__}"
    finally:
        sidecar.close()


def embed_texts(
    texts: list[str],
    *,
    model: str,
    endpoint: str,
    query: bool,
    timeout_seconds: float,
    dimensions: int | None = None,
) -> list[list[float]]:
    _require_loopback(endpoint)
    if not texts:
        return []
    values = [
        f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery: {text}" if query else text for text in texts
    ]
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/embed",
        data=json.dumps(
            {
                "model": model,
                "input": values,
                "truncate": True,
                "keep_alive": "30m",
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise LocalEmbeddingUnavailable(str(exc)) from exc
    if payload.get("error"):
        raise LocalEmbeddingUnavailable(str(payload["error"]))
    raw = payload.get("embeddings")
    if not isinstance(raw, list):
        raise LocalEmbeddingUnavailable("embedding response omitted embeddings")
    vectors: list[list[float]] = []
    for value in raw:
        if not isinstance(value, list) or not value:
            raise LocalEmbeddingUnavailable("embedding response contained an invalid vector")
        target_dimensions = dimensions or int(
            os.environ.get("OCBRAIN_EMBED_DIMENSIONS") or DEFAULT_EMBED_DIMENSIONS
        )
        converted = [float(item) for item in value]
        # Qwen3 embeddings are Matryoshka-trained: taking the leading dimensions
        # before normalization preserves their intended lower-dimensional form.
        vectors.append(_normalize(converted[:target_dimensions]))
    return vectors


def _require_loopback(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("embedding endpoint must be loopback HTTP; hosted fallback is prohibited")


def _ollama_model_metadata(endpoint: str, model: str) -> dict[str, str]:
    """Best-effort immutable model identity from the local Ollama registry."""
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/show",
        data=json.dumps({"model": model}, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return {}
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    digest = str(payload.get("digest") or "")
    if not digest:
        try:
            with urllib.request.urlopen(  # noqa: S310
                endpoint.rstrip("/") + "/api/tags", timeout=15
            ) as response:
                tags = json.loads(response.read())
            for item in tags.get("models", []):
                if item.get("name") == model or item.get("model") == model:
                    digest = str(item.get("digest") or "")
                    break
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
    return {
        "digest": digest or "unknown",
        "quantization": str(details.get("quantization_level") or "unknown"),
        "parameter_size": str(details.get("parameter_size") or "unknown"),
    }


def _normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise LocalEmbeddingUnavailable("embedding vector has zero norm")
    return [value / norm for value in values]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _encode_vector(values: list[float]) -> bytes:
    return array("f", values).tobytes()


def _decode_vector(value: bytes | memoryview) -> list[float]:
    result = array("f")
    result.frombytes(bytes(value))
    return list(result)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _document_text(row: sqlite3.Row) -> str:
    # The projection has already removed archive/catalog/path-only rows.  Keep
    # the derived embedding input deliberately simple and reproducible.
    return str(row["body"]).strip()


__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_EMBED_DIMENSIONS",
    "DEFAULT_OLLAMA_URL",
    "LocalEmbeddingUnavailable",
    "build_vector_index",
    "embed_texts",
    "semantic_neighbors",
    "vector_db_path",
    "vector_status",
]
