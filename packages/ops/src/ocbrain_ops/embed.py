"""Budget-capped, egress-audited OpenAI embeddings for the semantic layer (v0.3).

This is the write side of the semantic layer: it computes vector embeddings for
knowledge rows and stores them as ``float32`` blobs in ``knowledge.embedding`` so
:mod:`ocbrain.retrieve` can blend cosine similarity into retrieval and drive
vector attribution. It mirrors :mod:`ocbrain.judge` exactly for its hard
guarantees (R6/R8):

* **Private scope never leaves.** Rows whose composed scope (row + linked
  evidence) is ``private`` are dropped before any dispatch — never embedded,
  never sent — as are rows failing ``egress_allowed(..., 'hosted_teacher')``.
* **Redaction before send.** Every dispatched text passes through
  ``redact_secrets``.
* **Every batch is audited.** A row is written to ``egress_audits`` (included +
  rejected ids + payload hash) via ``record_egress_audit``.
* **Budget capped.** When today's spend reaches ``embed.daily_usd_cap`` the run
  records ``embed_runs.status='skipped_budget'`` and dispatches nothing.
* **Inert without a key.** With ``embed.api_key_env`` unset (or ``enabled=false``)
  the run records a ``skipped`` row and never touches the network.

Only *worthwhile* rows are embedded, and only when their vector is missing or
stale (``embedded_at`` NULL or older than ``updated_at``). Priority is
``inject=1`` first, then labeled rows, then retrieval-touched rows — the
never-referenced catalog backlog is never embedded, so spend is not wasted on it.

The API key is read from the process environment and never persisted, logged, or
printed. ``embed_runs`` stores only accounting (ids, counts, cost), never the
dispatched bodies.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ocbrain.db import knowledge_evidence, most_restrictive_scope, now_iso
from ocbrain.egress import record_egress_audit
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, egress_allowed
from ocbrain.text import redact_secrets
from ocbrain.vector import decode_embedding as decode_embedding
from ocbrain.vector import encode_embedding, knowledge_text

from ocbrain_ops.judge import scope_tag_for_privacy

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

# The hosted-egress target reused for the embeddings provider: embeddings go to
# the same hosted vendor as the judge, so the same egress gate applies (R8).
_EGRESS_TARGET = "hosted_teacher"

# Generous per-run row cap. The daily budget + batch loop are the real limiters;
# this just keeps the candidate SELECT bounded on a huge knowledge table.
_DEFAULT_ROW_CAP = 2000


# --------------------------------------------------------------------------- #
# Budget accounting
# --------------------------------------------------------------------------- #
def spent_today(conn: sqlite3.Connection, *, now: datetime | None = None) -> float:
    """Sum ``embed_runs.cost_usd`` for the current UTC day."""
    now = now or datetime.now(UTC)
    day = now.date().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM embed_runs WHERE substr(ts, 1, 10) = ?",
        (day,),
    ).fetchone()
    return float(row["spent"] or 0.0)


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def eligible_rows(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    limit: int = _DEFAULT_ROW_CAP,
) -> list[sqlite3.Row]:
    """Knowledge rows worth embedding, in priority order.

    Eligible rows are un-quarantined ``candidate``/``current`` rows whose vector
    is missing or stale (``embedding`` NULL, ``embedded_at`` NULL, or
    ``embedded_at < updated_at``) and that fall into one of three worthwhile
    tiers:

    1. ``inject=1`` — the injectable memory set (highest priority),
    2. labeled rows (``quality_label`` set),
    3. retrieval-touched rows (present in ``retrieval_uses``).

    A never-referenced, unlabeled, non-injected row — i.e. the archived catalog
    backlog — never matches, so embedding spend is never wasted on it. Private
    rows are excluded here and dropped again at dispatch (defense in depth).
    """
    return conn.execute(
        """
        SELECT k.*,
          CASE
            WHEN k.inject = 1 THEN 0
            WHEN k.quality_label IS NOT NULL THEN 1
            ELSE 2
          END AS embed_priority
        FROM knowledge k
        WHERE k.status IN ('candidate', 'current')
          AND k.quarantine_reason IS NULL
          AND k.privacy_scope != 'private'
          AND (
            k.embedding IS NULL
            OR k.embedded_at IS NULL
            OR k.embedded_at < k.updated_at
          )
          AND (
            k.inject = 1
            OR k.quality_label IS NOT NULL
            OR EXISTS (
              SELECT 1 FROM retrieval_uses ru WHERE ru.knowledge_id = k.id
            )
          )
          AND NOT (
            k.type = 'doc'
            AND COALESCE(k.origin, '') NOT IN ('human', 'harvest', 'loop')
            AND k.inject = 0
            AND NOT EXISTS (
              SELECT 1 FROM retrieval_uses ru2 WHERE ru2.knowledge_id = k.id
            )
          )
        ORDER BY embed_priority ASC, k.updated_at DESC, k.id ASC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()


def composed_scope(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    """Most-restrictive scope over the knowledge row and all linked evidence."""
    scopes = [row["privacy_scope"]]
    for evidence in knowledge_evidence(conn, row["id"]):
        scopes.append(evidence["privacy_scope"])
    return most_restrictive_scope(*scopes)


def build_embed_batch(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    *,
    context: ScopeContext | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(included, rejected)`` after scope + egress + redaction.

    ``included`` items carry the *redacted* text to dispatch; ``rejected`` items
    carry only ids + reason and are never sent. Private-scope rows and rows that
    fail the hosted-egress gate are rejected.
    """
    context = context or ScopeContext()
    included: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        scope = composed_scope(conn, row)
        if scope == "private":
            rejected.append({"id": row["id"], "reason": "private_scope"})
            continue
        scope_tag = scope_tag_for_privacy(scope)
        allowed, reason = egress_allowed(scope_tag, context, _EGRESS_TARGET)
        if not allowed:
            rejected.append({"id": row["id"], "reason": reason})
            continue
        text = redact_secrets(knowledge_text(row))
        if not text.strip():
            rejected.append({"id": row["id"], "reason": "empty_text"})
            continue
        included.append({"id": row["id"], "scope": scope, "text": text})
    return included, rejected


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def call_openai_embeddings(
    payload: dict[str, Any],
    *,
    api_key: str,
    model: str,
    url: str = OPENAI_EMBEDDINGS_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST an embeddings request via stdlib urllib (no third-party deps).

    Injected as ``call=`` in tests so the network is never touched there.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed https OpenAI endpoint
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"embedding call failed: {exc}") from exc


def _cost_usd(cfg: Any, model: str, usage: dict[str, Any]) -> float:
    price = cfg.embed.price_per_mtok.get(model)
    if not price:
        return 0.0
    total = usage.get("total_tokens")
    if total is None:
        total = usage.get("prompt_tokens", 0)
    return float(total or 0) / 1_000_000 * float(price)


def _parse_embeddings(response: dict[str, Any]) -> list[list[float]]:
    """Extract per-item vectors from an embeddings response, ordered by index."""
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if not isinstance(data, list):
        return []
    ordered = sorted(
        (d for d in data if isinstance(d, dict)),
        key=lambda d: d.get("index", 0),
    )
    vectors: list[list[float]] = []
    for item in ordered:
        emb = item.get("embedding")
        if isinstance(emb, list):
            vectors.append([float(x) for x in emb])
    return vectors


def _build_payload(model: str, included: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": model, "input": [item["text"] for item in included]}


def _write_egress_audit(
    conn: sqlite3.Connection,
    included: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> str:
    payload_text = "\n\n".join(item["text"] for item in included)
    result = {
        "target": _EGRESS_TARGET,
        "context": {},
        "query": None,
        "included": [
            {"id": item["id"], "scope": item["scope"], "body": item["text"]} for item in included
        ],
        "rejected": rejected,
        "payload_hash": sha256_text(payload_text),
    }
    return record_egress_audit(conn, result)


def _record_run(
    conn: sqlite3.Connection,
    *,
    status: str,
    items: int = 0,
    cost_usd: float = 0.0,
    error: str | None = None,
    request_hash: str | None = None,
    ts: str | None = None,
) -> str:
    ts = ts or now_iso()
    run_id = stable_id("embed", request_hash or status, ts)
    conn.execute(
        """
        INSERT OR IGNORE INTO embed_runs (id, ts, items, cost_usd, status, error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, ts, items, cost_usd, status, error),
    )
    return run_id


def _store_vectors(
    conn: sqlite3.Connection,
    included: list[dict[str, Any]],
    vectors: list[list[float]],
    *,
    ts: str,
) -> int:
    """Write ``float32`` blobs onto the matching knowledge rows. Returns count."""
    stored = 0
    for item, vector in zip(included, vectors, strict=False):
        if not vector:
            continue
        conn.execute(
            "UPDATE knowledge SET embedding = ?, embedded_at = ? WHERE id = ?",
            (encode_embedding(vector), ts, item["id"]),
        )
        stored += 1
    return stored


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def embed_pending(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    call: Any = call_openai_embeddings,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Embed pending knowledge rows within budget; store ``float32`` vectors.

    Returns a MaintenanceResult-shaped summary. Never raises for the ordinary
    disabled / no-key / budget / no-eligible paths.
    """
    now = now or datetime.now(UTC)
    # ``os.environ`` is a ``Mapping[str, str]`` (not a plain ``dict``); resolve
    # to a concrete, always-non-None mapping so the lookup below can never deref
    # ``None``.
    resolved_env: Mapping[str, str] = os.environ if env is None else env
    model = cfg.embed.model

    if not cfg.embed.enabled:
        _record_run(conn, status="skipped")
        return {"action": "embed", "changed": 0, "status": "skipped", "reason": "disabled"}

    api_key = resolved_env.get(cfg.embed.api_key_env)
    if not api_key:
        _record_run(conn, status="skipped")
        return {"action": "embed", "changed": 0, "status": "skipped", "reason": "no_api_key"}

    if spent_today(conn, now=now) >= cfg.embed.daily_usd_cap:
        _record_run(conn, status="skipped_budget")
        return {"action": "embed", "changed": 0, "status": "skipped_budget"}

    rows = eligible_rows(conn, cfg)
    if not rows:
        _record_run(conn, status="ok")
        return {"action": "embed", "changed": 0, "status": "ok"}

    total_cost = spent_today(conn, now=now)
    stored_total = 0
    batches = 0
    for start in range(0, len(rows), cfg.embed.batch_size):
        batch = rows[start : start + cfg.embed.batch_size]
        included, rejected = build_embed_batch(conn, batch)

        audit_id = _write_egress_audit(conn, included, rejected)
        # The audit is durable before any hosted call, and the single-writer
        # slot is free for MCP feedback/stallcheck while network I/O waits.
        conn.commit()
        if not included:
            _record_run(conn, status="skipped_egress", items=0, request_hash=audit_id)
            conn.commit()
            continue

        payload = _build_payload(model, included)
        request_hash = sha256_text(canonical_json(payload))
        response = call(payload, api_key=api_key, model=model)
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        cost = _cost_usd(cfg, model, usage)
        vectors = _parse_embeddings(response)

        ts = now_iso()
        stored = _store_vectors(conn, included, vectors, ts=ts)
        _record_run(
            conn,
            status="ok",
            items=stored,
            cost_usd=cost,
            request_hash=request_hash,
            ts=ts,
        )
        # Bound the second write window to one completed provider batch.
        conn.commit()
        stored_total += stored
        batches += 1
        total_cost += cost
        if total_cost >= cfg.embed.daily_usd_cap:
            break

    return {
        "action": "embed",
        "changed": stored_total,
        "status": "ok",
        "batches": batches,
    }


def embed_query(
    query: str,
    cfg: Any,
    *,
    call: Any = call_openai_embeddings,
    env: Mapping[str, str] | None = None,
) -> list[float]:
    """Embed a single query string, returning its vector (``[]`` when inert).

    A convenience for retrieval callers that need a query vector to blend against
    stored knowledge vectors. Inert (returns ``[]``) when embedding is disabled or
    no API key is present, so callers transparently fall back to lexical-only.
    Never audited or budget-charged: the query text is the caller's own, not
    stored knowledge, and is redacted before dispatch.
    """
    resolved_env: Mapping[str, str] = os.environ if env is None else env
    if not cfg.embed.enabled:
        return []
    api_key = resolved_env.get(cfg.embed.api_key_env)
    if not api_key:
        return []
    text = redact_secrets(query)
    if not text.strip():
        return []
    payload = {"model": cfg.embed.model, "input": [text]}
    response = call(payload, api_key=api_key, model=cfg.embed.model)
    vectors = _parse_embeddings(response)
    return vectors[0] if vectors else []
