from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.db import connect, get_knowledge, init_db, now_iso, upsert_knowledge
from ocbrain_ops.embed import (
    build_embed_batch,
    decode_embedding,
    eligible_rows,
    embed_pending,
    embed_query,
    encode_embedding,
    spent_today,
)

KEY_ENV = {"OPENAI_API_KEY": "sk-test-not-a-real-key-000000000000"}


def _cfg(tmp_path: Path, **embed_overrides):
    base = load_config(tmp_path / "cfg.json")
    # Functional embedding tests opt into the hosted lane explicitly; the
    # product default is fail-closed.
    settings = {"enabled": True, **embed_overrides}
    return replace(base, embed=replace(base.embed, **settings))


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _know(
    conn: sqlite3.Connection,
    predicate: str,
    *,
    scope: str = "workspace",
    value_text: str | None = None,
) -> str:
    return upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject="runtime",
        predicate=predicate,
        value_text=value_text or f"claim about {predicate}",
        status="current",
        privacy_scope=scope,
    )


def _set(conn: sqlite3.Connection, kid: str, **cols) -> None:
    assignments = ", ".join(f"{k}=?" for k in cols)
    conn.execute(f"UPDATE knowledge SET {assignments} WHERE id=?", (*cols.values(), kid))


def _touch_retrieval(conn: sqlite3.Connection, kid: str) -> None:
    conn.execute(
        "INSERT INTO retrieval_uses (id, knowledge_id, served_at) VALUES (?, ?, ?)",
        (f"ru:{kid}", kid, now_iso()),
    )


def _stub_embed(vector=None):
    """Fake embeddings endpoint: one deterministic vector per input text."""

    def call(payload, *, api_key, model):
        inputs = payload["input"]
        data = [
            {
                "index": i,
                "embedding": list(vector) if vector is not None else [float(len(t)), 1.0, 2.0, 3.0],
            }
            for i, t in enumerate(inputs)
        ]
        return {
            "data": data,
            "usage": {"prompt_tokens": 10 * len(inputs), "total_tokens": 10 * len(inputs)},
        }

    return call


def _boom_call(payload, *, api_key, model):  # pragma: no cover - must never run
    raise AssertionError("network call must not happen")


# --------------------------------------------------------------------------- #
# Vector codec
# --------------------------------------------------------------------------- #
def test_float32_round_trip() -> None:
    vec = [0.5, -1.25, 3.0, 0.0]
    blob = encode_embedding(vec)
    assert isinstance(blob, bytes)
    assert decode_embedding(blob) == vec
    assert decode_embedding(None) == []
    assert decode_embedding(b"") == []


# --------------------------------------------------------------------------- #
# Eligibility & priority
# --------------------------------------------------------------------------- #
def test_priority_inject_then_labeled_then_retrieval_and_catalog_excluded(
    tmp_path: Path,
) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)

    injected = _know(conn, "injected")
    _set(conn, injected, inject=1)
    labeled = _know(conn, "labeled")
    _set(conn, labeled, quality_label="good")
    touched = _know(conn, "touched")
    _touch_retrieval(conn, touched)
    # Never-referenced, unlabeled, non-injected candidate == catalog backlog.
    catalog = _know(conn, "catalog")

    ordered = [row["id"] for row in eligible_rows(conn, cfg)]
    assert ordered == [injected, labeled, touched]
    assert catalog not in ordered


def test_labeled_raw_catalog_doc_stays_unembedded_until_retrieved(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    catalog = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="raw-history",
        title="Raw history file",
        body_uri="/local/history.jsonl",
        doc_kind="history",
        status="current",
        origin="catalog",
    )
    _set(conn, catalog, quality_label="good")
    assert catalog not in {row["id"] for row in eligible_rows(conn, cfg)}
    _touch_retrieval(conn, catalog)
    assert catalog in {row["id"] for row in eligible_rows(conn, cfg)}


def test_stale_and_unembedded_are_eligible_fresh_are_not(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "k")
    _set(conn, kid, inject=1)

    # Unembedded -> eligible.
    assert kid in {r["id"] for r in eligible_rows(conn, cfg)}

    # Freshly embedded (embedded_at >= updated_at) -> not eligible.
    _set(conn, kid, embedding=encode_embedding([1.0]), embedded_at="2999-01-01T00:00:00")
    assert kid not in {r["id"] for r in eligible_rows(conn, cfg)}

    # Stale (embedded_at < updated_at) -> eligible again.
    _set(conn, kid, embedded_at="2000-01-01T00:00:00", updated_at="2001-01-01T00:00:00")
    assert kid in {r["id"] for r in eligible_rows(conn, cfg)}


# --------------------------------------------------------------------------- #
# Privacy / egress / redaction
# --------------------------------------------------------------------------- #
def test_private_never_eligible_or_dispatched(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    private = _know(conn, "priv", scope="private")
    _set(conn, private, inject=1)
    workspace = _know(conn, "work", scope="workspace")
    _set(conn, workspace, inject=1)

    # Private is excluded from the candidate set entirely.
    assert private not in {r["id"] for r in eligible_rows(conn, cfg)}

    # And dropped again at batch build (defense in depth), with a rejection reason.
    rows = [get_knowledge(conn, private), get_knowledge(conn, workspace)]
    included, rejected = build_embed_batch(conn, rows)
    assert {i["id"] for i in included} == {workspace}
    assert any(r["id"] == private and r["reason"] == "private_scope" for r in rejected)

    # End to end: after a run the private row still has no embedding.
    embed_pending(conn, cfg, call=_stub_embed(), env=KEY_ENV)
    assert get_knowledge(conn, private)["embedding"] is None
    assert get_knowledge(conn, workspace)["embedding"] is not None


def test_redaction_before_dispatch(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _know(
        conn,
        "leak",
        value_text="the token is sk-abcdefghijklmnopqrstuvwxyz012345 keep it",
    )
    included, _rejected = build_embed_batch(conn, [get_knowledge(conn, kid)])
    assert included
    assert "sk-abcdefghijklmnop" not in included[0]["text"]
    assert "[REDACTED]" in included[0]["text"]


def test_egress_audit_written_per_batch(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path, batch_size=1)
    for i in range(2):
        kid = _know(conn, f"k{i}")
        _set(conn, kid, inject=1)
    embed_pending(conn, cfg, call=_stub_embed(), env=KEY_ENV)
    # One egress audit per dispatched batch (batch_size=1 -> 2 batches).
    assert conn.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 2


def test_network_call_never_holds_sqlite_writer_lock(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "network-window")
    _set(conn, kid, inject=1)

    def observing_call(payload, *, api_key, model):
        observer = sqlite3.connect(tmp_path / "ocbrain.sqlite", timeout=0)
        observer.execute("BEGIN IMMEDIATE")
        # The egress audit was committed before dispatch, not lost to unlock.
        assert observer.execute("SELECT COUNT(*) FROM egress_audits").fetchone()[0] == 1
        observer.rollback()
        observer.close()
        return _stub_embed()(payload, api_key=api_key, model=model)

    result = embed_pending(conn, cfg, call=observing_call, env=KEY_ENV)
    assert result["changed"] == 1


# --------------------------------------------------------------------------- #
# Inert / budget paths
# --------------------------------------------------------------------------- #
def test_inert_without_api_key(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)
    result = embed_pending(conn, cfg, call=_boom_call, env={})
    assert result["status"] == "skipped"
    assert result["reason"] == "no_api_key"
    assert get_knowledge(conn, kid)["embedding"] is None


def test_inert_when_disabled(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path, enabled=False)
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)
    result = embed_pending(conn, cfg, call=_boom_call, env=KEY_ENV)
    assert result["status"] == "skipped"
    assert result["reason"] == "disabled"


def test_daily_budget_skip(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)
    conn.execute(
        "INSERT INTO embed_runs (id, ts, items, cost_usd, status) VALUES ('e0', ?, 0, ?, 'ok')",
        (now_iso(), cfg.embed.daily_usd_cap + 0.1),
    )
    result = embed_pending(conn, cfg, call=_boom_call, env=KEY_ENV)
    assert result["status"] == "skipped_budget"
    assert get_knowledge(conn, kid)["embedding"] is None


# --------------------------------------------------------------------------- #
# Happy path: storage, cost, idempotency
# --------------------------------------------------------------------------- #
def test_embeds_and_stores_float32_vector(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)

    result = embed_pending(conn, cfg, call=_stub_embed([1.0, 2.0, 3.0, 4.0]), env=KEY_ENV)
    assert result["status"] == "ok"
    assert result["changed"] == 1

    row = get_knowledge(conn, kid)
    assert row["embedding"] is not None
    assert row["embedded_at"] is not None
    assert decode_embedding(row["embedding"]) == [1.0, 2.0, 3.0, 4.0]


def test_cost_accounting(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path, price_per_mtok={"text-embedding-3-small": 0.02})
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)
    embed_pending(conn, cfg, call=_stub_embed(), env=KEY_ENV)
    # total_tokens = 10 (one input) -> 10 / 1e6 * 0.02
    assert abs(spent_today(conn) - (10 / 1_000_000 * 0.02)) < 1e-12


def test_reembed_is_idempotent(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    cfg = _cfg(tmp_path)
    kid = _know(conn, "a")
    _set(conn, kid, inject=1)
    first = embed_pending(conn, cfg, call=_stub_embed(), env=KEY_ENV)
    assert first["changed"] == 1
    # Nothing stale now -> a second run embeds nothing and never calls the API.
    second = embed_pending(conn, cfg, call=_boom_call, env=KEY_ENV)
    assert second["status"] == "ok"
    assert second["changed"] == 0


# --------------------------------------------------------------------------- #
# embed_query
# --------------------------------------------------------------------------- #
def test_embed_query_inert_without_key(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert embed_query("hello", cfg, call=_boom_call, env={}) == []


def test_embed_query_returns_vector(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    vec = embed_query("hello", cfg, call=_stub_embed([9.0, 8.0]), env=KEY_ENV)
    assert vec == [9.0, 8.0]
