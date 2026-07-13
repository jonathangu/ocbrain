from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ocbrain.db import (
    connect,
    get_knowledge,
    init_db,
    link_knowledge_evidence,
    log_retrieval_use,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.ids import content_hash
from ocbrain_ops.maintenance import archive_unreferenced_catalog


def _db(tmp_path: Path) -> sqlite3.Connection:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    return conn


def _catalog_doc(
    conn: sqlite3.Connection,
    slug: str,
    *,
    age_days: int = 30,
    origin: str = "autopilot",
    status: str = "current",
    inject: bool = False,
) -> str:
    kid = upsert_knowledge(
        conn, knowledge_type="doc", gate="auto", slug=slug,
        title=f"catalog {slug}", doc_kind="memory", status=status, origin=origin,
        inject=inject,
    )
    old = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    conn.execute("UPDATE knowledge SET updated_at=? WHERE id=?", (old, kid))
    return kid


def test_archives_old_never_referenced_catalog(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _catalog_doc(conn, "old-1", age_days=30)
    result = archive_unreferenced_catalog(conn, older_than_days=14, batch_cap=5000)
    assert result.changed == 1
    row = get_knowledge(conn, kid)
    assert row["status"] == "archived"
    assert row["invalidation_reason"] == "catalog_never_referenced"
    assert result.details[0] == {
        "id": kid,
        "from_status": "current",
        "to_status": "archived",
        "reason": "catalog_never_referenced",
    }


def test_keeps_referenced_catalog(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _catalog_doc(conn, "hot", age_days=30)
    log_retrieval_use(conn, kid, outcome="served")
    result = archive_unreferenced_catalog(conn, older_than_days=14)
    assert result.changed == 0
    assert get_knowledge(conn, kid)["status"] == "current"


def test_keeps_recent_catalog(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _catalog_doc(conn, "fresh", age_days=3)
    result = archive_unreferenced_catalog(conn, older_than_days=14)
    assert result.changed == 0


def test_keeps_distilled_and_human_docs(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _catalog_doc(conn, "wiki", age_days=90, origin="harvest")
    _catalog_doc(conn, "proc", age_days=90, origin="loop")
    _catalog_doc(conn, "hand", age_days=90, origin="human")
    result = archive_unreferenced_catalog(conn, older_than_days=14)
    assert result.changed == 0


def test_keeps_injected_catalog(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _catalog_doc(conn, "pinned", age_days=90, inject=True)
    result = archive_unreferenced_catalog(conn, older_than_days=14)
    assert result.changed == 0
    assert conn.execute("SELECT inject FROM knowledge WHERE id=?", (kid,)).fetchone()["inject"] == 1


def test_idempotent(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _catalog_doc(conn, "a", age_days=30)
    _catalog_doc(conn, "b", age_days=30)
    first = archive_unreferenced_catalog(conn, older_than_days=14)
    second = archive_unreferenced_catalog(conn, older_than_days=14)
    assert first.changed == 2
    assert second.changed == 0  # nothing left to archive


def test_batch_cap(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    for i in range(5):
        _catalog_doc(conn, f"c{i}", age_days=30)
    result = archive_unreferenced_catalog(conn, older_than_days=14, batch_cap=2)
    assert result.changed == 2
    remaining = conn.execute(
        "SELECT COUNT(*) FROM knowledge WHERE status='current'"
    ).fetchone()[0]
    assert remaining == 3


def test_zero_batch_cap_is_noop(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    _catalog_doc(conn, "x", age_days=30)
    result = archive_unreferenced_catalog(conn, older_than_days=14, batch_cap=0)
    assert result.changed == 0


def test_reversible_preserves_evidence_links(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    kid = _catalog_doc(conn, "rev", age_days=30)
    evd = upsert_evidence(
        conn, source_type="memory_file", claim="c", content_hash=content_hash("rev"),
        source_uri="file://rev", verifier_status="not_required",
    )
    link_knowledge_evidence(conn, kid, evd, relation="derived_from")
    archive_unreferenced_catalog(conn, older_than_days=14)
    # Evidence link survives -> un-archive is a pure status flip.
    links = conn.execute(
        "SELECT COUNT(*) FROM knowledge_evidence WHERE knowledge_id=?", (kid,)
    ).fetchone()[0]
    assert links == 1
    conn.execute("UPDATE knowledge SET status='current' WHERE id=?", (kid,))
    assert get_knowledge(conn, kid)["status"] == "current"
