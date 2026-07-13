from __future__ import annotations

import random
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

import ocbrain.v1_migration as v1_migration
from ocbrain.core_ops import sha256_file
from ocbrain.core_v1 import (
    LEGACY_IMPORT_KINDS,
    append_core_event,
    get_core_v1_belief,
    get_core_v1_evidence,
    init_core_v1,
    project_core_v1,
    search_core_v1,
    verify_event_chain,
)
from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.events import append_event
from ocbrain.ids import content_hash
from ocbrain.scope import ScopeContext, ScopeTag
from ocbrain.v1_migration import migrate_core_v1

PROJECT_SCOPE = ScopeTag("project", "project:ocbrain")


def _outputs(root: Path) -> dict[str, Path]:
    root.mkdir()
    return {
        "core": root / "core.sqlite",
        "training": root / "training.sqlite",
        "ops": root / "ops.sqlite",
        "archive": root / "archive.sqlite",
        "manifest": root / "manifest.json",
    }


def _migrate(source: Path, outputs: dict[str, Path]) -> dict[str, Any]:
    return migrate_core_v1(
        source,
        outputs["core"],
        outputs["archive"],
        outputs["manifest"],
        outputs["training"],
        outputs["ops"],
        batch_size=2,
        progress_interval=2,
    )


def _legacy_source(path: Path, *, evidence_collision: bool = False) -> str:
    conn = connect(path)
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="test",
        source_runtime="codex",
        source_uri="file:///legacy-proof.txt",
        content_hash=content_hash("legacy source artifact"),
        claim="Relational snapshot evidence body.",
        project="workspace",
        privacy_scope="workspace",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="adversarial-proof",
        title="Adversarial proof",
        body_uri="file:///legacy-proof.txt",
        doc_kind="closeout",
        status="current",
        confidence=0.8,
        content_hash=content_hash("legacy knowledge"),
        project="workspace",
        privacy_scope="workspace",
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="supports")
    if evidence_collision:
        append_event(
            conn,
            "evidence_recorded",
            {
                "evidence_id": evidence_id,
                "kind": "event-proof",
                "body": "Event-authoritative evidence body.",
                "artifact_ref": "file:///event-proof.txt",
                "scope": PROJECT_SCOPE.to_dict(),
            },
            writer="event-author",
        )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return evidence_id


def _table_rows(conn: sqlite3.Connection, table: str, order: str) -> list[tuple[Any, ...]]:
    columns = [str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")]
    names = ", ".join(f'"{name}"' for name in columns)
    return [tuple(row) for row in conn.execute(f"SELECT {names} FROM {table} ORDER BY {order}")]


def _semantic_projection(conn: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    return {
        "evidence_objects": _table_rows(conn, "evidence_objects", "evidence_id"),
        "current_beliefs": _table_rows(conn, "current_beliefs", "belief_id"),
        "belief_evidence": _table_rows(conn, "belief_evidence", "belief_id, evidence_id, relation"),
        "object_aliases": _table_rows(conn, "object_aliases", "alias_id"),
        "search_documents": _table_rows(conn, "search_documents", "doc_id"),
    }


def test_concurrent_appends_serialize_the_chain_head(tmp_path: Path) -> None:
    path = tmp_path / "core.sqlite"
    setup = connect(path)
    init_core_v1(setup)
    append_core_event(
        setup,
        "test_seed",
        {"value": "seed"},
        ts="2026-07-13T00:00:00+00:00",
    )
    setup.commit()
    setup.close()

    barrier = threading.Barrier(2)

    class _FrozenCursor:
        def __init__(self, row: sqlite3.Row | None) -> None:
            self.row = row

        def fetchone(self) -> sqlite3.Row | None:
            return self.row

    class _SynchronizedHeadConnection(sqlite3.Connection):
        _event_writer_reserved = False

        def execute(  # type: ignore[override]
            self, sql: str, parameters: tuple[Any, ...] = ()
        ) -> sqlite3.Cursor | _FrozenCursor:
            if sql.strip().upper() == "BEGIN IMMEDIATE":
                barrier.wait(timeout=5)
                cursor = super().execute(sql, parameters)
                self._event_writer_reserved = True
                return cursor
            cursor = super().execute(sql, parameters)
            if (
                not self._event_writer_reserved
                and "SELECT event_hash FROM brain_events ORDER BY rowid DESC" in sql
            ):
                row = cursor.fetchone()
                barrier.wait(timeout=5)
                return _FrozenCursor(row)
            return cursor

    successes: list[str] = []
    errors: list[BaseException] = []
    outcome_lock = threading.Lock()

    def _append(index: int) -> None:
        conn = sqlite3.connect(path, timeout=5, factory=_SynchronizedHeadConnection)
        conn.row_factory = sqlite3.Row
        try:
            event_id = append_core_event(
                conn,
                "test_concurrent",
                {"worker": index},
                writer=f"worker-{index}",
                ts=f"2026-07-13T00:00:0{index + 1}+00:00",
            )
            conn.commit()
            with outcome_lock:
                successes.append(event_id)
        except BaseException as exc:  # pragma: no cover - asserted below
            conn.rollback()
            with outcome_lock:
                errors.append(exc)
        finally:
            conn.close()

    workers = [threading.Thread(target=_append, args=(index,)) for index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert len(successes) == 2
    check = connect(path)
    try:
        assert verify_event_chain(check)["verified"] is True
        rows = list(
            check.execute(
                "SELECT prev_hash, COUNT(*) AS children FROM brain_events "
                "WHERE prev_hash IS NOT NULL GROUP BY prev_hash"
            )
        )
        assert all(int(row["children"]) == 1 for row in rows)
    finally:
        check.close()


def test_append_preserves_caller_transaction_and_autocommit_semantics(tmp_path: Path) -> None:
    path = tmp_path / "core.sqlite"
    owner = connect(path)
    init_core_v1(owner)
    owner.execute("INSERT INTO schema_meta VALUES ('caller-owned', 'uncommitted')")
    append_core_event(
        owner,
        "caller_transaction",
        {"value": "must roll back with caller"},
        ts="2026-07-13T00:00:00+00:00",
    )
    assert owner.in_transaction is True

    observer = connect(path)
    assert observer.execute(
        "SELECT COUNT(*) FROM schema_meta WHERE key='caller-owned'"
    ).fetchone()[0] == 0
    assert observer.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 0
    observer.close()
    owner.rollback()
    owner.close()

    autocommit = sqlite3.connect(path, isolation_level=None)
    autocommit.row_factory = sqlite3.Row
    append_core_event(
        autocommit,
        "autocommit_event",
        {"value": "visible immediately"},
        ts="2026-07-13T00:00:01+00:00",
    )
    assert autocommit.in_transaction is False
    autocommit.close()

    observer = connect(path)
    try:
        assert observer.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == 1
        assert verify_event_chain(observer)["verified"] is True
    finally:
        observer.close()


def test_randomized_incremental_projection_matches_full_rebuild(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    rng = random.Random(0x0CB1)
    sequence = 0

    def _append(kind: str, body: dict[str, Any]) -> str:
        nonlocal sequence
        sequence += 1
        event_id = append_core_event(
            conn,
            kind,
            body,
            writer="randomized-gate",
            ts=f"2026-07-13T00:{sequence // 60:02d}:{sequence % 60:02d}+00:00",
        )
        if rng.random() < 0.37:
            project_core_v1(conn)
        return event_id

    for index in range(24):
        evidence_id = f"evd:random:{index:02d}"
        _append(
            "evidence_recorded",
            {
                "evidence_id": evidence_id,
                "kind": "randomized",
                "body": f"Random evidence {index} zero-day café",
                "scope": PROJECT_SCOPE.to_dict(),
            },
        )
        proposal_id = _append(
            "compilation_proposed",
            {
                "belief_id": f"belief:random:{index:02d}",
                "body": f"Random belief {index} zero-day café",
                "evidence_ids": [evidence_id],
                "scope": PROJECT_SCOPE.to_dict(),
                "confidence": round(rng.uniform(0.45, 0.95), 4),
            },
        )
        _append(
            "compilation_decided",
            {"proposal_event_id": proposal_id, "decision": "approve", "actor": "gate"},
        )
        if index % 3 == 0:
            _append(
                "correction_recorded",
                {
                    "target_layer": "belief",
                    "target_id": f"belief:random:{index:02d}",
                    "op": "edit",
                    "body": f"Corrected random belief {index} zero-day café",
                    "author": "gate",
                },
            )
        if index % 5 == 0:
            _append(
                "scope_promoted",
                {
                    "belief_id": f"belief:random:{index:02d}",
                    "scope": ScopeTag("global", "global").to_dict(),
                    "approved_by": "gate",
                },
            )
        if index % 7 == 0:
            _append(
                "tombstone_recorded",
                {"target": f"belief:random:{index:02d}", "mode": "forget"},
            )

    project_core_v1(conn)
    incremental = _semantic_projection(conn)
    incremental_search = search_core_v1(
        conn,
        "zero-day café",
        context=ScopeContext(project="ocbrain"),
        cross_scope=True,
        limit=50,
    )
    project_core_v1(conn, full=True)
    rebuilt = _semantic_projection(conn)
    rebuilt_search = search_core_v1(
        conn,
        "zero-day café",
        context=ScopeContext(project="ocbrain"),
        cross_scope=True,
        limit=50,
    )

    assert incremental == rebuilt
    assert incremental_search == rebuilt_search
    assert verify_event_chain(conn)["verified"] is True
    conn.close()


def test_duplicate_full_migrations_have_identical_semantic_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _legacy_source(source)
    first = _outputs(tmp_path / "first")
    second = _outputs(tmp_path / "second")

    first_result = _migrate(source, first)
    second_result = _migrate(source, second)

    assert first_result["import_batch_id"] == second_result["import_batch_id"]
    first_conn = connect(first["core"])
    second_conn = connect(second["core"])
    try:
        placeholders = ",".join("?" for _ in LEGACY_IMPORT_KINDS)
        sql = (
            "SELECT event_seq, id, ts, kind, writer, session_id, body_json, "
            "body_hash, prev_hash, event_hash FROM brain_events "
            f"WHERE kind IN ({placeholders}) ORDER BY event_seq"
        )
        params = tuple(sorted(LEGACY_IMPORT_KINDS))
        assert [tuple(row) for row in first_conn.execute(sql, params)] == [
            tuple(row) for row in second_conn.execute(sql, params)
        ]
        assert _semantic_projection(first_conn) == _semantic_projection(second_conn)
        assert verify_event_chain(first_conn) == verify_event_chain(second_conn)
    finally:
        second_conn.close()
        first_conn.close()


def test_event_authority_survives_legacy_evidence_id_content_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    colliding_id = _legacy_source(source, evidence_collision=True)
    outputs = _outputs(tmp_path / "migration")
    _migrate(source, outputs)

    conn = connect(outputs["core"])
    try:
        collisions = list(
            conn.execute(
                "SELECT evidence_id, body FROM evidence_objects "
                "WHERE evidence_id=? OR evidence_id LIKE ? ORDER BY evidence_id",
                (colliding_id, f"legacy:{colliding_id}:%"),
            )
        )
        assert len(collisions) == 2
        assert {str(row["body"]) for row in collisions} == {
            "Event-authoritative evidence body.",
            "Relational snapshot evidence body.",
        }
        direct = get_core_v1_evidence(conn, colliding_id)
        assert direct is not None
        assert direct["canonical_id"] == colliding_id
        assert direct["body"] == "Event-authoritative evidence body."
        mapped = next(
            str(row["evidence_id"]) for row in collisions if str(row["evidence_id"]) != colliding_id
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM belief_evidence WHERE evidence_id=?", (mapped,)
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_later_canonical_events_displace_conflicting_legacy_aliases(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    legacy_evidence_id = "evd:reused"
    alternate_evidence_id = "legacy:evd:reused:old"
    append_core_event(
        conn,
        "legacy_evidence_imported",
        {
            "legacy_evidence_id": legacy_evidence_id,
            "canonical_evidence_id": alternate_evidence_id,
            "legacy_row_sha256": "old-row",
            "row": {"claim": "Old relational evidence.", "source_type": "legacy"},
            "scope": PROJECT_SCOPE.to_dict(),
        },
        ts="2026-07-13T00:00:00+00:00",
        project=True,
    )
    assert get_core_v1_evidence(conn, legacy_evidence_id)["canonical_id"] == (alternate_evidence_id)

    append_core_event(
        conn,
        "evidence_recorded",
        {
            "evidence_id": legacy_evidence_id,
            "kind": "event-proof",
            "body": "Later event-authored evidence.",
            "scope": PROJECT_SCOPE.to_dict(),
        },
        ts="2026-07-13T00:00:01+00:00",
        project=True,
    )
    direct_evidence = get_core_v1_evidence(conn, legacy_evidence_id)
    assert direct_evidence is not None
    assert direct_evidence["canonical_id"] == legacy_evidence_id
    assert direct_evidence["body"] == "Later event-authored evidence."

    legacy_belief_id = "know_reused"
    alternate_belief_id = "legacy:know_reused"
    append_core_event(
        conn,
        "legacy_knowledge_imported",
        {
            "legacy_knowledge_id": legacy_belief_id,
            "canonical_belief_id": alternate_belief_id,
            "legacy_row_sha256": "old-belief-row",
            "body": "Old relational belief.",
            "row": {"id": legacy_belief_id, "status": "current", "confidence": 0.6},
            "evidence_links": [],
            "scope": PROJECT_SCOPE.to_dict(),
        },
        ts="2026-07-13T00:00:02+00:00",
        project=True,
    )
    assert get_core_v1_belief(conn, legacy_belief_id)["canonical_id"] == (alternate_belief_id)
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": legacy_belief_id,
            "body": "Later event-authored belief.",
            "evidence_ids": [legacy_evidence_id],
            "scope": PROJECT_SCOPE.to_dict(),
            "confidence": 0.9,
        },
        ts="2026-07-13T00:00:03+00:00",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {"proposal_event_id": proposal, "decision": "approve", "actor": "gate"},
        ts="2026-07-13T00:00:04+00:00",
        project=True,
    )
    direct_belief = get_core_v1_belief(conn, legacy_belief_id)
    assert direct_belief is not None
    assert direct_belief["canonical_id"] == legacy_belief_id
    assert direct_belief["body"] == "Later event-authored belief."

    incremental = _semantic_projection(conn)
    project_core_v1(conn, full=True)
    assert _semantic_projection(conn) == incremental
    conn.close()


@pytest.mark.parametrize(
    "query",
    [
        "zero-day",
        "CAFÉ naïve coöperate",
        "foo_bar",
        'zero-day" OR secret:*',
        "table'); DROP TABLE search_documents; --",
        "NEAR(body:zero-day, 1)",
        "---zero-day---",
    ],
)
def test_fts_normalization_handles_unicode_hyphens_and_syntax(tmp_path: Path, query: str) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    evidence_id = "evd:fts"
    append_core_event(
        conn,
        "evidence_recorded",
        {
            "evidence_id": evidence_id,
            "kind": "test",
            "body": "Source",
            "scope": PROJECT_SCOPE.to_dict(),
        },
        ts="2026-07-13T00:00:00+00:00",
    )
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": "belief:fts",
            "body": "Zero-day CAFÉ naïve coöperate foo_bar table syntax proof.",
            "evidence_ids": [evidence_id],
            "scope": PROJECT_SCOPE.to_dict(),
            "confidence": 0.9,
        },
        ts="2026-07-13T00:00:01+00:00",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {"proposal_event_id": proposal, "decision": "approve", "actor": "gate"},
        ts="2026-07-13T00:00:02+00:00",
        project=True,
    )

    result = search_core_v1(conn, query, context=ScopeContext(project="ocbrain"), limit=5)
    assert [item["belief_id"] for item in result["items"]] == ["belief:fts"]
    assert (
        conn.execute("SELECT COUNT(*) FROM search_documents WHERE doc_id='belief:fts'").fetchone()[
            0
        ]
        == 1
    )
    conn.close()


def test_publish_failure_preserves_source_and_cleans_only_owned_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.sqlite"
    _legacy_source(source)
    source_hash = sha256_file(source)
    source_stat = source.stat()
    outputs = _outputs(tmp_path / "failed")
    sentinel = outputs["core"].with_name(".core.sqlite.operator-owned.tmp.keep")
    sentinel.write_text("operator-owned", encoding="utf-8")

    real_replace = v1_migration.os.replace
    replacements = 0

    def _fail_during_publish(source_path: str | Path, target_path: str | Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == 3:
            raise RuntimeError("injected publish failure")
        real_replace(source_path, target_path)

    monkeypatch.setattr(v1_migration.os, "replace", _fail_during_publish)
    with pytest.raises(RuntimeError, match="injected publish failure"):
        _migrate(source, outputs)

    assert sha256_file(source) == source_hash
    assert source.stat().st_size == source_stat.st_size
    assert source.stat().st_mtime_ns == source_stat.st_mtime_ns
    assert all(not path.exists() for path in outputs.values())
    assert sentinel.read_text(encoding="utf-8") == "operator-owned"
    leftovers = [
        path
        for path in outputs["core"].parent.iterdir()
        if path != sentinel and (path.name.endswith(".tmp") or ".tmp-" in path.name)
    ]
    assert leftovers == []
