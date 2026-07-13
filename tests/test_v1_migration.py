from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ocbrain.core_ops import sha256_file
from ocbrain.core_v1 import (
    CORE_V1_FTS_TABLES,
    CORE_V1_TABLES,
    assert_core_v1_inventory,
    get_core_v1_belief,
    get_core_v1_evidence,
    is_core_v1,
    project_core_v1,
    verify_event_chain,
)
from ocbrain.db import (
    connect,
    init_db,
    link_knowledge_evidence,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.events import append_event, decide_compilation, propose_compilation, record_evidence
from ocbrain.ids import content_hash
from ocbrain.scope import ScopeTag
from ocbrain.v1_migration import (
    OPS_TABLES,
    TRAINING_TABLES,
    event_prefix_sha256,
    migrate_core_v1,
    migration_plan,
)


def _source_database(path: Path, *, with_event_prefix: bool = False) -> tuple[str, str]:
    conn = connect(path)
    init_db(conn)
    claim = "Codex verified the proof body."
    # Deliberately model a source/file hash that differs from the claim-body hash.
    evidence_id = upsert_evidence(
        conn,
        source_type="closeout",
        source_runtime="codex",
        source_uri="file:///proof.txt",
        content_hash=content_hash("whole source artifact, not claim"),
        claim=claim,
        project="workspace",
        privacy_scope="workspace",
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug="proof",
        title="Proof",
        body_uri="file:///proof.txt",
        doc_kind="closeout",
        status="current",
        confidence=0.8,
        content_hash=content_hash("proof"),
        project="workspace",
        privacy_scope="workspace",
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="supports")
    if with_event_prefix:
        scope = ScopeTag("project", "project:ocbrain")
        synthetic_evidence_id = content_hash("synthetic")[:16]
        record_evidence(
            conn,
            body="Synthetic legacy backfill evidence.",
            kind="legacy_doc",
            scope=scope,
            writer="ocbrain-backfill",
        )
        proposal = propose_compilation(
            conn,
            belief_id=f"legacy:{knowledge_id}",
            body="Event-authored proof body wins over the relational snapshot.",
            evidence_ids=[synthetic_evidence_id],
            scope=scope,
            confidence=0.9,
            writer="ocbrain-backfill",
        )
        decide_compilation(
            conn,
            proposal_event_id=proposal,
            decision="approve",
            actor="ocbrain-backfill",
        )
        # Legacy corrections target know_* while the event belief is legacy:know_*.
        append_event(
            conn,
            "correction_recorded",
            {
                "target_layer": "knowledge",
                "target_id": knowledge_id,
                "op": "retract",
                "body": None,
                "author": "human:test",
                "hard": True,
            },
            writer="human:test",
        )
    conn.execute(
        "INSERT INTO autopilot_runs(id, started_at, finished_at, status, stages_json) "
        "VALUES ('auto_1', '2026-07-12T00:00:00+00:00', "
        "'2026-07-12T00:01:00+00:00', 'ok', '[]')"
    )
    conn.execute(
        "INSERT INTO loop_liveness(loop_id, run_id, last_heartbeat_at) "
        "VALUES ('stallcheck', 'run_1', '2026-07-12T00:00:00+00:00')"
    )
    conn.execute(
        "INSERT INTO dataset_examples("
        "id, dataset, content_hash, dedup_key, source_kind, evidence_ids, "
        "privacy_scope, quality_label, example_json, created_at, updated_at"
        ") VALUES ("
        "'dsx_1', 'sft', 'hash', 'dedup', 'codex_session', '[]', "
        "'workspace', 'good', '{}', "
        "'2026-07-12T00:00:00+00:00', '2026-07-12T00:00:00+00:00'"
        ")"
    )
    # Insert a lexically earlier primary key second so companion verification
    # cannot accidentally depend on source rowid order.
    conn.execute(
        "INSERT INTO dataset_examples("
        "id, dataset, content_hash, dedup_key, source_kind, evidence_ids, "
        "privacy_scope, quality_label, example_json, created_at, updated_at"
        ") VALUES ("
        "'dsx_0', 'sft', 'hash-0', 'dedup-0', 'codex_session', '[]', "
        "'workspace', 'good', '{}', "
        "'2026-07-12T00:00:00+00:00', '2026-07-12T00:00:00+00:00'"
        ")"
    )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    return evidence_id, knowledge_id


def _outputs(tmp_path: Path) -> dict[str, Path]:
    return {
        "core": tmp_path / "core.sqlite",
        "training": tmp_path / "training.sqlite",
        "ops": tmp_path / "ops.sqlite",
        "archive": tmp_path / "archive.sqlite",
        "manifest": tmp_path / "migration.json",
    }


def test_plan_is_read_only_and_reports_all_fresh_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source)
    paths = _outputs(tmp_path)

    result = migration_plan(
        source,
        paths["core"],
        paths["archive"],
        paths["manifest"],
        paths["training"],
        paths["ops"],
    )

    assert result["ready"] is True
    assert result["source_semantic_counts"]["knowledge"] == 1
    assert result["source_training_counts"]["dataset_examples"] == 2
    assert result["source_ops_counts"]["autopilot_runs"] == 1
    assert result["safety"]["automatic_activation"] is False
    assert result["safety"]["hosted_calls"] == 0
    assert not any(path.exists() for path in paths.values())


def test_event_authoritative_migration_and_companion_extracts(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    evidence_id, knowledge_id = _source_database(source, with_event_prefix=True)
    source_hash = sha256_file(source)
    paths = _outputs(tmp_path)

    result = migrate_core_v1(
        source,
        paths["core"],
        paths["archive"],
        paths["manifest"],
        paths["training"],
        paths["ops"],
        batch_size=2,
        progress_interval=2,
    )

    assert result["status"] == "verified"
    assert result["safety"]["live_database_replaced"] is False
    assert result["safety"]["live_database_repointed"] is False
    assert source_hash == sha256_file(source)
    assert json.loads(paths["manifest"].read_text())["format"] == result["format"]
    assert all(path.exists() for path in paths.values())
    assert all((path.stat().st_mode & 0o077) == 0 for path in paths.values())

    core = connect(paths["core"])
    archive = connect(paths["archive"])
    training = connect(paths["training"])
    ops = connect(paths["ops"])
    try:
        assert is_core_v1(core)
        assert_core_v1_inventory(core)
        actual = {
            row[0]
            for row in core.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert actual == set(CORE_V1_TABLES) | set(CORE_V1_FTS_TABLES)
        forbidden = {
            "evidence",
            "knowledge",
            "knowledge_evidence",
            "memory",
            *TRAINING_TABLES,
            *OPS_TABLES,
        }
        assert actual.isdisjoint(forbidden)
        assert archive.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0] == 1
        assert training.execute("SELECT COUNT(*) FROM dataset_examples").fetchone()[0] == 2
        assert ops.execute("SELECT COUNT(*) FROM autopilot_runs").fetchone()[0] == 1
        assert ops.execute("SELECT COUNT(*) FROM loop_liveness").fetchone()[0] == 1

        belief = get_core_v1_belief(core, knowledge_id)
        assert belief is not None
        assert belief["canonical_id"] == f"legacy:{knowledge_id}"
        assert belief["body"].startswith("Event-authored proof body")
        assert belief["status"] == "retracted"
        assert belief["serve"] == 0
        assert belief["scope"]["scope_id"] == "project:ocbrain"
        assert evidence_id in belief["evidence_ids"]

        evidence = get_core_v1_evidence(core, evidence_id)
        assert evidence is not None
        assert evidence["content_hash"] == content_hash(evidence["body"])
        assert evidence["source_content_hash"] != evidence["content_hash"]
        assert evidence["scope"]["scope_type"] == "legacy_unscoped"
        assert verify_event_chain(core)["verified"] is True
        assert result["core"]["verification"]["legacy_event_prefix"]["verified"] is True
    finally:
        ops.close()
        training.close()
        archive.close()
        core.close()


def test_migration_preserves_gapped_event_sequences_exactly(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source, with_event_prefix=True)
    source_conn = connect(source)
    # Rowid is not part of the legacy event hash. Move the second row to create
    # a gap and prove the migration copies the sequence explicitly.
    last_rowid = source_conn.execute("SELECT MAX(rowid) FROM brain_events").fetchone()[0]
    source_conn.execute("UPDATE brain_events SET rowid=20 WHERE rowid=?", (last_rowid,))
    source_conn.commit()
    source_sequences = [
        row[0] for row in source_conn.execute("SELECT rowid FROM brain_events ORDER BY rowid")
    ]
    source_prefix_sha = event_prefix_sha256(source_conn)
    source_conn.close()
    paths = _outputs(tmp_path)

    migrate_core_v1(
        source,
        paths["core"],
        paths["archive"],
        paths["manifest"],
        paths["training"],
        paths["ops"],
    )

    core = connect(paths["core"])
    try:
        imported = core.execute(
            "SELECT MIN(event_seq) FROM brain_events WHERE kind='legacy_evidence_imported'"
        ).fetchone()[0]
        copied_sequences = [
            row[0]
            for row in core.execute(
                "SELECT event_seq FROM brain_events WHERE event_seq < ? ORDER BY event_seq",
                (imported,),
            )
        ]
        assert copied_sequences == source_sequences
        assert event_prefix_sha256(core, through_seq=max(source_sequences)) == source_prefix_sha
    finally:
        core.close()


def test_corrupt_event_prefix_refuses_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source, with_event_prefix=True)
    conn = connect(source)
    conn.execute("UPDATE brain_events SET event_hash='corrupt' WHERE rowid=2")
    conn.commit()
    conn.close()
    paths = _outputs(tmp_path)

    with pytest.raises(RuntimeError, match="legacy event prefix is corrupt"):
        migrate_core_v1(
            source,
            paths["core"],
            paths["archive"],
            paths["manifest"],
            paths["training"],
            paths["ops"],
        )

    assert paths["core"].exists() is False
    assert paths["archive"].exists() is False
    assert paths["manifest"].exists() is False


def test_projection_rebuild_preserves_runtime_retrieval_and_closeout(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source)
    paths = _outputs(tmp_path)
    migrate_core_v1(
        source,
        paths["core"],
        paths["archive"],
        paths["manifest"],
        paths["training"],
        paths["ops"],
    )
    conn = connect(paths["core"])
    try:
        conn.execute(
            "INSERT INTO retrieval_uses(id, outcome, served_at) "
            "VALUES ('ret_runtime', 'used', '2026-07-13T00:00:00+00:00')"
        )
        conn.execute(
            """
            INSERT INTO task_closeouts(
              id, schema_version, closed_at, task_ref, status, summary,
              decision_impact, context_json, artifact_refs_json,
              verifier_refs_json, provenance_json, receipt_json, content_hash
            ) VALUES (
              'close_runtime', 'ocbrain.closeout.v1', '2026-07-13T00:00:01+00:00',
              'runtime', 'completed', 'done', 'informed', '{}', '[]', '[]',
              '{}', '{}', 'close-hash'
            )
            """
        )
        conn.execute(
            "INSERT INTO task_closeout_retrievals VALUES ('close_runtime','ret_runtime')"
        )
        conn.commit()

        project_core_v1(conn, full=True)
        conn.commit()

        assert conn.execute(
            "SELECT outcome FROM retrieval_uses WHERE id='ret_runtime'"
        ).fetchone()[0] == "used"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_closeout_retrievals "
            "WHERE closeout_id='close_runtime' AND retrieval_use_id='ret_runtime'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_v1_event_log_is_append_only(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source)
    paths = _outputs(tmp_path)
    migrate_core_v1(
        source,
        paths["core"],
        paths["archive"],
        paths["manifest"],
        paths["training"],
        paths["ops"],
    )
    conn = connect(paths["core"])
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE brain_events SET writer='tamper' WHERE event_seq=1")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM brain_events WHERE event_seq=1")
    finally:
        conn.close()


def test_migration_refuses_existing_outputs_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    _source_database(source)
    source_hash = sha256_file(source)
    paths = _outputs(tmp_path)
    paths["core"].write_bytes(b"do not replace")

    with pytest.raises(ValueError, match="not fresh"):
        migrate_core_v1(
            source,
            paths["core"],
            paths["archive"],
            paths["manifest"],
            paths["training"],
            paths["ops"],
        )

    assert paths["core"].read_bytes() == b"do not replace"
    assert sha256_file(source) == source_hash
