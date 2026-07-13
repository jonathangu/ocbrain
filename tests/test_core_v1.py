from __future__ import annotations

import json
import sqlite3

import pytest

from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import (
    CORE_V1_APPLICATION_ID,
    CORE_V1_USER_VERSION,
    append_core_event,
    conservative_legacy_scope,
    get_core_v1_belief,
    init_core_v1,
    project_core_v1,
    record_core_v1_evidence,
    record_core_v1_retrieval,
    search_core_v1,
    verify_event_chain,
)
from ocbrain.db import connect, init_db
from ocbrain.scope import ScopeContext, ScopeTag


def _seed_belief(conn: sqlite3.Connection, *, belief_id: str, scope: ScopeTag) -> str:
    evidence_id, _ = record_core_v1_evidence(
        conn,
        body=f"Source evidence for {belief_id}",
        kind="test",
        scope=scope,
        writer="test",
    )
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "body": f"The shared context bridge fact for {belief_id}",
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": 0.9,
        },
        writer="test",
    )
    append_core_event(
        conn,
        "compilation_decided",
        {
            "proposal_event_id": proposal,
            "decision": "approve",
            "actor": "test",
        },
        writer="test",
        project=True,
    )
    return evidence_id


def test_fresh_v1_markers_and_legacy_init_refusal(tmp_path) -> None:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    schema_before = list(
        conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
    )

    assert conn.execute("PRAGMA application_id").fetchone()[0] == CORE_V1_APPLICATION_ID
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CORE_V1_USER_VERSION
    with pytest.raises(ValueError, match="legacy init_db"):
        init_db(conn)
    assert list(
        conn.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name")
    ) == schema_before
    conn.close()


def test_init_core_v1_rejects_nonempty_or_mixed_database(tmp_path) -> None:
    conn = connect(tmp_path / "mixed.sqlite")
    conn.execute("CREATE TABLE foreign_state(id TEXT)")
    conn.commit()
    with pytest.raises(ValueError, match="existing schema"):
        init_core_v1(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='brain_events'"
    ).fetchone()[0] == 0
    conn.close()

    conn = connect(tmp_path / "marked.sqlite")
    init_core_v1(conn)
    conn.execute("CREATE TABLE leaked_companion(id TEXT)")
    conn.commit()
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        init_core_v1(conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='leaked_companion'"
    ).fetchone()[0] == 1
    conn.close()


def test_private_unscoped_legacy_rows_remain_confidential() -> None:
    scope = conservative_legacy_scope(
        {"project": "workspace", "privacy_scope": "private", "body_uri": None}
    )
    assert scope.scope_type == "legacy_unscoped"
    assert scope.visibility == "confidential"
    assert scope.egress_policy == "local_only"


def test_scope_filtering_happens_before_search_limit(tmp_path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    # More than the old prefilter limit of out-of-scope documents must not hide
    # the one matching in-scope document.
    for index in range(45):
        _seed_belief(
            conn,
            belief_id=f"belief:foreign:{index:02d}",
            scope=ScopeTag("project", "project:foreign"),
        )
    _seed_belief(
        conn,
        belief_id="belief:visible",
        scope=ScopeTag("project", "project:ocbrain"),
    )

    result = search_core_v1(
        conn,
        "shared context bridge fact",
        context=ScopeContext(project="ocbrain"),
        limit=1,
    )

    assert [item["belief_id"] for item in result["items"]] == ["belief:visible"]
    conn.close()


def test_full_projection_is_deterministic_and_preserves_runtime_receipts(tmp_path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="belief:receipt",
        scope=ScopeTag("project", "project:ocbrain"),
    )
    retrieval_id = record_core_v1_retrieval(
        conn,
        query="receipt",
        context={"project": "ocbrain"},
        items=[
            {"object_id": "belief:receipt", "object_kind": "belief", "score": 1.0},
            {"object_id": "belief:receipt", "object_kind": "belief", "score": 0.5},
        ],
        runtime="codex",
        task_ref="receipt-test",
        session_id="session-1",
    )
    record_closeout(
        conn,
        task_ref="receipt-test",
        status="completed",
        summary="verified",
        retrieval_use_ids=[retrieval_id],
        decision_impact="informed",
        verifier_refs=[
            {
                "uri": "pytest://test_core_v1",
                "kind": "pytest",
                "status": "passed",
            }
        ],
    )
    conn.commit()

    first = project_core_v1(conn, full=True)
    beliefs_first = [tuple(row) for row in conn.execute("SELECT * FROM current_beliefs")]
    cursor_first = tuple(conn.execute("SELECT * FROM projection_cursor").fetchone())
    second = project_core_v1(conn, full=True)
    beliefs_second = [tuple(row) for row in conn.execute("SELECT * FROM current_beliefs")]
    cursor_second = tuple(conn.execute("SELECT * FROM projection_cursor").fetchone())

    assert first["last_event_hash"] == second["last_event_hash"]
    assert beliefs_first == beliefs_second
    assert cursor_first == cursor_second
    assert conn.execute(
        "SELECT COUNT(*) FROM retrieval_items WHERE retrieval_use_id=?", (retrieval_id,)
    ).fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM task_closeout_retrievals WHERE retrieval_use_id=?",
        (retrieval_id,),
    ).fetchone()[0] == 1
    conn.close()


def test_cursor_anchor_tampering_fails_even_without_new_events(tmp_path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="belief:anchor",
        scope=ScopeTag("project", "project:ocbrain"),
    )
    conn.execute("UPDATE projection_cursor SET last_event_hash='wrong'")
    with pytest.raises(RuntimeError, match="cursor anchor"):
        project_core_v1(conn)
    assert verify_event_chain(conn)["verified"] is True
    conn.close()


def test_get_core_record_keeps_lifecycle_metadata_for_mcp_gate(tmp_path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    _seed_belief(
        conn,
        belief_id="belief:gate",
        scope=ScopeTag("project", "project:ocbrain"),
    )
    belief = get_core_v1_belief(conn, "belief:gate")
    assert belief is not None
    assert belief["status"] == "current"
    assert belief["serve"] == 1
    assert json.loads(belief["attributes_json"]) == {}
    conn.close()
