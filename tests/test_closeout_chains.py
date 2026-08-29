"""Closeout chains: the normalized task key, the parent pointer, and history.

Chains previously existed only as string equality on free-text ``task_ref`` — a
column carrying Linear ids, slugs, filesystem paths, raw UUIDs, and even literal
query strings. These tests pin the four properties that make the derived key
safe to index on: it is idempotent, it preserves case, it never refuses a
closeout, and it does not touch a row that was written before it existed.
"""

from __future__ import annotations

import sqlite3

import pytest

from ocbrain.closeout import MAX_TASK_REF_NORM, normalize_task_ref, record_closeout
from ocbrain.core_v1 import (
    CORE_V1_SCHEMA,
    CORE_V1_SCHEMA_VERSION,
    init_core_v1,
    migrate_core_v1_columns,
    record_core_v1_retrieval,
)
from ocbrain.db import connect, init_db
from ocbrain.scope import ScopeContext


def _core(tmp_path, name="chain-core.sqlite"):
    conn = connect(tmp_path / name)
    init_core_v1(conn)
    return conn


def _close(conn, task_ref, summary="Chain step recorded with a durable outcome.", **kwargs):
    return record_closeout(
        conn,
        task_ref=task_ref,
        status="completed",
        summary=summary,
        context=ScopeContext(project="ocbrain"),
        **kwargs,
    )


# --- normalization ---------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "COFASC-292",
        "  COFASC-292  ",
        "ocbrain:COFASC-292",
        "task:COFASC-292",
        "ocbrain:task:  COFASC-292",
        "COFASC-292\n",
    ],
)
def test_wrapper_prefixes_and_whitespace_fold_onto_one_key(raw):
    assert normalize_task_ref(raw) == "COFASC-292"


def test_internal_whitespace_collapses_to_single_spaces():
    assert normalize_task_ref("ship   the\tedge   worker") == "ship the edge worker"


@pytest.mark.parametrize(
    "raw",
    [
        "COFASC-292",
        "ocbrain:  Ship The Edge Worker ",
        "3EA24180-0230-4E8A-8C51-2F51F4AE5EEB",
        "/root",
        "",
        "ocbrain:",
        "x" * (MAX_TASK_REF_NORM + 50),
    ],
)
def test_normalization_is_idempotent(raw):
    once = normalize_task_ref(raw)
    assert normalize_task_ref(once) == once


def test_case_is_preserved_because_ids_are_case_significant():
    """Linear ids and UUIDs differ by case; folding could merge two tasks."""
    upper = "3EA24180-0230-4E8A-8C51-2F51F4AE5EEB"
    assert normalize_task_ref(upper) == upper
    assert normalize_task_ref("Cofasc-292") != normalize_task_ref("COFASC-292")


def test_a_reference_that_is_only_a_prefix_keeps_itself():
    """Otherwise every odd input would fold to '' and chain onto every other."""
    assert normalize_task_ref("ocbrain:") == "ocbrain:"
    assert normalize_task_ref("task: ") == "task:"


def test_length_is_bounded():
    assert len(normalize_task_ref("y" * 5000)) == MAX_TASK_REF_NORM


# --- chain block on the receipt -------------------------------------------


def test_previous_in_chain_is_the_latest_closeout_on_the_same_key(tmp_path):
    conn = _core(tmp_path)
    first = _close(conn, "COFASC-292")
    second = _close(conn, "ocbrain:COFASC-292")
    third = _close(conn, "  COFASC-292 ")

    assert first["chain"]["previous_in_chain"] is None
    assert second["chain"]["previous_in_chain"] == first["id"]
    assert third["chain"]["previous_in_chain"] == second["id"]
    assert third["task_ref"] == "COFASC-292"
    assert third["task_ref_norm"] == "COFASC-292"


def test_a_different_task_does_not_join_the_chain(tmp_path):
    conn = _core(tmp_path)
    _close(conn, "COFASC-292")
    other = _close(conn, "COFASC-293")
    assert other["chain"]["previous_in_chain"] is None


def test_parent_is_recorded_in_the_receipt_and_the_column(tmp_path):
    conn = _core(tmp_path)
    parent = _close(conn, "COFASC-292")
    child = _close(conn, "COFASC-292", parent_closeout_id=parent["id"])

    assert child["chain"]["parent_closeout_id"] == parent["id"]
    assert "parent_unresolved" not in child["chain"]
    stored = conn.execute(
        "SELECT parent_closeout_id, task_ref_norm FROM task_closeouts WHERE id=?",
        (child["id"],),
    ).fetchone()
    assert stored["parent_closeout_id"] == parent["id"]
    assert stored["task_ref_norm"] == "COFASC-292"


def test_an_unresolved_parent_is_flagged_not_refused(tmp_path):
    """A closeout must never be lost because the agent mistyped a parent id."""
    conn = _core(tmp_path)
    receipt = _close(conn, "COFASC-292", parent_closeout_id="close_deadbeefdeadbeef")

    assert receipt["chain"]["parent_unresolved"] is True
    assert receipt["chain"]["parent_closeout_id"] == "close_deadbeefdeadbeef"
    # The claim is kept in the receipt; the column stays NULL so no join can
    # follow a pointer to a row that does not exist.
    stored = conn.execute(
        "SELECT parent_closeout_id FROM task_closeouts WHERE id=?", (receipt["id"],)
    ).fetchone()
    assert stored["parent_closeout_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM task_closeouts").fetchone()[0] == 1


def test_no_parent_reports_no_unresolved_flag(tmp_path):
    conn = _core(tmp_path)
    receipt = _close(conn, "COFASC-292")
    assert receipt["chain"] == {"parent_closeout_id": None, "previous_in_chain": None}


# --- history ---------------------------------------------------------------


def test_historical_rows_are_not_backfilled(tmp_path):
    """Fold at write, never rewrite history: a pre-existing row keeps its NULL."""
    conn = _core(tmp_path)
    # A row as an older binary wrote it: no chain columns supplied at all.
    conn.execute(
        """
        INSERT INTO task_closeouts (
          id, schema_version, closed_at, task_ref, status, summary,
          decision_impact, context_json, artifact_refs_json,
          verifier_refs_json, provenance_json, receipt_json, content_hash
        ) VALUES ('close_history00000001', 'ocbrain.closeout.v1',
                  '2026-07-31T03:02:17.086596+00:00', 'COFASC-292', 'completed',
                  'Historical closeout.', 'unknown', '{}', '[]', '[]', '{}',
                  '{}', 'sha-history-1')
        """
    )
    conn.commit()

    fresh = _close(conn, "COFASC-292")

    assert fresh["chain"]["previous_in_chain"] is None
    historical = conn.execute(
        "SELECT parent_closeout_id, task_ref_norm FROM task_closeouts WHERE id=?",
        ("close_history00000001",),
    ).fetchone()
    assert historical["parent_closeout_id"] is None
    assert historical["task_ref_norm"] is None


def test_columns_appear_on_open_and_an_old_binary_is_unaffected(tmp_path):
    """The additive migration adds the columns; it never rewrites a row."""
    path = tmp_path / "old-core.sqlite"
    conn = connect(path)
    # Build a core exactly as it was before the chain columns existed. This
    # incorporates the portable fixture added to #37 after #40 first opened,
    # while calling init_core_v1 so the real migrate-before-index order is also
    # under test.
    old_schema = CORE_V1_SCHEMA
    for fragment, replacement in (
        # Stripped first, because it sits after `task_ref_norm` and the chain
        # fragment below anchors on that column being last.
        (
            ",\n  -- Write-time discipline, all three derived in ocbrain.closeout and NULL on\n"
            "  -- every historical row. `session_id_source` says on whose word `session_id`\n"
            "  -- was filled (harness_attested / agent_reported / server_connection / none);\n"
            "  -- `runtime_family` is the groupable form of the free-text `runtime` above,\n"
            "  -- which stays verbatim; `unresolved` is what the caller said did not work.\n"
            "  session_id_source TEXT,\n"
            "  runtime_family TEXT,\n"
            "  unresolved TEXT\n",
            "\n",
        ),
        (
            ",\n  provenance_json TEXT,\n"
            "  -- The folded form of `task_ref` above, so a retrieval and the closeout that\n"
            "  -- links it agree on which task they belong to. NULL on historical rows.\n"
            "  task_ref_norm TEXT,\n"
            "  -- On whose word `session_id` above was filled, exactly as on task_closeouts.\n"
            "  -- The identical defect lived here at four times the scale: 967 of 1,115\n"
            "  -- session ids on the live core were hand-written and none joined a\n"
            "  -- transcript. NULL on every historical row.\n"
            "  session_id_source TEXT\n",
            ",\n  provenance_json TEXT\n",
        ),
        (
            ",\n  -- Chain pointers. `parent_closeout_id` is the closeout this one continues,\n"
            "  -- written only when it resolved; `task_ref_norm` is the folded form of\n"
            "  -- `task_ref` above, which stays verbatim. Both are NULL on every row written\n"
            "  -- before they existed: the fold happens at write time and history is never\n"
            "  -- rewritten. See ocbrain.closeout.normalize_task_ref.\n"
            "  parent_closeout_id TEXT,\n"
            "  task_ref_norm TEXT\n",
            "\n",
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_task_closeouts_chain\n"
            "  ON task_closeouts(task_ref_norm, closed_at);\n",
            "",
        ),
    ):
        assert fragment in old_schema, "schema text moved; update this fixture"
        old_schema = old_schema.replace(fragment, replacement)
    assert "task_ref_norm" not in old_schema
    conn.executescript(old_schema)
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('core_schema', ?)",
        (CORE_V1_SCHEMA_VERSION,),
    )

    def columns(table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    assert not columns("task_closeouts") & {"parent_closeout_id", "task_ref_norm"}
    assert "task_ref_norm" not in columns("retrieval_uses")
    conn.execute(
        """
        INSERT INTO task_closeouts (
          id, schema_version, closed_at, task_ref, status, summary,
          decision_impact, context_json, artifact_refs_json,
          verifier_refs_json, provenance_json, receipt_json, content_hash
        ) VALUES ('close_before00000001', 'ocbrain.closeout.v1', '2026-08-01T00:00:00+00:00',
                  'COFASC-292', 'completed', 'Written before the columns existed.',
                  'unknown', '{}', '[]', '[]', '{}', '{}', 'sha-before-1')
        """
    )
    conn.commit()
    before = conn.execute(
        "SELECT receipt_json, content_hash FROM task_closeouts WHERE id=?",
        ("close_before00000001",),
    ).fetchone()

    init_core_v1(conn)
    conn.commit()

    assert columns("task_closeouts") >= {"parent_closeout_id", "task_ref_norm"}
    assert "task_ref_norm" in columns("retrieval_uses")
    assert migrate_core_v1_columns(conn) == []
    indexes = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert "idx_task_closeouts_chain" in indexes
    after = conn.execute(
        "SELECT receipt_json, content_hash, task_ref_norm FROM task_closeouts WHERE id=?",
        ("close_before00000001",),
    ).fetchone()
    assert after["receipt_json"] == before["receipt_json"]
    assert after["content_hash"] == before["content_hash"]
    assert after["task_ref_norm"] is None
    # And the append-only triggers survived the ALTERs.
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE task_closeouts SET summary='x' WHERE id='close_before00000001'")
    conn.rollback()


def test_retrieval_writes_stamp_the_same_normalized_key(tmp_path):
    """A read and the closeout that links it must agree on the task key."""
    conn = _core(tmp_path)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query="how do closeout chains work",
        context={"project": "ocbrain"},
        items=[],
        runtime="claude-code",
        task_ref="ocbrain:COFASC-292",
        session_id="sess-1",
    )
    conn.commit()
    row = conn.execute(
        "SELECT task_ref, task_ref_norm FROM retrieval_uses WHERE id=?", (retrieval_id,)
    ).fetchone()
    assert row["task_ref"] == "ocbrain:COFASC-292"
    assert row["task_ref_norm"] == "COFASC-292"
    assert row["task_ref_norm"] == _close(conn, "COFASC-292")["task_ref_norm"]


def test_chains_work_on_the_legacy_schema_too(tmp_path):
    """`record_closeout` still runs against a pre-v1 core; so must the chain."""
    conn = connect(tmp_path / "legacy.sqlite")
    init_db(conn)
    first = _close(conn, "task:COFASC-292")
    second = _close(conn, "COFASC-292")
    assert second["chain"]["previous_in_chain"] == first["id"]
