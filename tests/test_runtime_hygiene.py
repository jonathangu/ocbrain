"""Caller identity on the write path, and the read-side folder it replaced.

``canonical_runtime`` used to run on every retrieval write, collapsing a
free-text self-report into a guessed slug. Server-captured provenance made the
guess unnecessary for new rows, so the folder moved to ``scripts/procmine``
where the historical corpus still needs it. These tests pin both halves: the
write path records what the model said verbatim beside what the server
observed, and the relocated folder still folds the spellings it always did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocbrain import core_v1
from ocbrain.core_v1 import (
    append_core_event,
    init_core_v1,
    record_core_v1_retrieval,
)
from ocbrain.db import connect
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.provenance import Provenance
from ocbrain.scope import ScopeContext, ScopeTag

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from procmine.runtimes import canonical_runtime  # noqa: E402

SCOPE = ScopeTag(
    "project",
    "project:bountiful",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("codex-desktop", "codex"),
        ("Codex desktop", "codex"),
        ("Codex Desktop local macOS", "codex"),
        ("codex-desktop-heartbeat", "codex"),
        ("hermes-cron-mac-planner", "hermes"),
        ("Hermes Agent", "hermes"),
        ("macOS user Hermes gateway", "hermes"),
        ("cursor-subagent", "cursor"),
        ("claude-code", "claude-code"),
        ("macOS Telegram gateway", "telegram"),
        # No known client in the string: keep it legible rather than bucketing
        # every unrecognized runtime into one opaque label.
        ("local macOS + readonlyprod ClickHouse", "local-macos-readonlyprod-clickhouse"),
        ("mcp", "mcp"),
        ("  ", None),
        (None, None),
    ],
)
def test_canonical_runtime_collapses_client_spellings(reported, expected) -> None:
    assert canonical_runtime(reported) == expected


def test_recorded_retrieval_keeps_the_self_report_verbatim(tmp_path: Path) -> None:
    """No guessing on the write path, and no duplicate raw copy in the context.

    The stored runtime is exactly what the caller sent. Collapsing it to a slug
    here destroyed the only record of what was actually reported and forced a
    ``runtime_raw`` side-channel into ``context_json`` to put it back -- inside
    the value that feeds the retrieval ``stable_id``.
    """
    conn = _core(tmp_path)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query="probe",
        context={"project": "bountiful"},
        items=[],
        runtime="Codex desktop local macOS",
        task_ref=None,
        session_id=None,
    )
    conn.commit()

    row = conn.execute(
        "SELECT served_to_runtime, context_json FROM retrieval_uses WHERE id=?",
        (retrieval_id,),
    ).fetchone()
    assert row["served_to_runtime"] == "Codex desktop local macOS"
    assert "runtime_raw" not in row["context_json"]


def test_recorded_retrieval_separates_observed_identity_from_the_self_report(
    tmp_path: Path,
) -> None:
    conn = _core(tmp_path)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query="probe",
        context={"project": "bountiful"},
        items=[],
        runtime="whatever the model felt like typing",
        task_ref=None,
        session_id="a-human-slug-not-a-session",
        provenance=Provenance.capture(
            client_name="claude-code",
            env={"CLAUDE_CODE_SESSION_ID": "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7"},
        ),
    )
    conn.commit()

    row = conn.execute(
        "SELECT served_to_runtime, session_id, server_connection_id, "
        "client_session_hint, client_runtime_key, provenance_json, context_json "
        "FROM retrieval_uses WHERE id=?",
        (retrieval_id,),
    ).fetchone()
    # What the model said about its runtime, unaltered: that column is still
    # free text on purpose, and the fold happens read-side.
    assert row["served_to_runtime"] == "whatever the model felt like typing"
    # The session is not free text any more. The harness-attested id -- read by
    # the server from the MCP child's own environment -- fills the identity
    # column, and the model's slug is kept beside it as the claim it always was.
    assert row["session_id"] == "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7"
    identity = json.loads(row["provenance_json"])["session_identity"]
    assert identity["session_id_source"] == "harness_attested"
    assert identity["session_id_claim"] == "a-human-slug-not-a-session"
    # What the server saw, in its own columns.
    assert len(row["server_connection_id"]) == 32
    assert row["client_session_hint"] == "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7"
    assert row["client_runtime_key"] == "claude-code"
    assert "harness_attested" in row["provenance_json"]
    # And never mixed into the value the retrieval id is derived from.
    assert "server_connection_id" not in row["context_json"]


def test_the_retrieval_stable_id_ignores_provenance(tmp_path: Path, monkeypatch) -> None:
    """Two identical reads are one read, whoever served them.

    ``context_json`` is a ``stable_id`` input. If provenance leaked into it, a
    reconnect would change the id of an otherwise identical retrieval and the
    ledger would stop being content-addressed.
    """
    conn = _core(tmp_path)
    # Freeze served_at: it is the one legitimate reason two reads differ, and
    # this test is about the other inputs.
    monkeypatch.setattr(core_v1, "now_iso", lambda: "2026-08-25T00:00:00+00:00")
    ids = set()
    for connection in range(2):
        conn.execute("DELETE FROM retrieval_uses")
        ids.add(
            record_core_v1_retrieval(
                conn,
                query="probe",
                context={"project": "bountiful"},
                items=[],
                runtime="codex",
                task_ref=None,
                session_id=None,
                provenance=Provenance.capture(
                    client_name=f"client-{connection}",
                    env={"CLAUDE_CODE_SESSION_ID": f"session-{connection}"},
                    connection_id=f"connection-{connection}",
                ),
            )
        )
    conn.rollback()
    # Same served_at second, same query, same context, same items -> same id.
    assert len(ids) == 1


def _seed_servable_belief(conn, *, belief_id: str, body: str) -> None:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "curated_fact",
            "body": body,
            "evidence_ids": [],
            "scope": SCOPE.to_dict(),
            "confidence": 0.9,
            "attributes": {"source_quality": 0.95},
        },
        writer="test",
    )
    decide_proposal_v1(
        conn,
        proposal_event_id=proposal,
        decision="approve",
        actor="test",
        edited_body=None,
        reason="test seed",
    )


def test_source_expansion_bounds_the_issue_history(tmp_path: Path) -> None:
    """brain.source must not return an unbounded issuance list.

    context_source_handle_issues grows one row per (handle, retrieval) forever.
    The v1 path already windowed its inline list but nothing pinned it, so the
    bound was one refactor away from silently going missing. (The legacy v0
    expansion path was unbounded and is fixed alongside; it cannot load a v1
    handle, so it is not exercised here.)
    """
    from ocbrain.mcp_v1 import build_context_v1, expand_source_v1, record_context_v1
    from ocbrain.shared_context import ISSUED_BY_WINDOW

    conn = _core(tmp_path)
    belief_id = "curated:bountiful:hot-handle"
    body = "Exports finish before the morning report."
    _seed_servable_belief(conn, belief_id=belief_id, body=body)
    conn.commit()

    issued_total = ISSUED_BY_WINDOW + 5
    source_id = ""
    for _index in range(issued_total):
        packet, handles = build_context_v1(
            conn,
            "morning report exports",
            context=ScopeContext(project="bountiful"),
            limit=5,
            cross_scope=False,
            delivery_target="local_model",
        )
        assert handles
        record_context_v1(
            conn,
            packet,
            handles,
            context=ScopeContext(project="bountiful"),
            delivery_target="local_model",
        )
        source_id = handles[0]["id"]
    conn.commit()

    expanded = expand_source_v1(
        conn,
        source_id=source_id,
        context=ScopeContext(project="bountiful"),
        max_chars=2_000,
    )
    # Windowed, not truncated-to-nothing, and the total is still reported so a
    # caller can tell how much history it is not seeing.
    assert len(expanded["issued_by_retrieval_use_ids"]) == ISSUED_BY_WINDOW
    assert expanded["issued_by_count"] == issued_total
    assert issued_total > ISSUED_BY_WINDOW


def test_projection_does_not_duplicate_the_evidence_body(tmp_path: Path) -> None:
    """The body text belongs in one column, not three.

    evidence_objects.metadata_json carried a full copy of the event body, which
    already includes the text that sits in the same row's `body` column and in
    brain_events.body_json. On a real core that third copy was ~23% of the file.
    """
    from ocbrain.core_v1 import get_core_v1_evidence, record_core_v1_evidence

    conn = _core(tmp_path)
    body = "A distinctive evidence body that must be stored exactly once here."
    evidence_id, _event_id = record_core_v1_evidence(
        conn, body=body, kind="analysis_result", scope=SCOPE, writer="test"
    )
    conn.commit()

    stored = get_core_v1_evidence(conn, evidence_id)
    assert stored["body"] == body
    event_body = stored["metadata"]["event_body"]
    # Metadata that is not the text survives; the text itself does not.
    assert event_body["kind"] == "analysis_result"
    assert "body" not in event_body
    assert "body_omitted" in event_body
    row = conn.execute(
        "SELECT metadata_json FROM evidence_objects WHERE evidence_id=?", (evidence_id,)
    ).fetchone()
    assert body not in row["metadata_json"]
