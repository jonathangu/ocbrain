import json
import sqlite3

import pytest

from ocbrain.closeout import record_closeout
from ocbrain.db import connect, init_db, log_retrieval_use, now_iso
from ocbrain.mcp import handle_request


def _payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def _insert_global_belief(conn, *, belief_id="belief_shared_context"):
    conn.execute(
        """
        INSERT INTO current_beliefs (
          belief_id, body, scope_type, scope_id, visibility, egress_policy,
          confidence, confidence_band, evidence_ids, status, pinned,
          approved_event_id, last_event_id, last_compiled_at
        )
        VALUES (?, ?, 'global', 'global:doctrine', 'internal', 'hosted_ok',
                0.91, 'strong', '[]', 'current', 0, 'evt_test', 'evt_test', ?)
        """,
        (
            belief_id,
            "Coframe action priors preserve outcome semantics for future model transfer.",
            now_iso(),
        ),
    )
    conn.commit()


def test_context_envelope_and_issued_source_round_trip(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    _insert_global_belief(conn)

    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "brain.context",
            "arguments": {"query": "Coframe action priors outcome transfer"},
        },
    }
    first = _payload(handle_request(conn, request))
    second = _payload(handle_request(conn, request))

    assert first["schema_version"] == "ocbrain.context.v1"
    assert first["resolved_context"] == {}
    assert first["coverage"]["returned"] == 1
    assert first["coverage"]["source_handle_count"] == 1
    assert first["retrieval_use_status"] == "recorded"
    item = first["items"][0]
    assert set(item) == {
        "id",
        "kind",
        "excerpt",
        "scope",
        "score",
        "relevance",
        "confidence",
        "confidence_band",
        "status",
        "evidence_ids",
        "sources",
    }
    assert item["sources"][0]["id"] == second["items"][0]["sources"][0]["id"]

    source = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "brain.source",
                    "arguments": {"id": item["sources"][0]["id"]},
                },
            },
        )
    )
    assert source["schema_version"] == "ocbrain.source.v1"
    assert source["hash_verified"] is True
    assert source["content"].startswith("Coframe action priors")
    assert source["issued_by_retrieval_use_ids"] == [
        first["retrieval_use_id"],
        second["retrieval_use_id"],
    ]


def test_source_requires_issued_id_scope_match_and_current_hash(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    readme = repo / "README.md"
    readme.write_text(
        "# Coframe intelligence\n\n"
        "Coframe actions preserve outcome semantics and transfer priors across future websites.\n"
    )
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    context_response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.context",
                "arguments": {
                    "query": "Coframe actions outcome transfer priors",
                    "context": {"repo": str(repo)},
                },
            },
        },
    )
    context_payload = _payload(context_response)
    source_id = context_payload["items"][0]["sources"][0]["id"]

    unknown = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "brain.source", "arguments": {"id": "src_not_issued"}},
        },
    )
    assert "source handle not found" in unknown["error"]["message"]

    wrong_scope = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "brain.source",
                "arguments": {"id": source_id, "context": {"repo": str(tmp_path / 'other')}},
            },
        },
    )
    assert wrong_scope["error"]["code"] == -32001

    expanded = _payload(
        handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "brain.source",
                    "arguments": {"id": source_id, "context": {"repo": str(repo)}},
                },
            },
        )
    )
    assert "Coframe actions preserve" in expanded["content"]

    readme.write_text(readme.read_text() + "\nThe source changed after issuance.\n")
    changed = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "brain.source",
                "arguments": {"id": source_id, "context": {"repo": str(repo)}},
            },
        },
    )
    assert "source changed after issuance" in changed["error"]["message"]


def test_closeout_is_append_only_and_marks_decision_impact(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    retrieval_id = log_retrieval_use(
        conn,
        None,
        runtime="codex",
        task_ref="task-42",
        outcome="served",
    )
    conn.commit()

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.closeout",
                "arguments": {
                    "task_ref": "task-42",
                    "status": "completed",
                    "summary": "Implemented and checked the shared context contract.",
                    "retrieval_use_ids": [retrieval_id],
                    "decision_impact": "changed",
                    "artifact_refs": [{"uri": "repo://src/ocbrain/shared_context.py"}],
                    "actions": [
                        {
                            "mechanism": "code_edit",
                            "semantic_role": "correction",
                            "target": {"repo": "ocbrain", "path": "src/ocbrain/closeout.py"},
                            "context_before": {"defect": "outcomes lacked local semantics"},
                            "policy": {"runtime": "codex", "model": "future-compatible"},
                            "cost": {"reversible": True},
                        }
                    ],
                    "outcomes": [
                        {
                            "metric": "contract_tests",
                            "value": 20,
                            "unit": "passing_tests",
                            "role": "guardrail",
                            "baseline": 18,
                            "counterfactual": {"without_change": 18},
                            "uncertainty": {"kind": "exact"},
                            "interpretation": (
                                "Passing tests verify this local contract; they are not a "
                                "universal product reward."
                            ),
                        }
                    ],
                    "context": {"runtime": "codex", "session": "session-1"},
                },
            },
        },
    )
    receipt = _payload(response)
    assert receipt["schema_version"] == "ocbrain.closeout.v1"
    assert receipt["verification_status"] == "agent_reported"
    assert receipt["provenance"]["source"] == "agent_reported"
    assert receipt["retrieval_use_ids"] == [retrieval_id]
    assert receipt["actions"][0]["schema_version"] == "ocbrain.action.v1"
    assert receipt["actions"][0]["semantic_role"] == "correction"
    assert receipt["outcomes"][0]["schema_version"] == "ocbrain.outcome.v1"
    assert receipt["outcomes"][0]["counterfactual"] == {"without_change": 18}
    assert conn.execute(
        "SELECT affected_decision FROM retrieval_uses WHERE id = ?", (retrieval_id,)
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT retrieval_use_id FROM task_closeout_retrievals WHERE closeout_id = ?",
        (receipt["id"],),
    ).fetchone()[0] == retrieval_id

    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(
            "UPDATE task_closeouts SET summary = 'rewritten' WHERE id = ?",
            (receipt["id"],),
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM task_closeouts WHERE id = ?", (receipt["id"],))
    conn.rollback()


@pytest.mark.parametrize(
    ("impact", "expected"),
    [("informed", 1), ("none", 0), ("unknown", None)],
)
def test_closeout_decision_impact_mapping(tmp_path, impact, expected):
    conn = connect(tmp_path / f"{impact}.sqlite")
    init_db(conn)
    retrieval_id = log_retrieval_use(conn, None, task_ref=impact, outcome="served")
    record_closeout(
        conn,
        task_ref=impact,
        status="completed",
        summary="Recorded outcome.",
        retrieval_use_ids=[retrieval_id],
        decision_impact=impact,
    )
    conn.commit()
    value = conn.execute(
        "SELECT affected_decision FROM retrieval_uses WHERE id = ?", (retrieval_id,)
    ).fetchone()[0]
    assert value == expected


def test_blocked_closeout_requires_awaiting(tmp_path):
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "brain.closeout",
                "arguments": {
                    "task_ref": "blocked-task",
                    "status": "blocked",
                    "summary": "Blocked.",
                },
            },
        },
    )
    assert "blocked closeouts require awaiting" in response["error"]["message"]
