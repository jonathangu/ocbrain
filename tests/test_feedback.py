from __future__ import annotations

from ocbrain.db import connect, init_db, log_retrieval_use, update_retrieval_use_feedback
from ocbrain.events import record_evidence
from ocbrain.feedback import feedback_coverage, infer_retrieval_outcomes
from ocbrain.scope import ScopeContext


def _db(tmp_path):
    conn = connect(tmp_path / "feedback.sqlite")
    init_db(conn)
    return conn


def test_infers_used_from_later_same_session_evidence(tmp_path):
    conn = _db(tmp_path)
    retrieval_id = log_retrieval_use(
        conn,
        None,
        runtime="codex",
        task_ref="task-1",
        query_text="what is the release contract",
        served_ids=["belief_release_contract"],
        session_id="session-1",
    )
    conn.commit()
    record_evidence(
        conn,
        body="The verified release completed.",
        context=ScopeContext(session="session-1"),
        writer="codex",
        session_id="session-1",
    )
    conn.commit()

    result = infer_retrieval_outcomes(conn)
    row = conn.execute(
        "SELECT outcome, feedback_source FROM retrieval_uses WHERE id = ?",
        (retrieval_id,),
    ).fetchone()
    assert result["changed"] == 1
    assert row["outcome"] == "used"
    assert row["feedback_source"] == "inferred_session_evidence"


def test_explicit_feedback_is_never_overwritten_by_inference(tmp_path):
    conn = _db(tmp_path)
    retrieval_id = log_retrieval_use(
        conn,
        None,
        runtime="claude",
        task_ref="task-2",
        served_ids=["belief_x"],
        session_id="session-2",
    )
    update_retrieval_use_feedback(
        conn,
        retrieval_id,
        outcome="irrelevant",
        note="wrong project",
    )
    record_evidence(
        conn,
        body="Later work happened.",
        context=ScopeContext(session="session-2"),
        writer="claude",
        session_id="session-2",
    )
    conn.commit()

    result = infer_retrieval_outcomes(conn)
    row = conn.execute(
        "SELECT outcome, feedback_source FROM retrieval_uses WHERE id = ?",
        (retrieval_id,),
    ).fetchone()
    assert result["changed"] == 0
    assert row["outcome"] == "irrelevant"
    assert row["feedback_source"] == "explicit"
    assert feedback_coverage(conn)["coverage"] == 1.0
