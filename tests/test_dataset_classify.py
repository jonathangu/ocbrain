from __future__ import annotations

from ocbrain.db import connect, init_db, upsert_evidence
from ocbrain_training.dataset.classify import DPO_CONTRAST_GATE_VERSION, classify_examples
from ocbrain_training.dataset.quality import store_example


def _db(tmp_path):
    conn = connect(tmp_path / "classify.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="test",
        source_runtime="local",
        source_uri="test://classify",
        content_hash="source",
        claim="classification fixture",
    )
    return conn, evidence_id


def _store(conn, evidence_id, *, dataset, sender_verified=True, label="good"):
    response = "A clear and sufficiently detailed response written for this test fixture."
    if dataset == "dpo":
        body = {
            "input": {"messages": [{"role": "user", "content": "Explain the decision."}]},
            "preferred_output": [{"role": "assistant", "content": response}],
            "non_preferred_output": [
                {
                    "role": "assistant",
                    "content": "Take an unrelated action without explaining any tradeoff.",
                }
            ],
        }
    else:
        body = {
            "messages": [
                {"role": "user", "content": "Explain the decision."},
                {"role": "assistant", "content": response},
            ]
        }
    return store_example(
        conn,
        dataset=dataset,
        source_kind="openclaw_session" if dataset != "dpo" else "correction_event",
        source_uri=f"test://{dataset}/{sender_verified}/{label}",
        evidence_ids=[evidence_id],
        privacy_scope="workspace",
        body=body,
        metadata={
            "sender_verified": sender_verified,
            "gate": "strict" if dataset == "dpo" else None,
            "contrast_gate_version": (DPO_CONTRAST_GATE_VERSION if dataset == "dpo" else None),
        },
        target_text=response,
        base_label=label,
        base_confidence=0.9,
    )


def test_classifies_weights_streams_and_retrieval_boundary(tmp_path):
    conn, evidence_id = _db(tmp_path)
    persona = _store(conn, evidence_id, dataset="persona")
    sft = _store(conn, evidence_id, dataset="sft")
    dpo = _store(conn, evidence_id, dataset="dpo")
    conn.commit()

    result = classify_examples(conn)
    rows = {
        row["id"]: row["train_class"]
        for row in conn.execute("SELECT id, train_class FROM dataset_examples")
    }
    assert result["changed"] == 3
    assert rows[persona["id"]] == "train_voice"
    assert rows[sft["id"]] == "train_skill"
    assert rows[dpo["id"]] == "train_judgment"


def test_persona_styled_assistant_text_without_verified_author_is_excluded(tmp_path):
    conn, evidence_id = _db(tmp_path)
    row = _store(conn, evidence_id, dataset="persona", sender_verified=False)
    conn.commit()

    classify_examples(conn)
    stored = conn.execute(
        "SELECT train_class, train_class_reason FROM dataset_examples WHERE id = ?",
        (row["id"],),
    ).fetchone()
    assert stored["train_class"] == "exclude"
    assert stored["train_class_reason"] == "persona_author_not_verified"
