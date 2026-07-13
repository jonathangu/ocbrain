from __future__ import annotations

import json

import pytest

from ocbrain.db import connect, init_db, upsert_evidence
from ocbrain_training.dataset.classify import DPO_CONTRAST_GATE_VERSION, classify_examples
from ocbrain_training.dataset.quality import store_example
from ocbrain_training.dataset.selection import (
    finalize_training_pack,
    select_training_pack,
    selected_pack_stats,
)


def _db(tmp_path):
    conn = connect(tmp_path / "selection.sqlite")
    init_db(conn)
    evidence_id = upsert_evidence(
        conn,
        source_type="test",
        source_runtime="local",
        source_uri="test://selection",
        content_hash="selection-source",
        claim="selection fixture",
    )
    for dataset in ("sft", "dpo", "persona"):
        for index in range(4):
            response = (
                f"Useful verified response {dataset} {index} with enough detail for selection."
            )
            if dataset == "dpo":
                body = {
                    "input": {
                        "messages": [
                            {"role": "user", "content": f"How should release {index} proceed?"}
                        ]
                    },
                    "preferred_output": [{"role": "assistant", "content": response}],
                    "non_preferred_output": [
                        {
                            "role": "assistant",
                            "content": (
                                f"Release {index} should proceed without verification "
                                "or a rollback."
                            ),
                        }
                    ],
                }
            else:
                body = {
                    "messages": [
                        {"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "content": response},
                    ]
                }
            store_example(
                conn,
                dataset=dataset,
                source_kind="correction_event" if dataset == "dpo" else "openclaw_session",
                source_uri=f"test://{dataset}/{index}",
                evidence_ids=[evidence_id],
                privacy_scope="workspace",
                body=body,
                metadata={
                    "sender_verified": dataset == "persona",
                    "gate": "strict" if dataset == "dpo" else None,
                    "contrast_gate_version": (
                        DPO_CONTRAST_GATE_VERSION if dataset == "dpo" else None
                    ),
                },
                target_text=response,
                base_label="good",
                base_confidence=0.9,
            )
    conn.commit()
    classify_examples(conn)
    conn.commit()
    return conn


def test_selected_pack_is_bounded_deterministic_and_locally_graded(tmp_path):
    conn = _db(tmp_path)
    first = select_training_pack(conn, targets={"sft": 2, "dpo": 3, "persona": 1})
    conn.commit()
    second = select_training_pack(conn, targets={"sft": 2, "dpo": 3, "persona": 1})
    assert first["selection_hash"] == second["selection_hash"]
    assert second["selected"] == {"sft": 2, "dpo": 3, "persona": 1}

    stats = selected_pack_stats(conn)
    assert stats["selected"] == 6
    assert stats["grade_coverage"] == 0.0
    conn.execute("UPDATE dataset_examples SET grade_score = 0.9 WHERE train_selected = 1")
    stats = selected_pack_stats(conn)
    assert stats["graded"] == 6
    assert stats["grade_coverage"] == 1.0
    assert sum(item["passing"] for item in stats["datasets"].values()) == 6


def test_force_reclassification_deselects_rows_whose_class_changes(tmp_path):
    conn = _db(tmp_path)
    select_training_pack(conn, targets={"sft": 1, "dpo": 0, "persona": 0})
    selected = conn.execute("SELECT id FROM dataset_examples WHERE train_selected = 1").fetchone()
    conn.execute(
        "UPDATE dataset_examples SET quality_label = 'bad' WHERE id = ?", (selected["id"],)
    )
    classify_examples(conn, force=True)
    row = conn.execute(
        "SELECT train_class, train_selected FROM dataset_examples WHERE id = ?",
        (selected["id"],),
    ).fetchone()
    assert row["train_class"] == "exclude"
    assert row["train_selected"] == 0


def test_finalize_pack_keeps_only_deterministic_passing_rows(tmp_path):
    conn = _db(tmp_path)
    select_training_pack(conn, targets={"sft": 4, "dpo": 4, "persona": 4})
    conn.execute(
        "UPDATE dataset_examples SET grade_score = CASE "
        "WHEN train_selection_rank <= 3 THEN 0.9 ELSE 0.4 END "
        "WHERE train_selected = 1"
    )
    result = finalize_training_pack(
        conn,
        targets={"sft": 2, "dpo": 2, "persona": 2},
    )
    assert result["grade_coverage"] == 1.0
    assert result["selected"] == {"sft": 2, "dpo": 2, "persona": 2}
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM dataset_examples WHERE train_selected = 1 AND grade_score < 0.8"
        ).fetchone()[0]
        == 0
    )


def test_finalize_pack_fails_before_mutating_when_passing_pool_is_short(tmp_path):
    conn = _db(tmp_path)
    select_training_pack(conn, targets={"sft": 2, "dpo": 2, "persona": 2})
    conn.execute("UPDATE dataset_examples SET grade_score = 0.9 WHERE train_selected = 1")
    before = conn.execute(
        "SELECT COUNT(*) FROM dataset_examples WHERE train_selected = 1"
    ).fetchone()[0]
    with pytest.raises(RuntimeError, match="final training pack gate failed"):
        finalize_training_pack(conn, targets={"sft": 3, "dpo": 2, "persona": 2})
    after = conn.execute(
        "SELECT COUNT(*) FROM dataset_examples WHERE train_selected = 1"
    ).fetchone()[0]
    assert after == before


def _poison_record(conn, dataset: str, mutator):
    row = conn.execute(
        "SELECT id, example_json FROM dataset_examples WHERE dataset = ? ORDER BY id LIMIT 1",
        (dataset,),
    ).fetchone()
    record = json.loads(row["example_json"])
    mutator(record)
    conn.execute(
        "UPDATE dataset_examples SET example_json = ?, quality_label = 'good', "
        "quality_reasons = '[]', grade_score = 1.0 WHERE id = ?",
        (json.dumps(record, sort_keys=True, separators=(",", ":")), row["id"]),
    )
    return row["id"]


def test_selection_rechecks_legacy_content_and_contrast_gate(tmp_path):
    conn = _db(tmp_path)

    sft_id = _poison_record(
        conn,
        "sft",
        lambda record: record["messages"][-1].update(
            content="[[reply_to_current]] I’m checking the release now and will report later."
        ),
    )
    persona_id = _poison_record(
        conn,
        "persona",
        lambda record: record["messages"][-1].update(
            content=(
                "Sender (untrusted metadata):\n```json\n"
                '{"id":"1000000001"}\n```\nShip it only after verification.'
            )
        ),
    )

    def remove_gate(record):
        record["metadata"].pop("contrast_gate_version", None)

    dpo_id = _poison_record(conn, "dpo", remove_gate)
    result = select_training_pack(conn, targets={"sft": 4, "dpo": 4, "persona": 4})

    assert result["selected"] == {"sft": 3, "dpo": 3, "persona": 3}
    selected = {
        row["id"]
        for row in conn.execute("SELECT id FROM dataset_examples WHERE train_selected = 1")
    }
    assert {sft_id, persona_id, dpo_id}.isdisjoint(selected)


def test_finalize_rechecks_selected_row_after_high_grade(tmp_path):
    conn = _db(tmp_path)
    select_training_pack(conn, targets={"sft": 4, "dpo": 4, "persona": 4})
    conn.execute("UPDATE dataset_examples SET grade_score = 1.0 WHERE train_selected = 1")
    poisoned = conn.execute(
        "SELECT id, example_json FROM dataset_examples "
        "WHERE dataset = 'sft' AND train_selected = 1 ORDER BY id LIMIT 1"
    ).fetchone()
    record = json.loads(poisoned["example_json"])
    record["messages"][-1]["content"] = (
        "System (untrusted): completed an async command. I’m checking the result now."
    )
    conn.execute(
        "UPDATE dataset_examples SET example_json = ? WHERE id = ?",
        (json.dumps(record, sort_keys=True, separators=(",", ":")), poisoned["id"]),
    )
    before = conn.execute(
        "SELECT COUNT(*) FROM dataset_examples WHERE train_selected = 1"
    ).fetchone()[0]

    with pytest.raises(RuntimeError, match="final training pack gate failed"):
        finalize_training_pack(conn, targets={"sft": 4, "dpo": 4, "persona": 4})

    after = conn.execute(
        "SELECT COUNT(*) FROM dataset_examples WHERE train_selected = 1"
    ).fetchone()[0]
    assert after == before
