from __future__ import annotations

import json
from pathlib import Path

from ocbrain.config import OcbrainConfig
from ocbrain.dataset.pilot import (
    prepare_blind_pairs,
    prepare_pilot,
    record_training_result,
    score_blind_ratings,
)
from ocbrain.dataset.quality import store_example
from ocbrain.db import connect, init_db
from ocbrain.events import canonical_json


def _db(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    return conn


def _store_chat(conn, dataset: str, index: int):
    target = (
        f"Voice sample {index} makes a specific recommendation, names the tradeoff, "
        "and stays direct without generic assistant framing or invented precision."
    )
    body = {
        "messages": [
            {"role": "user", "content": f"How should I handle decision {index}?"},
            {"role": "assistant", "content": target},
        ]
    }
    result = store_example(
        conn,
        dataset=dataset,
        source_kind="codex_session",
        source_uri=f"/local/{dataset}-{index}.jsonl",
        evidence_ids=[f"evd_{dataset}_{index}"],
        privacy_scope="workspace",
        body=body,
        metadata={"session_id": f"session-{index}"},
        target_text=target,
        base_label="good",
        base_confidence=0.9,
        occurred_at=f"2026-06-{(index % 28) + 1:02d}T00:00:00Z",
    )
    conn.execute("UPDATE dataset_examples SET grade_score = 0.9 WHERE id = ?", (result["id"],))


def _store_dpo(conn, index: int):
    chosen = (
        f"Preferred answer {index} is concrete, correct, and explains the important "
        "tradeoff clearly."
    )
    result = store_example(
        conn,
        dataset="dpo",
        source_kind="correction_event",
        source_uri=f"/local/dpo-{index}.jsonl",
        evidence_ids=[f"evd_dpo_{index}"],
        privacy_scope="workspace",
        body={
            "input": {"messages": [{"role": "user", "content": f"question {index}"}]},
            "preferred_output": [{"role": "assistant", "content": chosen}],
            "non_preferred_output": [
                {"role": "assistant", "content": "Rejected response is vague and wrong."}
            ],
        },
        metadata={},
        target_text=chosen,
        base_label="good",
        base_confidence=0.9,
    )
    conn.execute("UPDATE dataset_examples SET grade_score = 0.9 WHERE id = ?", (result["id"],))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_eval_pack_exists_before_train_and_is_deterministic(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(25):
        _store_chat(conn, "persona", i)
    for i in range(4):
        _store_chat(conn, "sft", 100 + i)
        _store_dpo(conn, i)
    conn.commit()
    root = tmp_path / "pilot"

    first = prepare_pilot(
        conn,
        cfg=OcbrainConfig(),
        output_dir=root,
        min_grade=0.8,
        base_model="local/test-4bit",
        base_model_source="upstream/test",
        base_model_revision="abc123",
    )
    assert first["eval_ready"] is True
    assert first["eval_prompt_count"] == 20
    assert first["heldout_count"] == 20
    assert first["train_counts"]["persona"] == 5
    assert first["training_ready"] is True
    assert first["mlx_train_count"] + first["mlx_valid_count"] == 9
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    assert manifest["mlx"]["base_model_revision"] == "abc123"
    for split in ("train", "valid"):
        path = root / "mlx" / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            prompt_chars = sum(len(message["content"]) for message in row["messages"][:-1])
            assert prompt_chars <= manifest["mlx"]["prompt_char_limit"]
    manifest_before = (root / "pilot-manifest.json").read_bytes()
    prompts_before = (root / "eval" / "prompts.jsonl").read_bytes()
    second = prepare_pilot(
        conn,
        cfg=OcbrainConfig(),
        output_dir=root,
        min_grade=0.8,
        base_model="local/test-4bit",
        base_model_source="upstream/test",
        base_model_revision="abc123",
    )
    assert second["eval_ready"] is True
    assert (root / "pilot-manifest.json").read_bytes() == manifest_before
    assert (root / "eval" / "prompts.jsonl").read_bytes() == prompts_before
    assert len(_read_jsonl(root / "eval" / "references.jsonl")) == 20
    assert all("metadata" not in row for row in _read_jsonl(root / "train" / "sft.jsonl"))


def test_blind_randomization_and_scoring(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(22):
        _store_chat(conn, "persona", i)
    _store_chat(conn, "sft", 99)
    _store_dpo(conn, 1)
    conn.commit()
    root = tmp_path / "pilot"
    prepare_pilot(conn, cfg=OcbrainConfig(), output_dir=root, min_grade=0.8)

    prompts = _read_jsonl(root / "eval" / "prompts.jsonl")
    candidates = root / "candidate.jsonl"
    candidates.write_text(
        "\n".join(
            canonical_json({"eval_id": row["eval_id"], "response": "candidate response"})
            for row in prompts
        )
        + "\n"
    )
    blind = prepare_blind_pairs(root, candidates)
    assert blind["pairs"] == 20
    key = json.loads((root / "eval" / "blind_key.json").read_text())
    ratings = root / "ratings.jsonl"
    dimensions = ["voice_fidelity", "taste_alignment", "naturalness", "specificity"]
    ratings.write_text(
        "\n".join(
            canonical_json(
                {
                    "eval_id": eval_id,
                    "winner": mapping["candidate_side"],
                    "scores": {
                        mapping["reference_side"]: {name: 3 for name in dimensions},
                        mapping["candidate_side"]: {name: 4 for name in dimensions},
                    },
                }
            )
            for eval_id, mapping in key["items"].items()
        )
        + "\n"
    )
    report = score_blind_ratings(root, ratings)
    assert report["outcomes"] == {"reference": 0, "candidate": 20, "tie": 0}
    assert report["candidate_win_rate_decided"] == 1.0
    assert report["dimensions"]["voice_fidelity"] == {
        "reference": 3.0,
        "candidate": 4.0,
    }


def test_record_training_requires_and_hashes_adapter(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(22):
        _store_chat(conn, "persona", i)
    _store_chat(conn, "sft", 99)
    _store_dpo(conn, 1)
    conn.commit()
    root = tmp_path / "pilot"
    prepare_pilot(
        conn,
        cfg=OcbrainConfig(),
        output_dir=root,
        min_grade=0.8,
        base_model="local/test-4bit",
    )
    adapters = root / "adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"adapter-bytes")
    (adapters / "adapter_config.json").write_text('{"rank":8}\n')
    result = record_training_result(
        root,
        iterations=25,
        train_loss=2.2,
        validation_loss=4.5,
        exit_code=0,
    )
    assert result["training_completed"] is True
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    assert manifest["training_started"] is True
    assert manifest["training_result"]["iterations"] == 25
    assert len(manifest["training_result"]["adapter"]["sha256"]) == 64
