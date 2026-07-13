from __future__ import annotations

import json
from pathlib import Path

from ocbrain.config import OcbrainConfig
from ocbrain.db import connect, init_db
from ocbrain.events import canonical_json
from ocbrain_training.dataset.classify import DPO_CONTRAST_GATE_VERSION
from ocbrain_training.dataset.pilot import (
    _eligible_rows,
    prepare_blind_pairs,
    prepare_multiblind,
    prepare_pilot,
    record_training_result,
    score_blind_ratings,
    score_multiblind,
)
from ocbrain_training.dataset.quality import store_example


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
        metadata={
            "session_id": f"session-{index}",
            "sender_verified": dataset == "persona",
        },
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
        metadata={
            "gate": "strict",
            "contrast_gate_version": DPO_CONTRAST_GATE_VERSION,
        },
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


def test_second_pilot_reuses_frozen_eval_bytes_and_heldout_bar(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(25):
        _store_chat(conn, "persona", i)
    _store_chat(conn, "sft", 99)
    _store_dpo(conn, 1)
    conn.commit()
    first = tmp_path / "pilot-v1"
    prepare_pilot(conn, cfg=OcbrainConfig(), output_dir=first, min_grade=0.8)

    for i in range(25, 35):
        _store_chat(conn, "persona", i)
    conn.commit()
    second = tmp_path / "pilot-v2"
    result = prepare_pilot(
        conn,
        cfg=OcbrainConfig(),
        output_dir=second,
        min_grade=0.8,
        eval_from=first,
        seed="ocbrain-voice-pilot-v2-training",
        training_iterations=50,
    )

    assert result["eval_prompt_count"] == 20
    for name in ("prompts.jsonl", "references.jsonl", "rubric.json"):
        assert (second / "eval" / name).read_bytes() == (first / "eval" / name).read_bytes()
    first_manifest = json.loads((first / "pilot-manifest.json").read_text())
    second_manifest = json.loads((second / "pilot-manifest.json").read_text())
    assert second_manifest["heldout_content_hash"] == first_manifest["heldout_content_hash"]
    assert second_manifest["frozen_eval"]["reused"] is True
    assert second_manifest["mlx"]["trainer_argv"][-1] == "50"

    heldout_ids = {
        row["source_example_id"] for row in _read_jsonl(first / "eval" / "references.jsonl")
    }
    heldout_hashes = {
        row["content_hash"]
        for row in conn.execute("SELECT id, content_hash FROM dataset_examples").fetchall()
        if row["id"] in heldout_ids
    }
    for split in ("sft", "dpo", "persona"):
        for record in _read_jsonl(second / "train" / f"{split}.jsonl"):
            # The exported body intentionally drops metadata, so compare its
            # canonical body hash to the DB content hashes reserved by v1.
            from ocbrain.ids import content_hash

            assert content_hash(canonical_json(record)) not in heldout_hashes


def test_v04_quality_gate_preserves_legacy_sentinel_and_requires_train_classes(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(45):
        _store_chat(conn, "persona", i)
    _store_chat(conn, "sft", 99)
    _store_dpo(conn, 1)
    conn.execute(
        "UPDATE dataset_examples SET train_class = CASE dataset "
        "WHEN 'persona' THEN 'train_voice' WHEN 'sft' THEN 'train_skill' "
        "WHEN 'dpo' THEN 'train_judgment' END, "
        "train_selected = CASE WHEN dataset = 'persona' THEN 0 ELSE 1 END"
    )
    conn.execute(
        "UPDATE dataset_examples SET train_selected = 1 WHERE id IN "
        "(SELECT id FROM dataset_examples WHERE dataset = 'persona' ORDER BY id LIMIT 5)"
    )
    conn.commit()
    sentinel = tmp_path / "pilot-v2"
    prepare_pilot(conn, cfg=OcbrainConfig(), output_dir=sentinel, min_grade=0.8)

    root = tmp_path / "pilot-v3"
    result = prepare_pilot(
        conn,
        cfg=OcbrainConfig(),
        output_dir=root,
        min_grade=0.8,
        eval_prompts=20,
        quality_gates=True,
        minimum_train_counts={"sft": 1, "dpo": 1, "persona": 5},
        sentinel_from=sentinel,
        base_model="local/test-4bit",
    )

    assert result["training_ready"] is True
    for source_name, copied_name in (
        ("prompts.jsonl", "legacy-sentinel-prompts.jsonl"),
        ("references.jsonl", "legacy-sentinel-references.jsonl"),
        ("rubric.json", "legacy-sentinel-rubric.json"),
    ):
        assert (root / "eval" / copied_name).read_bytes() == (
            sentinel / "eval" / source_name
        ).read_bytes()
    manifest = json.loads((root / "pilot-manifest.json").read_text())
    assert manifest["quality_gate"]["passed"] is True
    assert manifest["legacy_sentinel"]["prompt_count"] == 20


def test_pilot_eligibility_rechecks_legacy_content_and_dpo_gate(tmp_path: Path):
    conn = _db(tmp_path)
    _store_chat(conn, "sft", 1)
    _store_chat(conn, "persona", 2)
    _store_dpo(conn, 3)
    conn.execute(
        "UPDATE dataset_examples SET train_class = CASE dataset "
        "WHEN 'sft' THEN 'train_skill' WHEN 'persona' THEN 'train_voice' "
        "WHEN 'dpo' THEN 'train_judgment' END"
    )

    sft = conn.execute(
        "SELECT id, example_json FROM dataset_examples WHERE dataset = 'sft'"
    ).fetchone()
    sft_record = json.loads(sft["example_json"])
    sft_record["messages"][-1]["content"] = (
        "[[reply_to_current]] I’m checking the release now and will report later."
    )
    conn.execute(
        "UPDATE dataset_examples SET example_json = ?, grade_score = 1.0 WHERE id = ?",
        (canonical_json(sft_record), sft["id"]),
    )

    dpo = conn.execute(
        "SELECT id, example_json FROM dataset_examples WHERE dataset = 'dpo'"
    ).fetchone()
    dpo_record = json.loads(dpo["example_json"])
    dpo_record["metadata"].pop("contrast_gate_version")
    conn.execute(
        "UPDATE dataset_examples SET example_json = ?, grade_score = 1.0 WHERE id = ?",
        (canonical_json(dpo_record), dpo["id"]),
    )

    assert _eligible_rows(conn, "sft", min_grade=0.8) == []
    assert _eligible_rows(conn, "dpo", min_grade=0.8) == []
    assert len(_eligible_rows(conn, "persona", min_grade=0.8)) == 1


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


def test_four_way_blind_pack_and_scoring(tmp_path: Path):
    conn = _db(tmp_path)
    for i in range(22):
        _store_chat(conn, "persona", i)
    _store_chat(conn, "sft", 99)
    _store_dpo(conn, 1)
    conn.commit()
    root = tmp_path / "pilot"
    prepare_pilot(conn, cfg=OcbrainConfig(), output_dir=root, min_grade=0.8)
    prompts = _read_jsonl(root / "eval" / "prompts.jsonl")
    response_sets = {}
    for source in ("base", "tuned", "frontier"):
        path = tmp_path / f"{source}.jsonl"
        path.write_text(
            "\n".join(
                canonical_json({"eval_id": row["eval_id"], "response": f"{source} response"})
                for row in prompts
            )
            + "\n"
        )
        response_sets[source] = path
    result = prepare_multiblind(root, response_sets)
    assert result["items"] == 20
    key = json.loads((root / "eval" / "multiblind-key.json").read_text())
    ratings = tmp_path / "multiratings.jsonl"
    ratings.write_text(
        "\n".join(
            canonical_json(
                {
                    "eval_id": eval_id,
                    "ranking": [
                        next(label for label, source in mapping.items() if source == wanted)
                        for wanted in ("tuned", "jonathan", "frontier", "base")
                    ],
                }
            )
            for eval_id, mapping in key["items"].items()
        )
        + "\n"
    )
    report = score_multiblind(root, ratings)
    assert report["first_place"]["tuned"] == 20
    assert report["mean_rank"] == {
        "base": 4.0,
        "frontier": 3.0,
        "jonathan": 2.0,
        "tuned": 1.0,
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
