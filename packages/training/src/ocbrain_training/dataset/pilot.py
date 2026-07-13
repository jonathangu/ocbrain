"""Eval-before-train pack for the first local fine-tune pilot.

Preparation is deterministic and local-only. It selects a held-out persona set
before writing any training file, creates twenty voice/taste prompts and private
references, and excludes every held-out content hash from all training streams.
Later helpers randomize real-operator vs model responses for blind scoring.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id

from ocbrain_training.dataset.classify import classify_record

RUBRIC = {
    "scale": "1 (poor) to 5 (excellent)",
    "dimensions": {
        "voice_fidelity": "Sounds like one coherent person rather than generic assistant prose.",
        "taste_alignment": (
            "Makes the same kind of prioritization and tradeoffs the operator values."
        ),
        "naturalness": "Feels direct and human, without imitation artifacts or canned framing.",
        "specificity": (
            "Uses concrete judgment and enough detail to be useful without fake precision."
        ),
    },
    "winner_rule": "Choose A, B, or tie after scoring both responses independently.",
}

MLX_LM_GIT_COMMIT = "a790972f0f844d81067ed45c28b524220a10c019"
MLX_MAX_PROMPT_CHARS = 1600


def _jsonl(rows: Iterable[dict[str, Any]]) -> str:
    materialized = [canonical_json(row) for row in rows]
    return ("\n".join(materialized) + "\n") if materialized else ""


def _atomic_write(path: Path, payload: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return {
        "path": str(path),
        "bytes": len(payload.encode("utf-8")),
        "sha256": sha256_text(payload),
    }


def _default_output_dir(conn: sqlite3.Connection, cfg: OcbrainConfig) -> Path:
    configured = Path(cfg.dataset.export_dir).expanduser()
    if configured.is_absolute():
        return configured / "pilot-v1"
    row = conn.execute("PRAGMA database_list").fetchone()
    db_file = row["file"] if row is not None else ""
    if db_file:
        return Path(db_file).resolve().parent / configured.name / "pilot-v1"
    return configured / "pilot-v1"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON objects in {path.name}")
            rows.append(value)
    return rows


def _eligible_rows(
    conn: sqlite3.Connection,
    dataset: str,
    *,
    min_grade: float,
    train_class: str | None = None,
    selected_only: bool = False,
) -> list[sqlite3.Row]:
    class_clause = " AND train_class = ?" if train_class else ""
    selected_clause = " AND train_selected = 1" if selected_only else ""
    params: tuple[Any, ...] = (
        (dataset, min_grade, train_class) if train_class else (dataset, min_grade)
    )
    rows = list(
        conn.execute(
            f"""
            SELECT id, dataset, content_hash, example_json, grade_score,
                   grade_model, grade_prompt_version, train_class, train_selected,
                   source_kind, privacy_scope, quality_label, quality_reasons
            FROM dataset_examples
            WHERE dataset = ?
              AND quality_label = 'good'
              AND privacy_scope != 'private'
              AND grade_score >= ?
              {class_clause}
              {selected_clause}
            ORDER BY id
            """,  # noqa: S608 - optional clause is a fixed internal literal
            params,
        )
    )
    expected_class = (
        train_class
        or {
            "sft": "train_skill",
            "dpo": "train_judgment",
            "persona": "train_voice",
        }[dataset]
    )
    eligible: list[sqlite3.Row] = []
    for row in rows:
        current_class, _reason = classify_record(row)
        if current_class == expected_class:
            eligible.append(row)
    return eligible


def _persona_eval_parts(record: dict[str, Any]) -> tuple[list[dict[str, Any]], str] | None:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        target = message.get("content")
        prompt = messages[:index]
        if isinstance(target, str) and target.strip() and prompt:
            clean_prompt = [m for m in prompt if isinstance(m, dict)]
            if clean_prompt:
                return clean_prompt, target
    return None


def _record_body(example_json: str) -> dict[str, Any]:
    record = json.loads(example_json)
    if not isinstance(record, dict):
        raise ValueError("dataset example is not a JSON object")
    return {key: value for key, value in record.items() if key != "metadata"}


def _mlx_chat_record(body: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """Bound chat context while preserving the final assistant completion.

    MLX-LM truncates the tokenized sequence from the right at ``max_seq_length``.
    With prompt masking, an oversized prompt can consume the whole window and
    leave zero loss-bearing tokens (NaN loss). Drop oldest context messages until
    the conservative character budget is met; reject an indivisible long prompt.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, False
    clean = [dict(message) for message in messages if isinstance(message, dict)]
    if len(clean) < 2 or clean[-1].get("role") != "assistant":
        return None, False
    target = clean[-1].get("content")
    if not isinstance(target, str) or len(target.strip()) < 80:
        return None, False
    prompt = clean[:-1]
    original_count = len(prompt)
    while len(prompt) > 1 and sum(len(str(m.get("content") or "")) for m in prompt) > (
        MLX_MAX_PROMPT_CHARS
    ):
        prompt.pop(0)
    prompt_chars = sum(len(str(message.get("content") or "")) for message in prompt)
    if prompt_chars > MLX_MAX_PROMPT_CHARS:
        return None, original_count != len(prompt)
    return {"messages": [*prompt, clean[-1]]}, original_count != len(prompt)


def prepare_pilot(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    output_dir: str | Path | None = None,
    min_grade: float | None = None,
    eval_prompts: int = 20,
    seed: str = "ocbrain-voice-pilot-v1",
    base_model: str | None = None,
    base_model_source: str | None = None,
    base_model_revision: str | None = None,
    eval_from: str | Path | None = None,
    training_iterations: int = 25,
    quality_gates: bool = False,
    minimum_train_counts: dict[str, int] | None = None,
    sentinel_from: str | Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic private pilot pack, refusing train-first states."""
    cfg = cfg or load_config()
    if eval_prompts < 20:
        raise ValueError("the voice/taste pilot requires at least 20 eval prompts")
    threshold = min_grade
    if threshold is None:
        threshold = cfg.dataset.export_min_grade
    if threshold is None:
        threshold = 0.8
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_grade must be between 0 and 1")
    if training_iterations < 1:
        raise ValueError("training_iterations must be positive")

    root = (
        Path(output_dir).expanduser() if output_dir is not None else _default_output_dir(conn, cfg)
    )
    required_classes = {
        "sft": "train_skill",
        "dpo": "train_judgment",
        "persona": "train_voice",
    }
    minimums = dict(minimum_train_counts or {"sft": 1000, "dpo": 200, "persona": 300})
    sentinel_hashes: set[str] = set()
    sentinel_eval: dict[str, Any] | None = None
    if sentinel_from is not None:
        sentinel_root = Path(sentinel_from).expanduser()
        sentinel_prompts = (sentinel_root / "eval" / "prompts.jsonl").read_text(encoding="utf-8")
        sentinel_references = (sentinel_root / "eval" / "references.jsonl").read_text(
            encoding="utf-8"
        )
        sentinel_rubric = (sentinel_root / "eval" / "rubric.json").read_text(encoding="utf-8")
        sentinel_prompt_rows = _load_jsonl(sentinel_root / "eval" / "prompts.jsonl")
        sentinel_reference_rows = _load_jsonl(sentinel_root / "eval" / "references.jsonl")
        if len(sentinel_prompt_rows) < 20 or len(sentinel_prompt_rows) != len(
            sentinel_reference_rows
        ):
            raise RuntimeError("legacy sentinel must contain matched prompts/references")
        if json.loads(sentinel_rubric) != RUBRIC:
            raise RuntimeError("legacy sentinel rubric changed")
        sentinel_source_ids = [row.get("source_example_id") for row in sentinel_reference_rows]
        if any(not isinstance(value, str) for value in sentinel_source_ids):
            raise RuntimeError("legacy sentinel references lost source example ids")
        placeholders = ",".join("?" for _ in sentinel_source_ids)
        sentinel_rows = conn.execute(
            f"SELECT id, content_hash FROM dataset_examples WHERE id IN ({placeholders})",  # noqa: S608
            tuple(sentinel_source_ids),
        ).fetchall()
        if len(sentinel_rows) != len(set(sentinel_source_ids)):
            raise RuntimeError("legacy sentinel source examples are missing")
        sentinel_hashes = {str(row["content_hash"]) for row in sentinel_rows}
        sentinel_files = {
            "prompts": _atomic_write(
                root / "eval" / "legacy-sentinel-prompts.jsonl", sentinel_prompts
            ),
            "references": _atomic_write(
                root / "eval" / "legacy-sentinel-references.jsonl", sentinel_references
            ),
            "rubric": _atomic_write(root / "eval" / "legacy-sentinel-rubric.json", sentinel_rubric),
        }
        sentinel_eval = {
            "source_name": sentinel_root.name,
            "prompt_count": len(sentinel_prompt_rows),
            "files": sentinel_files,
            "content_hash": sha256_text(canonical_json(sorted(sentinel_hashes))),
        }
    frozen_eval: dict[str, Any] | None = None
    if eval_from is not None:
        source_root = Path(eval_from).expanduser()
        source_eval = source_root / "eval"
        source_payloads = {
            "prompts": (source_eval / "prompts.jsonl").read_text(encoding="utf-8"),
            "references": (source_eval / "references.jsonl").read_text(encoding="utf-8"),
            "rubric": (source_eval / "rubric.json").read_text(encoding="utf-8"),
        }
        prompts = _load_jsonl(source_eval / "prompts.jsonl")
        references = _load_jsonl(source_eval / "references.jsonl")
        if len(prompts) < 20 or len(prompts) != len(references):
            raise RuntimeError("frozen eval must contain at least 20 matched prompts/references")
        if json.loads(source_payloads["rubric"]) != RUBRIC:
            raise RuntimeError("frozen eval rubric does not match the unchanged pilot rubric")
        prompt_ids = {row.get("eval_id") for row in prompts}
        reference_ids = {row.get("eval_id") for row in references}
        if None in prompt_ids or prompt_ids != reference_ids:
            raise RuntimeError("frozen eval prompt/reference ids do not match")
        source_ids = [row.get("source_example_id") for row in references]
        if any(not isinstance(value, str) for value in source_ids):
            raise RuntimeError("frozen references must retain source example ids")
        placeholders = ",".join("?" for _ in source_ids)
        heldout_rows = conn.execute(
            f"SELECT id, content_hash FROM dataset_examples WHERE id IN ({placeholders})",
            tuple(source_ids),
        ).fetchall()
        if len(heldout_rows) != len(set(source_ids)):
            raise RuntimeError("frozen eval source examples are missing from the dataset")
        heldout_hashes = {row["content_hash"] for row in heldout_rows}
        source_manifest_path = source_root / "pilot-manifest.json"
        if source_manifest_path.is_file():
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            expected = source_manifest.get("heldout_content_hash")
            actual = sha256_text(canonical_json(sorted(heldout_hashes)))
            if expected and expected != actual:
                raise RuntimeError("frozen eval held-out hash no longer matches its manifest")
        filenames = {
            "prompts": "prompts.jsonl",
            "references": "references.jsonl",
            "rubric": "rubric.json",
        }
        eval_files = {
            name: _atomic_write(root / "eval" / filenames[name], payload)
            for name, payload in source_payloads.items()
        }
        frozen_eval = {
            "reused": True,
            "source_name": source_root.name,
            "file_sha256": {name: value["sha256"] for name, value in eval_files.items()},
        }
    else:
        persona_rows = _eligible_rows(
            conn,
            "persona",
            min_grade=threshold,
            train_class=required_classes["persona"] if quality_gates else None,
        )
        eval_candidates: list[tuple[str, sqlite3.Row, list[dict[str, Any]], str]] = []
        for row in persona_rows:
            record = json.loads(row["example_json"])
            parts = _persona_eval_parts(record)
            if parts is None:
                continue
            prompt, reference = parts
            # Prefer locally graded voice rows outside the finalized training
            # pack, so the 100-case evaluation does not consume its own 300-row
            # training minimum.
            digest = sha256_text(f"{seed}:{row['content_hash']}")
            rank = f"{int(row['train_selected'])}:{digest}"
            if row["content_hash"] not in sentinel_hashes:
                eval_candidates.append((rank, row, prompt, reference))
        eval_candidates.sort(key=lambda item: item[0])
        if len(eval_candidates) < eval_prompts:
            raise RuntimeError(
                f"eval-before-train gate: need {eval_prompts} graded persona prompts, "
                f"found {len(eval_candidates)}"
            )

        heldout = eval_candidates[:eval_prompts]
        heldout_hashes = {row["content_hash"] for _, row, _, _ in heldout}
        prompts = []
        references = []
        for _, row, prompt, reference in heldout:
            eval_id = stable_id("eval", seed, row["id"])
            prompts.append({"eval_id": eval_id, "messages": prompt})
            references.append(
                {
                    "eval_id": eval_id,
                    "reference_response": reference,
                    "source_example_id": row["id"],
                }
            )

        # The heldout set exists in memory before any training path is opened.
        eval_files = {
            "prompts": _atomic_write(root / "eval" / "prompts.jsonl", _jsonl(prompts)),
            "references": _atomic_write(root / "eval" / "references.jsonl", _jsonl(references)),
            "rubric": _atomic_write(root / "eval" / "rubric.json", canonical_json(RUBRIC) + "\n"),
        }

    heldout_hashes |= sentinel_hashes
    if quality_gates:
        selected_grade_gaps = {
            str(row["dataset"]): int(row["n"])
            for row in conn.execute(
                """
                SELECT dataset, COUNT(*) AS n
                FROM dataset_examples
                WHERE train_selected = 1 AND grade_score IS NULL
                GROUP BY dataset
                """
            )
        }
        if selected_grade_gaps:
            raise RuntimeError(
                "v0.4 selected pack is not 100% locally graded: "
                + canonical_json(selected_grade_gaps)
            )
        preflight_counts = {
            dataset: sum(
                1
                for row in _eligible_rows(
                    conn,
                    dataset,
                    min_grade=threshold,
                    train_class=required_classes[dataset],
                    selected_only=True,
                )
                if row["content_hash"] not in heldout_hashes
            )
            for dataset in required_classes
        }
        missing = {
            dataset: {"required": int(minimums[dataset]), "found": preflight_counts[dataset]}
            for dataset in required_classes
            if preflight_counts[dataset] < int(minimums[dataset])
        }
        if missing:
            raise RuntimeError("v0.4 corpus quality gate failed: " + canonical_json(missing))

    train_files: dict[str, dict[str, Any]] = {}
    train_counts: dict[str, int] = {}
    grade_sources: dict[str, list[dict[str, str]]] = {}
    chat_records: list[tuple[str, dict[str, Any]]] = []
    mlx_trimmed = 0
    mlx_rejected = 0
    for dataset in ("sft", "dpo", "persona"):
        records: list[dict[str, Any]] = []
        eligible = _eligible_rows(
            conn,
            dataset,
            min_grade=threshold,
            train_class=required_classes[dataset] if quality_gates else None,
            selected_only=quality_gates,
        )
        sources = {
            (str(row["grade_model"] or "unknown"), str(row["grade_prompt_version"] or "unknown"))
            for row in eligible
        }
        grade_sources[dataset] = [
            {"model": model, "prompt_version": prompt_version}
            for model, prompt_version in sorted(sources)
        ]
        for row in eligible:
            if row["content_hash"] in heldout_hashes:
                continue
            body = _record_body(row["example_json"])
            records.append(body)
            if dataset in {"sft", "persona"}:
                mlx_body, trimmed = _mlx_chat_record(body)
                if mlx_body is None:
                    mlx_rejected += 1
                    continue
                mlx_trimmed += int(trimmed)
                rank = sha256_text(f"{seed}:mlx-valid:{canonical_json(mlx_body)}")
                chat_records.append((rank, mlx_body))
        train_counts[dataset] = len(records)
        train_files[dataset] = _atomic_write(root / "train" / f"{dataset}.jsonl", _jsonl(records))

    chat_records.sort(key=lambda item: item[0])
    valid_count = min(50, max(1, len(chat_records) // 10)) if len(chat_records) >= 10 else 0
    valid_chat = [record for _, record in chat_records[:valid_count]]
    train_chat = [record for _, record in chat_records[valid_count:]]
    mlx_files = {
        "train": _atomic_write(root / "mlx" / "train.jsonl", _jsonl(train_chat)),
    }
    if valid_chat:
        mlx_files["valid"] = _atomic_write(root / "mlx" / "valid.jsonl", _jsonl(valid_chat))

    mlx_source = f"mlx-lm[train] @ git+https://github.com/ml-explore/mlx-lm.git@{MLX_LM_GIT_COMMIT}"
    training_ready = bool(
        base_model
        and train_chat
        and len(prompts) >= eval_prompts
        and (not quality_gates or all(train_counts[key] >= minimums[key] for key in minimums))
    )

    manifest = {
        "version": 2 if quality_gates else 1,
        "seed": seed,
        "min_grade": threshold,
        "eval_built_before_train": True,
        "eval_ready": len(prompts) >= 20,
        "training_started": False,
        "eval_prompt_count": len(prompts),
        "heldout_content_hash": sha256_text(canonical_json(sorted(heldout_hashes))),
        "train_counts": train_counts,
        "grade_sources": grade_sources,
        "mlx": {
            "ready": training_ready,
            "base_model": base_model,
            "base_model_source": base_model_source,
            "base_model_revision": base_model_revision,
            "format": "chat",
            "train_count": len(train_chat),
            "valid_count": len(valid_chat),
            "prompt_char_limit": MLX_MAX_PROMPT_CHARS,
            "context_trimmed_count": mlx_trimmed,
            "rejected_count": mlx_rejected,
            "files": mlx_files,
            "dpo_included": False,
            "dpo_note": "DPO stays as a separate preference artifact; MLX-LM LoRA trains chat SFT.",
            "trainer_source": mlx_source,
            "trainer_argv": [
                "uvx",
                "--from",
                mlx_source,
                "mlx_lm.lora",
                "--model",
                base_model or "<select-local-base-model>",
                "--train",
                "--data",
                str(root / "mlx"),
                "--adapter-path",
                str(root / "adapters"),
                "--mask-prompt",
                "--batch-size",
                "1",
                "--num-layers",
                "8",
                "--max-seq-length",
                "2048",
                "--grad-checkpoint",
                "--steps-per-report",
                "5",
                "--steps-per-eval",
                "10",
                "--save-every",
                "5",
                "--seed",
                "20260709",
                "--iters",
                str(training_iterations),
            ],
        },
        "eval_files": eval_files,
        "frozen_eval": frozen_eval,
        "legacy_sentinel": sentinel_eval,
        "quality_gate": {
            "enabled": quality_gates,
            "minimum_train_counts": minimums if quality_gates else None,
            "required_classes": required_classes if quality_gates else None,
            "passed": bool(quality_gates),
        },
        "train_files": train_files,
        "blind_protocol": {
            "candidate_format": {"eval_id": "eval_...", "response": "model text"},
            "ratings_format": {
                "eval_id": "eval_...",
                "winner": "a|b|tie",
                "scores": {"a": {"dimension": 1}, "b": {"dimension": 1}},
            },
        },
    }
    manifest_file = _atomic_write(root / "pilot-manifest.json", canonical_json(manifest) + "\n")
    return {
        "action": "dataset-pilot-prepare",
        "changed": len(train_files) + len(eval_files) + len(mlx_files) + 1,
        "output_dir": str(root),
        "manifest_path": manifest_file["path"],
        "eval_ready": manifest["eval_ready"],
        "eval_prompt_count": len(prompts),
        "train_counts": train_counts,
        "heldout_count": len(heldout_hashes),
        "training_ready": training_ready,
        "mlx_train_count": len(train_chat),
        "mlx_valid_count": len(valid_chat),
    }


def prepare_blind_pairs(
    pilot_dir: str | Path,
    candidate_responses: str | Path,
    *,
    seed: str = "ocbrain-blind-v1",
) -> dict[str, Any]:
    root = Path(pilot_dir).expanduser()
    prompts = _load_jsonl(root / "eval" / "prompts.jsonl")
    references = {
        row["eval_id"]: row["reference_response"]
        for row in _load_jsonl(root / "eval" / "references.jsonl")
    }
    candidates = {
        row["eval_id"]: row["response"]
        for row in _load_jsonl(Path(candidate_responses).expanduser())
        if isinstance(row.get("response"), str)
    }
    expected = {row["eval_id"] for row in prompts}
    if set(references) != expected or set(candidates) != expected:
        raise ValueError("candidate/reference eval ids must exactly match the prepared prompts")

    pairs: list[dict[str, Any]] = []
    key: dict[str, Any] = {"seed": seed, "items": {}}
    for prompt in prompts:
        eval_id = prompt["eval_id"]
        reference_side = "a" if int(sha256_text(f"{seed}:{eval_id}"), 16) % 2 == 0 else "b"
        candidate_side = "b" if reference_side == "a" else "a"
        outputs = {reference_side: references[eval_id], candidate_side: candidates[eval_id]}
        pairs.append(
            {
                "eval_id": eval_id,
                "messages": prompt["messages"],
                "response_a": outputs["a"],
                "response_b": outputs["b"],
                "rubric": RUBRIC,
            }
        )
        key["items"][eval_id] = {
            "reference_side": reference_side,
            "candidate_side": candidate_side,
        }

    pairs_file = _atomic_write(root / "eval" / "blind_pairs.jsonl", _jsonl(pairs))
    key_file = _atomic_write(root / "eval" / "blind_key.json", canonical_json(key) + "\n")
    return {
        "action": "dataset-pilot-blind",
        "changed": 2,
        "pairs": len(pairs),
        "pairs_path": pairs_file["path"],
        "key_path": key_file["path"],
    }


def prepare_multiblind(
    pilot_dir: str | Path,
    response_sets: dict[str, str | Path],
    *,
    seed: str = "ocbrain-multiblind-v1",
) -> dict[str, Any]:
    """Randomize Jonathan/base/tuned/frontier answers into a four-way blind pack."""
    required = {"base", "tuned", "frontier"}
    if set(response_sets) != required:
        raise ValueError("four-way blind eval requires base, tuned, and frontier responses")
    root = Path(pilot_dir).expanduser()
    references = {
        row["eval_id"]: row["reference_response"]
        for row in _load_jsonl(root / "eval" / "references.jsonl")
    }
    prompts = _load_jsonl(root / "eval" / "prompts.jsonl")
    answers: dict[str, dict[str, str]] = {
        eval_id: {"jonathan": text} for eval_id, text in references.items()
    }
    for source, path in response_sets.items():
        rows = _load_jsonl(Path(path).expanduser())
        values = {str(row.get("eval_id")): row.get("response") for row in rows}
        if set(values) != set(references) or any(
            not isinstance(value, str) for value in values.values()
        ):
            raise ValueError(f"{source} responses do not match the frozen eval ids")
        for eval_id, value in values.items():
            answers[eval_id][source] = str(value)

    items: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    labels = ["a", "b", "c", "d"]
    for prompt in prompts:
        eval_id = str(prompt["eval_id"])
        sources = sorted(answers[eval_id], key=lambda name: sha256_text(f"{seed}:{eval_id}:{name}"))
        mapping = dict(zip(labels, sources, strict=True))
        key[eval_id] = mapping
        items.append(
            {
                "eval_id": eval_id,
                "messages": prompt["messages"],
                "responses": {label: answers[eval_id][source] for label, source in mapping.items()},
            }
        )
    item_file = _atomic_write(root / "eval" / "multiblind-items.jsonl", _jsonl(items))
    key_file = _atomic_write(
        root / "eval" / "multiblind-key.json",
        canonical_json({"seed": seed, "items": key}) + "\n",
    )
    return {
        "action": "dataset-pilot-multiblind",
        "items": len(items),
        "sources": sorted({"jonathan", *required}),
        "item_file": item_file,
        "key_file": key_file,
    }


def score_multiblind(
    pilot_dir: str | Path,
    ratings_path: str | Path,
) -> dict[str, Any]:
    root = Path(pilot_dir).expanduser()
    key = json.loads((root / "eval" / "multiblind-key.json").read_text(encoding="utf-8"))
    mappings = key.get("items") if isinstance(key, dict) else None
    if not isinstance(mappings, dict):
        raise ValueError("multiblind key is missing")
    ratings = _load_jsonl(Path(ratings_path).expanduser())
    if {str(row.get("eval_id")) for row in ratings} != set(mappings):
        raise ValueError("multiblind ratings do not match the frozen eval ids")
    sources = {source for mapping in mappings.values() for source in mapping.values()}
    first_place = {source: 0 for source in sorted(sources)}
    rank_sum = {source: 0.0 for source in sorted(sources)}
    score_sum: dict[str, dict[str, float]] = {source: {} for source in sorted(sources)}
    score_count: dict[str, dict[str, int]] = {source: {} for source in sorted(sources)}
    for rating in ratings:
        eval_id = str(rating["eval_id"])
        mapping = mappings[eval_id]
        ranking = rating.get("ranking")
        if not isinstance(ranking, list) or set(ranking) != set(mapping):
            raise ValueError(f"invalid four-way ranking for {eval_id}")
        for index, label in enumerate(ranking, 1):
            source = mapping[label]
            rank_sum[source] += index
            if index == 1:
                first_place[source] += 1
        raw_scores = rating.get("scores") or {}
        for label, dimensions in raw_scores.items():
            if label not in mapping or not isinstance(dimensions, dict):
                continue
            source = mapping[label]
            for dimension, value in dimensions.items():
                score_sum[source][dimension] = score_sum[source].get(dimension, 0.0) + float(value)
                score_count[source][dimension] = score_count[source].get(dimension, 0) + 1
    report = {
        "items": len(ratings),
        "first_place": first_place,
        "mean_rank": {
            source: round(rank_sum[source] / len(ratings), 4) for source in sorted(sources)
        },
        "dimensions": {
            source: {
                dimension: round(total / score_count[source][dimension], 4)
                for dimension, total in sorted(score_sum[source].items())
            }
            for source in sorted(sources)
        },
    }
    report_file = _atomic_write(
        root / "eval" / "multiblind-report.json", canonical_json(report) + "\n"
    )
    return {"action": "dataset-pilot-multiscore", **report, "report_file": report_file}


def score_blind_ratings(
    pilot_dir: str | Path,
    ratings_path: str | Path,
) -> dict[str, Any]:
    root = Path(pilot_dir).expanduser()
    key = json.loads((root / "eval" / "blind_key.json").read_text(encoding="utf-8"))
    items = key.get("items") if isinstance(key, dict) else None
    if not isinstance(items, dict):
        raise ValueError("blind key is invalid")
    ratings = _load_jsonl(Path(ratings_path).expanduser())
    if {row.get("eval_id") for row in ratings} != set(items):
        raise ValueError("ratings eval ids must exactly match the blind key")

    outcomes = {"reference": 0, "candidate": 0, "tie": 0}
    dimension_values: dict[str, dict[str, list[float]]] = {
        name: {"reference": [], "candidate": []} for name in RUBRIC["dimensions"]
    }
    for rating in ratings:
        eval_id = rating["eval_id"]
        winner = str(rating.get("winner") or "").lower()
        mapping = items[eval_id]
        if winner == "tie":
            outcomes["tie"] += 1
        elif winner in {"a", "b"}:
            role = "reference" if winner == mapping["reference_side"] else "candidate"
            outcomes[role] += 1
        else:
            raise ValueError("winner must be a, b, or tie")

        scores = rating.get("scores")
        if not isinstance(scores, dict):
            continue
        for role in ("reference", "candidate"):
            side = mapping[f"{role}_side"]
            side_scores = scores.get(side)
            if not isinstance(side_scores, dict):
                continue
            for dimension in dimension_values:
                value = side_scores.get(dimension)
                if isinstance(value, (int, float)) and 1 <= float(value) <= 5:
                    dimension_values[dimension][role].append(float(value))

    dimensions: dict[str, dict[str, float | None]] = {}
    for dimension, roles in dimension_values.items():
        dimensions[dimension] = {
            role: round(sum(values) / len(values), 3) if values else None
            for role, values in roles.items()
        }
    decided = outcomes["reference"] + outcomes["candidate"]
    report = {
        "items": len(ratings),
        "outcomes": outcomes,
        "candidate_win_rate_decided": round(outcomes["candidate"] / decided, 4)
        if decided
        else None,
        "dimensions": dimensions,
    }
    report_file = _atomic_write(root / "eval" / "blind_report.json", canonical_json(report) + "\n")
    return {
        "action": "dataset-pilot-score",
        "changed": 1,
        "report_path": report_file["path"],
        **report,
    }


def record_training_result(
    pilot_dir: str | Path,
    *,
    iterations: int,
    train_loss: float,
    validation_loss: float,
    exit_code: int,
) -> dict[str, Any]:
    """Record a verified local trainer result after adapter files exist."""
    root = Path(pilot_dir).expanduser()
    manifest_path = root / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter = root / "adapters" / "adapters.safetensors"
    adapter_config = root / "adapters" / "adapter_config.json"
    if exit_code != 0:
        raise ValueError("cannot mark training complete with a nonzero exit code")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not adapter.is_file() or not adapter_config.is_file():
        raise FileNotFoundError("verified adapter weights/config are required")
    adapter_bytes = adapter.read_bytes()
    completed_at = datetime.now(UTC).isoformat(timespec="microseconds")
    manifest["training_started"] = True
    manifest["training_completed"] = True
    manifest["training_result"] = {
        "completed_at": completed_at,
        "iterations": iterations,
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "exit_code": exit_code,
        "adapter": {
            "path": str(adapter),
            "bytes": len(adapter_bytes),
            "sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        },
        "adapter_config": {
            "path": str(adapter_config),
            "sha256": sha256_text(adapter_config.read_text(encoding="utf-8")),
        },
    }
    written = _atomic_write(manifest_path, canonical_json(manifest) + "\n")
    return {
        "action": "dataset-pilot-record-training",
        "changed": 1,
        "manifest_path": written["path"],
        "training_completed": True,
        "iterations": iterations,
        "adapter_path": str(adapter),
        "adapter_bytes": len(adapter_bytes),
    }
