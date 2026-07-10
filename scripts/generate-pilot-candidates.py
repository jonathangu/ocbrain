#!/usr/bin/env python3
"""Generate local adapter responses for a prepared ocbrain blind-eval pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected objects in {path}")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    root = args.pilot_dir.expanduser()
    manifest = json.loads((root / "pilot-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("eval_ready") is not True or manifest.get("training_completed") is not True:
        raise RuntimeError("pilot eval and verified training must be complete before generation")
    mlx = manifest.get("mlx")
    if not isinstance(mlx, dict) or not mlx.get("base_model"):
        raise RuntimeError("pilot manifest has no local base model")
    result = manifest.get("training_result")
    adapter_info = result.get("adapter") if isinstance(result, dict) else None
    if not isinstance(adapter_info, dict) or not adapter_info.get("path"):
        raise RuntimeError("pilot manifest has no verified adapter path")

    prompts = _load_jsonl(root / "eval" / "prompts.jsonl")
    mx.random.seed(args.seed)
    model, tokenizer = load(
        str(mlx["base_model"]),
        adapter_path=str(Path(adapter_info["path"]).parent),
    )
    sampler = make_sampler(temp=args.temperature)
    candidates = []
    for row in prompts:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError("eval prompt has no messages list")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        candidates.append({"eval_id": row["eval_id"], "response": response})

    payload = "\n".join(_canonical(row) for row in candidates) + "\n"
    path = root / "eval" / "candidate-responses.jsonl"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    print(
        _canonical(
            {
                "action": "dataset-pilot-generate",
                "candidates": len(candidates),
                "path": str(path),
                "bytes": len(payload.encode("utf-8")),
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "local_only": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
