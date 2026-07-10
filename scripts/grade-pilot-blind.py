#!/usr/bin/env python3
"""Rate an ocbrain blind-eval pack with a loopback-only Ollama judge."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

from ocbrain.dataset.grade import require_loopback_endpoint
from ocbrain.events import canonical_json


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected objects in {path}")
        rows.append(value)
    return rows


def _score(value: Any, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} score") from exc
    if not 1 <= score <= 5:
        raise ValueError(f"{label} score outside 1..5")
    return round(score, 3)


def _normalize(raw: Any, eval_id: str, dimensions: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("judge response is not an object")
    winner = str(raw.get("winner") or "").lower()
    if winner not in {"a", "b", "tie"}:
        raise ValueError("winner must be a, b, or tie")
    scores = raw.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("judge response has no scores object")
    normalized_scores: dict[str, dict[str, float]] = {}
    for side in ("a", "b"):
        side_scores = scores.get(side)
        if not isinstance(side_scores, dict):
            raise ValueError(f"judge response has no {side} scores")
        normalized_scores[side] = {
            dimension: _score(side_scores.get(dimension), f"{side}.{dimension}")
            for dimension in dimensions
        }
    return {
        "eval_id": eval_id,
        "winner": winner,
        "scores": normalized_scores,
        "explanation": str(raw.get("explanation") or "")[:300],
    }


def _rate(
    endpoint: str,
    model: str,
    pair: dict[str, Any],
    *,
    timeout: int,
) -> dict[str, Any]:
    rubric = pair.get("rubric")
    dimensions_raw = rubric.get("dimensions") if isinstance(rubric, dict) else None
    if not isinstance(dimensions_raw, dict) or not dimensions_raw:
        raise ValueError("blind pair has no rubric dimensions")
    dimensions = list(dimensions_raw)
    schema = {
        "winner": "a|b|tie",
        "scores": {
            side: {dimension: "number 1..5" for dimension in dimensions}
            for side in ("a", "b")
        },
        "explanation": "short string",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict blind evaluator of voice and taste fidelity. "
                "The two responses are anonymized. Do not guess authorship. Score A and B "
                "independently against the prompt and rubric, then choose A, B, or tie. "
                "Prefer concise, specific judgment over generic assistant prose. Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": canonical_json(
                {
                    "messages": pair.get("messages"),
                    "response_a": pair.get("response_a"),
                    "response_b": pair.get("response_b"),
                    "rubric": rubric,
                    "required_schema": schema,
                }
            ),
        },
    ]
    body = canonical_json(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        envelope = json.loads(response.read().decode("utf-8"))
    message = envelope.get("message") if isinstance(envelope, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("local judge returned no message content")
    return _normalize(json.loads(content), str(pair["eval_id"]), dimensions)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(canonical_json(row) for row in rows) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default="qwen3.6:35b-a3b-q4_K_M")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    endpoint = require_loopback_endpoint(args.endpoint)
    root = args.pilot_dir.expanduser()
    pairs = _load_jsonl(root / "eval" / "blind_pairs.jsonl")
    output = root / "eval" / "blind-ratings.jsonl"
    ratings = [] if args.force or not output.exists() else _load_jsonl(output)
    completed = {row.get("eval_id") for row in ratings}
    for pair in pairs:
        eval_id = pair.get("eval_id")
        if eval_id in completed:
            continue
        last_error: Exception | None = None
        for _attempt in range(2):
            try:
                rating = _rate(endpoint, args.model, pair, timeout=args.timeout)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
            ratings.append(rating)
            completed.add(eval_id)
            _write_jsonl(output, ratings)
            print(canonical_json({"rated": len(ratings), "total": len(pairs)}), flush=True)
            break
        else:
            raise RuntimeError(f"failed to rate {eval_id}: {last_error}")

    if completed != {pair.get("eval_id") for pair in pairs}:
        raise RuntimeError("ratings do not exactly cover the blind pack")
    print(
        canonical_json(
            {
                "action": "dataset-pilot-rate-local",
                "items": len(ratings),
                "judge_model": args.model,
                "local_only": True,
                "path": str(output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
