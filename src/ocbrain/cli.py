from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocbrain.classifier import classify_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocbrain-closeout",
        description="Dry-run classify a completed artifact into durable knowledge candidates.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Markdown artifact to classify")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        raise SystemExit(f"input does not exist: {args.input}")

    candidates = classify_artifact(args.input)
    payload = {"input": str(args.input), "candidates": [item.to_dict() for item in candidates]}
    indent = 2 if args.pretty else None
    print(json.dumps(payload, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
