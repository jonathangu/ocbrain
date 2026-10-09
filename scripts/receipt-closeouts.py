#!/usr/bin/env python3
"""File one OCBrain closeout per finished receipt found under the given roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from ocbrain.runtime_call import invoke

STATUS_RE = re.compile(r"^[#*\-\s]*STATUS:\s*\**\s*(DONE|BLOCKED|FAILED)\b")
PR_RE = re.compile(r"https://github\.com/[\w.-]+/[\w.-]+/pull/\d+")
TITLE_PREFIX_RE = re.compile(r"^Receipt\s*[:—-]\s*", re.IGNORECASE)
COMMAND_RE = re.compile(r"`[^`]+`")
CLOSEOUT_STATUS = {"DONE": "completed", "BLOCKED": "blocked", "FAILED": "failed"}
VERIFIER_STATUS = {"DONE": "passed", "BLOCKED": "unknown", "FAILED": "failed"}
UNRESOLVED_LIMIT = 500
SUMMARY_LIMIT = 300
PR_LIMIT = 10


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--roots", required=True, help="colon-separated directories")
    parser.add_argument("--project", default="workspace")
    parser.add_argument("--since-days", type=float, default=7.0)
    parser.add_argument(
        "--state", type=Path, default=Path.home() / ".ocbrain" / "receipt-closeouts.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _receipt_files(roots: list[Path], since_days: float) -> list[Path]:
    cutoff = time.time() - since_days * 86400
    found: dict[Path, None] = {}
    for root in roots:
        resolved_root = root.resolve()
        if not resolved_root.is_dir():
            continue
        for path in sorted(resolved_root.rglob("*.md")):
            name = path.name
            if not name.startswith(("RECEIPT-", "builder-")):
                continue
            real = path.resolve()
            if not real.is_file() or not real.is_relative_to(resolved_root):
                continue
            if real.stat().st_mtime < cutoff:
                continue
            found[real] = None
    return list(found)


def _final_status(lines: list[str]) -> tuple[int, re.Match[str]] | None:
    hit = None
    for index, line in enumerate(lines):
        match = STATUS_RE.match(line)
        if match:
            hit = (index, match)
    return hit


def _summary(lines: list[str], fallback: str) -> str:
    for line in lines:
        if line.lstrip().startswith("#"):
            text = TITLE_PREFIX_RE.sub("", line.lstrip("# \t").strip()).strip()
            if text:
                return text[:SUMMARY_LIMIT]
    return fallback


def _unresolved(lines: list[str], index: int, match: re.Match[str], word: str) -> str:
    parts = [lines[index][match.end() :].strip(" *:-\t")]
    parts.extend(line.strip() for line in lines[index + 1 :] if line.strip())
    text = " ".join(part for part in parts if part)[:UNRESOLVED_LIMIT].strip()
    return text or f"receipt reports {word} without detail"


def _awaiting(lines: list[str], index: int, unresolved: str) -> str:
    for line in lines[index + 1 :]:
        if COMMAND_RE.search(line):
            return line.strip()[:UNRESOLVED_LIMIT]
    return unresolved


def _plan(path: Path, project: str, raw: bytes) -> dict[str, Any] | None:
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hit = _final_status(lines)
    if hit is None:
        return None
    index, match = hit
    word = match.group(1)
    task_ref = f"{path.parent.name}/{path.stem}".lower()
    digest = hashlib.sha256(raw).hexdigest()
    uri = f"file://{path}"
    pr_urls = list(dict.fromkeys(PR_RE.findall(text)))[:PR_LIMIT]
    arguments: dict[str, Any] = {
        "task_ref": task_ref,
        "status": CLOSEOUT_STATUS[word],
        "summary": _summary(lines, path.stem),
        "context": {
            "project": project,
            "runtime": "cli",
            "client": "receipt-harvest",
            "task": task_ref,
        },
        "runtime_detail": "receipt-closeouts",
        "artifact_refs": [
            {"kind": "receipt", "uri": uri, "sha256": digest},
            *({"kind": "pr", "uri": url} for url in pr_urls),
        ],
        "verifier_refs": [
            {
                "kind": "receipt_status",
                "uri": f"{uri}#status",
                "status": VERIFIER_STATUS[word],
                "detail": lines[index].strip()[:UNRESOLVED_LIMIT],
            }
        ],
    }
    if word != "DONE":
        unresolved = _unresolved(lines, index, match, word)
        arguments["unresolved"] = unresolved
        if word == "BLOCKED":
            arguments["awaiting"] = _awaiting(lines, index, unresolved)
    return {"arguments": arguments, "sha256": digest}


def _load_state(path: Path) -> dict[str, dict[str, str]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _save_state(path: Path, state: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = [Path(item) for item in args.roots.split(":") if item]
    state = _load_state(args.state)
    counts = {"scanned": 0, "filed": 0, "skipped_unchanged": 0, "skipped_no_status": 0, "errors": 0}
    for path in _receipt_files(roots, args.since_days):
        counts["scanned"] += 1
        try:
            plan = _plan(path, args.project, path.read_bytes())
            if plan is None:
                counts["skipped_no_status"] += 1
                continue
            previous = state.get(str(path)) or {}
            if previous.get("sha256") == plan["sha256"]:
                counts["skipped_unchanged"] += 1
                continue
            arguments = plan["arguments"]
            if previous.get("closeout_id"):
                arguments["parent_closeout_id"] = previous["closeout_id"]
            if args.dry_run:
                print(json.dumps(arguments, sort_keys=True))
                continue
            result = invoke(args.db, "brain.closeout", arguments)
            state[str(path)] = {"sha256": plan["sha256"], "closeout_id": str(result["id"])}
            _save_state(args.state, state)
            counts["filed"] += 1
        except Exception as exc:
            counts["errors"] += 1
            print(f"receipt-closeouts: {path}: {exc}", file=sys.stderr)
    print(json.dumps(counts))
    return 1 if counts["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
