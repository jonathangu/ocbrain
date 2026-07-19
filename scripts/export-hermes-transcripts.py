#!/usr/bin/env python3
"""Export Hermes session transcripts (state.db) to JSONL files for ocbrain harvest.

Hermes keeps its real transcript store in ``~/.hermes/state.db`` (SQLite),
which ocbrain's file-based harvest cannot read directly. This exporter dumps
each session to ``~/.hermes/sessions/export/hermes-<session_id>.jsonl`` — a
path below ``.hermes`` so ``history_runtime()`` attributes it to ``hermes``.

Writes are content-compared before replacing, so unchanged sessions keep their
mtime and ocbrain's fingerprint gate skips them on recurring runs.

Usage: export-hermes-transcripts.py [--db PATH] [--out DIR] [--max-file-bytes N]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_DB = Path.home() / ".hermes" / "state.db"
DEFAULT_OUT = Path.home() / ".hermes" / "sessions" / "export"


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat()


def export_session(conn: sqlite3.Connection, session_id: str, meta: dict, max_bytes: int) -> str:
    """Render one session as JSONL text (bounded to max_bytes)."""
    rows = conn.execute(
        "SELECT role, content, tool_name, timestamp FROM messages "
        "WHERE session_id = ? AND active = 1 ORDER BY timestamp, id",
        (session_id,),
    )
    lines = [json.dumps({"_meta": meta}, ensure_ascii=False)]
    total = len(lines[0]) + 1
    for role, content, tool_name, ts in rows:
        if not content:
            continue
        record = {
            "role": role,
            "timestamp": iso(ts),
            "content": content,
        }
        if tool_name:
            record["tool"] = tool_name
        line = json.dumps(record, ensure_ascii=False)
        if total + len(line) + 1 > max_bytes:
            lines.append(json.dumps({"_truncated": True}))
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) + "\n"


def write_if_changed(path: Path, text: str) -> bool:
    """Write only when content differs; preserves mtime for fingerprint gating."""
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except OSError:
            pass
    path.write_text(text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-file-bytes", type=int, default=200_000)
    args = parser.parse_args()

    if not args.db.exists():
        print(json.dumps({"exported": 0, "reason": f"no state.db at {args.db}"}))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        sessions = conn.execute(
            "SELECT id, source, display_name, model, started_at, ended_at, message_count "
            "FROM sessions WHERE message_count > 0"
        ).fetchall()
        exported = unchanged = 0
        for sid, source, display, model, started, ended, msg_count in sessions:
            meta = {
                "session_id": sid,
                "source": source,
                "display_name": display,
                "model": model,
                "started_at": iso(started),
                "ended_at": iso(ended),
                "message_count": msg_count,
            }
            text = export_session(conn, sid, meta, args.max_file_bytes)
            if write_if_changed(args.out / f"hermes-{sid}.jsonl", text):
                exported += 1
            else:
                unchanged += 1
    finally:
        conn.close()
    print(json.dumps({"exported": exported, "unchanged": unchanged, "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
