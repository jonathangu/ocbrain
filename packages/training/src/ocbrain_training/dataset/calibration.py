"""Import private human judge calibration with reasons and ideal corrections."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.events import sha256_text
from ocbrain.text import find_probable_secret_leaks


def import_calibrations(
    conn: sqlite3.Connection,
    path: str | Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser()
    payload = source.read_bytes()
    source_hash = sha256_text(payload.decode("utf-8"))
    timestamp = (now or datetime.now(UTC)).isoformat(timespec="microseconds")
    imported = complete = incomplete = 0
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"calibration line {line_number} is not an object")
        eval_id = str(row.get("eval_id") or "").strip()
        winner = str(row.get("winner") or "").lower()
        if not eval_id or winner not in {"a", "b", "tie", "neither"}:
            raise ValueError(f"calibration line {line_number} has invalid id/winner")
        reason = row.get("reason") or row.get("critique")
        ideal = row.get("ideal_response")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"calibration line {line_number} reason is not text")
        if ideal is not None and not isinstance(ideal, str):
            raise ValueError(f"calibration line {line_number} ideal_response is not text")
        if ideal and find_probable_secret_leaks(ideal):
            raise ValueError(f"calibration line {line_number} contains a probable secret")
        ideal_source = str(row.get("ideal_response_source") or ("human" if ideal else "")) or None
        status = "complete" if reason and ideal and ideal_source == "human" else "incomplete"
        conn.execute(
            """
            INSERT INTO dataset_calibrations (
              eval_id, winner, reason, ideal_response, ideal_response_source,
              preference_strength, labeled_by, labeled_at, source_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(eval_id) DO UPDATE SET
              winner=excluded.winner, reason=excluded.reason,
              ideal_response=excluded.ideal_response,
              ideal_response_source=excluded.ideal_response_source,
              preference_strength=excluded.preference_strength,
              labeled_by=excluded.labeled_by, labeled_at=excluded.labeled_at,
              source_hash=excluded.source_hash, status=excluded.status
            """,
            (
                eval_id,
                winner,
                reason,
                ideal,
                ideal_source,
                row.get("preference_strength"),
                row.get("labeled_by"),
                row.get("labeled_on") or timestamp,
                source_hash,
                status,
            ),
        )
        imported += 1
        complete += int(status == "complete")
        incomplete += int(status == "incomplete")
    return {
        "action": "dataset-calibration-import",
        "changed": imported,
        "complete": complete,
        "incomplete": incomplete,
        "source_hash": source_hash,
        "local_only": True,
        "contains_calibration_text": False,
    }
