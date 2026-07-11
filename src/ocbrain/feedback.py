"""Retrieval-feedback coverage and conservative automatic outcome inference.

Explicit agent feedback remains the highest-quality signal.  This module adds a
floor for sessions that omit it: a served result becomes ``used`` only when a
later evidence/correction event in the same session exists, or when a later
event explicitly mentions one of the served object ids.  Inferred rows are
marked separately and can never masquerade as human/agent feedback.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from ocbrain.db import now_iso


def feedback_coverage(conn: sqlite3.Connection) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT outcome, COALESCE(feedback_source, '') AS source, COUNT(*) AS n "
        "FROM retrieval_uses GROUP BY outcome, COALESCE(feedback_source, '')"
    ).fetchall()
    total = sum(int(row["n"]) for row in rows)
    pending = sum(int(row["n"]) for row in rows if row["outcome"] in {"served", "unknown"})
    explicit = sum(int(row["n"]) for row in rows if row["source"] == "explicit")
    inferred = sum(int(row["n"]) for row in rows if str(row["source"]).startswith("inferred_"))
    covered = total - pending
    instrumented = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome NOT IN ('served','unknown') THEN 1 ELSE 0 END) AS covered
        FROM retrieval_uses
        WHERE query_text IS NOT NULL OR served_ids_json IS NOT NULL
        """
    ).fetchone()
    instrumented_total = int(instrumented["total"] or 0)
    instrumented_covered = int(instrumented["covered"] or 0)
    return {
        "action": "retrieval-feedback-stats",
        "total": total,
        "covered": covered,
        "pending": pending,
        "coverage": round(covered / total, 4) if total else 0.0,
        "explicit": explicit,
        "inferred": inferred,
        "instrumented_total": instrumented_total,
        "instrumented_covered": instrumented_covered,
        "instrumented_coverage": (
            round(instrumented_covered / instrumented_total, 4) if instrumented_total else 0.0
        ),
        "by_outcome": {
            str(row["outcome"]): sum(
                int(item["n"]) for item in rows if item["outcome"] == row["outcome"]
            )
            for row in rows
        },
    }


def _served_ids(value: str | None) -> list[str]:
    try:
        parsed = json.loads(value or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_outcome(event: sqlite3.Row, served_ids: set[str]) -> tuple[str, str] | None:
    body = str(event["body_json"] or "")
    if served_ids and any(item in body for item in served_ids):
        if event["kind"] == "correction_recorded":
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {}
            if parsed.get("op") in {"mark_wrong", "retract"} or parsed.get("hard"):
                return "harmful", "inferred_exact_correction"
        return "used", "inferred_exact_reference"
    if event["kind"] in {"evidence_recorded", "correction_recorded"}:
        return "used", "inferred_session_evidence"
    return None


def infer_retrieval_outcomes(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    window_hours: int = 24,
    limit: int = 500,
) -> dict[str, Any]:
    """Infer outcomes for pending retrievals without overwriting explicit labels."""
    now = now or datetime.now(UTC)
    rows = conn.execute(
        """
        SELECT id, served_at, session_id, served_ids_json
        FROM retrieval_uses
        WHERE outcome = 'served'
          AND feedback_source IS NULL
        ORDER BY served_at ASC
        LIMIT ?
        """,
        (max(1, int(limit)),),
    ).fetchall()
    changed = 0
    exact = 0
    session = 0
    harmful = 0
    for row in rows:
        served_at = _parse_ts(row["served_at"])
        if served_at > now:
            continue
        until = min(now, served_at + timedelta(hours=max(1, window_hours)))
        served_ids = set(_served_ids(row["served_ids_json"]))
        params: list[Any] = [row["served_at"], until.isoformat()]
        id_clauses = ["body_json LIKE ?" for _ in served_ids]
        id_params = [f"%{item}%" for item in sorted(served_ids)]
        clause = ""
        if row["session_id"]:
            exact_clause = " OR ".join(id_clauses)
            clause = " AND (session_id = ?" + (f" OR {exact_clause}" if exact_clause else "") + ")"
            params.append(row["session_id"])
            params.extend(id_params)
        elif served_ids:
            clause = " AND (" + " OR ".join(id_clauses) + ")"
            params.extend(id_params)
        else:
            continue
        events = conn.execute(
            "SELECT kind, body_json, session_id FROM brain_events "
            "WHERE ts > ? AND ts <= ?" + clause + " ORDER BY ts ASC LIMIT 50",
            params,
        ).fetchall()
        inferred: tuple[str, str] | None = None
        for event in events:
            # A session match permits the conservative evidence-event fallback;
            # a global exact-id scan does not.
            if row["session_id"] and event["session_id"] != row["session_id"]:
                body = str(event["body_json"] or "")
                if not any(item in body for item in served_ids):
                    continue
            inferred = _event_outcome(event, served_ids)
            if inferred:
                break
        if inferred is None:
            continue
        outcome, source = inferred
        cursor = conn.execute(
            """
            UPDATE retrieval_uses
            SET outcome = ?, feedback_source = ?, feedback_at = ?,
                note = COALESCE(note, 'automatic outcome inference')
            WHERE id = ? AND outcome = 'served' AND feedback_source IS NULL
            """,
            (outcome, source, now_iso(), row["id"]),
        )
        if cursor.rowcount <= 0:
            continue
        changed += 1
        harmful += int(outcome == "harmful")
        exact += int(source in {"inferred_exact_reference", "inferred_exact_correction"})
        session += int(source == "inferred_session_evidence")
    return {
        "action": "infer-retrieval-feedback",
        "changed": changed,
        "examined": len(rows),
        "exact": exact,
        "session": session,
        "harmful": harmful,
        "coverage": feedback_coverage(conn),
    }
