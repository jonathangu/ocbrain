from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ocbrain.db import upsert_evidence
from ocbrain.ids import content_hash


@dataclass(frozen=True)
class MaintenanceResult:
    action: str
    changed: int
    details: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "changed": self.changed, "details": self.details}


def prune_knowledge(
    conn: sqlite3.Connection,
    *,
    ttl_days: int = 30,
    archive_stale_days: int | None = None,
    now: datetime | None = None,
) -> MaintenanceResult:
    timestamp = now or datetime.now(UTC)
    stale_before = (timestamp - timedelta(days=ttl_days)).isoformat()
    details: list[dict[str, Any]] = []

    stale_rows = list(
        conn.execute(
            """
            SELECT knowledge.id, knowledge.type, knowledge.status, knowledge.updated_at
            FROM knowledge
            LEFT JOIN retrieval_uses ON retrieval_uses.knowledge_id = knowledge.id
            WHERE knowledge.status IN ('candidate','current')
              AND knowledge.updated_at < ?
            GROUP BY knowledge.id
            HAVING COUNT(retrieval_uses.id) = 0
            ORDER BY knowledge.updated_at ASC, knowledge.id ASC
            """,
            (stale_before,),
        )
    )
    for row in stale_rows:
        conn.execute(
            """
            UPDATE knowledge
            SET status = 'stale',
                invalidation_reason = 'stale',
                updated_at = ?
            WHERE id = ?
            """,
            (timestamp.isoformat(), row["id"]),
        )
        details.append(
            {
                "id": row["id"],
                "from_status": row["status"],
                "to_status": "stale",
                "reason": "ttl_unreferenced",
            }
        )

    if archive_stale_days is not None:
        archive_before = (timestamp - timedelta(days=archive_stale_days)).isoformat()
        archive_rows = list(
            conn.execute(
                """
                SELECT id, updated_at
                FROM knowledge
                WHERE status = 'stale' AND updated_at < ?
                ORDER BY updated_at ASC, id ASC
                """,
                (archive_before,),
            )
        )
        for row in archive_rows:
            conn.execute(
                """
                UPDATE knowledge
                SET status = 'archived',
                    invalidation_reason = COALESCE(invalidation_reason, 'stale'),
                    updated_at = ?
                WHERE id = ?
                """,
                (timestamp.isoformat(), row["id"]),
            )
            details.append(
                {
                    "id": row["id"],
                    "from_status": "stale",
                    "to_status": "archived",
                    "reason": "stale_archive_ttl",
                }
            )

    return MaintenanceResult("prune", len(details), details)


def heal_conflicts(
    conn: sqlite3.Connection,
    *,
    numeric_threshold: float = 0.0,
    now: datetime | None = None,
) -> MaintenanceResult:
    timestamp = now or datetime.now(UTC)
    details: list[dict[str, Any]] = []
    keys = list(
        conn.execute(
            """
            SELECT subject, predicate, COALESCE(project, '') AS project_key, COUNT(*) AS count
            FROM knowledge
            WHERE status = 'current' AND type = 'value'
            GROUP BY subject, predicate, COALESCE(project, '')
            HAVING COUNT(*) > 1
            ORDER BY subject ASC, predicate ASC, project_key ASC
            """
        )
    )
    for key in keys:
        rows = list(
            conn.execute(
                """
                SELECT *
                FROM knowledge
                WHERE status = 'current'
                  AND type = 'value'
                  AND subject IS ?
                  AND predicate IS ?
                  AND COALESCE(project, '') = ?
                ORDER BY COALESCE(confidence, 0) DESC, updated_at DESC, id ASC
                """,
                (key["subject"], key["predicate"], key["project_key"]),
            )
        )
        if not values_conflict(rows, numeric_threshold=numeric_threshold):
            continue
        winner = rows[0]
        losers = rows[1:]
        claim = (
            f"Heal superseded {len(losers)} conflicting current values for "
            f"{winner['subject']}::{winner['predicate']}."
        )
        evidence_id = upsert_evidence(
            conn,
            source_type="correction",
            source_runtime="ocbrain",
            source_uri=f"ocbrain://maintenance/heal/{winner['id']}/{timestamp.isoformat()}",
            content_hash=content_hash(json.dumps([dict(row) for row in rows], sort_keys=True)),
            claim=claim,
            verifier_status="not_required",
            project=winner["project"],
            privacy_scope=winner["privacy_scope"],
            occurred_at=timestamp.isoformat(),
        )
        for loser in losers:
            conn.execute(
                """
                UPDATE knowledge
                SET status = 'superseded',
                    superseded_by = ?,
                    invalidation_reason = 'contradicted',
                    updated_at = ?
                WHERE id = ?
                """,
                (winner["id"], timestamp.isoformat(), loser["id"]),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO knowledge_evidence (
                  knowledge_id, evidence_id, relation, created_at
                )
                VALUES (?, ?, 'supersedes', ?)
                """,
                (winner["id"], evidence_id, timestamp.isoformat()),
            )
            details.append(
                {
                    "winner": winner["id"],
                    "superseded": loser["id"],
                    "subject": winner["subject"],
                    "predicate": winner["predicate"],
                    "evidence_id": evidence_id,
                }
            )
    return MaintenanceResult("heal", len(details), details)


def values_conflict(rows: list[sqlite3.Row], *, numeric_threshold: float) -> bool:
    if len(rows) < 2:
        return False
    numeric_values = [row["value_numeric"] for row in rows if row["value_numeric"] is not None]
    if len(numeric_values) == len(rows):
        return max(numeric_values) - min(numeric_values) > numeric_threshold
    text_values = {row["value_text"] for row in rows if row["value_text"] is not None}
    bool_values = {row["value_bool"] for row in rows if row["value_bool"] is not None}
    return len(text_values) > 1 or len(bool_values) > 1


def sync_loop_liveness_from_runner(
    conn: sqlite3.Connection,
    runner_ledger: Path,
) -> int:
    runner = sqlite3.connect(f"file:{runner_ledger}?mode=ro", uri=True)
    runner.row_factory = sqlite3.Row
    try:
        rows = list(runner.execute("SELECT * FROM loop_liveness"))
    finally:
        runner.close()
    for row in rows:
        conn.execute(
            """
            INSERT INTO loop_liveness (
              loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
              expected_interval_seconds, deadman_due_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(loop_id, run_id) DO UPDATE SET
              last_heartbeat_at = excluded.last_heartbeat_at,
              last_ledger_write_at = excluded.last_ledger_write_at,
              expected_interval_seconds = excluded.expected_interval_seconds,
              deadman_due_at = excluded.deadman_due_at
            """,
            (
                row["loop_id"],
                row["run_id"],
                row["last_heartbeat_at"],
                row["last_ledger_write_at"],
                row["expected_interval_seconds"],
                row["deadman_due_at"],
            ),
        )
    return len(rows)


def check_loop_liveness(
    conn: sqlite3.Connection,
    *,
    runner_ledger: Path | None = None,
    now: datetime | None = None,
) -> MaintenanceResult:
    if runner_ledger is not None:
        sync_loop_liveness_from_runner(conn, runner_ledger)
    timestamp = now or datetime.now(UTC)
    details: list[dict[str, Any]] = []
    rows = list(
        conn.execute(
            """
            SELECT *
            FROM loop_liveness
            WHERE deadman_due_at IS NOT NULL
            ORDER BY deadman_due_at ASC, loop_id ASC, run_id ASC
            """
        )
    )
    for row in rows:
        due_at = parse_datetime(row["deadman_due_at"])
        if due_at is None or due_at > timestamp:
            continue
        tripwire_kinds = liveness_tripwire_kinds(row, due_at=due_at)
        for kind in tripwire_kinds:
            claim = liveness_claim(kind, row)
            evidence_id = upsert_evidence(
                conn,
                source_type="loop_tripwire",
                source_runtime="ocbrain",
                source_uri=f"ocbrain://loop/{row['loop_id']}/{row['run_id']}/liveness/{kind}",
                content_hash=content_hash(
                    json.dumps(
                        {
                            "kind": kind,
                            "loop_id": row["loop_id"],
                            "run_id": row["run_id"],
                            "deadman_due_at": row["deadman_due_at"],
                            "last_heartbeat_at": row["last_heartbeat_at"],
                            "last_ledger_write_at": row["last_ledger_write_at"],
                        },
                        sort_keys=True,
                    )
                ),
                claim=claim,
                verifier_status="not_required",
                loop_tags={
                    "loop_id": row["loop_id"],
                    "run_id": row["run_id"],
                    "tripwire": kind,
                },
                privacy_scope="workspace",
                occurred_at=timestamp.isoformat(),
            )
            details.append(
                {
                    "loop_id": row["loop_id"],
                    "run_id": row["run_id"],
                    "tripwire": kind,
                    "evidence_id": evidence_id,
                }
            )
    return MaintenanceResult("liveness-check", len(details), details)


def liveness_tripwire_kinds(row: sqlite3.Row, *, due_at: datetime) -> list[str]:
    kinds: list[str] = []
    heartbeat_at = parse_datetime(row["last_heartbeat_at"])
    ledger_write_at = parse_datetime(row["last_ledger_write_at"])
    if heartbeat_at is None or heartbeat_at <= due_at:
        kinds.append("heartbeat_starved")
    if ledger_write_at is None:
        kinds.append("no_ledger_writes")
    elif ledger_write_at <= due_at:
        kinds.append("stale_running_item")
    return kinds


def liveness_claim(kind: str, row: sqlite3.Row) -> str:
    loop = row["loop_id"]
    run = row["run_id"] or "unknown-run"
    if kind == "heartbeat_starved":
        return f"Loop {loop} run {run} missed its deadman heartbeat."
    if kind == "no_ledger_writes":
        return f"Loop {loop} run {run} crossed deadman without runner ledger writes."
    return f"Loop {loop} run {run} has stale running ledger activity."


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
