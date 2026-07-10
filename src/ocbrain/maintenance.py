from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ocbrain.db import upsert_evidence
from ocbrain.ids import content_hash, stable_id


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
    unhelpful_ttl_days: int | None = 14,
    archive_stale_days: int | None = None,
    now: datetime | None = None,
) -> MaintenanceResult:
    timestamp = now or datetime.now(UTC)
    stale_before = (timestamp - timedelta(days=ttl_days)).isoformat()
    unhelpful_before = (
        (timestamp - timedelta(days=unhelpful_ttl_days)).isoformat()
        if unhelpful_ttl_days is not None
        else None
    )
    details: list[dict[str, Any]] = []

    stale_rows = list(
        conn.execute(
            """
            SELECT
              knowledge.id,
              knowledge.type,
              knowledge.status,
              knowledge.updated_at,
              COUNT(retrieval_uses.id) AS retrieval_count,
              SUM(
                CASE
                  WHEN retrieval_uses.outcome IN ('improved','helpful','used')
                  THEN 1 ELSE 0
                END
              ) AS useful_count
            FROM knowledge
            LEFT JOIN retrieval_uses ON retrieval_uses.knowledge_id = knowledge.id
            WHERE knowledge.status IN ('candidate','current')
            GROUP BY knowledge.id
            HAVING
              (
                knowledge.updated_at < ?
                AND retrieval_count = 0
              )
              OR (
                ? IS NOT NULL
                AND knowledge.updated_at < ?
                AND retrieval_count > 0
                AND useful_count = 0
              )
            ORDER BY knowledge.updated_at ASC, knowledge.id ASC
            """,
            (stale_before, unhelpful_before, unhelpful_before),
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
                "reason": (
                    "ttl_unreferenced"
                    if row["retrieval_count"] == 0
                    else "ttl_served_without_usefulness"
                ),
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


# Raw imported catalog documents (memory / history / file catalog) carry a
# non-distilled ``origin``. Review-/loop-distilled docs (wiki/procedure) carry
# ``harvest``/``loop`` and human docs carry ``human`` — those are never archived
# by the backlog sweep.
_CATALOG_ARCHIVE_REASON = "catalog_never_referenced"


def archive_unreferenced_catalog(
    conn: sqlite3.Connection,
    *,
    older_than_days: int,
    batch_cap: int = 5000,
    now: datetime | None = None,
) -> MaintenanceResult:
    """Archive never-referenced, stale catalog *doc* rows (v0.3 backlog sweep).

    The judge and rebuild both pay for a 101k file-catalog doc backlog that no
    retrieval has ever touched. This sweep moves such rows to
    ``status='archived'`` so they leave the working set.

    Guarantees:

    * **Reversible.** Only ``status`` (+ an ``invalidation_reason`` marker)
      changes — the row, its evidence links, body, and embeddings are preserved,
      so un-archiving is a pure status flip.
    * **Idempotent.** Already-``archived`` rows are never re-selected, so a re-run
      over an unchanged DB archives nothing.
    * **Scoped.** Only raw ``doc`` imports (``origin`` not in harvest/loop/human),
      idle longer than ``older_than_days``, with zero ``retrieval_uses`` rows, and
      not currently injected, are eligible. Batch-capped at ``batch_cap``.
    """
    timestamp = now or datetime.now(UTC)
    if batch_cap <= 0:
        return MaintenanceResult("archive-catalog", 0, [])
    cutoff = (timestamp - timedelta(days=older_than_days)).isoformat()
    rows = list(
        conn.execute(
            """
            SELECT k.id, k.status
            FROM knowledge AS k
            WHERE k.type = 'doc'
              AND k.status IN ('candidate', 'current', 'stale')
              AND COALESCE(k.inject, 0) = 0
              AND (k.origin IS NULL OR k.origin NOT IN ('harvest', 'loop', 'human'))
              AND k.updated_at < ?
              AND NOT EXISTS (
                SELECT 1 FROM retrieval_uses ru WHERE ru.knowledge_id = k.id
              )
            ORDER BY k.updated_at ASC, k.id ASC
            LIMIT ?
            """,
            (cutoff, batch_cap),
        )
    )
    details: list[dict[str, Any]] = []
    for row in rows:
        conn.execute(
            """
            UPDATE knowledge
            SET status = 'archived',
                invalidation_reason = COALESCE(invalidation_reason, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (_CATALOG_ARCHIVE_REASON, timestamp.isoformat(), row["id"]),
        )
        details.append(
            {
                "id": row["id"],
                "from_status": row["status"],
                "to_status": "archived",
                "reason": _CATALOG_ARCHIVE_REASON,
            }
        )
    return MaintenanceResult("archive-catalog", len(details), details)


def _emit_heal_signal(
    conn: sqlite3.Connection,
    *,
    knowledge_id: str,
    role: str,
    polarity: str,
    weight: float,
    winner_id: str,
    loser_id: str,
    evidence_id: str,
    timestamp: str,
) -> str:
    """Persist a ``heal_superseded`` signal (spec §5.2 / §2.2 maintenance row).

    Uses the shared ``stable_id('sig', source, source_ref, kind, content_hash)``
    recipe so a re-run of :func:`heal_conflicts` collapses to the same row
    (INSERT OR IGNORE), keeping the emission idempotent.
    """
    details = {"role": role, "winner": winner_id, "loser": loser_id}
    details_json = json.dumps(details, sort_keys=True, separators=(",", ":"))
    source_ref = f"{winner_id}:{loser_id}"
    signal_id = stable_id(
        "sig", "maintenance", source_ref, "heal_superseded", content_hash(details_json)
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO signal_events (
          id, kind, polarity, weight, source, source_ref, session_key,
          knowledge_id, evidence_id, details, occurred_at, created_at
        )
        VALUES (?, 'heal_superseded', ?, ?, 'maintenance', ?, NULL, ?, ?, ?, ?, ?)
        """,
        (
            signal_id,
            polarity,
            weight,
            source_ref,
            knowledge_id,
            evidence_id,
            details_json,
            timestamp,
            timestamp,
        ),
    )
    return signal_id


def heal_conflicts(
    conn: sqlite3.Connection,
    *,
    numeric_threshold: float = 0.0,
    now: datetime | None = None,
    on_supersede: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
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
            # Loser loses (bad 0.4); winner is reinforced (good 0.3) — spec §5.2.
            _emit_heal_signal(
                conn,
                knowledge_id=loser["id"],
                role="loser",
                polarity="bad",
                weight=0.4,
                winner_id=winner["id"],
                loser_id=loser["id"],
                evidence_id=evidence_id,
                timestamp=timestamp.isoformat(),
            )
            _emit_heal_signal(
                conn,
                knowledge_id=winner["id"],
                role="winner",
                polarity="good",
                weight=0.3,
                winner_id=winner["id"],
                loser_id=loser["id"],
                evidence_id=evidence_id,
                timestamp=timestamp.isoformat(),
            )
            if on_supersede is not None:
                on_supersede(dict(winner), dict(loser))
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
        if due_at is None:
            detail = _record_liveness_tripwire(conn, row, "invalid_deadman_due", timestamp)
            if detail is not None:
                details.append(detail)
            continue
        if due_at > timestamp:
            continue
        tripwire_kinds = liveness_tripwire_kinds(row, due_at=due_at)
        for kind in tripwire_kinds:
            detail = _record_liveness_tripwire(conn, row, kind, timestamp)
            if detail is not None:
                details.append(detail)
    return MaintenanceResult("liveness-check", len(details), details)


def _record_liveness_tripwire(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    kind: str,
    timestamp: datetime,
) -> dict[str, Any] | None:
    """Write one state-change tripwire, not one copy per sweep.

    ``deadman_due_at`` moves whenever the producer refreshes its scan. Including
    it in the hash created a new evidence row for the same unchanged dead loop on
    every maintenance pass. Heartbeat/ledger state is the durable condition; a
    later recovery and second death changes those fields and produces new
    evidence. Malformed due values include the raw value so distinct corruption
    remains distinguishable.
    """
    source_uri = f"ocbrain://loop/{row['loop_id']}/{row['run_id']}/liveness/{kind}"
    payload = {
        "kind": kind,
        "loop_id": row["loop_id"],
        "run_id": row["run_id"],
        "last_heartbeat_at": row["last_heartbeat_at"],
        "last_ledger_write_at": row["last_ledger_write_at"],
    }
    if kind == "invalid_deadman_due":
        payload["deadman_due_at"] = row["deadman_due_at"]
    digest = content_hash(json.dumps(payload, sort_keys=True))
    existing = conn.execute(
        "SELECT id FROM evidence WHERE source_uri = ? AND content_hash = ? LIMIT 1",
        (source_uri, digest),
    ).fetchone()
    if existing is not None:
        return None
    evidence_id = upsert_evidence(
        conn,
        source_type="loop_tripwire",
        source_runtime="ocbrain",
        source_uri=source_uri,
        content_hash=digest,
        claim=liveness_claim(kind, row),
        verifier_status="not_required",
        loop_tags={
            "loop_id": row["loop_id"],
            "run_id": row["run_id"],
            "tripwire": kind,
        },
        privacy_scope="workspace",
        occurred_at=timestamp.isoformat(),
    )
    return {
        "loop_id": row["loop_id"],
        "run_id": row["run_id"],
        "tripwire": kind,
        "evidence_id": evidence_id,
    }


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
    if kind == "invalid_deadman_due":
        return f"Loop {loop} run {run} has an invalid deadman deadline."
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
