"""Automatic safeguards for ocbrain v0.2 (spec §5.5, §5.6, §5.1-6).

The human approval gate is gone. Its replacement is a set of *automatic* safeguards
that never enqueue a human:

* :func:`quarantine_knowledge` / :func:`release_quarantine` — a status change encoded
  additively via the ``knowledge.quarantine_reason`` column plus a tripwire evidence
  row and a ``correction_recorded`` audit event. Never a human queue.
* :data:`TRIPWIRES` + :func:`run_tripwires` — the six auto-quarantine tripwires.
* :func:`scan_evidence_for_injection` — injection scan over new third-party evidence.
* :func:`auto_decide_compilations` — the automatic replacement for the human
  compilation-decision gate.

Stdlib-only; ``MaintenanceResult``-shaped returns, matching the maintenance module.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from ocbrain.config import QuarantineConfig
from ocbrain.db import (
    get_knowledge,
    link_knowledge_evidence,
    now_iso,
    upsert_evidence,
)
from ocbrain.events import (
    append_event,
    decide_compilation,
    hard_blocked_belief,
    list_compilation_proposals,
    rebuild_projection,
)
from ocbrain.ids import content_hash
from ocbrain.maintenance import MaintenanceResult
from ocbrain.text import find_probable_injection, find_probable_secret_leaks

# Evidence source types that carry unvalidated third-party content (spec §5.5).
THIRD_PARTY_SOURCE_TYPES = frozenset(
    {
        "openclaw_history_file",
        "codex_history_file",
        "claude_history_file",
        "unknown_history_file",
        "session_harvest",
        "web",
    }
)


# --------------------------------------------------------------------------- #
# Shared low-level helpers
# --------------------------------------------------------------------------- #
def _get_watermark(conn: sqlite3.Connection, domain: str, stream: str) -> str | None:
    row = conn.execute(
        "SELECT watermark FROM harvest_watermarks WHERE domain = ? AND stream = ?",
        (domain, stream),
    ).fetchone()
    return row["watermark"] if row else None


def _set_watermark(
    conn: sqlite3.Connection, domain: str, stream: str, watermark: str
) -> None:
    conn.execute(
        """
        INSERT INTO harvest_watermarks (domain, stream, watermark, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(domain, stream) DO UPDATE SET
          watermark = excluded.watermark,
          updated_at = excluded.updated_at
        """,
        (domain, stream, watermark, now_iso()),
    )


def _quarantine_cfg(cfg: Any) -> QuarantineConfig:
    if cfg is None:
        return QuarantineConfig()
    section = getattr(cfg, "quarantine", None)
    if section is not None:
        return section
    return cfg


# --------------------------------------------------------------------------- #
# Quarantine (spec §5.5)
# --------------------------------------------------------------------------- #
def quarantine_knowledge(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    reason: str,
    actor: str = "ocbrain-autopilot",
    detail: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> bool:
    """Auto-quarantine a knowledge row (spec §5.5) — a status change, never a queue.

    Sets ``inject = 0``, demotes ``current -> candidate``, stamps ``quarantine_reason``,
    writes an ``autopilot_tripwire`` evidence row linked ``relation='contradicts'``
    (the ``check_loop_liveness`` tripwire-as-evidence pattern), and records a
    ``correction_recorded`` event (op ``demote``) so the hash-chained audit trail
    stays intact. Returns ``False`` if the row does not exist.
    """
    row = get_knowledge(conn, knowledge_id)
    if row is None:
        return False
    timestamp = (now or datetime.now(UTC)).isoformat()
    new_status = "candidate" if row["status"] == "current" else row["status"]
    conn.execute(
        """
        UPDATE knowledge
        SET status = ?, inject = 0, quarantine_reason = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_status, reason, timestamp, knowledge_id),
    )
    payload: dict[str, Any] = {
        "knowledge_id": knowledge_id,
        "reason": reason,
        "actor": actor,
    }
    if detail:
        payload["detail"] = detail
    evidence_id = upsert_evidence(
        conn,
        source_type="autopilot_tripwire",
        source_runtime="ocbrain",
        source_uri=f"ocbrain://safeguards/quarantine/{knowledge_id}/{reason}",
        content_hash=content_hash(json.dumps(payload, sort_keys=True)),
        claim=f"Auto-quarantine {knowledge_id}: {reason}.",
        verifier_status="not_required",
        privacy_scope=row["privacy_scope"],
        occurred_at=timestamp,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="contradicts")
    append_event(
        conn,
        "correction_recorded",
        {
            "target_layer": "knowledge",
            "target_id": knowledge_id,
            "op": "demote",
            "body": reason,
            "author": actor,
            "hard": False,
        },
        writer=actor,
    )
    return True


def release_quarantine(
    conn: sqlite3.Connection,
    knowledge_id: str,
    *,
    actor: str,
    reason: str,
    now: datetime | None = None,
) -> bool:
    """The only path back out of quarantine (spec §5.5).

    Clears ``quarantine_reason`` and records a ``correction_recorded`` audit event.
    Does not re-promote the row — admission/promotion runs through the normal
    autonomous path afterwards. Returns ``False`` if the row is missing or not
    currently quarantined.
    """
    row = get_knowledge(conn, knowledge_id)
    if row is None or row["quarantine_reason"] is None:
        return False
    timestamp = (now or datetime.now(UTC)).isoformat()
    conn.execute(
        "UPDATE knowledge SET quarantine_reason = NULL, updated_at = ? WHERE id = ?",
        (timestamp, knowledge_id),
    )
    append_event(
        conn,
        "correction_recorded",
        {
            "target_layer": "knowledge",
            "target_id": knowledge_id,
            "op": "release",
            "body": reason,
            "author": actor,
            "hard": False,
        },
        writer=actor,
    )
    return True


# --------------------------------------------------------------------------- #
# Injection scan (spec §5.6, autopilot stage 4)
# --------------------------------------------------------------------------- #
def scan_evidence_for_injection(
    conn: sqlite3.Connection, *, limit: int = 1000
) -> MaintenanceResult:
    """Scan new evidence for injection, watermarked by rowid (spec §5.6, stage 4).

    Third-party evidence (:data:`THIRD_PARTY_SOURCE_TYPES`) is scanned with
    :func:`find_probable_injection`; a hit stamps ``injection_scan_status='flagged'``
    and records the pattern names in ``injection_scan_hits``. Trusted evidence is
    marked ``clean`` without scanning. The ``injection_scan`` rowid watermark makes
    the pass idempotent.
    """
    watermark = int(_get_watermark(conn, "injection_scan", "evidence") or 0)
    rows = conn.execute(
        """
        SELECT rowid AS rid, id, claim, source_type
        FROM evidence
        WHERE rowid > ? AND injection_scan_status IS NULL
        ORDER BY rowid ASC
        LIMIT ?
        """,
        (watermark, limit),
    ).fetchall()
    details: list[dict[str, Any]] = []
    scanned = 0
    max_rowid = watermark
    for row in rows:
        max_rowid = max(max_rowid, row["rid"])
        scanned += 1
        if row["source_type"] not in THIRD_PARTY_SOURCE_TYPES:
            conn.execute(
                "UPDATE evidence SET injection_scan_status = 'clean' WHERE id = ?",
                (row["id"],),
            )
            continue
        hits = find_probable_injection(row["claim"] or "")
        status = "flagged" if hits else "clean"
        conn.execute(
            "UPDATE evidence SET injection_scan_status = ?, injection_scan_hits = ? WHERE id = ?",
            (status, json.dumps(hits) if hits else None, row["id"]),
        )
        if hits:
            details.append({"evidence_id": row["id"], "hits": hits})
    if rows:
        _set_watermark(conn, "injection_scan", "evidence", str(max_rowid))
    return MaintenanceResult("injection-scan", scanned, details)


# --------------------------------------------------------------------------- #
# Tripwire registry (spec §5.5)
# --------------------------------------------------------------------------- #
def _knowledge_text(row: sqlite3.Row) -> str:
    return f"{row['value_text'] or ''}\n{row['title'] or ''}"


def _tw_injection_suspected(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    if find_probable_injection(_knowledge_text(row)):
        return "injection_suspected"
    for evidence in conn.execute(
        """
        SELECT e.source_type AS source_type, e.injection_scan_status AS scan_status
        FROM knowledge_evidence ke
        JOIN evidence e ON e.id = ke.evidence_id
        WHERE ke.knowledge_id = ?
        """,
        (row["id"],),
    ):
        if (
            evidence["source_type"] in THIRD_PARTY_SOURCE_TYPES
            and evidence["scan_status"] == "flagged"
        ):
            return "injection_suspected"
    return None


def _tw_secret_leak(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    if find_probable_secret_leaks(_knowledge_text(row)):
        return "secret_leak"
    return None


def _tw_bad_feedback_spike(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    window_start = (now - timedelta(days=cfg.bad_feedback_window_days)).isoformat()
    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM retrieval_uses
        WHERE knowledge_id = ?
          AND outcome IN ('harmful', 'failed')
          AND served_at >= ?
        """,
        (row["id"], window_start),
    ).fetchone()[0]
    if count >= cfg.bad_feedback_count:
        return "bad_feedback_spike"
    return None


def _tw_hard_correction(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    if hard_blocked_belief(conn, row["id"]):
        return "hard_correction"
    return None


def _tw_contradiction_thrash(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    window_start = (now - timedelta(days=cfg.thrash_window_days)).isoformat()
    count = conn.execute(
        """
        SELECT COUNT(*)
        FROM knowledge_evidence
        WHERE knowledge_id = ?
          AND relation IN ('contradicts', 'supersedes')
          AND created_at >= ?
        """,
        (row["id"], window_start),
    ).fetchone()[0]
    if count >= cfg.thrash_count:
        return "contradiction_thrash"
    return None


def _tw_prescriptive_unverified_serving(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: QuarantineConfig, now: datetime
) -> str | None:
    if not (row["prescriptive"] or row["type"] == "capability"):
        return None
    if not row["inject"]:
        return None
    passed = conn.execute(
        """
        SELECT COUNT(*)
        FROM knowledge_evidence ke
        JOIN evidence e ON e.id = ke.evidence_id
        WHERE ke.knowledge_id = ? AND e.verifier_status = 'passed'
        """,
        (row["id"],),
    ).fetchone()[0]
    if passed:
        return None
    approvals = conn.execute(
        """
        SELECT COUNT(*)
        FROM signal_events
        WHERE knowledge_id = ?
          AND kind IN ('user_approval', 'user_thanks')
          AND polarity = 'good'
        """,
        (row["id"],),
    ).fetchone()[0]
    if approvals:
        return None
    return "prescriptive_unverified_serving"


# Ordered registry — the six auto-quarantine tripwires (spec §5.5).
TRIPWIRES: tuple[tuple[str, Any], ...] = (
    ("injection_suspected", _tw_injection_suspected),
    ("secret_leak", _tw_secret_leak),
    ("bad_feedback_spike", _tw_bad_feedback_spike),
    ("hard_correction", _tw_hard_correction),
    ("contradiction_thrash", _tw_contradiction_thrash),
    ("prescriptive_unverified_serving", _tw_prescriptive_unverified_serving),
)


def run_tripwires(
    conn: sqlite3.Connection,
    cfg: Any = None,
    *,
    limit: int = 1000,
    now: datetime | None = None,
) -> MaintenanceResult:
    """Fire the tripwire registry over candidate/current rows touched since watermark.

    Each eligible (not-quarantined) row is checked against every tripwire; the first
    that fires auto-quarantines the row. Watermarked on ``knowledge.updated_at`` so a
    row re-enters when it changes. Idempotent: quarantined rows drop out of the query.
    """
    quarantine_cfg = _quarantine_cfg(cfg)
    timestamp = now or datetime.now(UTC)
    watermark = _get_watermark(conn, "tripwires", "knowledge") or ""
    rows = conn.execute(
        """
        SELECT *
        FROM knowledge
        WHERE status IN ('candidate', 'current')
          AND quarantine_reason IS NULL
          AND updated_at > ?
        ORDER BY updated_at ASC, id ASC
        LIMIT ?
        """,
        (watermark, limit),
    ).fetchall()
    details: list[dict[str, Any]] = []
    max_watermark = watermark
    for row in rows:
        if row["updated_at"] and row["updated_at"] > max_watermark:
            max_watermark = row["updated_at"]
        for slug, predicate in TRIPWIRES:
            reason = predicate(conn, row, quarantine_cfg, timestamp)
            if reason:
                quarantine_knowledge(
                    conn,
                    row["id"],
                    reason=reason,
                    actor="ocbrain-autopilot",
                    detail={"tripwire": slug},
                    now=timestamp,
                )
                details.append({"id": row["id"], "tripwire": slug, "reason": reason})
                break
    if rows and max_watermark:
        _set_watermark(conn, "tripwires", "knowledge", max_watermark)
    return MaintenanceResult("tripwires", len(details), details)


# --------------------------------------------------------------------------- #
# Automatic compilation decisions (spec §5.1-6) — replaces the human decide gate
# --------------------------------------------------------------------------- #
def auto_decide_compilations(
    conn: sqlite3.Connection,
    *,
    actor: str = "ocbrain-autopilot",
    limit: int = 500,
) -> MaintenanceResult:
    """Auto-decide every undecided compilation proposal (spec §5.1-6).

    ``shadow`` (the ready-made quarantine analog) when the body flags injection, the
    belief is hard-blocked, or the reward band is ``discard``; otherwise ``approve``.
    Every decision is written with ``rebuild=False`` and a single
    :func:`rebuild_projection` runs at the end (the fold is O(all events)).
    """
    proposals = list_compilation_proposals(conn, include_decided=False, limit=limit)
    details: list[dict[str, Any]] = []
    for proposal in proposals:
        body = proposal.get("body") or ""
        belief_id = proposal.get("belief_id")
        reward_band = proposal.get("reward_band")
        if (
            find_probable_injection(body)
            or (belief_id and hard_blocked_belief(conn, belief_id))
            or reward_band == "discard"
        ):
            decision = "shadow"
        else:
            decision = "approve"
        decide_compilation(
            conn,
            proposal_event_id=proposal["proposal_event_id"],
            decision=decision,
            actor=actor,
            rebuild=False,
            check_existing=True,
        )
        details.append(
            {
                "proposal_event_id": proposal["proposal_event_id"],
                "belief_id": belief_id,
                "decision": decision,
            }
        )
    if details:
        rebuild_projection(conn)
    return MaintenanceResult("auto-decide", len(details), details)
