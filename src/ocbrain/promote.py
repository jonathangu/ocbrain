"""Auto-promotion, injection-gating, and decay/demotion of memory (spec §5.7).

Promotion decides which ``current`` knowledge rows get ``inject=1`` (i.e. reach
the injectable ``memory`` view). A row is eligible only if it is labeled good
with enough confidence (or clears the sparse-signal bootstrap exception), scans
clean for injection/secret leaks, and — for the risky class (prescriptive /
capability / high-risk) — carries passed-verifier evidence or an explicit
approval signal. Eligible rows are ranked by ``promote_score``; the top
``promote.max_injected`` win, subject to a ``build_excerpt`` char-budget dry-run
(``promote.max_chars``). Human-origin injected rows are pinned and never demoted
by score. Decay halves the score of memory served-but-never-useful within the
decay window; demotion drops inject on rows that turn bad/low-confidence or get
quarantined.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

from ocbrain.autolabel import USEFUL_OUTCOMES
from ocbrain.db import SCOPE_RANK, knowledge_evidence, now_iso
from ocbrain.excerpt import build_excerpt
from ocbrain.text import find_probable_injection, find_probable_secret_leaks

APPROVAL_KINDS = ("user_approval", "user_thanks")
RUNTIME = "ocbrain-autopilot"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def promote_score(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    cfg: Any,
    *,
    now: datetime | None = None,
) -> float:
    """Composite promotion score (§5.7).

    ``0.4·quality_confidence + 0.25·recency + 0.2·use_rate + 0.15·scope_bonus``.
    """
    now = now or datetime.now(UTC)
    quality_conf = float(row["quality_confidence"] or 0.0)
    recency = _recency(row["updated_at"], now, cfg.promote.decay_days)
    served, useful = _retrieval_counts(conn, row["id"])
    use_rate = useful / max(1, served)
    scope_bonus = SCOPE_RANK.get(row["privacy_scope"], 1) / 3.0
    return (
        0.4 * quality_conf
        + 0.25 * recency
        + 0.2 * use_rate
        + 0.15 * scope_bonus
    )


def _recency(updated_at: str | None, now: datetime, decay_days: int) -> float:
    ts = _parse_ts(updated_at)
    if ts is None or decay_days <= 0:
        return 1.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / decay_days)


def _retrieval_counts(conn: sqlite3.Connection, knowledge_id: str) -> tuple[int, int]:
    served = conn.execute(
        "SELECT COUNT(*) FROM retrieval_uses WHERE knowledge_id = ?", (knowledge_id,)
    ).fetchone()[0]
    placeholders = ",".join("?" for _ in USEFUL_OUTCOMES)
    useful = conn.execute(
        f"SELECT COUNT(*) FROM retrieval_uses "  # noqa: S608 - fixed literal tuple
        f"WHERE knowledge_id = ? AND outcome IN ({placeholders})",
        (knowledge_id, *USEFUL_OUTCOMES),
    ).fetchone()[0]
    return int(served), int(useful)


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
def injection_clean(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """Scan body + ALL linked evidence claims for injection/secret leaks (§5.6)."""
    body = " ".join(
        str(x) for x in (row["title"], row["subject"], row["predicate"], row["value_text"]) if x
    )
    if find_probable_injection(body) or find_probable_secret_leaks(body):
        return False
    for evidence in knowledge_evidence(conn, row["id"]):
        claim = str(evidence["claim"] or "")
        if find_probable_injection(claim) or find_probable_secret_leaks(claim):
            return False
    return True


def _has_passed_verifier(conn: sqlite3.Connection, knowledge_id: str) -> bool:
    return any(
        e["verifier_status"] == "passed" for e in knowledge_evidence(conn, knowledge_id)
    )


def _has_approval_signal(conn: sqlite3.Connection, knowledge_id: str) -> bool:
    placeholders = ",".join("?" for _ in APPROVAL_KINDS)
    row = conn.execute(
        f"SELECT 1 FROM signal_events "  # noqa: S608 - fixed literal tuple
        f"WHERE knowledge_id = ? AND polarity = 'good' AND kind IN ({placeholders}) LIMIT 1",
        (knowledge_id, *APPROVAL_KINDS),
    ).fetchone()
    return row is not None


def promotion_eligible(conn: sqlite3.Connection, row: sqlite3.Row, cfg: Any) -> bool:
    """All four gates of §5.7 must hold for ``inject=1``."""
    if row["status"] != "current" or row["quarantine_reason"] is not None:
        return False

    quality_conf = row["quality_confidence"] or 0.0
    confidence = row["confidence"] or 0.0
    normal = row["quality_label"] == "good" and quality_conf >= cfg.promote.min_confidence
    bootstrap = (
        confidence >= cfg.promote.bootstrap_min_confidence
        and _has_passed_verifier(conn, row["id"])
    )
    if not (normal or bootstrap):
        return False

    if not injection_clean(conn, row):
        return False

    risky = (
        row["prescriptive"] == 1
        or row["type"] == "capability"
        or row["risk"] in ("high", "critical")
    )
    if risky and not (
        _has_passed_verifier(conn, row["id"]) or _has_approval_signal(conn, row["id"])
    ):
        return False
    return True


# --------------------------------------------------------------------------- #
# Human-memory bootstrap (v0.3)
# --------------------------------------------------------------------------- #
def _risky_row(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """Risky class per §5.7: prescriptive / capability / high-risk."""
    risky = (
        row["prescriptive"] == 1
        or row["type"] == "capability"
        or row["risk"] in ("high", "critical")
    )
    if not risky:
        return False
    return not (
        _has_passed_verifier(conn, row["id"]) or _has_approval_signal(conn, row["id"])
    )


def _human_bootstrap_cfg(cfg: Any) -> tuple[bool, set[str], int]:
    hb = getattr(cfg.promote, "human_bootstrap", None) or {}
    enabled = bool(hb.get("enabled"))
    sources = {str(s) for s in (hb.get("sources") or [])}
    try:
        cap = int(hb.get("cap") or 0)
    except (TypeError, ValueError):
        cap = 0
    return enabled, sources, cap


def _has_bootstrap_source(
    conn: sqlite3.Connection, knowledge_id: str, sources: set[str]
) -> bool:
    """True if any linked evidence carries a human-vetted bootstrap source_type."""
    if not sources:
        return False
    ordered = sorted(sources)
    placeholders = ",".join("?" for _ in ordered)
    row = conn.execute(
        f"SELECT 1 FROM knowledge_evidence ke "  # noqa: S608 - fixed placeholder count
        f"JOIN evidence e ON e.id = ke.evidence_id "
        f"WHERE ke.knowledge_id = ? AND e.source_type IN ({placeholders}) LIMIT 1",
        (knowledge_id, *ordered),
    ).fetchone()
    return row is not None


def human_bootstrap_eligible(
    conn: sqlite3.Connection, row: sqlite3.Row, cfg: Any, sources: set[str]
) -> bool:
    """A curated human-memory row may earn ``inject=1`` WITHOUT a judge label.

    Still gated by the injection/secret scan and the risky-class rule (§5.7); the
    char budget is enforced afterwards by :func:`_enforce_char_budget`.
    """
    if row["status"] != "current" or row["quarantine_reason"] is not None:
        return False
    if not _has_bootstrap_source(conn, row["id"], sources):
        return False
    if not injection_clean(conn, row):
        return False
    if _risky_row(conn, row):
        return False
    return True


# --------------------------------------------------------------------------- #
# Promotion / demotion
# --------------------------------------------------------------------------- #
def promote_to_memory(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    now: datetime | None = None,
    runtime: str = RUNTIME,
) -> dict[str, Any]:
    """Re-rank memory: promote top-N eligible rows, demote score losers (§5.7)."""
    now = now or datetime.now(UTC)
    rows = conn.execute(
        "SELECT * FROM knowledge WHERE status = 'current' AND quarantine_reason IS NULL"
    ).fetchall()

    scored: dict[str, float] = {}
    for row in rows:
        score = promote_score(conn, row, cfg, now=now)
        scored[row["id"]] = score
        conn.execute(
            "UPDATE knowledge SET promote_score = ? WHERE id = ?", (score, row["id"])
        )

    eligible = [row for row in rows if promotion_eligible(conn, row, cfg)]
    eligible.sort(key=lambda r: (scored[r["id"]], r["id"]), reverse=True)
    selected = eligible[: cfg.promote.max_injected]
    selected_ids = {row["id"] for row in selected}

    # Human-memory bootstrap (v0.3): curated, human-vetted rows may be injected
    # WITHOUT a judge label, capped at ``human_bootstrap.cap``, on top of the
    # score-ranked winners. Injection/secret scan + risky-class rules still hold;
    # the char budget below is the final arbiter.
    hb_enabled, hb_sources, hb_cap = _human_bootstrap_cfg(cfg)
    if hb_enabled and hb_cap > 0 and hb_sources:
        bootstrap = [
            row
            for row in rows
            if row["id"] not in selected_ids
            and human_bootstrap_eligible(conn, row, cfg, hb_sources)
        ]
        bootstrap.sort(key=lambda r: (scored[r["id"]], r["id"]), reverse=True)
        for row in bootstrap[:hb_cap]:
            selected.append(row)
            selected_ids.add(row["id"])

    promoted = 0
    demoted = 0
    for row in selected:
        if row["inject"] != 1:
            conn.execute(
                "UPDATE knowledge SET inject = 1, updated_at = ? WHERE id = ?",
                (now_iso(), row["id"]),
            )
            promoted += 1
    for row in rows:
        if (
            row["inject"] == 1
            and row["id"] not in selected_ids
            and row["origin"] != "human"
        ):
            conn.execute(
                "UPDATE knowledge SET inject = 0, updated_at = ? WHERE id = ?",
                (now_iso(), row["id"]),
            )
            demoted += 1

    overflow = _enforce_char_budget(conn, cfg, runtime)
    return {
        "action": "promote",
        "changed": promoted + demoted + overflow,
        "promoted": promoted,
        "demoted": demoted + overflow,
    }


def _enforce_char_budget(conn: sqlite3.Connection, cfg: Any, runtime: str) -> int:
    """Demote lowest-score non-human rows until the excerpt fits ``max_chars``."""
    demoted = 0
    while True:
        block = build_excerpt(conn, runtime, limit=cfg.promote.max_injected)
        if len(block) <= cfg.promote.max_chars:
            return demoted
        victim = conn.execute(
            """
            SELECT id FROM knowledge
            WHERE status = 'current' AND inject = 1 AND origin != 'human'
            ORDER BY COALESCE(promote_score, -1) ASC, id ASC
            LIMIT 1
            """
        ).fetchone()
        if victim is None:
            return demoted
        conn.execute(
            "UPDATE knowledge SET inject = 0, updated_at = ? WHERE id = ?",
            (now_iso(), victim["id"]),
        )
        demoted += 1


def demote_and_decay(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Drop inject on turned-bad rows; halve score of stale-served memory (§5.7)."""
    now = now or datetime.now(UTC)
    demoted = 0
    for row in conn.execute(
        "SELECT * FROM knowledge WHERE inject = 1 AND origin != 'human'"
    ).fetchall():
        bad_label = row["quality_label"] in ("bad", "neutral")
        low_conf = row["quality_confidence"] is not None and row["quality_confidence"] < 0.4
        quarantined = row["quarantine_reason"] is not None
        if bad_label or low_conf or quarantined:
            conn.execute(
                "UPDATE knowledge SET inject = 0, updated_at = ? WHERE id = ?",
                (now_iso(), row["id"]),
            )
            demoted += 1

    decayed = 0
    cutoff = (now - timedelta(days=cfg.promote.decay_days)).isoformat()
    for row in conn.execute(
        "SELECT id, promote_score FROM knowledge "
        "WHERE status = 'current' AND promote_score IS NOT NULL"
    ).fetchall():
        served = conn.execute(
            "SELECT COUNT(*) FROM retrieval_uses WHERE knowledge_id = ? AND served_at >= ?",
            (row["id"], cutoff),
        ).fetchone()[0]
        if served == 0:
            continue
        placeholders = ",".join("?" for _ in USEFUL_OUTCOMES)
        useful = conn.execute(
            f"SELECT COUNT(*) FROM retrieval_uses "  # noqa: S608 - fixed literal tuple
            f"WHERE knowledge_id = ? AND served_at >= ? AND outcome IN ({placeholders})",
            (row["id"], cutoff, *USEFUL_OUTCOMES),
        ).fetchone()[0]
        if useful == 0:
            conn.execute(
                "UPDATE knowledge SET promote_score = ? WHERE id = ?",
                (row["promote_score"] * 0.5, row["id"]),
            )
            decayed += 1

    return {
        "action": "demote_decay",
        "changed": demoted + decayed,
        "demoted": demoted,
        "decayed": decayed,
    }


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
