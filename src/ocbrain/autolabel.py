"""Signal mining, attribution, and decayed label fold (spec §5.2–5.3).

This module owns the automatic quality-labeling half of ocbrain v0.2: it mines
*signals* (small, source-backed good/bad/neutral votes) out of the operational
substrate — ``retrieval_uses`` feedback, ``brain_events`` corrections, evidence
verifier status, the ``learning.db`` GATE/CORRECTION ledger, the commitments
file, and the cron run log — persists them idempotently to ``signal_events``,
attributes the unattached ones to knowledge rows, and folds them (with time
decay and hard-bad precedence) into ``quality_label`` / ``quality_confidence``
on the ``knowledge`` table.

Every miner is watermarked (``harvest_watermarks``) so re-running is cheap, and
every signal has a stable id (``INSERT OR IGNORE``) so re-emission is harmless.
Session signals are emitted by :mod:`ocbrain.review`; the ``llm_judge`` verdict
signals are emitted by :mod:`ocbrain.judge`. Both import :func:`record_signal`
and :class:`Signal` from here.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ocbrain.dataset.batching import DatasetWriteBatch
from ocbrain.db import now_iso, search
from ocbrain.events import canonical_json
from ocbrain.ids import content_hash, stable_id

DOMAIN = "autolabel"

# retrieval_uses.outcome -> (polarity, weight) per the taxonomy (§5.2). Outcomes
# not listed here ('served', 'neutral', 'unknown') carry no label signal.
_RETRIEVAL_MAP: dict[str, tuple[str, float]] = {
    "improved": ("good", 0.5),
    "helpful": ("good", 0.4),
    "used": ("good", 0.4),
    "harmful": ("bad", 0.9),
    "failed": ("bad", 0.6),
    "irrelevant": ("neutral", 0.2),
    "ignored": ("neutral", 0.1),
}

# Retrieval outcomes that count as a "useful" hit (used by promote.py too).
USEFUL_OUTCOMES = ("improved", "helpful", "used")

_POLARITY_SIGN = {"good": 1.0, "bad": -1.0, "neutral": 0.0}

# Keys in a signal's details JSON that carry free text worth FTS-attributing.
_ATTRIBUTION_TEXT_KEYS = (
    "claim",
    "content",
    "prevention_rule",
    "content_snippet",
    "text",
    "snippet",
    "body",
)


@dataclass(frozen=True)
class Signal:
    """One source-backed good/bad/neutral vote (spec §5.2).

    ``details`` participates in the stable id (via canonical JSON), so two
    genuinely-distinct observations from the same ``(source, source_ref, kind)``
    still get distinct ids, while a re-mine of the same observation collapses.
    """

    kind: str
    polarity: str  # 'good' | 'bad' | 'neutral'
    weight: float
    source: str
    source_ref: str
    session_key: str | None = None
    knowledge_id: str | None = None
    evidence_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    occurred_at: str | None = None


def signal_id(signal: Signal) -> str:
    """Deterministic id: ``sig_<hash(source, source_ref, kind, details)>`` (§1.2)."""
    details_json = canonical_json(signal.details or {})
    return stable_id(
        "sig",
        signal.source,
        signal.source_ref,
        signal.kind,
        content_hash(details_json),
    )


def record_signal(conn: sqlite3.Connection, signal: Signal) -> str:
    """Persist ``signal`` idempotently to ``signal_events``; return its stable id.

    Uses ``INSERT OR IGNORE`` on the stable id so a repeated mine of the same
    observation is a no-op. Shared by the miners here, :mod:`ocbrain.review`, and
    :mod:`ocbrain.judge`.
    """
    sig_id = signal_id(signal)
    created_at = now_iso()
    conn.execute(
        """
        INSERT OR IGNORE INTO signal_events (
          id, kind, polarity, weight, source, source_ref, session_key,
          knowledge_id, evidence_id, details, occurred_at, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sig_id,
            signal.kind,
            signal.polarity,
            float(signal.weight),
            signal.source,
            signal.source_ref,
            signal.session_key,
            signal.knowledge_id,
            signal.evidence_id,
            canonical_json(signal.details or {}),
            signal.occurred_at,
            created_at,
        ),
    )
    return sig_id


# --------------------------------------------------------------------------- #
# Watermarks
# --------------------------------------------------------------------------- #
def get_watermark(conn: sqlite3.Connection, domain: str, stream: str) -> str | None:
    row = conn.execute(
        "SELECT watermark FROM harvest_watermarks WHERE domain = ? AND stream = ?",
        (domain, stream),
    ).fetchone()
    return row["watermark"] if row else None


def set_watermark(
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


def _int_watermark(conn: sqlite3.Connection, stream: str) -> int:
    raw = get_watermark(conn, DOMAIN, stream)
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def _deadline(time_budget_seconds: float | None) -> float | None:
    return None if time_budget_seconds is None else time.monotonic() + time_budget_seconds


def _out_of_time(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


# --------------------------------------------------------------------------- #
# Miners
# --------------------------------------------------------------------------- #
def mine_retrieval_signals(
    conn: sqlite3.Connection,
    *,
    limit: int = 2000,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Mine ``retrieval_feedback`` signals from ``retrieval_uses`` (§5.2).

    Watermarked by rowid; each retrieval use carries its ``knowledge_id`` so
    these signals are attributed at birth.
    """
    stream = "retrieval_uses"
    watermark = _int_watermark(conn, stream)
    deadline = _deadline(time_budget_seconds)
    emitted = 0
    last_rowid = watermark
    rows = conn.execute(
        """
        SELECT rowid AS rid, knowledge_id, outcome, note, served_at
        FROM retrieval_uses
        WHERE rowid > ?
        ORDER BY rowid ASC
        LIMIT ?
        """,
        (watermark, limit),
    ).fetchall()
    for row in rows:
        if _out_of_time(deadline):
            break
        mapping = _RETRIEVAL_MAP.get(row["outcome"])
        if mapping and row["knowledge_id"]:
            polarity, weight = mapping
            record_signal(
                conn,
                Signal(
                    kind="retrieval_feedback",
                    polarity=polarity,
                    weight=weight,
                    source="retrieval",
                    source_ref=f"retrieval_uses:{row['rid']}",
                    knowledge_id=row["knowledge_id"],
                    details={"outcome": row["outcome"]},
                    occurred_at=row["served_at"],
                ),
            )
            emitted += 1
        last_rowid = row["rid"]
    if last_rowid > watermark:
        set_watermark(conn, DOMAIN, stream, str(last_rowid))
    return {"action": "mine_retrieval", "changed": emitted, "watermark": last_rowid}


def mine_event_signals(
    conn: sqlite3.Connection,
    *,
    limit: int = 2000,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Mine ``hard_correction_event`` and ``verifier_result`` signals (§5.2).

    Two independently-watermarked streams: ``brain_events`` (hard corrections)
    and evidence verifier status joined through ``knowledge_evidence``.
    """
    deadline = _deadline(time_budget_seconds)
    emitted = 0

    # Stream 1: hard corrections in brain_events.
    ev_stream = "brain_events"
    ev_watermark = _int_watermark(conn, ev_stream)
    ev_last = ev_watermark
    events = conn.execute(
        """
        SELECT rowid AS rid, id, body_json
        FROM brain_events
        WHERE rowid > ? AND kind = 'correction_recorded'
        ORDER BY rowid ASC
        LIMIT ?
        """,
        (ev_watermark, limit),
    ).fetchall()
    # Advance the watermark past every scanned correction row, even ones that do
    # not yield a signal, so we never re-scan them.
    max_scanned = conn.execute(
        "SELECT MAX(rowid) AS m FROM brain_events WHERE rowid > ? LIMIT 1",
        (ev_watermark,),
    ).fetchone()
    for event in events:
        if _out_of_time(deadline):
            max_scanned = None  # do not skip unscanned rows
            break
        body = _loads(event["body_json"])
        if body.get("hard") and body.get("target_layer") in {"knowledge", "belief"}:
            record_signal(
                conn,
                Signal(
                    kind="hard_correction_event",
                    polarity="bad",
                    weight=1.0,
                    source="events",
                    source_ref=f"brain_events:{event['rid']}",
                    knowledge_id=body.get("target_id"),
                    details={"op": body.get("op"), "hard": True},
                ),
            )
            emitted += 1
        ev_last = event["rid"]
    if max_scanned and max_scanned["m"] is not None:
        ev_last = max(ev_last, int(max_scanned["m"]))
    if ev_last > ev_watermark:
        set_watermark(conn, DOMAIN, ev_stream, str(ev_last))

    # Stream 2: verifier results on evidence linked to knowledge.
    vf_stream = "evidence_verifier"
    vf_watermark = _int_watermark(conn, vf_stream)
    vf_last = vf_watermark
    evidence_rows = conn.execute(
        """
        SELECT e.rowid AS rid, e.id AS evidence_id, e.verifier_status,
               ke.knowledge_id
        FROM evidence e
        JOIN knowledge_evidence ke ON ke.evidence_id = e.id
        WHERE e.rowid > ? AND e.verifier_status IN ('passed', 'failed')
        ORDER BY e.rowid ASC
        LIMIT ?
        """,
        (vf_watermark, limit),
    ).fetchall()
    for row in evidence_rows:
        if _out_of_time(deadline):
            break
        polarity, weight = (
            ("good", 0.5) if row["verifier_status"] == "passed" else ("bad", 0.6)
        )
        record_signal(
            conn,
            Signal(
                kind="verifier_result",
                polarity=polarity,
                weight=weight,
                source="events",
                source_ref=f"evidence:{row['evidence_id']}",
                knowledge_id=row["knowledge_id"],
                evidence_id=row["evidence_id"],
                details={"verifier_status": row["verifier_status"]},
            ),
        )
        emitted += 1
        vf_last = row["rid"]
    if vf_last > vf_watermark:
        set_watermark(conn, DOMAIN, vf_stream, str(vf_last))

    return {"action": "mine_events", "changed": emitted}


def mine_learning_db(
    conn: sqlite3.Connection,
    *,
    learning_db_path: str | None = None,
    limit: int = 2000,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Mine pre-labeled negatives from the ``learning.db`` GATE ledger (§5.2).

    ``learnings`` active GATE/CORRECTION rows become ``learning_gate_rule`` bad
    signals (and each ``prevention_rule`` an ``origin='harvest'`` prescriptive
    knowledge candidate); ``gate_violations`` become ``gate_violation`` bad
    signals with the offending snippet kept in details for later DPO mining.

    The learning DB is a *separate* sqlite file (tests pass a synthetic one).
    Missing file / missing tables are tolerated — the miner is a no-op then.
    """
    if not learning_db_path:
        return {"action": "mine_learning_db", "changed": 0, "skipped": "no_path"}
    from pathlib import Path

    path = Path(learning_db_path).expanduser()
    if not path.exists():
        return {"action": "mine_learning_db", "changed": 0, "skipped": "missing"}

    deadline = _deadline(time_budget_seconds)
    emitted = 0
    ext = sqlite3.connect(path)
    ext.row_factory = sqlite3.Row
    try:
        tables = {
            r["name"]
            for r in ext.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "learnings" in tables:
            emitted += _mine_learnings_table(conn, ext, limit=limit, deadline=deadline)
        if "gate_violations" in tables:
            emitted += _mine_gate_violations_table(
                conn, ext, limit=limit, deadline=deadline
            )
    finally:
        ext.close()
    return {"action": "mine_learning_db", "changed": emitted}


def _mine_learnings_table(
    conn: sqlite3.Connection,
    ext: sqlite3.Connection,
    *,
    limit: int,
    deadline: float | None,
) -> int:
    from ocbrain.db import upsert_knowledge
    from ocbrain.text import claim_key

    cols = {r["name"] for r in ext.execute("PRAGMA table_info(learnings)")}
    stream = "learning.db:learnings"
    watermark = _int_watermark(conn, stream)
    emitted = 0
    last = watermark
    rows = ext.execute(
        f"SELECT rowid AS rid, * FROM learnings WHERE rowid > ? "  # noqa: S608
        f"ORDER BY rowid ASC LIMIT {int(limit)}",
        (watermark,),
    ).fetchall()
    for row in rows:
        if _out_of_time(deadline):
            break
        keys = row.keys()
        rule_type = (_get(row, "rule_type", keys) or _get(row, "type", keys) or "").upper()
        status = (_get(row, "status", keys) or "active").lower()
        if status not in {"active", ""} or rule_type not in {"GATE", "CORRECTION"}:
            last = row["rid"]
            continue
        conf = _as_float(_get(row, "confidence", keys), 0.6)
        recurrence = _as_int(_get(row, "recurrence", keys), 1)
        weight = conf * min(1.0, 0.5 + 0.1 * recurrence)
        prevention = _get(row, "prevention_rule", keys) or ""
        body = prevention or _get(row, "body", keys) or _get(row, "content", keys) or ""
        record_signal(
            conn,
            Signal(
                kind="learning_gate_rule",
                polarity="bad",
                weight=weight,
                source="learning_db",
                source_ref=f"learnings:{row['rid']}",
                details={
                    "rule_type": rule_type,
                    "prevention_rule": prevention,
                    "content": body,
                },
            ),
        )
        emitted += 1
        # Each prevention rule is also a prescriptive knowledge candidate (§5.2).
        if prevention.strip():
            upsert_knowledge(
                conn,
                knowledge_type="value",
                gate="auto",
                subject=f"learning:{rule_type.lower()}",
                predicate=claim_key(prevention, limit=80),
                value_text=prevention.strip(),
                status="candidate",
                prescriptive=True,
                risk="medium",
                confidence=conf,
                privacy_scope="workspace",
                origin="harvest",
                actor="ocbrain-autolabel",
            )
        last = row["rid"]
    del cols
    if last > watermark:
        set_watermark(conn, DOMAIN, stream, str(last))
    return emitted


def _mine_gate_violations_table(
    conn: sqlite3.Connection,
    ext: sqlite3.Connection,
    *,
    limit: int,
    deadline: float | None,
) -> int:
    stream = "learning.db:gate_violations"
    watermark = _int_watermark(conn, stream)
    emitted = 0
    last = watermark
    rows = ext.execute(
        f"SELECT rowid AS rid, * FROM gate_violations WHERE rowid > ? "  # noqa: S608
        f"ORDER BY rowid ASC LIMIT {int(limit)}",
        (watermark,),
    ).fetchall()
    for row in rows:
        if _out_of_time(deadline):
            break
        keys = row.keys()
        snippet = _get(row, "content_snippet", keys) or _get(row, "snippet", keys) or ""
        record_signal(
            conn,
            Signal(
                kind="gate_violation",
                polarity="bad",
                weight=0.7,
                source="learning_db",
                source_ref=f"gate_violations:{row['rid']}",
                details={"content_snippet": snippet},
            ),
        )
        emitted += 1
        last = row["rid"]
    if last > watermark:
        set_watermark(conn, DOMAIN, stream, str(last))
    return emitted


def mine_commitments(
    conn: sqlite3.Connection,
    *,
    commitments_path: str | None = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Mine ``commitment_outcome`` signals from ``commitments.json`` (§5.2).

    Terminal statuses vote: completed/fulfilled good 0.5; expired/missed bad 0.5.
    Stable signal ids make re-reading the file harmless.
    """
    del time_budget_seconds
    entries = _load_json_records(commitments_path, container_keys=("commitments", "items"))
    if entries is None:
        return {"action": "mine_commitments", "changed": 0, "skipped": "missing"}
    good = {"completed", "fulfilled", "done", "kept"}
    bad = {"expired", "missed", "broken", "failed"}
    emitted = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").lower()
        if status in good:
            polarity, weight = "good", 0.5
        elif status in bad:
            polarity, weight = "bad", 0.5
        else:
            continue
        ident = str(entry.get("id") or entry.get("commitment_id") or index)
        record_signal(
            conn,
            Signal(
                kind="commitment_outcome",
                polarity=polarity,
                weight=weight,
                source="commitments",
                source_ref=f"commitment:{ident}",
                details={"status": status},
                occurred_at=entry.get("updated_at") or entry.get("due_at"),
            ),
        )
        emitted += 1
    return {"action": "mine_commitments", "changed": emitted}


def mine_cron_state(
    conn: sqlite3.Connection,
    *,
    cron_state_path: str | None = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Mine ``cron_run`` signals from ``jobs-state.json`` (§5.2).

    ok+delivered good 0.3; error bad 0.5. Watermark is the per-job
    ``updatedAtMs`` folded into the stable id, so unchanged jobs never re-emit.
    """
    del time_budget_seconds
    jobs = _load_cron_jobs(cron_state_path)
    if jobs is None:
        return {"action": "mine_cron_state", "changed": 0, "skipped": "missing"}
    emitted = 0
    for job_id, job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or job.get("lastStatus") or "").lower()
        delivered = bool(job.get("delivered", job.get("lastDelivered", True)))
        updated = str(job.get("updatedAtMs") or job.get("updated_at") or "")
        if status in {"ok", "success", "completed"} and delivered:
            polarity, weight = "good", 0.3
        elif status in {"error", "failed", "failure"}:
            polarity, weight = "bad", 0.5
        else:
            continue
        record_signal(
            conn,
            Signal(
                kind="cron_run",
                polarity=polarity,
                weight=weight,
                source="cron",
                source_ref=f"cron:{job_id}:{updated}",
                details={"status": status, "delivered": delivered},
            ),
        )
        emitted += 1
    return {"action": "mine_cron_state", "changed": emitted}


# --------------------------------------------------------------------------- #
# Attribution
# --------------------------------------------------------------------------- #
def attribute_signals(
    conn: sqlite3.Connection,
    *,
    limit: int = 2000,
    write_batch: DatasetWriteBatch | None = None,
    time_budget_seconds: float | None = None,
) -> set[str]:
    """Attach ``knowledge_id IS NULL`` signals to knowledge via claim_key + FTS.

    Session-only signals (source='session') are intentionally left unattributed —
    they label dataset examples by ``session_key`` (§5.2). Returns the set of
    knowledge ids newly attributed so the caller can force-fold them.
    """
    from ocbrain.text import claim_key

    rows = conn.execute(
        """
        SELECT id, details
        FROM signal_events
        WHERE knowledge_id IS NULL AND source != 'session'
        ORDER BY rowid ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    attributed: set[str] = set()
    deadline = _deadline(time_budget_seconds)
    for row in rows:
        if _out_of_time(deadline):
            break
        details = _loads(row["details"])
        text = _attribution_text(details)
        if not text:
            continue
        hits = search(conn, text, limit=1)
        if not hits:
            continue
        doc_id = hits[0]["doc_id"]
        if not str(doc_id).startswith("know"):
            continue
        # Cheap precision guard: require lexical overlap between the signal text
        # and the matched row's snippet/title (claim_key token intersection).
        matched_text = f"{hits[0]['title']} {hits[0]['snippet']}"
        if not _tokens_overlap(claim_key(text), claim_key(matched_text)):
            continue
        if write_batch is not None:
            write_batch.ensure()
        conn.execute(
            "UPDATE signal_events SET knowledge_id = ? WHERE id = ?",
            (doc_id, row["id"]),
        )
        if write_batch is not None:
            write_batch.operation()
            # The next operation is another potentially expensive FTS query.
            # Release SQLite's writer slot before doing that read.
            write_batch.flush()
        attributed.add(doc_id)
    return attributed


# --------------------------------------------------------------------------- #
# Label fold
# --------------------------------------------------------------------------- #
def label_from_signals(
    signals: list[sqlite3.Row | dict[str, Any]],
    cfg: Any,
    *,
    now: datetime | None = None,
) -> tuple[str, float, float, float]:
    """Pure decayed fold: return ``(label, confidence, S, M)`` (§5.3).

    ``S`` is the decayed signed score, ``M`` the decayed mass. Hard-bad
    precedence: any bad signal with weight >= ``labels.hard_bad_weight`` forces
    ``bad`` outright at that weight's confidence. The LLM judge cannot override
    this (its signals are ordinary weight-0.4 votes).
    """
    now = now or datetime.now(UTC)
    labels = cfg.labels
    score = 0.0
    mass = 0.0
    hard_bad = 0.0
    n = 0
    for signal in signals:
        polarity = _field(signal, "polarity")
        weight = float(_field(signal, "weight") or 0.0)
        occurred = _field(signal, "occurred_at") or _field(signal, "created_at")
        decay = _decay(occurred, now, labels.half_life_days)
        sign = _POLARITY_SIGN.get(polarity, 0.0)
        score += sign * weight * decay
        mass += weight * decay
        n += 1
        if polarity == "bad" and weight >= labels.hard_bad_weight:
            hard_bad = max(hard_bad, weight)

    if hard_bad > 0:
        return "bad", min(0.95, hard_bad), score, mass

    if mass <= 0 or n == 0:
        return "neutral", 0.0, score, mass

    ratio = score / mass
    if mass >= labels.min_mass and ratio >= labels.good_threshold:
        label = "good"
    elif mass >= labels.min_mass and ratio <= labels.bad_threshold:
        label = "bad"
    else:
        label = "neutral"
    confidence = min(0.95, abs(score) / mass * (n / (n + 1)))
    return label, confidence, score, mass


def fold_labels(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    now: datetime | None = None,
    extra_knowledge_ids: set[str] | None = None,
    write_batch: DatasetWriteBatch | None = None,
) -> dict[str, Any]:
    """Recompute ``quality_label`` for knowledge touched by new signals (§5.3).

    Watermarked by ``signal_events`` rowid. ``extra_knowledge_ids`` (e.g. rows
    just attributed, whose signal rowid predates the watermark) are folded too.
    On a flip to ``bad`` for an injected non-human row, inject is dropped
    immediately (the tripwire sweep + promotion re-rank run later in the stage).
    """
    now = now or datetime.now(UTC)
    stream = "signal_events"
    watermark = _int_watermark(conn, stream)
    max_rowid = conn.execute("SELECT MAX(rowid) AS m FROM signal_events").fetchone()["m"]

    affected: set[str] = set(extra_knowledge_ids or set())
    for row in conn.execute(
        """
        SELECT DISTINCT knowledge_id
        FROM signal_events
        WHERE rowid > ? AND knowledge_id IS NOT NULL
        """,
        (watermark,),
    ):
        affected.add(row["knowledge_id"])

    changed = 0
    for knowledge_id in affected:
        krow = conn.execute(
            "SELECT origin, inject FROM knowledge WHERE id = ?", (knowledge_id,)
        ).fetchone()
        if krow is None:
            continue
        signals = conn.execute(
            "SELECT polarity, weight, occurred_at, created_at "
            "FROM signal_events WHERE knowledge_id = ?",
            (knowledge_id,),
        ).fetchall()
        if not signals:
            continue
        label, confidence, _s, _m = label_from_signals(signals, cfg, now=now)
        drop_inject = (
            label == "bad"
            and krow["inject"] == 1
            and krow["origin"] != "human"
        )
        if write_batch is not None:
            write_batch.ensure()
        conn.execute(
            """
            UPDATE knowledge
            SET quality_label = ?,
                quality_confidence = ?,
                quality_updated_at = ?,
                inject = CASE WHEN ? THEN 0 ELSE inject END
            WHERE id = ?
            """,
            (label, confidence, now_iso(), 1 if drop_inject else 0, knowledge_id),
        )
        if write_batch is not None:
            write_batch.operation()
        changed += 1

    if max_rowid is not None and max_rowid > watermark:
        if write_batch is not None:
            write_batch.ensure()
        set_watermark(conn, DOMAIN, stream, str(max_rowid))
        if write_batch is not None:
            write_batch.operation()
    if write_batch is not None:
        write_batch.flush()
    return {"action": "fold_labels", "changed": changed}


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
def autolabel(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    now: datetime | None = None,
    run_judge: bool = True,
    judge_call: Any = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Full autolabel stage: mine -> attribute -> fold -> judge -> fold (§4.1 st.7).

    ``run_judge`` is best-effort: it is inert without an API key and never
    raises into the pipeline. Returns a MaintenanceResult-shaped summary.
    """
    stages: dict[str, Any] = {}
    started = time.monotonic()

    def remaining_budget() -> float | None:
        if time_budget_seconds is None:
            return None
        return max(0.0, time_budget_seconds - (time.monotonic() - started))

    stages["retrieval"] = mine_retrieval_signals(
        conn, time_budget_seconds=remaining_budget()
    )
    conn.commit()
    stages["events"] = mine_event_signals(
        conn, time_budget_seconds=remaining_budget()
    )
    conn.commit()
    stages["learning_db"] = mine_learning_db(
        conn,
        learning_db_path=cfg.dataset.learning_db,
        time_budget_seconds=remaining_budget(),
    )
    conn.commit()
    stages["commitments"] = mine_commitments(
        conn, commitments_path=cfg.dataset.commitments_path
    )
    conn.commit()
    stages["cron"] = mine_cron_state(conn, cron_state_path=cfg.dataset.cron_state_path)
    conn.commit()

    attribute_batch = DatasetWriteBatch(
        conn,
        max_operations=1,
        max_seconds=cfg.dataset.write_batch_seconds,
    )
    attributed = attribute_signals(
        conn,
        write_batch=attribute_batch,
        time_budget_seconds=remaining_budget(),
    )
    stages["attribute"] = {"action": "attribute", "changed": len(attributed)}
    fold_batch = DatasetWriteBatch(
        conn,
        max_operations=cfg.dataset.write_batch_size,
        max_seconds=cfg.dataset.write_batch_seconds,
    )
    stages["fold"] = fold_labels(
        conn,
        cfg,
        now=now,
        extra_knowledge_ids=attributed,
        write_batch=fold_batch,
    )

    if run_judge:
        try:
            from ocbrain.judge import judge_ambiguous

            kwargs: dict[str, Any] = {"now": now}
            if judge_call is not None:
                kwargs["call"] = judge_call
            stages["judge"] = judge_ambiguous(conn, cfg, **kwargs)
            fold2_batch = DatasetWriteBatch(
                conn,
                max_operations=cfg.dataset.write_batch_size,
                max_seconds=cfg.dataset.write_batch_seconds,
            )
            stages["fold2"] = fold_labels(conn, cfg, now=now, write_batch=fold2_batch)
        except Exception as exc:  # noqa: BLE001 - judge must never break the stage
            stages["judge"] = {"action": "judge", "error": str(exc)}

    changed = sum(int(s.get("changed", 0)) for s in stages.values() if isinstance(s, dict))
    batches = [attribute_batch, fold_batch]
    if "fold2_batch" in locals():
        batches.append(fold2_batch)
    metrics = [batch.metrics() for batch in batches]
    writer_lock = {
        "operations": sum(item["operations"] for item in metrics),
        "batches_committed": sum(item["batches_committed"] for item in metrics),
        "lock_wait_seconds": round(sum(item["lock_wait_seconds"] for item in metrics), 6),
        "max_lock_wait_seconds": max(item["max_lock_wait_seconds"] for item in metrics),
        "writer_lock_seconds": round(sum(item["writer_lock_seconds"] for item in metrics), 6),
        "max_writer_lock_seconds": max(item["max_writer_lock_seconds"] for item in metrics),
    }
    return {
        "action": "autolabel",
        "changed": changed,
        "stages": stages,
        "writer_lock": writer_lock,
    }


# --------------------------------------------------------------------------- #
# Shared helpers (also imported by judge.py / promote.py)
# --------------------------------------------------------------------------- #
def signals_for(conn: sqlite3.Connection, knowledge_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT polarity, weight, occurred_at, created_at, kind "
        "FROM signal_events WHERE knowledge_id = ?",
        (knowledge_id,),
    ).fetchall()


def decayed_mass(
    signals: list[sqlite3.Row | dict[str, Any]],
    cfg: Any,
    *,
    now: datetime | None = None,
) -> float:
    now = now or datetime.now(UTC)
    half_life = cfg.labels.half_life_days
    total = 0.0
    for signal in signals:
        weight = float(_field(signal, "weight") or 0.0)
        occurred = _field(signal, "occurred_at") or _field(signal, "created_at")
        total += weight * _decay(occurred, now, half_life)
    return total


def _decay(occurred: str | None, now: datetime, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    ts = _parse_ts(occurred)
    if ts is None:
        return 1.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return 0.5 ** (age_days / half_life_days)


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


def _attribution_text(details: dict[str, Any]) -> str:
    for key in _ATTRIBUTION_TEXT_KEYS:
        value = details.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _tokens_overlap(a: str, b: str) -> bool:
    ta = {t for t in a.split() if len(t) >= 3}
    tb = {t for t in b.split() if len(t) >= 3}
    return bool(ta & tb)


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    import json

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _load_json_records(
    path: str | None, *, container_keys: tuple[str, ...]
) -> list[Any] | None:
    if not path:
        return None
    from pathlib import Path

    file = Path(path).expanduser()
    if not file.exists():
        return None
    import json

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in container_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return list(data.values())
    return None


def _load_cron_jobs(path: str | None) -> list[tuple[str, Any]] | None:
    if not path:
        return None
    from pathlib import Path

    file = Path(path).expanduser()
    if not file.exists():
        return None
    import json

    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        jobs = data.get("jobs", data)
        if isinstance(jobs, dict):
            return list(jobs.items())
        if isinstance(jobs, list):
            return [(str(j.get("id", i)), j) for i, j in enumerate(jobs)]
    if isinstance(data, list):
        return [(str(j.get("id", i)), j) for i, j in enumerate(data)]
    return None


def _field(row: sqlite3.Row | dict[str, Any], key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _get(row: sqlite3.Row, key: str, keys: Any) -> Any:
    return row[key] if key in keys else None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
