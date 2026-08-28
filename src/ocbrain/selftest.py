"""Standing health scorecard for one OCBrain core.

Every health question this brain has ever been asked -- is retrieval answering,
did the alias table hold, is the correction pathway actually being used, is the
fleet still harvesting -- has been answered by a throwaway script written fresh
in a scratch directory and then lost. This module is that script, made a command
and given thresholds, so the answer is a number with a verdict attached rather
than a paragraph of recollection.

Three properties are non-negotiable, because a selftest without them is worse
than none:

**Read-only.** The core is opened ``mode=ro`` and refused if that is impossible.
A measurement that can corrupt what it measures is not a measurement. The
optional vector sidecar is opened the same way.

**Honest degradation.** A missing sidecar, a provenance column an older core
never grew, a runtime with no rows: each produces ``not_measured`` and a reason.
Never a silent zero. This is the same posture the retrieval path takes when it
abstains rather than pad a packet, and for the same reason -- a zero that means
"nothing here" and a zero that means "could not look" are different facts, and
collapsing them is how a dashboard starts lying.

**Thresholds with provenance.** :data:`THRESHOLDS` is the single table, every
entry naming where its number came from. Several are measurements taken on this
machine and are cited as the dated baselines they are; several are judgement
calls and say so. ``docs/THRESHOLDS.md`` carries the long form. A threshold
nobody can trace is a threshold people learn to mute.

The command exits non-zero when any metric is ``alarm``, so it can be a cron
gate. ``watch`` never fails the gate -- it is the band that exists so the gate
does not have to fire before a human has a chance to look.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCORECARD_SCHEMA = "ocbrain.selftest.v1"

# Status vocabulary. ``not_measured`` is deliberately not a failure: it says the
# question could not be asked here, which is a fact about the install and not a
# fault in the corpus.
OK = "ok"
WATCH = "watch"
ALARM = "alarm"
NOT_MEASURED = "not_measured"

HIGHER_BETTER = "higher_better"
LOWER_BETTER = "lower_better"
INFO = "info"

# Removal within this many days of mint is what "pollution" means: a belief that
# did not survive long enough to have been worth minting. Fourteen days is the
# TARL convention, adopted deliberately rather than derived here.
POLLUTION_HORIZON_DAYS = 14

# Calibration asks whether a confidence band predicts survival. A belief needs a
# full horizon behind it before it can be scored, so the cohort is everything
# minted at least this long ago.
CALIBRATION_HORIZON_DAYS = 30

# Corpus-wide near-duplicate threshold. Deliberately NOT the same number as
# ``mcp_v1.ADVISORY_COSINE_THRESHOLD`` (0.90), which gates what one served packet
# warns about. This is the lower, more sensitive bar for a standing corpus-health
# sweep: a pair at 0.88 is not worth interrupting a read for, but is worth
# knowing about when counting how much of the corpus says the same thing twice.
DUPLICATE_COSINE_THRESHOLD = 0.88

# A harvest gap longer than this on a runtime that is otherwise live is the
# alarm the Hermes fleet's fourteen dark days should have tripped.
HARVEST_SILENCE_ALARM_HOURS = 48.0

# What counts as a *live* harvest stream, and therefore as something whose
# silence is news. This corpus carries ninety-odd runtime spellings, most of them
# one-shot lane labels ("cursor-opus5-lane-r2", one row, last July). Alarming per
# label would make this row permanently red and therefore permanently ignored.
# A stream is live if it wrote at least ``HARVEST_MIN_ROWS`` rows within
# ``HARVEST_LIVE_DAYS``; the alarm band is then the gap between the silence
# threshold and the liveness window -- "was harvesting this week, has not in two
# days" -- which is exactly the shape of a fleet going dark.
HARVEST_LIVE_DAYS = 7
HARVEST_MIN_ROWS = 3

# Provenance and trace-join rates are meaningless over rows written before the
# server could stamp them. Below this many post-capture rows the sample is too
# thin to carry a verdict, and saying so beats reporting a percentage of eleven.
MIN_PROVENANCE_SAMPLE = 20

# Writers whose corrections are machine-issued: the hygiene sweep, the wiki
# curator under operator approval, the seal-truth compiler. Metric C8 is about
# what *agents* do with the correction pathway, and folding these in would drown
# the signal in 700-odd maintenance retractions.
MACHINE_WRITER_PREFIXES = ("maintenance:", "operator-approved:", "deterministic:")

_UUID = re.compile(r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")

_REMOVAL_OPS = ("retract", "mark_wrong", "supersede")


class SelftestError(RuntimeError):
    """The scorecard could not be produced at all."""


@dataclass(frozen=True)
class Threshold:
    """One documented verdict boundary.

    ``source`` is required and is the whole point: it names the measurement,
    convention, or admitted guess the number came from, so a reader deciding
    whether to trust an ``alarm`` can trace it without archaeology.
    """

    direction: str
    ok: float | None
    watch: float | None
    source: str

    def classify(self, value: float | None) -> str:
        if value is None:
            return NOT_MEASURED
        if self.direction == INFO:
            return OK
        if self.direction == HIGHER_BETTER:
            if self.ok is not None and value >= self.ok:
                return OK
            if self.watch is not None and value >= self.watch:
                return WATCH
            return ALARM
        if self.ok is not None and value <= self.ok:
            return OK
        if self.watch is not None and value <= self.watch:
            return WATCH
        return ALARM

    def describe(self) -> str:
        if self.direction == INFO:
            return "informational"
        arrow = ">=" if self.direction == HIGHER_BETTER else "<="
        return f"ok {arrow} {_fmt_number(self.ok)}, watch {arrow} {_fmt_number(self.watch)}"


# --------------------------------------------------------------------------- #
# The threshold table.
#
# One place, every number sourced. Measurements taken on this install are dated,
# because a measurement without a date becomes folklore. Guesses are labelled
# "judgement" and are exactly that -- a starting line to calibrate away from, not
# an authority. docs/THRESHOLDS.md is the long form of this table.
# --------------------------------------------------------------------------- #
THRESHOLDS: dict[str, Threshold] = {
    "answer_rate": Threshold(
        HIGHER_BETTER,
        0.40,
        0.25,
        "Measured 45.8% all-time (917/2001 retrievals) and 47.4% over the 30 days "
        "to 2026-08-25 on the live core. Bands set just under the measured floor "
        "so ordinary variation does not fire. Judgement, anchored on measurement.",
    ),
    "scope_reachability": Threshold(
        HIGHER_BETTER,
        0.85,
        0.70,
        "Regression guard on the 2026-08-24 alias-table repair, recorded at "
        "25.1% before and 95.2% after. Reproduced here at 22.5% -> 87.2% "
        "all-time and 91.8% over 30 days under this module's definition (see "
        "docs/THRESHOLDS.md for why the two differ). Alarm band sits far below "
        "the repaired floor and far above the broken one, so only a real "
        "regression crosses it.",
    ),
    "zero_result_rate": Threshold(
        LOWER_BETTER,
        0.60,
        0.75,
        "Complement of answer_rate; same measurement, stated so the census has a "
        "verdict of its own.",
    ),
    "pollution_rate": Threshold(
        LOWER_BETTER,
        0.15,
        0.30,
        "TARL convention (removal within 14 days of mint). The bands are "
        "judgement: no pre-v2.2 measurement of this exists, because supersession "
        "did not exist to distinguish replacement from deletion.",
    ),
    "structured_removal_share": Threshold(
        HIGHER_BETTER,
        0.80,
        0.50,
        "Should trend to 1.0 now that supersession exists. Judgement, set to the "
        "same bar as correction_adoption because they measure the same shift "
        "from opposite ends.",
    ),
    "conflict_preservation": Threshold(
        HIGHER_BETTER,
        1.0,
        0.99,
        "Target is exactly 100%: nothing the correction pathway touches may "
        "become unreachable. Any loss at all is an alarm; the watch band exists "
        "only so a single in-flight row does not page before a human looks.",
    ),
    "calibration_gap": Threshold(
        LOWER_BETTER,
        0.20,
        0.35,
        "Largest |mean stated confidence - observed 30-day survival| across bands. "
        "Judgement: no prior calibration measurement exists for this corpus. This "
        "measured 0.668 on the live core at 2026-08-25 and the cause is known -- "
        "auto-compiled receipt and transcript beliefs are minted at moderate "
        "confidence and swept by the hygiene pass at a median of 1.5 days, so the "
        "moderate band's survival is 0.0 across 658 beliefs. The fix is upstream "
        "(mint receipts at low confidence, or do not mint them as beliefs), not a "
        "wider threshold. See the per-band removed_by breakdown.",
    ),
    "duplicate_key_clusters": Threshold(
        LOWER_BETTER,
        0.0,
        3.0,
        "attributes.key is a wiki fact's identity and is meant to be unique "
        "across serving beliefs, so the correct count is zero by construction. "
        "The watch band tolerates a curator cascade caught mid-flight.",
    ),
    "correction_adoption": Threshold(
        HIGHER_BETTER,
        0.80,
        0.50,
        "Brief-specified target (>80%). Historical baseline is 0/11: every "
        "agent-issued correction in this core's life was a bare retract with the "
        "replacement buried in the correction body. Verified on the live core at "
        "2026-08-25 -- 11 agent corrections, all op=retract, zero supersedes.",
    ),
    "lossy_supersession_share": Threshold(
        LOWER_BETTER,
        0.15,
        0.40,
        "Share of machine-authored supersessions whose successor drops a "
        "checkable token (an issue ref, backticked literal, path, flag, "
        "identifier, or figure) the predecessor carried. The 2026-08-26 backlog "
        "triage found 15 of 28 curator-proposed supersessions would have "
        "silently destroyed checkable facts; the landed population measures "
        "46/82 (56%) with this extractor, 2026-08-26 -- an honest alarm at "
        "ship. A refresh that legitimately updates a count moves this too, "
        "which is why the ok band is not zero. Judgement.",
    ),
    "pending_supersede_age_hours": Threshold(
        LOWER_BETTER,
        72.0,
        168.0,
        "A pending supersession leaves the stale belief serving until an admin "
        "decides, so age is the cost. Three days to notice, a week to alarm. "
        "Judgement.",
    ),
    "contradictions_nonempty_rate": Threshold(
        HIGHER_BETTER,
        0.01,
        0.0,
        "Provably 0.0 for this core's entire life before v2.2: the declared pass "
        "had no writer, so every packet ever served carried an empty "
        "contradictions list. Any non-zero value is the proof the writer exists; "
        "the bar is deliberately just above zero because most packets legitimately "
        "carry no conflict, and a high rate would itself be bad news.",
    ),
    "provenance_coverage": Threshold(
        HIGHER_BETTER,
        0.90,
        0.50,
        "Measured over rows written since provenance capture began on this core, "
        "not over the whole window: server-captured provenance landed with PR #33 "
        "on 2026-08-24 and no earlier row could ever have carried it, so including "
        "them would measure the migration date rather than whether clients are "
        "stamping. Judgement: once every client has reconnected, near-total "
        "coverage is the expectation.",
    ),
    "closeout_trace_join_rate": Threshold(
        HIGHER_BETTER,
        0.50,
        0.20,
        "Historically 3.7% (13 closeouts identity-joining to a transcript, "
        "recorded in ocbrain.provenance's module docstring). The brief is explicit "
        "that this should climb for new rows only, so the gated figure is measured "
        "over closeouts written since provenance capture began; the window and "
        "all-time figures are reported beside it and are not gated. Judgement.",
    ),
    "harvest_silence_hours": Threshold(
        LOWER_BETTER,
        HARVEST_SILENCE_ALARM_HOURS,
        HARVEST_SILENCE_ALARM_HOURS,
        "The Hermes fleet once went dark for 14 days and nothing noticed. 48h is "
        "the brief's number and is a judgement. Applied only to live streams "
        f"(>= {HARVEST_MIN_ROWS} rows in the last {HARVEST_LIVE_DAYS} days), "
        "because per-label freshness across ninety historical runtime spellings "
        "is noise (see docs/THRESHOLDS.md).",
    ),
    "near_duplicate_clusters": Threshold(
        INFO,
        None,
        None,
        "Reported, never gated. No calibrated bar exists for how much semantic "
        "overlap a healthy corpus carries, and inventing one would be authority "
        "this measurement has not earned. Track it against a saved --baseline.",
    ),
    "pending_supersede_depth": Threshold(
        INFO,
        None,
        None,
        "Reported, never gated: depth alone is not a fault, and the cost of a "
        "pending supersession is carried by pending_supersede_age_hours, which is. "
        "The value counts distinct targets and the display carries the raw "
        "proposal count beside it, because raw depth alone read as ordinary "
        "backlog while a proposal loop grew it without bound.",
    ),
    "db_size_mb": Threshold(
        INFO,
        None,
        None,
        "Reported, never gated. A core has no correct size; the signal is the "
        "trend against a saved --baseline.",
    ),
    "rows_added_in_window": Threshold(
        INFO,
        None,
        None,
        "Reported, never gated. PR #33 cut projected growth from ~106 MB/month to "
        "~14 MB/month by storing transcript evidence as file pointers; rows added "
        "per table is the honest single-snapshot growth signal, because bytes "
        "cannot be attributed to a table without a full page scan.",
    ),
    "integrity": Threshold(
        HIGHER_BETTER,
        1.0,
        1.0,
        "SQLite quick_check plus foreign_key_check. Binary: either the core is "
        "structurally sound or it is not.",
    ),
    "vector_sidecar_lag_events": Threshold(
        LOWER_BETTER,
        500.0,
        5000.0,
        "Events appended since the sidecar was built. The sidecar only indexes "
        "serving beliefs, and most events are evidence rather than compilations, "
        "so a lag of a few hundred is normal drift rather than staleness. "
        "Judgement.",
    ),
    "briefing_determinism": Threshold(
        HIGHER_BETTER,
        1.0,
        1.0,
        "Binary by construction. `brain.briefing` promises byte-identical output "
        "for the same scope and corpus state, and a harness that reorients "
        "through it every iteration inherits that promise. Anything below 1.0 "
        "means the promise is already broken, so there is no watch band to sit "
        "in.",
    ),
    "briefing_budget_compliance": Threshold(
        HIGHER_BETTER,
        1.0,
        1.0,
        "Also binary. The budget is a hard ceiling the renderer enforces before "
        "it spends anything on items; a briefing over budget means the skeleton "
        "reservation is wrong, not that a scope got busy.",
    ),
    "goal_pointer_resolution": Threshold(
        HIGHER_BETTER,
        1.0,
        0.80,
        "Share of open goals whose source_pointer still resolves on this "
        "machine. A goal is a pointer to a spec in the repo; when the spec moves "
        "and the goal does not, the goal is pointing at nothing. Watch rather "
        "than alarm at 80% because a pointer written on another machine can "
        "legitimately not resolve on this one. Judgement.",
    ),
    "goal_open_age_days": Threshold(
        LOWER_BETTER,
        14.0,
        45.0,
        "Age of the oldest open goal. Goal drift is a distinct failure that "
        "pass/fail benchmarks cannot see (arXiv 2608.06663, 1,547 papers), and "
        "an objective nobody has closed or abandoned in six weeks is the "
        "observable form of it. Judgement: no measurement exists yet because "
        "goals ship with this release.",
    ),
}


@dataclass
class Metric:
    """One measured question, its verdict, and how to read it."""

    key: str
    section: str
    label: str
    value: float | None = None
    display: str | None = None
    status: str = OK
    reason: str | None = None
    basis: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        threshold = THRESHOLDS.get(self.key)
        payload: dict[str, Any] = {
            "key": self.key,
            "section": self.section,
            "label": self.label,
            "value": self.value,
            "status": self.status,
        }
        if self.display is not None:
            payload["display"] = self.display
        if self.reason:
            payload["reason"] = self.reason
        if self.basis:
            payload["basis"] = self.basis
        if threshold is not None:
            payload["threshold"] = {
                "direction": threshold.direction,
                "ok": threshold.ok,
                "watch": threshold.watch,
                "source": threshold.source,
            }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def _measured(
    key: str,
    section: str,
    label: str,
    value: float | None,
    *,
    display: str | None = None,
    basis: str | None = None,
    **detail: Any,
) -> Metric:
    """Build a metric and let its threshold decide the verdict."""
    threshold = THRESHOLDS.get(key)
    status = threshold.classify(value) if threshold else (OK if value is not None else NOT_MEASURED)
    return Metric(
        key=key,
        section=section,
        label=label,
        value=value,
        display=display,
        status=status,
        basis=basis,
        detail={k: v for k, v in detail.items() if v is not None},
    )


def _unmeasured(key: str, section: str, label: str, reason: str, **detail: Any) -> Metric:
    """A question this install cannot answer, and why. Never a silent zero."""
    return Metric(
        key=key,
        section=section,
        label=label,
        value=None,
        status=NOT_MEASURED,
        reason=reason,
        detail={k: v for k, v in detail.items() if v is not None},
    )


# --------------------------------------------------------------------------- #
# Connection handling
# --------------------------------------------------------------------------- #


def open_readonly(path: Path) -> sqlite3.Connection:
    """Open a core strictly read-only, or refuse.

    ``mode=ro`` is the guarantee; ``query_only`` is the belt to its braces and
    also covers the case of a URI that some future caller builds differently.
    A selftest that can write to the thing it measures is worthless, so this
    raises rather than falling back to a writable handle.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise SelftestError(f"core database does not exist: {resolved}")
    try:
        conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as error:  # pragma: no cover - platform/permission dependent
        raise SelftestError(f"cannot open core read-only: {resolved}: {error}") from error
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only = 1")
    except sqlite3.Error as error:  # pragma: no cover - defensive
        conn.close()
        raise SelftestError(f"cannot enforce read-only on {resolved}: {error}") from error
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _scalar(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Any:
    row = conn.execute(sql, tuple(params)).fetchone()
    return None if row is None else row[0]


# --------------------------------------------------------------------------- #
# Section A -- retrieval health
# --------------------------------------------------------------------------- #


def _serving_scopes(conn: sqlite3.Connection) -> set[str]:
    from ocbrain.scope import resolve_scope_alias

    return {
        resolve_scope_alias(str(row[0]))
        for row in conn.execute(
            "SELECT DISTINCT scope_id FROM current_beliefs WHERE serve=1 AND status='current'"
        )
    }


def _retrieval_rows(conn: sqlite3.Connection, cutoff: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT id, context_json, served_ids_json, served_at, client_runtime_key, "
            "server_connection_id, session_id, client_session_hint "
            "FROM retrieval_uses WHERE served_at >= ? ORDER BY served_at",
            (cutoff,),
        )
    )


def _scope_reach(conn: sqlite3.Connection, rows: Iterable[sqlite3.Row]) -> dict[str, Any]:
    """Classify each retrieval by whether its passed scope can reach the corpus.

    The caller's compatible set always contains ``global:doctrine``, which is
    occupied, so counting it would make reachability trivially 100% and measure
    nothing. Only the scopes the caller actually *named* -- project, repo,
    client, task, session -- are considered, folded and alias-resolved exactly
    as the serving path folds them. A caller that named no scope at all is a
    third category, not a failure.
    """
    from ocbrain.scope import ScopeContext, resolve_scope_alias

    served = _serving_scopes(conn)
    reachable: list[sqlite3.Row] = []
    unreachable: list[sqlite3.Row] = []
    unscoped: list[sqlite3.Row] = []
    for row in rows:
        try:
            context = json.loads(row["context_json"] or "{}")
        except (TypeError, ValueError):
            context = {}
        scope_context = ScopeContext.from_dict(context if isinstance(context, dict) else {})
        named = {
            value
            for value in scope_context.compatible_scope_ids()
            if not value.startswith("global:")
        }
        if not named:
            unscoped.append(row)
        elif any(resolve_scope_alias(value) in served for value in named):
            reachable.append(row)
        else:
            unreachable.append(row)
    return {
        "reachable": reachable,
        "unreachable": unreachable,
        "unscoped": unscoped,
        "serving_scopes": sorted(served),
    }


def _answered(row: sqlite3.Row) -> bool:
    """Whether one recorded retrieval served at least one item.

    ``served_ids_json`` is the receipt written in the same statement as the row
    itself; ``retrieval_items`` is the normalized copy. They agree on all 2001
    rows of the live core, and this reads the receipt because it survives a
    core whose item rows were never backfilled.
    """
    raw = row["served_ids_json"]
    if not raw:
        return False
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return bool(parsed)


def _passed_project(row: sqlite3.Row) -> str:
    try:
        context = json.loads(row["context_json"] or "{}")
    except (TypeError, ValueError):
        return "(unparseable)"
    if not isinstance(context, dict):
        return "(unparseable)"
    value = context.get("project")
    text = str(value).strip() if value is not None else ""
    return text or "(none passed)"


def _section_a(conn: sqlite3.Connection, cutoff: str) -> list[Metric]:
    if not _table_exists(conn, "retrieval_uses"):
        reason = "this core has no retrieval_uses table"
        return [
            _unmeasured("answer_rate", "A", "Answer rate", reason),
            _unmeasured("scope_reachability", "A", "Scope reachability", reason),
            _unmeasured("zero_result_rate", "A", "Zero-result rate", reason),
        ]
    rows = _retrieval_rows(conn, cutoff)
    if not rows:
        reason = "no retrievals recorded in the window"
        return [
            _unmeasured("answer_rate", "A", "Answer rate", reason, retrievals=0),
            _unmeasured("scope_reachability", "A", "Scope reachability", reason, retrievals=0),
            _unmeasured("zero_result_rate", "A", "Zero-result rate", reason, retrievals=0),
        ]

    reach = _scope_reach(conn, rows)
    answered = [row for row in rows if _answered(row)]
    answer_rate = len(answered) / len(rows)

    def split(bucket: str) -> dict[str, Any] | None:
        subset = reach[bucket]
        if not subset:
            return None
        hit = sum(1 for row in subset if _answered(row))
        return {"retrievals": len(subset), "answered": hit, "rate": round(hit / len(subset), 4)}

    zero_rows = [row for row in rows if not _answered(row)]
    census: dict[str, int] = {}
    for row in zero_rows:
        project = _passed_project(row)
        census[project] = census.get(project, 0) + 1
    top = sorted(census.items(), key=lambda pair: (-pair[1], pair[0]))[:8]

    reach_denominator = len(reach["reachable"]) + len(reach["unreachable"])
    if reach_denominator:
        reachability = len(reach["reachable"]) / reach_denominator
        reach_metric = _measured(
            "scope_reachability",
            "A",
            "Scope reachability",
            reachability,
            display=f"{reachability:.1%}",
            basis="passed scope folded and alias-resolved against occupied serving scopes",
            scoped_retrievals=reach_denominator,
            reachable=len(reach["reachable"]),
            unreachable=len(reach["unreachable"]),
            no_scope_passed=len(reach["unscoped"]),
            serving_scopes=reach["serving_scopes"],
        )
    else:
        reach_metric = _unmeasured(
            "scope_reachability",
            "A",
            "Scope reachability",
            "every retrieval in the window passed no scope at all",
            no_scope_passed=len(reach["unscoped"]),
        )

    zero_rate = len(zero_rows) / len(rows)
    return [
        _measured(
            "answer_rate",
            "A",
            "Answer rate",
            answer_rate,
            display=f"{answer_rate:.1%}",
            basis="retrievals serving >=1 item / retrievals in window",
            retrievals=len(rows),
            answered=len(answered),
            reachable_scope=split("reachable"),
            unreachable_scope=split("unreachable"),
            no_scope_passed=split("unscoped"),
        ),
        reach_metric,
        _measured(
            "zero_result_rate",
            "A",
            "Zero-result rate",
            zero_rate,
            display=f"{zero_rate:.1%} ({len(zero_rows)})",
            basis="retrievals serving no items / retrievals in window",
            zero_result_count=len(zero_rows),
            top_passed_projects=[{"project": name, "count": n} for name, n in top],
        ),
    ]


# --------------------------------------------------------------------------- #
# Section B -- corpus quality
# --------------------------------------------------------------------------- #


def _mint_times(conn: sqlite3.Connection) -> dict[str, str]:
    """Belief id -> the timestamp of the decision that approved it."""
    return {
        str(row["belief_id"]): str(row["ts"])
        for row in conn.execute(
            "SELECT b.belief_id AS belief_id, e.ts AS ts "
            "FROM current_beliefs b JOIN brain_events e ON e.id = b.approved_event_id"
        )
        if row["ts"]
    }


def _removals(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Belief id -> its earliest removal, and whether that removal was structured.

    A structured removal is a supersession: it carries a forward pointer, so a
    reader holding the old id is walked to the replacement instead of refused.
    A bare retract or a tombstone destroys the trail, which is exactly the shape
    v2.2 exists to replace.
    """
    removals: dict[str, dict[str, Any]] = {}

    def note(target: str, ts: str, op: str, structured: bool) -> None:
        target = str(target or "").strip()
        if not target:
            return
        existing = removals.get(target)
        if existing is None or ts < existing["ts"]:
            removals[target] = {"ts": ts, "op": op, "structured": structured}

    placeholders = ",".join("?" for _ in _REMOVAL_OPS)
    for row in conn.execute(
        "SELECT ts, json_extract(body_json, '$.target_id') AS target, "
        "json_extract(body_json, '$.op') AS op FROM brain_events "
        f"WHERE kind='correction_recorded' AND json_extract(body_json, '$.op') IN ({placeholders})",
        _REMOVAL_OPS,
    ):
        op = str(row["op"] or "")
        note(row["target"], str(row["ts"]), op, op == "supersede")
    for row in conn.execute(
        "SELECT ts, json_extract(body_json, '$.target') AS target FROM brain_events "
        "WHERE kind='tombstone_recorded'"
    ):
        note(row["target"], str(row["ts"]), "tombstone", False)
    return removals


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _pollution(conn: sqlite3.Connection, cutoff: str) -> list[Metric]:
    mints = _mint_times(conn)
    removals = _removals(conn)
    cohort = [(bid, ts) for bid, ts in mints.items() if ts >= cutoff]
    if not cohort:
        reason = "no beliefs were approved in the window"
        return [
            _unmeasured("pollution_rate", "B", "Memory pollution rate", reason, minted=0),
            _unmeasured(
                "structured_removal_share", "B", "Structured removal share", reason, removed=0
            ),
        ]
    horizon = timedelta(days=POLLUTION_HORIZON_DAYS)
    polluted: list[dict[str, Any]] = []
    for belief_id, mint_ts in cohort:
        removal = removals.get(belief_id)
        if removal is None:
            continue
        minted_at = _parse_ts(mint_ts)
        removed_at = _parse_ts(removal["ts"])
        if minted_at is None or removed_at is None or removed_at < minted_at:
            continue
        if removed_at - minted_at <= horizon:
            polluted.append({"belief_id": belief_id, **removal})
    rate = len(polluted) / len(cohort)
    metrics = [
        _measured(
            "pollution_rate",
            "B",
            "Memory pollution rate",
            rate,
            display=f"{rate:.1%} ({len(polluted)}/{len(cohort)})",
            basis=f"approved in window and removed within {POLLUTION_HORIZON_DAYS}d of mint",
            minted_in_window=len(cohort),
            removed_within_horizon=len(polluted),
            horizon_days=POLLUTION_HORIZON_DAYS,
        )
    ]
    if not polluted:
        metrics.append(
            _unmeasured(
                "structured_removal_share",
                "B",
                "Structured removal share",
                "no beliefs minted in the window were removed within the horizon",
                minted_in_window=len(cohort),
            )
        )
        return metrics
    structured = sum(1 for item in polluted if item["structured"])
    share = structured / len(polluted)
    metrics.append(
        _measured(
            "structured_removal_share",
            "B",
            "Structured removal share",
            share,
            display=f"{share:.1%} ({structured}/{len(polluted)})",
            basis="of horizon removals, the share carrying a superseded_by pointer",
            structured=structured,
            removals=len(polluted),
        )
    )
    return metrics


def _conflict_preservation(conn: sqlite3.Connection, cutoff: str) -> Metric:
    """Every detected conflict must leave its losing side reachable.

    Two pathways produce a conflict. A supersession retires one belief and points
    at its replacement; the loser stays readable through ``mode=as_stored`` only
    if its row survives *and* carries the validity window that says which era it
    belonged to. An ``annotate``-written ``contradicts`` pair asserts two beliefs
    disagree without retiring either, so both sides must simply still be there.

    Target is 100%. Anything the correction pathway touches and then loses is a
    fact the brain can no longer show its working for.
    """
    checked: list[dict[str, Any]] = []

    for row in conn.execute(
        "SELECT ts, json_extract(body_json, '$.target_id') AS target, "
        "json_extract(body_json, '$.successor_id') AS successor FROM brain_events "
        "WHERE kind='correction_recorded' AND json_extract(body_json, '$.op')='supersede' "
        "AND ts >= ?",
        (cutoff,),
    ):
        loser = str(row["target"] or "").strip()
        if not loser:
            continue
        stored = conn.execute(
            "SELECT status, attributes_json FROM current_beliefs WHERE belief_id=?", (loser,)
        ).fetchone()
        if stored is None:
            checked.append({"kind": "supersede", "belief_id": loser, "preserved": False,
                            "why": "row absent"})
            continue
        attributes = _json_object(stored["attributes_json"])
        has_window = bool(str(attributes.get("valid_until") or "").strip())
        has_pointer = bool(str(attributes.get("superseded_by") or "").strip())
        checked.append(
            {
                "kind": "supersede",
                "belief_id": loser,
                "preserved": has_window and has_pointer,
                "why": None if (has_window and has_pointer) else "missing validity window/pointer",
            }
        )

    for row in conn.execute(
        "SELECT belief_id, attributes_json FROM current_beliefs "
        "WHERE json_extract(attributes_json, '$.contradicts') IS NOT NULL"
    ):
        attributes = _json_object(row["attributes_json"])
        others = attributes.get("contradicts")
        if not isinstance(others, list):
            continue
        for other in others:
            other_id = str(other or "").strip()
            if not other_id:
                continue
            present = _scalar(
                conn, "SELECT 1 FROM current_beliefs WHERE belief_id=? LIMIT 1", (other_id,)
            )
            checked.append(
                {
                    "kind": "contradicts",
                    "belief_id": other_id,
                    "preserved": present is not None,
                    "why": None if present is not None else "row absent",
                }
            )

    if not checked:
        return _unmeasured(
            "conflict_preservation",
            "B",
            "Conflict preservation",
            "no supersessions in the window and no contradicts annotations in the corpus",
            conflicts=0,
        )
    preserved = sum(1 for item in checked if item["preserved"])
    rate = preserved / len(checked)
    lost = [item for item in checked if not item["preserved"]][:8]
    return _measured(
        "conflict_preservation",
        "B",
        "Conflict preservation",
        rate,
        display=f"{rate:.1%} ({preserved}/{len(checked)})",
        basis="losing side still stored with a validity window (supersede) or still present "
        "(contradicts)",
        conflicts=len(checked),
        preserved=preserved,
        unreachable_sample=lost or None,
    )


def _json_object(raw: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _calibration(conn: sqlite3.Connection, now: datetime) -> Metric:
    """Does a confidence band predict survival?

    The cohort is every belief minted at least one horizon ago, so each one has
    had a full 30 days in which to be removed. The comparison is against the mean
    *stated* confidence in the band rather than an invented band midpoint: the
    stored number is what the brain actually claimed, and inventing a midpoint
    would be scoring the brain against a number nobody wrote down.
    """
    mints = _mint_times(conn)
    removals = _removals(conn)
    horizon = timedelta(days=CALIBRATION_HORIZON_DAYS)
    boundary = now - horizon
    bands: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT belief_id, confidence, confidence_band FROM current_beliefs"
    ):
        belief_id = str(row["belief_id"])
        minted_at = _parse_ts(mints.get(belief_id))
        if minted_at is None or minted_at > boundary:
            continue
        band = str(row["confidence_band"] or "unknown")
        entry = bands.setdefault(
            band, {"n": 0, "survived": 0, "confidence_sum": 0.0, "removed_by": {}}
        )
        entry["n"] += 1
        entry["confidence_sum"] += float(row["confidence"] or 0.0)
        removal = removals.get(belief_id)
        removed_at = _parse_ts(removal["ts"]) if removal else None
        if removed_at is None or removed_at - minted_at > horizon:
            entry["survived"] += 1
        else:
            # Which pathway retired it. Without this the band's survival number
            # says "these beliefs did not last" and leaves the reader to guess
            # whether that means refuted, replaced, or swept as low-value -- three
            # very different findings that a single rate cannot distinguish.
            op = str(removal["op"])
            entry["removed_by"][op] = entry["removed_by"].get(op, 0) + 1
    if not bands:
        return _unmeasured(
            "calibration_gap",
            "B",
            "Calibration gap",
            f"no beliefs are older than the {CALIBRATION_HORIZON_DAYS}-day survival horizon",
        )
    detail: list[dict[str, Any]] = []
    worst = 0.0
    worst_band = None
    for band, entry in sorted(bands.items()):
        survival = entry["survived"] / entry["n"]
        stated = entry["confidence_sum"] / entry["n"]
        gap = abs(stated - survival)
        detail.append(
            {
                "band": band,
                "beliefs": entry["n"],
                "mean_confidence": round(stated, 4),
                "survival_rate": round(survival, 4),
                "gap": round(gap, 4),
                "removed_by": entry["removed_by"],
            }
        )
        if gap > worst:
            worst = gap
            worst_band = band
    return _measured(
        "calibration_gap",
        "B",
        "Calibration gap",
        worst,
        display=f"{worst:.3f} ({worst_band})",
        basis=f"max |mean stated confidence - {CALIBRATION_HORIZON_DAYS}d survival| across bands",
        widest_band=worst_band,
        bands=detail,
    )


def _load_serving_vectors(
    conn: sqlite3.Connection, belief_ids: set[str]
) -> tuple[dict[str, list[float]], str | None, dict[str, Any]]:
    """Read the optional local sidecar. Stand down silently when it is absent.

    Returns ``(vectors, reason_it_is_missing, sidecar_meta)``. A reason and empty
    vectors is the honest answer; an empty mapping with no reason would be a lie
    that reads as "no duplicates found".
    """
    from ocbrain.hybrid import VECTOR_SCHEMA_VERSION, connection_path, vector_db_path
    from ocbrain.vector import decode_embedding

    core_path = connection_path(conn)
    if core_path is None:
        return {}, "core connection has no filesystem path", {}
    path = vector_db_path(core_path)
    if not path.is_file():
        return {}, f"no vector sidecar at {path}", {}
    try:
        sidecar = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return {}, f"vector sidecar unreadable: {error}", {}
    try:
        sidecar.row_factory = sqlite3.Row
        meta = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM meta")}
        if meta.get("schema_version") != VECTOR_SCHEMA_VERSION:
            return (
                {},
                f"vector sidecar schema is {meta.get('schema_version')!r}, "
                f"expected {VECTOR_SCHEMA_VERSION!r}",
                meta,
            )
        vectors: dict[str, list[float]] = {}
        for row in sidecar.execute("SELECT belief_id, vector FROM belief_vectors"):
            belief_id = str(row["belief_id"])
            if belief_id not in belief_ids:
                continue
            decoded = decode_embedding(row["vector"])
            if decoded:
                vectors[belief_id] = decoded
        return vectors, None, meta
    except (OSError, sqlite3.Error, ValueError) as error:
        return {}, f"vector sidecar unreadable: {error}", {}
    finally:
        sidecar.close()


def _normalize(vector: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return None
    return [value / norm for value in vector]


def _cluster(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Connected components over a pair list, so a chain counts as one cluster."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in pairs:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[str, list[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(members) for members in groups.values()), key=lambda m: (-len(m), m[0]))


def _duplicates(conn: sqlite3.Connection) -> list[Metric]:
    rows = list(
        conn.execute(
            "SELECT belief_id, json_extract(attributes_json, '$.key') AS attribute_key "
            "FROM current_beliefs WHERE serve=1 AND status='current'"
        )
    )
    serving = {str(row["belief_id"]) for row in rows}
    by_key: dict[str, list[str]] = {}
    for row in rows:
        key = str(row["attribute_key"] or "").strip()
        if key:
            by_key.setdefault(key, []).append(str(row["belief_id"]))
    key_clusters = [
        {"key": key, "belief_ids": sorted(ids)}
        for key, ids in sorted(by_key.items())
        if len(ids) > 1
    ]
    metrics = [
        _measured(
            "duplicate_key_clusters",
            "B",
            "Duplicate attributes.key clusters",
            float(len(key_clusters)),
            display=str(len(key_clusters)),
            basis="serving beliefs sharing one attributes.key",
            serving_beliefs=len(serving),
            keyed_beliefs=sum(len(ids) for ids in by_key.values()),
            clusters=key_clusters[:8] or None,
        )
    ]

    vectors, reason, meta = _load_serving_vectors(conn, serving)
    if reason is not None:
        metrics.append(
            _unmeasured(
                "near_duplicate_clusters",
                "B",
                "Near-duplicate clusters (cosine)",
                reason,
                serving_beliefs=len(serving),
            )
        )
        return metrics
    normalized = {}
    for belief_id, vector in vectors.items():
        unit = _normalize(vector)
        if unit is not None:
            normalized[belief_id] = unit
    ids = sorted(normalized)
    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(ids):
        left_vector = normalized[left]
        for right in ids[index + 1 :]:
            right_vector = normalized[right]
            if len(left_vector) != len(right_vector):
                continue
            cosine = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
            if cosine >= DUPLICATE_COSINE_THRESHOLD:
                pairs.append((left, right))
    clusters = _cluster(pairs)
    coverage = len(normalized) / len(serving) if serving else 0.0
    metrics.append(
        Metric(
            key="near_duplicate_clusters",
            section="B",
            label="Near-duplicate clusters (cosine)",
            value=float(len(clusters)),
            display=str(len(clusters)),
            status=OK,
            basis=f"serving beliefs with cosine >= {DUPLICATE_COSINE_THRESHOLD} in the sidecar",
            detail={
                "threshold": DUPLICATE_COSINE_THRESHOLD,
                "embedded_serving_beliefs": len(normalized),
                "serving_beliefs": len(serving),
                "embedding_coverage": round(coverage, 4),
                "sidecar_model": meta.get("model"),
                "clusters": [{"belief_ids": members} for members in clusters[:8]] or None,
            },
        )
    )
    return metrics


# --------------------------------------------------------------------------- #
# Section C -- the correction pathway
# --------------------------------------------------------------------------- #


def _is_machine_writer(writer: str) -> bool:
    return writer.startswith(MACHINE_WRITER_PREFIXES)


def _correction_adoption(conn: sqlite3.Connection, cutoff: str) -> Metric:
    """Are agents using the structured pathway, or still bare-retracting?

    The old shape was a retract with the replacement fact typed into the
    correction body -- a field nothing indexes and nothing serves. Every one of
    the eleven agent-issued corrections this core had before v2.2 was that shape.
    """
    rows = list(
        conn.execute(
            "SELECT writer, json_extract(body_json, '$.op') AS op FROM brain_events "
            "WHERE kind='correction_recorded' AND ts >= ?",
            (cutoff,),
        )
    )
    agent_rows = [row for row in rows if not _is_machine_writer(str(row["writer"] or ""))]
    if not agent_rows:
        return _unmeasured(
            "correction_adoption",
            "C",
            "Correction adoption (structured)",
            "no agent-issued corrections in the window",
            machine_corrections=len(rows),
        )
    structured = sum(1 for row in agent_rows if str(row["op"] or "") == "supersede")
    rate = structured / len(agent_rows)
    shapes: dict[str, int] = {}
    for row in agent_rows:
        op = str(row["op"] or "(none)")
        shapes[op] = shapes.get(op, 0) + 1
    return _measured(
        "correction_adoption",
        "C",
        "Correction adoption (structured)",
        rate,
        display=f"{rate:.1%} ({structured}/{len(agent_rows)})",
        basis="agent-issued corrections with op=supersede / all agent-issued corrections",
        agent_corrections=len(agent_rows),
        structured=structured,
        machine_corrections=len(rows) - len(agent_rows),
        shapes=shapes,
    )


# Something a reader could look up, run, or verify: an issue ref (#3495), a
# backticked literal, a path, a dotted/underscored/hyphenated identifier, a
# flag, a figure (with its % or unit prefix intact), a slash pair (A/A), or an
# acronym. Extends the deslop checkable-content arms from a presence test into
# a set extractor -- deslop asks "is there anything checkable", this asks
# "which checkable things, exactly", because a rewording is judged by the
# difference of the two sets.
_CHECKABLE_TOKEN_RE = re.compile(
    r"(?:#\d+|`[^`\n]+`|[~/][\w./-]+|--\w[\w-]*|\w+(?:[._-]\w+)+"
    r"|\b\d+(?:\.\d+)?%?\b|\b[A-Z][\w]*/[A-Z][\w]*\b|\b[A-Z]{2,}\b)"
)


def _checkable_tokens(body: str) -> set[str]:
    return set(_CHECKABLE_TOKEN_RE.findall(body or ""))


def _lossy_supersessions(conn: sqlite3.Connection, cutoff: str) -> Metric:
    """Do machine rewordings preserve the facts a reader could check?

    A machine-authored supersession -- a curator refresh or a compactor merge --
    claims to restate or consolidate, not to correct. When its successor drops a
    PR reference, a profile list, or a named figure the predecessor carried, the
    corpus got smoother and knows less. The 2026-08-26 triage found this was the
    COMMON case among pending curator proposals, not the exception. Agent-issued
    supersessions are excluded: a correction is SUPPOSED to drop the tokens of
    the fact it refutes.
    """
    rows = list(
        conn.execute(
            "SELECT e.writer AS writer,"
            "       json_extract(e.body_json, '$.target_id') AS old_id,"
            "       o.body AS old_body, n.body AS new_body"
            " FROM brain_events e"
            " JOIN current_beliefs o ON o.belief_id = json_extract(e.body_json, '$.target_id')"
            " JOIN current_beliefs n ON n.belief_id = json_extract(e.body_json, '$.successor_id')"
            " WHERE e.kind='correction_recorded'"
            "   AND json_extract(e.body_json, '$.op')='supersede'"
            "   AND e.ts >= ?",
            (cutoff,),
        )
    )
    machine = [row for row in rows if _is_machine_writer(str(row["writer"] or ""))]
    if not machine:
        return _unmeasured(
            "lossy_supersession_share",
            "C",
            "Machine rewordings that drop checkable tokens",
            "no machine-authored supersessions in the window",
            agent_supersessions=len(rows),
        )
    by_writer: dict[str, dict[str, int]] = {}
    lossy_ids: list[str] = []
    for row in machine:
        writer = str(row["writer"] or "")
        bucket = by_writer.setdefault(writer, {"pairs": 0, "lossy": 0})
        bucket["pairs"] += 1
        dropped = _checkable_tokens(str(row["old_body"])) - _checkable_tokens(
            str(row["new_body"])
        )
        if dropped:
            bucket["lossy"] += 1
            if len(lossy_ids) < 5:
                lossy_ids.append(str(row["old_id"]))
    lossy = sum(bucket["lossy"] for bucket in by_writer.values())
    share = lossy / len(machine)
    return _measured(
        "lossy_supersession_share",
        "C",
        "Machine rewordings that drop checkable tokens",
        share,
        display=f"{share:.1%} ({lossy}/{len(machine)})",
        basis=(
            "machine-authored supersessions whose successor body lost at least "
            "one checkable token / all machine-authored supersessions"
        ),
        by_writer=by_writer,
        sample_lossy_targets=lossy_ids,
        agent_supersessions=len(rows) - len(machine),
    )


def _pending_queue(conn: sqlite3.Connection, now: datetime) -> list[Metric]:
    """Distinct beliefs awaiting a supersede decision, and the age of the oldest.

    An undecided proposal carrying ``attributes.supersedes`` *is* the pending
    correction -- there is no second table and no new status -- so this reads
    exactly what ``mcp_v1.pending_supersede_count`` reads, and adds the age the
    count alone cannot express. The stale belief keeps serving until an admin
    decides, so age is the real cost.

    The headline is *distinct targets*, with the raw proposal count beside it.
    Raw depth alone reported 283 on the live core and looked like ordinary
    backlog; it was 33 beliefs, one of them proposed twelve times by a loop that
    re-proposed the same supersessions every hour. A metric whose number hides
    unbounded growth is worse than no metric, so both are shown and neither can
    be read without the other.
    """
    rows = list(
        conn.execute(
            "SELECT p.id AS id, p.ts AS ts, p.writer AS writer, "
            "json_extract(p.body_json, '$.attributes.supersedes') AS target "
            "FROM brain_events p "
            "WHERE p.kind='compilation_proposed' "
            "AND json_extract(p.body_json, '$.attributes.supersedes') IS NOT NULL "
            "AND NOT EXISTS (SELECT 1 FROM brain_events d WHERE d.kind='compilation_decided' "
            "AND json_extract(d.body_json, '$.proposal_event_id') = p.id) ORDER BY p.ts"
        )
    )
    targets = {str(row["target"]) for row in rows}
    depth = Metric(
        key="pending_supersede_depth",
        section="C",
        label="Pending supersede queue depth",
        value=float(len(targets)),
        display=f"{len(targets)} distinct ({len(rows)} proposals)",
        status=OK,
        basis=(
            "distinct beliefs targeted by undecided compilation proposals carrying "
            "attributes.supersedes, and the raw proposal count"
        ),
        detail={
            "pending": len(rows),
            "proposals": len(rows),
            "distinct_targets": len(targets),
        },
    )
    if not rows:
        return [
            depth,
            _unmeasured(
                "pending_supersede_age_hours",
                "C",
                "Oldest pending supersede",
                "the pending supersede queue is empty",
                pending=0,
            ),
        ]
    oldest = _parse_ts(rows[0]["ts"])
    if oldest is None:
        return [
            depth,
            _unmeasured(
                "pending_supersede_age_hours",
                "C",
                "Oldest pending supersede",
                "the oldest pending proposal has an unparseable timestamp",
                pending=len(rows),
            ),
        ]
    age_hours = max((now - oldest).total_seconds() / 3600.0, 0.0)
    return [
        depth,
        _measured(
            "pending_supersede_age_hours",
            "C",
            "Oldest pending supersede",
            age_hours,
            display=f"{age_hours:.1f}h",
            basis="now minus the timestamp of the oldest undecided supersede proposal",
            pending=len(rows),
            oldest_proposal_event=str(rows[0]["id"]),
            oldest_proposed_at=str(rows[0]["ts"]),
            oldest_writer=str(rows[0]["writer"] or ""),
        ),
    ]


def _contradiction_rate(conn: sqlite3.Connection, cutoff: str) -> Metric:
    """How many served packets in the window carried at least one contradiction.

    ``contradictions[]`` is computed at serve time and never persisted -- there
    is no column holding it -- so this is a *reconstruction*: the two declared
    signals are recomputed over each packet's stored ``served_ids_json`` against
    the corpus as it stands now. It answers "does the writer produce anything",
    which is the question that matters given the pathway is new, but it is not a
    replay of what each packet literally shipped, and the ``basis`` field says so.

    The signals and their cap come from :mod:`ocbrain.mcp_v1` rather than being
    restated here, so the two cannot drift apart. The comparison is batched --
    keys and vectors loaded once for the whole corpus instead of once per packet
    -- because per-packet sidecar opens would put this well past its time budget.
    """
    from ocbrain.mcp_v1 import ADVISORY_COSINE_THRESHOLD, MAX_ADVISORY_PAIR_ITEMS

    rows = _retrieval_rows(conn, cutoff)
    if not rows:
        return _unmeasured(
            "contradictions_nonempty_rate",
            "C",
            "Packets carrying contradictions",
            "no retrievals recorded in the window",
        )
    keys = {
        str(row["belief_id"]): str(row["attribute_key"] or "").strip()
        for row in conn.execute(
            "SELECT belief_id, json_extract(attributes_json, '$.key') AS attribute_key "
            "FROM current_beliefs"
        )
    }
    declared: dict[str, set[str]] = {}
    for row in conn.execute(
        "SELECT belief_id, attributes_json FROM current_beliefs "
        "WHERE json_extract(attributes_json, '$.contradicts') IS NOT NULL "
        "OR json_extract(attributes_json, '$.contradiction_ids') IS NOT NULL"
    ):
        attributes = _json_object(row["attributes_json"])
        others = attributes.get("contradicts") or attributes.get("contradiction_ids") or []
        if isinstance(others, list):
            declared[str(row["belief_id"])] = {str(value) for value in others if value}

    all_ids = set(keys)
    vectors, vector_reason, _meta = _load_serving_vectors(conn, all_ids)
    normalized: dict[str, list[float]] = {}
    for belief_id, vector in vectors.items():
        unit = _normalize(vector)
        if unit is not None:
            normalized[belief_id] = unit

    carried = 0
    reasons: dict[str, int] = {}
    skipped = 0
    for row in rows:
        try:
            served = json.loads(row["served_ids_json"] or "[]")
        except (TypeError, ValueError):
            served = []
        ids = [str(value) for value in served if value]
        if len(ids) < 2:
            continue
        found: set[str] = set()
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                if right in declared.get(left, ()) or left in declared.get(right, ()):
                    found.add("explicit_compiler_metadata")
                    continue
                if len(ids) > MAX_ADVISORY_PAIR_ITEMS:
                    continue
                left_key = keys.get(left) or ""
                if left_key and left_key == (keys.get(right) or ""):
                    found.add("duplicate_key")
                elif left in normalized and right in normalized:
                    left_vector, right_vector = normalized[left], normalized[right]
                    if len(left_vector) != len(right_vector):
                        continue
                    cosine = sum(a * b for a, b in zip(left_vector, right_vector, strict=True))
                    if cosine >= ADVISORY_COSINE_THRESHOLD:
                        found.add("embedding_similarity")
        if len(ids) > MAX_ADVISORY_PAIR_ITEMS:
            skipped += 1
        if found:
            carried += 1
            for reason in found:
                reasons[reason] = reasons.get(reason, 0) + 1
    rate = carried / len(rows)
    basis = (
        "RECONSTRUCTED: contradictions[] is computed at serve time and not stored, so the "
        "declared and advisory signals are recomputed over each packet's stored served_ids "
        "against the corpus as it stands now"
    )
    return _measured(
        "contradictions_nonempty_rate",
        "C",
        "Packets carrying contradictions",
        rate,
        display=f"{rate:.1%} ({carried}/{len(rows)})",
        basis=basis,
        packets=len(rows),
        packets_with_contradictions=carried,
        by_reason=reasons or None,
        advisory_pass_skipped_oversized_packets=skipped or None,
        embedding_signal_unavailable=vector_reason,
    )


# --------------------------------------------------------------------------- #
# Section D -- plumbing
# --------------------------------------------------------------------------- #


def _capture_start(conn: sqlite3.Connection) -> str | None:
    """When this core first stamped a server-observed connection id.

    Rows written before this could not have carried provenance no matter how
    well-behaved the client was, so including them measures the migration date
    rather than whether clients are stamping now. ``None`` means capture has
    never happened here, which is a different answer from "coverage is zero".
    """
    candidates = [
        _scalar(
            conn,
            "SELECT MIN(served_at) FROM retrieval_uses WHERE server_connection_id IS NOT NULL",
        ),
        _scalar(
            conn,
            "SELECT MIN(closed_at) FROM task_closeouts WHERE server_connection_id IS NOT NULL",
        ),
    ]
    stamps = [str(value) for value in candidates if value]
    return min(stamps) if stamps else None


def _provenance(conn: sqlite3.Connection, cutoff: str) -> Metric:
    retrieval_columns = _columns(conn, "retrieval_uses")
    closeout_columns = _columns(conn, "task_closeouts")
    if "server_connection_id" not in retrieval_columns or (
        "server_connection_id" not in closeout_columns
    ):
        return _unmeasured(
            "provenance_coverage",
            "D",
            "Provenance coverage",
            "this core predates the server_connection_id column (PR #33)",
        )
    start = _capture_start(conn)
    if start is None:
        return _unmeasured(
            "provenance_coverage",
            "D",
            "Provenance coverage",
            "no row on this core has ever carried a server_connection_id",
        )
    boundary = max(start, cutoff)

    def tally(table: str, column: str) -> list[sqlite3.Row]:
        return list(
            conn.execute(
                "SELECT COALESCE(client_runtime_key, '(unattributed)') AS runtime_key, "  # noqa: S608
                "COUNT(*) AS n, SUM(server_connection_id IS NOT NULL) AS covered "
                f"FROM {table} WHERE {column} >= ? GROUP BY runtime_key",
                (boundary,),
            )
        )

    retrievals = tally("retrieval_uses", "served_at")
    closeouts = tally("task_closeouts", "closed_at")
    total = sum(int(row["n"]) for row in retrievals) + sum(int(row["n"]) for row in closeouts)
    window_total = int(
        (_scalar(conn, "SELECT COUNT(*) FROM retrieval_uses WHERE served_at >= ?", (cutoff,)) or 0)
        + (
            _scalar(conn, "SELECT COUNT(*) FROM task_closeouts WHERE closed_at >= ?", (cutoff,))
            or 0
        )
    )
    if total < MIN_PROVENANCE_SAMPLE:
        return _unmeasured(
            "provenance_coverage",
            "D",
            "Provenance coverage",
            f"only {total} rows written since provenance capture began; "
            f"{MIN_PROVENANCE_SAMPLE} needed for a verdict",
            capture_started_at=start,
            rows_since_capture=total,
            rows_in_window=window_total,
        )
    covered = sum(int(row["covered"] or 0) for row in retrievals) + sum(
        int(row["covered"] or 0) for row in closeouts
    )
    rate = covered / total
    breakdown: dict[str, dict[str, Any]] = {}
    for label, rowset in (("retrievals", retrievals), ("closeouts", closeouts)):
        for row in rowset:
            key = str(row["runtime_key"])
            entry = breakdown.setdefault(key, {"retrievals": 0, "closeouts": 0, "covered": 0})
            entry[label] += int(row["n"])
            entry["covered"] += int(row["covered"] or 0)
    return _measured(
        "provenance_coverage",
        "D",
        "Provenance coverage",
        rate,
        display=f"{rate:.1%} ({covered}/{total})",
        basis="retrievals + closeouts written since provenance capture began, carrying "
        "server_connection_id",
        capture_started_at=start,
        rows_since_capture=total,
        rows_in_window=window_total,
        covered=covered,
        by_client_runtime_key=breakdown,
    )


def _transcript_ids(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.stem for path in root.rglob("*.jsonl")}


def _closeout_join(
    conn: sqlite3.Connection, cutoff: str, transcript_root: Path | None
) -> Metric:
    """Do closeouts name a session that resolves to a real transcript?

    The join is what makes a closeout's tool-call trace minable later, and it has
    only ever worked when the recorded session id is byte-identical to a
    transcript filename. Without a transcript root on this machine the question
    cannot be asked, and saying so beats reporting a zero that means "did not
    look".
    """
    columns = _columns(conn, "task_closeouts")
    if not columns:
        return _unmeasured(
            "closeout_trace_join_rate", "D", "Closeout to trace join", "no task_closeouts table"
        )
    root = transcript_root or (Path.home() / ".claude" / "projects")
    transcripts = _transcript_ids(root)
    if not transcripts:
        return _unmeasured(
            "closeout_trace_join_rate",
            "D",
            "Closeout to trace join",
            f"no transcripts found under {root}",
            transcript_root=str(root),
        )
    hint_column = "client_session_hint" if "client_session_hint" in columns else "NULL"
    rows = list(
        conn.execute(
            f"SELECT closed_at, session_id, {hint_column} AS hint FROM task_closeouts"  # noqa: S608
        )
    )
    if not rows:
        return _unmeasured(
            "closeout_trace_join_rate", "D", "Closeout to trace join", "no closeouts recorded"
        )

    def joins(row: sqlite3.Row) -> bool:
        for candidate in (row["session_id"], row["hint"]):
            text = str(candidate or "").strip()
            if text and text in transcripts:
                return True
        return False

    def uuid_shaped(row: sqlite3.Row) -> bool:
        candidates = (row["session_id"], row["hint"])
        return any(_UUID.match(str(value or "").strip()) for value in candidates)

    window = [row for row in rows if str(row["closed_at"] or "") >= cutoff]
    all_time_rate = sum(1 for row in rows if joins(row)) / len(rows)
    window_rate = (sum(1 for row in window if joins(row)) / len(window)) if window else None

    # The brief is explicit that this climbs for new rows only, so the gated
    # figure is the post-capture one. The window and all-time figures ride along
    # unbadged, because the historical 3.7% is the baseline this is measured
    # against and hiding it would make the improvement unfalsifiable.
    start = _capture_start(conn)
    recent = [row for row in rows if start and str(row["closed_at"] or "") >= start]
    shared = {
        "window_closeouts": len(window),
        "window_rate": round(window_rate, 4) if window_rate is not None else None,
        "all_time_closeouts": len(rows),
        "all_time_rate": round(all_time_rate, 4),
        "window_uuid_shaped": sum(1 for row in window if uuid_shaped(row)),
        "transcripts_on_disk": len(transcripts),
        "transcript_root": str(root),
        "capture_started_at": start,
    }
    if len(recent) < MIN_PROVENANCE_SAMPLE:
        return _unmeasured(
            "closeout_trace_join_rate",
            "D",
            "Closeout to trace join",
            f"only {len(recent)} closeouts written since provenance capture began; "
            f"{MIN_PROVENANCE_SAMPLE} needed for a verdict",
            closeouts_since_capture=len(recent),
            **shared,
        )
    joined = sum(1 for row in recent if joins(row))
    rate = joined / len(recent)
    return _measured(
        "closeout_trace_join_rate",
        "D",
        "Closeout to trace join",
        rate,
        display=f"{rate:.1%} ({joined}/{len(recent)})",
        basis="closeouts written since provenance capture began whose session id or harness "
        "hint names a transcript file",
        closeouts_since_capture=len(recent),
        joined_since_capture=joined,
        **shared,
    )


def _harvest(conn: sqlite3.Connection, now: datetime) -> Metric:
    """Newest evidence per source runtime, and any live stream gone quiet.

    Only *live* streams are tracked -- see :data:`HARVEST_LIVE_DAYS` for why
    per-label freshness across every historical runtime spelling is noise rather
    than signal. The alarm this exists for is a stream that was harvesting and
    stopped, and that is what the liveness filter leaves behind.
    """
    if not _table_exists(conn, "evidence_objects"):
        return _unmeasured("harvest_silence_hours", "D", "Harvest freshness", "no evidence table")
    live_cutoff = (now - timedelta(days=HARVEST_LIVE_DAYS)).isoformat()
    all_runtimes = int(
        _scalar(conn, "SELECT COUNT(DISTINCT COALESCE(source_runtime, '')) FROM evidence_objects")
        or 0
    )
    rows = list(
        conn.execute(
            "SELECT COALESCE(source_runtime, '(unattributed)') AS runtime, COUNT(*) AS n, "
            "MAX(recorded_at) AS newest FROM evidence_objects WHERE recorded_at >= ? "
            "GROUP BY runtime HAVING n >= ?",
            (live_cutoff, HARVEST_MIN_ROWS),
        )
    )
    tracked: list[dict[str, Any]] = []
    for row in rows:
        parsed = _parse_ts(row["newest"])
        if parsed is None:
            continue
        silence = max((now - parsed).total_seconds() / 3600.0, 0.0)
        tracked.append(
            {
                "runtime": str(row["runtime"]),
                "rows_in_live_window": int(row["n"]),
                "newest": str(row["newest"]),
                "silent_hours": round(silence, 1),
            }
        )
    if not tracked:
        return _unmeasured(
            "harvest_silence_hours",
            "D",
            "Harvest freshness",
            f"no runtime wrote >= {HARVEST_MIN_ROWS} evidence rows in the last "
            f"{HARVEST_LIVE_DAYS} days",
            distinct_runtimes_all_time=all_runtimes,
        )
    tracked.sort(key=lambda item: -item["silent_hours"])
    worst = tracked[0]
    silent = [item for item in tracked if item["silent_hours"] > HARVEST_SILENCE_ALARM_HOURS]
    return _measured(
        "harvest_silence_hours",
        "D",
        "Harvest freshness",
        float(worst["silent_hours"]),
        display=f"{worst['silent_hours']:.1f}h ({worst['runtime']})",
        basis=f"longest silence among live streams (>= {HARVEST_MIN_ROWS} rows in the last "
        f"{HARVEST_LIVE_DAYS} days)",
        live_streams=len(tracked),
        distinct_runtimes_all_time=all_runtimes,
        silent_streams=silent or None,
        streams=tracked[:8],
    )


_GROWTH_TABLES = (
    ("brain_events", "ts"),
    ("evidence_objects", "recorded_at"),
    ("retrieval_uses", "served_at"),
    ("task_closeouts", "closed_at"),
    ("current_beliefs", "last_compiled_at"),
)


def _storage(conn: sqlite3.Connection, cutoff: str, since_days: int) -> list[Metric]:
    page_count = int(_scalar(conn, "PRAGMA page_count") or 0)
    page_size = int(_scalar(conn, "PRAGMA page_size") or 0)
    size_mb = page_count * page_size / (1024 * 1024)
    tables: dict[str, dict[str, int]] = {}
    for table, column in _GROWTH_TABLES:
        if not _table_exists(conn, table):
            continue
        total = int(_scalar(conn, f"SELECT COUNT(*) FROM {table}") or 0)  # noqa: S608
        recent = int(
            _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE {column} >= ?", (cutoff,)) or 0  # noqa: S608
        )
        tables[table] = {"rows": total, f"added_{since_days}d": recent}
    metrics = [
        Metric(
            key="db_size_mb",
            section="D",
            label="Core size",
            value=round(size_mb, 2),
            display=f"{size_mb:,.1f} MB",
            status=OK,
            basis="page_count * page_size",
            detail={"page_count": page_count, "page_size": page_size, "tables": tables},
        )
    ]

    if not tables:
        metrics.append(
            _unmeasured(
                "rows_added_in_window",
                "D",
                f"Rows added ({since_days}d)",
                "no timestamped tables present on this core",
            )
        )
        return metrics

    # Deliberately rows, not an estimated megabyte figure. Prorating size by row
    # share reads as authoritative and is not: evidence bodies are orders of
    # magnitude larger than retrieval receipts, so the estimate was wrong by a
    # factor of seven against the known ~14 MB/month post-PR-#33 figure. A real
    # byte delta needs two snapshots, which is what --baseline is for.
    recent_rows = sum(entry[f"added_{since_days}d"] for entry in tables.values())
    metrics.append(
        _measured(
            "rows_added_in_window",
            "D",
            f"Rows added ({since_days}d)",
            float(recent_rows),
            display=f"{recent_rows:,}",
            basis="sum of timestamped rows added across tracked tables; per-table detail below, "
            "byte-level delta needs --baseline",
            rows_total=sum(entry["rows"] for entry in tables.values()),
            tables=tables,
        )
    )
    return metrics


def _sidecar_freshness(conn: sqlite3.Connection) -> Metric:
    from ocbrain.hybrid import VECTOR_SCHEMA_VERSION, connection_path, vector_db_path

    core_path = connection_path(conn)
    if core_path is None:
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            "core connection has no filesystem path",
        )
    path = vector_db_path(core_path)
    if not path.is_file():
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            f"no vector sidecar at {path}",
            expected_path=str(path),
        )
    try:
        sidecar = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            f"vector sidecar unreadable: {error}",
        )
    try:
        meta = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM meta")}
        rows = int(_scalar(sidecar, "SELECT COUNT(*) FROM belief_vectors") or 0)
    except sqlite3.Error as error:
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            f"vector sidecar unreadable: {error}",
        )
    finally:
        sidecar.close()
    if meta.get("schema_version") != VECTOR_SCHEMA_VERSION:
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            f"sidecar schema {meta.get('schema_version')!r} != {VECTOR_SCHEMA_VERSION!r}",
        )
    built_at_seq = meta.get("core_event_seq")
    head = int(_scalar(conn, "SELECT COALESCE(MAX(event_seq), 0) FROM brain_events") or 0)
    if built_at_seq is None:
        return _unmeasured(
            "vector_sidecar_lag_events",
            "D",
            "Vector sidecar freshness",
            "sidecar records no core_event_seq",
            core_head_event_seq=head,
        )
    lag = max(head - int(built_at_seq), 0)
    return _measured(
        "vector_sidecar_lag_events",
        "D",
        "Vector sidecar freshness",
        float(lag),
        display=f"{lag} events behind",
        basis="core head event_seq minus the seq the sidecar was built at",
        sidecar_path=str(path),
        sidecar_rows=rows,
        sidecar_built_at=meta.get("built_at"),
        sidecar_model=meta.get("model"),
        core_head_event_seq=head,
    )


def _integrity(conn: sqlite3.Connection) -> Metric:
    """Structural soundness. ``quick_check`` rather than ``integrity_check``.

    ``quick_check`` skips the exhaustive per-row index cross-check, which is what
    makes the full check take tens of seconds on a 180 MB core. Everything this
    command exists to catch -- a torn page, a corrupt b-tree, a broken foreign
    key -- it still catches, and a check too slow to run hourly does not get run.
    """
    try:
        quick = [str(row[0]) for row in conn.execute("PRAGMA quick_check")]
    except sqlite3.Error as error:
        return _unmeasured("integrity", "D", "Integrity", f"quick_check failed: {error}")
    try:
        fk = list(conn.execute("PRAGMA foreign_key_check"))
    except sqlite3.Error as error:
        return _unmeasured("integrity", "D", "Integrity", f"foreign_key_check failed: {error}")
    ok = quick == ["ok"] and not fk
    return _measured(
        "integrity",
        "D",
        "Integrity",
        1.0 if ok else 0.0,
        display="ok" if ok else "FAILED",
        basis="PRAGMA quick_check + PRAGMA foreign_key_check",
        quick_check=quick[:5],
        foreign_key_violations=len(fk),
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


_BRIEFING_SCOPE_TYPES = ("project", "repo", "client", "task")


def _briefing_scope_key(scope_type: str, value: Any) -> tuple[str, str] | None:
    """Normalize one scope the same way a real briefing caller does."""
    if scope_type not in _BRIEFING_SCOPE_TYPES:
        return None
    from ocbrain.scope import ScopeContext

    context = ScopeContext.from_dict({scope_type: value})
    normalized = getattr(context, scope_type)
    if not normalized:
        return None
    return scope_type, normalized


def _briefing_scope_label(scope: tuple[str, str]) -> str:
    """Keep the historical bare-project return while identifying other kinds."""
    scope_type, value = scope
    if scope_type == "project" and value.partition(":")[0] not in _BRIEFING_SCOPE_TYPES:
        return value
    return f"{scope_type}:{value}"


def _briefing_scope_context(scope: str):
    """Turn one sampled label into the context accepted by ``build_briefing``."""
    from ocbrain.scope import ScopeContext

    scope_type, separator, value = scope.partition(":")
    if separator and scope_type in _BRIEFING_SCOPE_TYPES:
        return ScopeContext.from_dict({scope_type: value})
    return ScopeContext(project=scope)


def _briefing_scopes(conn: sqlite3.Connection, *, limit: int = 5) -> list[str]:
    """Choose deterministic briefing scopes, with currently open work first.

    Goal state comes from the serving/current fold in ``current_beliefs`` and
    specifically from the folded goal attribute ``status``. The belief row's
    own ``status='current'`` only says that the belief is current; it does not
    mean the goal is still open.

    Open-goal scopes are sorted first. Spare slots are filled by other goal or
    closeout scopes, with busier closeout scopes first and the normalized scope
    as a total tie-breaker. A scope present in more than one source is sampled
    once. Ordinary project labels remain bare for compatibility; other
    supported scope kinds carry their ``type:`` prefix so the caller can
    reconstruct context.
    """
    if limit <= 0:
        return []

    active: set[tuple[str, str]] = set()
    goal_scopes: set[tuple[str, str]] = set()
    if _table_exists(conn, "current_beliefs"):
        rows = conn.execute(
            "SELECT belief_id, scope_type, scope_id, attributes_json "
            "FROM current_beliefs "
            "WHERE serve=1 AND status='current' AND belief_type='goal' "
            "ORDER BY scope_type, scope_id, belief_id"
        )
        for row in rows:
            scope_type = str(row["scope_type"] or "")
            scope_id = str(row["scope_id"] or "")
            prefix = f"{scope_type}:"
            value = scope_id[len(prefix) :] if scope_id.startswith(prefix) else scope_id
            scope = _briefing_scope_key(scope_type, value)
            if scope is None:
                continue
            goal_scopes.add(scope)
            attributes = json.loads(row["attributes_json"] or "{}")
            if str(attributes.get("status") or "open") == "open":
                active.add(scope)

    closeout_counts: dict[tuple[str, str], int] = {}
    if _table_exists(conn, "task_closeouts"):
        from ocbrain.scope import ScopeContext

        for row in conn.execute(
            "SELECT context_json, COUNT(*) AS n FROM task_closeouts "
            "GROUP BY context_json ORDER BY context_json"
        ):
            raw_context = json.loads(row[0] or "{}")
            if not isinstance(raw_context, dict):
                continue
            context = ScopeContext.from_dict(raw_context)
            scope = next(
                (
                    (scope_type, value)
                    for scope_type in _BRIEFING_SCOPE_TYPES
                    if (value := getattr(context, scope_type))
                ),
                None,
            )
            if scope is not None:
                closeout_counts[scope] = closeout_counts.get(scope, 0) + int(row["n"])

    scope_order = {scope_type: index for index, scope_type in enumerate(_BRIEFING_SCOPE_TYPES)}

    def stable_key(scope: tuple[str, str]) -> tuple[int, str]:
        return scope_order[scope[0]], scope[1]

    ordered_active = sorted(active, key=stable_key)
    remaining = (goal_scopes | closeout_counts.keys()) - active
    ordered_remaining = sorted(
        remaining,
        key=lambda scope: (-closeout_counts.get(scope, 0), *stable_key(scope)),
    )
    return [_briefing_scope_label(scope) for scope in (*ordered_active, *ordered_remaining)[:limit]]


def _harness_surface(conn: sqlite3.Connection, now: datetime) -> list[Metric]:
    """Section E: is the loop-facing surface keeping its two promises?

    Determinism and boundedness are the whole contract, so they are measured
    rather than asserted once in a unit test against a fixture. A unit test says
    the code can be deterministic; this says the corpus an operator actually has
    does not break it.

    Two precision-side metrics the harness literature asks for are deliberately
    absent here, because they already exist elsewhere in this scorecard and a
    second name for the same number is how a scorecard stops being read:
    ``pollution_rate`` (section B) is the false-positive injection measure --
    beliefs approved and then removed within a horizon -- and ``zero_result_rate``
    with ``calibration_gap`` (sections A and B) together are the abstention
    calibration. Nothing is faked in to fill a row.
    """
    from ocbrain.briefing import DEFAULT_BRIEFING_BUDGET_CHARS, build_briefing, list_goals

    scopes = _briefing_scopes(conn)
    if not scopes:
        reason = "no scope has a goal or a closeout to brief on"
        return [
            _unmeasured("briefing_determinism", "E", "Briefing determinism", reason, scopes=0),
            _unmeasured(
                "briefing_budget_compliance", "E", "Briefing within budget", reason, scopes=0
            ),
            _unmeasured("goal_pointer_resolution", "E", "Goal spec pointers resolve", reason),
            _unmeasured("goal_open_age_days", "E", "Oldest open goal", reason),
        ]

    identical = 0
    within_budget = 0
    divergent: list[str] = []
    oversized: list[str] = []
    for scope in scopes:
        context = _briefing_scope_context(scope)
        first = build_briefing(conn, context=context)
        second = build_briefing(conn, context=context)
        if first["text"] == second["text"]:
            identical += 1
        else:
            divergent.append(scope)
        if first["used_chars"] <= first["budget_chars"]:
            within_budget += 1
        else:
            oversized.append(scope)

    metrics = [
        _measured(
            "briefing_determinism",
            "E",
            "Briefing determinism",
            identical / len(scopes),
            display=f"{identical}/{len(scopes)} scopes byte-identical",
            basis="brain.briefing called twice per scope, bytes compared",
            scopes=len(scopes),
            divergent_scopes=divergent or None,
        ),
        _measured(
            "briefing_budget_compliance",
            "E",
            "Briefing within budget",
            within_budget / len(scopes),
            display=f"{within_budget}/{len(scopes)} within {DEFAULT_BRIEFING_BUDGET_CHARS} chars",
            basis="rendered characters against the declared budget",
            scopes=len(scopes),
            oversized_scopes=oversized or None,
        ),
    ]

    goals: list[dict[str, Any]] = []
    for scope in scopes:
        goals.extend(
            list_goals(conn, context=_briefing_scope_context(scope), status="open", limit=0)
        )
    unique = {goal["goal_id"]: goal for goal in goals}
    if not unique:
        metrics.append(
            _unmeasured(
                "goal_pointer_resolution",
                "E",
                "Goal spec pointers resolve",
                "no goals are open in the sampled scopes",
                open_goals=0,
            )
        )
        metrics.append(
            _unmeasured(
                "goal_open_age_days",
                "E",
                "Oldest open goal",
                "no goals are open in the sampled scopes",
                open_goals=0,
            )
        )
        return metrics

    unresolved = [
        goal["goal_id"] for goal in unique.values() if goal.get("warning") is not None
    ]
    metrics.append(
        _measured(
            "goal_pointer_resolution",
            "E",
            "Goal spec pointers resolve",
            (len(unique) - len(unresolved)) / len(unique),
            display=f"{len(unique) - len(unresolved)}/{len(unique)} resolve",
            basis="source_pointer.path checked on this filesystem",
            open_goals=len(unique),
            unresolved=unresolved or None,
        )
    )
    ages: list[tuple[float, str]] = []
    for goal in unique.values():
        opened = _parse_ts(goal.get("opened_at"))
        if opened is not None:
            ages.append((max((now - opened).total_seconds() / 86400.0, 0.0), goal["goal_id"]))
    if not ages:
        metrics.append(
            _unmeasured(
                "goal_open_age_days",
                "E",
                "Oldest open goal",
                "no open goal carries a parseable opened_at",
                open_goals=len(unique),
            )
        )
        return metrics
    age, oldest_id = max(ages)
    metrics.append(
        _measured(
            "goal_open_age_days",
            "E",
            "Oldest open goal",
            age,
            display=f"{age:.1f}d",
            basis="now minus opened_at of the longest-open goal",
            open_goals=len(unique),
            oldest_goal_id=oldest_id,
        )
    )
    return metrics


SECTION_TITLES = {
    "A": "Retrieval health",
    "B": "Corpus quality",
    "C": "Correction pathway",
    "D": "Plumbing",
    "E": "Harness surface",
}


def run_selftest(
    conn: sqlite3.Connection,
    *,
    since_days: int = 30,
    now: datetime | None = None,
    transcript_root: Path | None = None,
) -> dict[str, Any]:
    """Measure one core and return the scorecard."""
    started = time.monotonic()
    now = now or datetime.now(UTC)
    cutoff = (now - timedelta(days=since_days)).isoformat()

    metrics: list[Metric] = []
    metrics.extend(_section_a(conn, cutoff))
    metrics.extend(_pollution(conn, cutoff))
    metrics.append(_conflict_preservation(conn, cutoff))
    metrics.append(_calibration(conn, now))
    metrics.extend(_duplicates(conn))
    metrics.append(_correction_adoption(conn, cutoff))
    metrics.append(_lossy_supersessions(conn, cutoff))
    metrics.extend(_pending_queue(conn, now))
    metrics.append(_contradiction_rate(conn, cutoff))
    metrics.append(_provenance(conn, cutoff))
    metrics.append(_closeout_join(conn, cutoff, transcript_root))
    metrics.append(_harvest(conn, now))
    metrics.extend(_storage(conn, cutoff, since_days))
    metrics.append(_sidecar_freshness(conn))
    metrics.append(_integrity(conn))
    metrics.extend(_harness_surface(conn, now))

    tally = {OK: 0, WATCH: 0, ALARM: 0, NOT_MEASURED: 0}
    for metric in metrics:
        tally[metric.status] = tally.get(metric.status, 0) + 1
    worst = ALARM if tally[ALARM] else (WATCH if tally[WATCH] else OK)
    return {
        "schema_version": SCORECARD_SCHEMA,
        "generated_at": now.isoformat(),
        "window": {"since_days": since_days, "cutoff": cutoff},
        "core_schema": _scalar(
            conn, "SELECT value FROM schema_meta WHERE key='core_schema'"
        )
        if _table_exists(conn, "schema_meta")
        else None,
        "status": worst,
        "tally": tally,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "metrics": [metric.to_dict() for metric in metrics],
    }


def exit_code(scorecard: dict[str, Any]) -> int:
    """Non-zero when any metric alarmed, so this can be a cron gate.

    ``watch`` deliberately does not fail. The whole point of a middle band is to
    give a human a chance to look before the gate starts failing, and a gate that
    fires on every yellow is a gate that gets muted.
    """
    return 1 if scorecard.get("tally", {}).get(ALARM) else 0


# --------------------------------------------------------------------------- #
# Rendering and diffing
# --------------------------------------------------------------------------- #

_STATUS_MARK = {OK: "  ", WATCH: " ~", ALARM: "!!", NOT_MEASURED: " ?"}
_STATUS_LABEL = {OK: "OK", WATCH: "WATCH", ALARM: "ALARM", NOT_MEASURED: "not measured"}


def _fmt_number(value: float | None) -> str:
    if value is None:
        return "-"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _display(metric: dict[str, Any]) -> str:
    if metric.get("status") == NOT_MEASURED:
        return "-"
    if metric.get("display"):
        return str(metric["display"])
    return _fmt_number(metric.get("value"))


def render_pretty(scorecard: dict[str, Any]) -> str:
    """A scorecard a human reads top to bottom and stops at the first marker."""
    metrics = scorecard.get("metrics", [])
    label_width = max([len(str(m.get("label", ""))) for m in metrics] + [24])
    value_width = max([len(_display(m)) for m in metrics] + [10])
    status_width = max(len(text) for text in _STATUS_LABEL.values())

    lines: list[str] = []
    window = scorecard.get("window", {})
    lines.append("OCBrain selftest")
    lines.append(
        f"  window {window.get('since_days')}d to {scorecard.get('generated_at', '')}"
        f"   elapsed {scorecard.get('elapsed_seconds')}s"
    )
    tally = scorecard.get("tally", {})
    lines.append(
        f"  verdict {str(scorecard.get('status', '')).upper()}"
        f"   ok {tally.get(OK, 0)}"
        f"   watch {tally.get(WATCH, 0)}"
        f"   alarm {tally.get(ALARM, 0)}"
        f"   not measured {tally.get(NOT_MEASURED, 0)}"
    )
    lines.append("")

    rule = "-" * (3 + label_width + 2 + value_width + 2 + status_width + 2 + 28)
    for section in ("A", "B", "C", "D"):
        rows = [m for m in metrics if m.get("section") == section]
        if not rows:
            continue
        lines.append(f"{section}. {SECTION_TITLES[section].upper()}")
        lines.append(rule)
        for metric in rows:
            status = str(metric.get("status", OK))
            threshold = metric.get("threshold") or {}
            if status == NOT_MEASURED:
                note = _ellipsize(str(metric.get("reason", "")), 28)
            elif threshold.get("direction") == INFO:
                note = "informational"
            else:
                arrow = ">=" if threshold.get("direction") == HIGHER_BETTER else "<="
                note = f"{arrow} {_fmt_number(threshold.get('ok'))}"
            lines.append(
                f"{_STATUS_MARK.get(status, '  ')} "
                f"{str(metric.get('label', '')):<{label_width}}  "
                f"{_display(metric):>{value_width}}  "
                f"{_STATUS_LABEL.get(status, status):<{status_width}}  "
                f"{note}"
            )
        lines.append("")

    flagged = [m for m in metrics if m.get("status") in (WATCH, ALARM)]
    if flagged:
        lines.append("FLAGGED")
        lines.append(rule)
        for metric in flagged:
            lines.append(
                f"{_STATUS_MARK.get(str(metric.get('status')), '  ')} "
                f"{metric.get('label')}: {_display(metric)} "
                f"({_STATUS_LABEL.get(str(metric.get('status')))})"
            )
            threshold = metric.get("threshold") or {}
            if threshold.get("source"):
                for chunk in _wrap(str(threshold["source"]), 92):
                    lines.append(f"     {chunk}")
        lines.append("")

    unmeasured = [m for m in metrics if m.get("status") == NOT_MEASURED]
    if unmeasured:
        lines.append("NOT MEASURED")
        lines.append(rule)
        for metric in unmeasured:
            wrapped = _wrap(f"{metric.get('label')}: {metric.get('reason')}", 92)
            lines.append(f"   {wrapped[0]}" if wrapped else f"   {metric.get('label')}")
            lines.extend(f"     {chunk}" for chunk in wrapped[1:])
        lines.append("")

    diff = scorecard.get("baseline_diff")
    if diff:
        lines.append("DIFF VS BASELINE")
        lines.append(rule)
        lines.append(f"   baseline generated {diff.get('baseline_generated_at')}")
        for change in diff.get("changes", []):
            delta = change.get("delta")
            delta_text = "-" if delta is None else f"{delta:+.4g}"
            lines.append(
                f"   {str(change.get('label')):<{label_width}}  "
                f"{_fmt_number(change.get('baseline_value')):>10} -> "
                f"{_fmt_number(change.get('value')):<10} "
                f"{delta_text:>10}  "
                f"{change.get('baseline_status')} -> {change.get('status')}"
            )
        if not diff.get("changes"):
            lines.append("   no metric changed")
        for key in diff.get("added", []):
            lines.append(f"   + {key} (new in this run)")
        for key in diff.get("removed", []):
            lines.append(f"   - {key} (absent from this run)")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _ellipsize(text: str, width: int) -> str:
    """Truncate with a marker, so a clipped reason cannot read as a whole one."""
    return text if len(text) <= width else f"{text[: width - 3].rstrip()}..."


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def diff_scorecards(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """What moved since a saved scorecard.

    Reports every metric whose value or verdict changed. A metric that appeared
    or vanished between runs is listed separately rather than silently dropped:
    a metric that stopped being measurable is itself news.
    """
    base_by_key = {str(m.get("key")): m for m in baseline.get("metrics", [])}
    cur_by_key = {str(m.get("key")): m for m in current.get("metrics", [])}
    changes: list[dict[str, Any]] = []
    for key, metric in cur_by_key.items():
        previous = base_by_key.get(key)
        if previous is None:
            continue
        value = metric.get("value")
        old_value = previous.get("value")
        status = metric.get("status")
        old_status = previous.get("status")
        if value == old_value and status == old_status:
            continue
        delta = None
        if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
            delta = round(float(value) - float(old_value), 6)
        changes.append(
            {
                "key": key,
                "label": metric.get("label"),
                "baseline_value": old_value,
                "value": value,
                "delta": delta,
                "baseline_status": old_status,
                "status": status,
                "regressed": _regressed(key, old_status, status),
            }
        )
    changes.sort(key=lambda item: (not item["regressed"], str(item["key"])))
    return {
        "baseline_generated_at": baseline.get("generated_at"),
        "baseline_window": baseline.get("window"),
        "changes": changes,
        "added": sorted(set(cur_by_key) - set(base_by_key)),
        "removed": sorted(set(base_by_key) - set(cur_by_key)),
        "regressions": sum(1 for change in changes if change["regressed"]),
    }


_SEVERITY = {OK: 0, NOT_MEASURED: 1, WATCH: 2, ALARM: 3}


def _regressed(key: str, old_status: Any, new_status: Any) -> bool:
    del key
    return _SEVERITY.get(str(new_status), 0) > _SEVERITY.get(str(old_status), 0)


__all__ = [
    "ALARM",
    "NOT_MEASURED",
    "OK",
    "SCORECARD_SCHEMA",
    "THRESHOLDS",
    "WATCH",
    "Metric",
    "SelftestError",
    "Threshold",
    "diff_scorecards",
    "exit_code",
    "open_readonly",
    "render_pretty",
    "run_selftest",
]
