"""Stage 5: turn mined gotchas into scoped beliefs, deterministically.

A gotcha is a one-sentence claim about one step signature — "``tool:wait_agent``
times out on 207 of 357 calls, and the recurring next move is
``tool:list_agents``, which works 92% of the time". It is step-scoped, needs no
closeout, and has three orders of magnitude more support than any procedure this
corpus can carry. That makes it the half of procedural memory worth shipping
first.

Three things this deliberately is **not**:

*Not auto-compile.* That mechanism was deleted. Nothing here promotes evidence on
its own recognisance; every belief is an explicit ``compilation_proposed`` plus
an explicit approval, written by a named non-human actor.

*Not the curator.* The curator asks a hosted model to write a claim. Here the
wording is generated from the counts by :func:`procmine.dag.mine_gotchas`, so the
sentence cannot drift from the evidence and no model call is made. Mining is
offline and local.

*Not incremental.* Every run recomputes the whole claim from the whole corpus and
republishes it under a **stable belief id** derived from the signature and the
scope. A re-mint therefore converges on the same row rather than adding one, and
no statistic is ever incremented in place — the same recompute-and-replace rule
the rest of the brain follows for derived state.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ocbrain.core_v1 import append_core_event, record_core_v1_evidence
from ocbrain.db import connect
from ocbrain.ids import stable_id
from ocbrain.mcp_v1 import decide_proposal_v1
from ocbrain.scope import ScopeTag, fold_scope_id, resolve_scope_alias
from ocbrain.text import find_probable_secret_leaks, redact_secrets

from . import __version__
from .dag import (
    MIN_GOTCHA_CALLS,
    MIN_GOTCHA_FAILURE_RATE,
    mine_gotchas,
    mine_repairs,
    step_reliability,
)
from .episodes import BRAIN_DB, join_episodes, load_episodes, mining_set

MINER_VERSION = f"procmine/{__version__}"
WRITER = f"procmine:{__version__}"
BELIEF_TYPE = "gotcha"
EVIDENCE_KIND = "procmine_gotcha"
ATTRIBUTE_KIND = "gotcha.v1"

# At most twelve a run. Exactly twelve signatures clear the thresholds on the
# current corpus, and a cap that is not also a promise about the future is the
# only thing standing between a mining bug and a serving corpus full of noise.
MINT_LIMIT = 12

# A gotcha describes tooling, and tooling changes. Six weeks is long enough that
# a quarterly workflow is not swept mid-cycle and short enough that a claim
# nobody re-mined stops being served. `expired` is one of the two hygiene
# classes that may retire a belief unattended.
VALID_FOR_DAYS = 45

# Confidence is a Bayesian shrink of the *repair* success rate toward a coin
# flip. The failure rate is the claim, not the uncertainty: a step that fails 46%
# of 357 calls fails 46% of the time, full stop. What a reader actually gambles
# on is the remedy the claim names, so that is what confidence scores. A gotcha
# with no recurring repair lands at exactly 0.5 — trust the warning, not a way
# out of it.
CONFIDENCE_PRIOR_RATE = 0.5
CONFIDENCE_PRIOR_WEIGHT = 8.0

# How well the *scope attribution* is evidenced, not how well the counts are.
# The counts come from the whole corpus; only the project a gotcha is filed
# under depends on the closeout-to-trace join, and identity beats timing.
_JOIN_TIER_QUALITY = {
    "exact": 0.90,
    "uuid": 0.85,
    "temporal-context": 0.75,
    "temporal": 0.70,
    "temporal-ambiguous": 0.55,
}
# No episode carries this signature at all: the finding is corpus-wide and
# label-free, which is a real provenance and a weaker one than a joined episode.
LABEL_FREE_QUALITY = 0.65

FALLBACK_SCOPE_ID = "project:workspace"
# Bound the stats snapshot so one pathological session cannot write a megabyte
# of evidence body.
MAX_EVIDENCE_BYTES = 4_000


def _now() -> datetime:
    return datetime.now(UTC)


def gotcha_belief_id(signature: str, scope_id: str) -> str:
    """The stable identity of one gotcha.

    Derived from the signature and the scope and nothing else — not the counts,
    not the date, not the run. That is what makes a re-mint a replacement rather
    than an increment.
    """
    return stable_id("belief", BELIEF_TYPE, signature, scope_id)


def shrunk_confidence(successes: int, trials: int) -> float:
    if trials <= 0:
        return CONFIDENCE_PRIOR_RATE
    numerator = successes + CONFIDENCE_PRIOR_WEIGHT * CONFIDENCE_PRIOR_RATE
    return round(numerator / (trials + CONFIDENCE_PRIOR_WEIGHT), 4)


def _repair_trials(gotcha: dict[str, Any], repairs: dict[str, Any]) -> tuple[int, int]:
    """``(successes, trials)`` for whichever remedy recurs, repair before retry."""
    for key in ("repairs", "retries"):
        for row in repairs.get(key, []):
            if row["failing_step"] == gotcha["step"]:
                return int(row["repair_succeeded"]), int(row["pairs"])
    return 0, 0


def _signature_attribution(
    episodes: list[Any],
) -> dict[str, dict[str, Any]]:
    """Per signature: which project its episodes belong to, and how well joined.

    Built once over the mining set rather than per gotcha, because the mining
    set is small and the signature space is not.
    """
    table: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        quality = _JOIN_TIER_QUALITY.get(episode.join_tier)
        if quality is None:
            continue
        project = (episode.project or "").strip()
        for signature in {str(event["arg_signature"]) for event in episode.events}:
            bucket = table.setdefault(signature, {"projects": {}, "qualities": []})
            bucket["qualities"].append(quality)
            if project:
                bucket["projects"][project] = bucket["projects"].get(project, 0) + 1
    return table


def _scope_for(signature: str, attribution: dict[str, dict[str, Any]]) -> tuple[ScopeTag, float]:
    bucket = attribution.get(signature)
    if bucket is None or not bucket["qualities"]:
        return _scope_tag(FALLBACK_SCOPE_ID), LABEL_FREE_QUALITY
    projects: dict[str, int] = bucket["projects"]
    quality = round(sum(bucket["qualities"]) / len(bucket["qualities"]), 4)
    if not projects:
        return _scope_tag(FALLBACK_SCOPE_ID), quality
    # Dominant project, with the name as a tiebreaker so two equally-supported
    # projects do not make the scope depend on dict ordering.
    dominant = sorted(projects.items(), key=lambda item: (-item[1], item[0]))[0][0]
    scope_id = dominant if dominant.startswith("project:") else f"project:{dominant}"
    # Closeouts store whatever project string the writing agent typed on the
    # day, and historical rows keep those variants forever. Retrieval reaches a
    # stored scope only after folding the caller's string through the alias
    # table, so a gotcha minted into the raw variant lands where no caller can
    # match it. Canonicalize at write time -- the same repair the read side got.
    scope_id = resolve_scope_alias(fold_scope_id(scope_id))
    return _scope_tag(scope_id), quality


def _scope_tag(scope_id: str) -> ScopeTag:
    # local_only: a gotcha names internal tooling and internal step shapes. It is
    # never worth a hosted round trip, and a mined artifact must not widen its
    # own egress.
    return ScopeTag(
        "project",
        scope_id,
        visibility="internal",
        egress_policy="local_only",
        provenance="procmine",
    )


def _safe_text(text: str) -> str | None:
    """Redact, then refuse what still trips the detector.

    ``procmine.normalize`` already applies this to every fragment that reaches a
    signature or an error fingerprint. Running it again on the assembled claim
    and stats body is the cheap half of never being the component that wrote a
    secret into the serving corpus.
    """
    cleaned = redact_secrets(text)
    return None if find_probable_secret_leaks(cleaned) else cleaned


def _stats_snapshot(gotcha: dict[str, Any], successes: int, trials: int) -> dict[str, Any]:
    return {
        "schema_version": "ocbrain.procmine.gotcha-stats.v1",
        "step": gotcha["step"],
        "calls": gotcha["calls"],
        "failures": gotcha["failures"],
        "failure_rate": gotcha["failure_rate"],
        "sessions": gotcha["sessions"],
        "dominant_failure_class": gotcha["dominant_failure_class"],
        "by_runtime": gotcha["by_runtime"],
        "top_errors": gotcha["top_errors"][:3],
        "receipt_sessions": gotcha["receipt_sessions"][:5],
        "repair_successes": successes,
        "repair_trials": trials,
        "thresholds": {
            "min_calls": MIN_GOTCHA_CALLS,
            "min_failure_rate": MIN_GOTCHA_FAILURE_RATE,
        },
        "miner_version": MINER_VERSION,
    }


def build_candidates(
    traces: list[dict[str, Any]],
    *,
    brain_db: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Everything a mint would write, computed without touching the brain.

    Split from :func:`mint_gotchas` so the report-only path and the applying
    path cannot disagree about what would be written.
    """
    cap = MINT_LIMIT if limit is None else max(0, min(limit, MINT_LIMIT))
    repairs = mine_repairs(traces)
    reliability = step_reliability(traces)
    gotchas = mine_gotchas(reliability, repairs, limit=cap)

    attribution: dict[str, dict[str, Any]] = {}
    try:
        episodes = load_episodes(brain_db or BRAIN_DB)
    except sqlite3.Error:
        episodes = []
    if episodes:
        join_episodes(episodes, traces)
        mining_episodes, _attrition = mining_set(episodes)
        attribution = _signature_attribution(mining_episodes)

    mined_at = _now()
    valid_until = (mined_at + timedelta(days=VALID_FOR_DAYS)).isoformat()
    candidates: list[dict[str, Any]] = []
    for gotcha in gotchas:
        scope, quality = _scope_for(gotcha["step"], attribution)
        successes, trials = _repair_trials(gotcha, repairs)
        body = _safe_text(gotcha["claim"])
        snapshot = _stats_snapshot(gotcha, successes, trials)
        evidence_body = _safe_text(json.dumps(snapshot, sort_keys=True))
        if body is None or evidence_body is None:
            # A claim that still trips the leak detector after redaction is
            # dropped, never shipped. Same rule the signature layer follows.
            continue
        candidates.append(
            {
                "belief_id": gotcha_belief_id(gotcha["step"], scope.scope_id),
                "signature": gotcha["step"],
                "body": body,
                "scope": scope,
                "confidence": shrunk_confidence(successes, trials),
                "evidence_body": evidence_body[:MAX_EVIDENCE_BYTES],
                "attributes": {
                    "kind": ATTRIBUTE_KIND,
                    "signature": gotcha["step"],
                    "support": {
                        "calls": gotcha["calls"],
                        "failures": gotcha["failures"],
                        "sessions": gotcha["sessions"],
                        "repair_successes": successes,
                        "repair_trials": trials,
                    },
                    "mined_at": mined_at.isoformat(),
                    "miner_version": MINER_VERSION,
                    "lifecycle": "current",
                    "valid_until": valid_until,
                    "source_quality": quality,
                },
            }
        )
    return candidates


def mint_gotchas(
    traces: list[dict[str, Any]],
    *,
    brain_db: Path | None = None,
    apply: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Report, or write, one belief per mined gotcha.

    Report-only by default. ``apply`` is what the scheduled stage passes, and it
    is gated behind its own environment variable there, because a miner that
    writes to the serving corpus on every cycle without anyone asking is exactly
    the failure mode the deleted auto-compile path had.
    """
    candidates = build_candidates(traces, brain_db=brain_db, limit=limit)
    result: dict[str, Any] = {
        "schema_version": "ocbrain.procmine.mint.v1",
        "miner_version": MINER_VERSION,
        "writer": WRITER,
        "limit": MINT_LIMIT if limit is None else limit,
        "candidates": [
            {
                "belief_id": candidate["belief_id"],
                "signature": candidate["signature"],
                "scope_id": candidate["scope"].scope_id,
                "confidence": candidate["confidence"],
                "body": candidate["body"],
            }
            for candidate in candidates
        ],
        "applied": [],
        "blocked": [],
    }
    if not apply:
        result["applied_mode"] = "report_only"
        return result

    result["applied_mode"] = "apply"
    conn = connect(brain_db or BRAIN_DB)
    try:
        for candidate in candidates:
            outcome = _write_gotcha(conn, candidate)
            result[outcome[0]].append(outcome[1])
        conn.commit()
    finally:
        conn.close()
    return result


def _write_gotcha(
    conn: sqlite3.Connection, candidate: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    scope: ScopeTag = candidate["scope"]
    belief_id = candidate["belief_id"]
    evidence_id, _event_id = record_core_v1_evidence(
        conn,
        body=candidate["evidence_body"],
        kind=EVIDENCE_KIND,
        scope=scope,
        writer=WRITER,
        artifact_ref=f"procmine:gotcha:{candidate['signature']}",
    )
    proposal_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "schema_version": "ocbrain.compilation.v1",
            "subject": {"kind": "belief", "id": belief_id},
            "belief_id": belief_id,
            "belief_type": BELIEF_TYPE,
            "body": candidate["body"],
            "evidence_ids": [evidence_id],
            "scope": scope.to_dict(),
            "confidence": candidate["confidence"],
            "attributes": candidate["attributes"],
        },
        writer=WRITER,
    )
    try:
        decide_proposal_v1(
            conn,
            proposal_event_id=proposal_id,
            decision="approve",
            actor=WRITER,
            edited_body=None,
            reason="deterministic mint: claim generated from counts, not written",
        )
    except PermissionError as exc:
        # Retracted or tombstoned by a human. A miner must not walk that back.
        return "blocked", {"belief_id": belief_id, "reason": str(exc)}
    return "applied", {
        "belief_id": belief_id,
        "evidence_id": evidence_id,
        "scope_id": scope.scope_id,
        "confidence": candidate["confidence"],
    }


def gotcha_thresholds() -> dict[str, Any]:
    return {
        "min_calls": MIN_GOTCHA_CALLS,
        "min_failure_rate": MIN_GOTCHA_FAILURE_RATE,
        "mint_limit": MINT_LIMIT,
        "valid_for_days": VALID_FOR_DAYS,
    }


__all__ = [
    "BELIEF_TYPE",
    "EVIDENCE_KIND",
    "MINER_VERSION",
    "MINT_LIMIT",
    "VALID_FOR_DAYS",
    "WRITER",
    "build_candidates",
    "gotcha_belief_id",
    "gotcha_thresholds",
    "mint_gotchas",
    "shrunk_confidence",
]
