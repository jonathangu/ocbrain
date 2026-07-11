"""Budget-capped, egress-audited hosted LLM judge (spec §5.4, R6).

The judge is an *optional* tie-breaker: it looks only at knowledge rows the
decayed fold left ambiguous (``quality_label='neutral'`` with real conflicting
mass) plus zero-signal promotion candidates, and asks a hosted model for a
good/neutral/bad verdict. Its verdicts fold back as ordinary ``llm_judge``
signals (weight ``judge.signal_weight`` = 0.4) — they can never override
hard-human signals (§5.3 hard-bad precedence).

Hard guarantees (R6/R8):

* **Private scope never leaves.** Rows whose composed scope is ``private`` are
  dropped before any dispatch, as are rows failing ``egress_allowed(..., 'hosted_teacher')``.
* **Redaction before send.** Every dispatched body passes through ``redact_secrets``.
* **Every batch is audited.** A row is written to ``egress_audits`` (included +
  rejected ids + payload hash) via ``record_egress_audit``.
* **Budget capped.** When today's spend reaches ``judge.daily_usd_cap`` the run
  records ``judge_runs.status='skipped_budget'`` and dispatches nothing.
* **Inert without a key.** With ``judge.api_key_env`` unset (or ``enabled=false``)
  the judge records a ``skipped`` run and never touches the network.
* **Verdicts only at rest.** ``judge_runs.response_json`` stores the parsed
  verdicts, never the dispatched bodies. The API key is read from the process
  environment and never persisted, logged, or printed.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from typing import Any

from ocbrain.autolabel import Signal, decayed_mass, record_signal, signals_for
from ocbrain.db import knowledge_evidence, most_restrictive_scope, now_iso
from ocbrain.egress import record_egress_audit
from ocbrain.events import canonical_json, sha256_text
from ocbrain.ids import stable_id
from ocbrain.scope import (
    DEFAULT_GLOBAL_SCOPE_ID,
    ScopeContext,
    ScopeTag,
    egress_allowed,
)
from ocbrain.text import redact_secrets

logger = logging.getLogger(__name__)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

_JUDGE_SYSTEM = (
    "You are a strict knowledge-quality judge. For each item decide whether the "
    "claim is a durable, correct, useful fact worth keeping. Reply with ONLY a "
    "JSON array of objects, one per input id, each: "
    '{"id": <id>, "label": "good"|"neutral"|"bad", "confidence": <0..1>, '
    '"rationale": <short string>}. No prose outside the JSON.'
)

_VERDICT_POLARITY = {"good": "good", "bad": "bad", "neutral": "neutral"}

# --------------------------------------------------------------------------- #
# Candidate targeting (v0.3)
# --------------------------------------------------------------------------- #
# The overnight run drained hosted-judge spend grading a 101k never-referenced
# file-catalog doc backlog. Targeting restricts the graded set to rows with real
# provenance: each ``judge.targeting['sources']`` token maps to a SQL predicate
# over the ``knowledge`` alias ``k``; a row is a candidate if it satisfies ANY
# enabled predicate. Unknown tokens are ignored. With no known token the filter
# is inert (matches everything) so the judge never silently grades nothing.
#
# * ``retrieval_touched`` — the row has been served at least once (any
#   ``retrieval_uses`` row). Worth judging regardless of how it was minted.
# * ``lesson`` — review-/loop-distilled knowledge (``origin`` in ``harvest`` or
#   ``loop``), i.e. lessons extracted from work, not raw file imports.
# * ``session_derived`` — distilled (non-``doc``) knowledge (values / capabilities
#   surfaced from working sessions), as opposed to raw catalog documents.
_TARGETING_PREDICATES: dict[str, str] = {
    "retrieval_touched": ("EXISTS (SELECT 1 FROM retrieval_uses ru WHERE ru.knowledge_id = k.id)"),
    "lesson": "k.origin IN ('harvest', 'loop')",
    "session_derived": "k.type <> 'doc'",
}

# A "catalog doc" is a raw imported document (memory / history / file catalog),
# NOT a distilled wiki/procedure — those carry ``origin`` harvest/loop. When
# ``exclude_catalog_docs`` is set, catalog docs that no retrieval has ever
# touched are dropped from the graded set. Human-authored docs are never
# excluded here.
_CATALOG_DOC_PREDICATE = (
    "k.type = 'doc' "
    "AND (k.origin IS NULL OR k.origin NOT IN ('harvest', 'loop', 'human')) "
    "AND NOT EXISTS (SELECT 1 FROM retrieval_uses ru2 WHERE ru2.knowledge_id = k.id)"
)


def _targeting_clause(cfg: Any) -> str:
    """SQL fragment (may be empty) restricting judge candidates per targeting cfg.

    Returned text is composed only of trusted config-keyed literals, never user
    input, so it is safe to inline into the candidate query.
    """
    targeting = getattr(cfg.judge, "targeting", None) or {}
    sources = targeting.get("sources") or []
    exclude_catalog = bool(targeting.get("exclude_catalog_docs"))

    clauses: list[str] = []
    predicates = [
        _TARGETING_PREDICATES[token] for token in sources if token in _TARGETING_PREDICATES
    ]
    if predicates:
        clauses.append("(" + " OR ".join(predicates) + ")")
    if exclude_catalog:
        clauses.append("NOT (" + _CATALOG_DOC_PREDICATE + ")")
    if not clauses:
        return ""
    return " AND " + " AND ".join(clauses)


def spent_today(conn: sqlite3.Connection, *, now: datetime | None = None) -> float:
    """Sum ``judge_runs.cost_usd`` for the current UTC day."""
    now = now or datetime.now(UTC)
    day = now.date().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM judge_runs WHERE substr(ts, 1, 10) = ?",
        (day,),
    ).fetchone()
    return float(row["spent"] or 0.0)


def scope_tag_for_privacy(scope: str) -> ScopeTag:
    """Map the relational privacy ladder to a ScopeTag for egress checks (R8).

    ``private`` becomes a confidential/local-only tag (hosted egress denied);
    everything else maps to the hosted-eligible global doctrine scope. Callers
    still drop ``private`` explicitly before dispatch — this keeps the
    ``egress_allowed`` gate honest as a second line of defense.
    """
    if scope == "private":
        return ScopeTag(
            scope_type="global",
            scope_id=DEFAULT_GLOBAL_SCOPE_ID,
            visibility="confidential",
            egress_policy="local_only",
        )
    return ScopeTag(
        scope_type="global",
        scope_id=DEFAULT_GLOBAL_SCOPE_ID,
        visibility="internal",
        egress_policy="hosted_ok",
    )


def composed_scope(conn: sqlite3.Connection, row: sqlite3.Row) -> str:
    """Most-restrictive scope over the knowledge row and all linked evidence."""
    scopes = [row["privacy_scope"]]
    for evidence in knowledge_evidence(conn, row["id"]):
        scopes.append(evidence["privacy_scope"])
    return most_restrictive_scope(*scopes)


def _row_text(row: sqlite3.Row) -> str:
    parts = [
        row["title"],
        row["subject"],
        row["predicate"],
        row["value_text"],
    ]
    return " ".join(str(p) for p in parts if p)


def eligible_rows(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    now: datetime | None = None,
) -> list[sqlite3.Row]:
    """Rows worth judging: ambiguous-with-mass neutrals + zero-signal candidates.

    Restricted to provenance-bearing rows per ``judge.targeting`` so hosted spend
    is never wasted grading the never-referenced file-catalog doc backlog. Ordered
    by ``promote_score`` desc, capped at ``judge.per_run_item_cap``.
    """
    now = now or datetime.now(UTC)
    candidates = conn.execute(
        f"""
        SELECT k.*
        FROM knowledge AS k
        WHERE k.quarantine_reason IS NULL
          AND k.status IN ('candidate', 'current')
          AND (k.quality_label = 'neutral' OR k.quality_label IS NULL)
          {_targeting_clause(cfg)}
        ORDER BY COALESCE(k.promote_score, -1) DESC,
                 COALESCE(k.confidence, 0) DESC,
                 k.id ASC
        """  # noqa: S608 - fragment is composed only of trusted config-keyed literals
    ).fetchall()

    chosen: list[sqlite3.Row] = []
    for row in candidates:
        signals = signals_for(conn, row["id"])
        mass = decayed_mass(signals, cfg, now=now)
        if row["quality_label"] == "neutral":
            # Ambiguous only if there is real conflicting evidence.
            if mass >= 0.3:
                chosen.append(row)
        elif not signals:
            # Zero-signal promotion candidate — the judge bootstraps a label.
            chosen.append(row)
        if len(chosen) >= cfg.judge.per_run_item_cap:
            break
    return chosen


def build_judge_batch(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    *,
    context: ScopeContext | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(included, rejected)`` items after scope + egress + redaction.

    ``included`` items carry the *redacted* text to dispatch; ``rejected`` items
    carry only ids + reason and are never sent.
    """
    context = context or ScopeContext()
    included: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        scope = composed_scope(conn, row)
        if scope == "private":
            rejected.append({"id": row["id"], "reason": "private_scope"})
            continue
        scope_tag = scope_tag_for_privacy(scope)
        allowed, reason = egress_allowed(scope_tag, context, "hosted_teacher")
        if not allowed:
            rejected.append({"id": row["id"], "reason": reason})
            continue
        redacted = redact_secrets(_row_text(row))
        included.append({"id": row["id"], "scope": scope, "text": redacted})
    return included, rejected


def call_openai(
    payload: dict[str, Any],
    *,
    api_key: str,
    model: str,
    url: str = OPENAI_URL,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """POST a chat-completions request via stdlib urllib (no third-party deps).

    Injected as ``call=`` in tests so the network is never touched there.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed https OpenAI endpoint
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:  # pragma: no cover - network failure path
        raise RuntimeError(f"judge call failed: {exc}") from exc


def _is_timeout_error(exc: BaseException) -> bool:
    """True for transient read/socket timeouts worth one retry.

    ``call_openai`` lets a socket read timeout surface as ``TimeoutError``
    (``socket.timeout`` is its alias) and wraps ``URLError`` connection failures
    into a ``RuntimeError`` whose message carries the underlying reason, so we
    match both the type and the 'timed out' text.
    """
    if isinstance(exc, TimeoutError):
        return True
    return "timed out" in str(exc).lower()


def _call_with_timeout_retry(
    call: Any,
    payload: dict[str, Any],
    *,
    api_key: str,
    model: str,
    sleep: Callable[[float], None] = time.sleep,
    max_retries: int = 1,
    backoff_seconds: float = 2.0,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Invoke ``call`` with bounded exponential retry on timeout errors.

    Non-timeout failures (and a timeout that persists past the retry budget)
    propagate unchanged.
    """
    attempt = 0
    while True:
        try:
            return call(payload, api_key=api_key, model=model)
        except Exception as exc:  # noqa: BLE001 - re-raised unless a retryable timeout
            if attempt >= max_retries or not _is_timeout_error(exc):
                raise
            delay = backoff_seconds * (2**attempt)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise
            attempt += 1
            logger.warning(
                "judge: hosted call timed out (attempt %d/%d); retrying after backoff",
                attempt,
                max_retries + 1,
            )
            sleep(delay)


def _knowledge_exists(conn: sqlite3.Connection, knowledge_id: Any) -> bool:
    """Whether ``knowledge_id`` still names a live ``knowledge`` row."""
    return (
        conn.execute("SELECT 1 FROM knowledge WHERE id = ? LIMIT 1", (knowledge_id,)).fetchone()
        is not None
    )


def _cost_usd(cfg: Any, model: str, usage: dict[str, Any]) -> float:
    prices = cfg.judge.price_per_mtok.get(model)
    if not prices:
        return 0.0
    prompt = float(usage.get("prompt_tokens", 0) or 0)
    completion = float(usage.get("completion_tokens", 0) or 0)
    return prompt / 1_000_000 * float(prices.get("prompt", 0.0)) + completion / 1_000_000 * float(
        prices.get("completion", 0.0)
    )


def _parse_verdicts(response: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return []
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("verdicts") or parsed.get("items") or []
    if not isinstance(parsed, list):
        return []
    verdicts: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict) and item.get("id") and item.get("label"):
            verdicts.append(item)
    return verdicts


def _record_run(
    conn: sqlite3.Connection,
    *,
    status: str,
    model: str,
    item_count: int = 0,
    request_hash: str | None = None,
    usage: dict[str, Any] | None = None,
    cost_usd: float = 0.0,
    verdicts: list[dict[str, Any]] | None = None,
    egress_audit_id: str | None = None,
    ts: str | None = None,
) -> str:
    ts = ts or now_iso()
    run_id = stable_id("judge", request_hash or status, ts)
    usage = usage or {}
    conn.execute(
        """
        INSERT OR IGNORE INTO judge_runs (
          id, ts, model, status, item_count, request_hash, prompt_tokens,
          completion_tokens, cost_usd, response_json, egress_audit_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            ts,
            model,
            status,
            item_count,
            request_hash,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            cost_usd,
            canonical_json(verdicts) if verdicts is not None else None,
            egress_audit_id,
        ),
    )
    return run_id


def judge_ambiguous(
    conn: sqlite3.Connection,
    cfg: Any,
    *,
    call: Any = call_openai,
    now: datetime | None = None,
    env: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Judge ambiguous rows within budget; fold verdicts as ``llm_judge`` signals.

    Returns a MaintenanceResult-shaped summary. Never raises for the ordinary
    disabled / no-key / budget / no-eligible paths.
    """
    now = now or datetime.now(UTC)
    # ``os.environ`` is a ``Mapping[str, str]`` (not a plain ``dict``); resolve
    # to a concrete, always-non-None mapping so the lookup below can never deref
    # ``None``.
    resolved_env: Mapping[str, str] = os.environ if env is None else env
    model = cfg.judge.model
    deadline = None if time_budget_seconds is None else time.monotonic() + time_budget_seconds

    if not cfg.judge.enabled:
        _record_run(conn, status="skipped", model=model)
        return {"action": "judge", "changed": 0, "status": "skipped", "reason": "disabled"}

    api_key = resolved_env.get(cfg.judge.api_key_env)
    if not api_key:
        _record_run(conn, status="skipped", model=model)
        return {"action": "judge", "changed": 0, "status": "skipped", "reason": "no_api_key"}

    if spent_today(conn, now=now) >= cfg.judge.daily_usd_cap:
        _record_run(conn, status="skipped_budget", model=model)
        return {"action": "judge", "changed": 0, "status": "skipped_budget"}

    rows = eligible_rows(conn, cfg, now=now)
    if not rows:
        _record_run(conn, status="ok", model=model)
        return {"action": "judge", "changed": 0, "status": "ok"}

    total_cost = spent_today(conn, now=now)
    folded = 0
    skipped_missing = 0
    batches = 0
    for start in range(0, len(rows), cfg.judge.batch_size):
        batch = rows[start : start + cfg.judge.batch_size]
        included, rejected = build_judge_batch(conn, batch)

        audit_id = _write_egress_audit(conn, included, rejected)
        # Make the audit durable and release SQLite before hosted inference.
        conn.commit()
        if not included:
            _record_run(
                conn,
                status="skipped_egress",
                model=model,
                item_count=0,
                egress_audit_id=audit_id,
            )
            conn.commit()
            continue

        payload = _build_payload(cfg, model, included)
        request_hash = sha256_text(canonical_json(payload))
        provider_call = call
        if call is call_openai:
            provider_call = partial(call_openai, timeout=max(1.0, float(cfg.judge.timeout_seconds)))
        response = _call_with_timeout_retry(
            provider_call,
            payload,
            api_key=api_key,
            model=model,
            sleep=sleep,
            max_retries=max(0, int(cfg.judge.timeout_max_retries)),
            backoff_seconds=max(0.0, float(cfg.judge.retry_backoff_seconds)),
            deadline=deadline,
        )
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        cost = _cost_usd(cfg, model, usage)
        verdicts = _parse_verdicts(response)

        _record_run(
            conn,
            status="ok",
            model=model,
            item_count=len(included),
            request_hash=request_hash,
            usage=usage,
            cost_usd=cost,
            verdicts=verdicts,
            egress_audit_id=audit_id,
            ts=now_iso(),
        )
        fold_n, skip_n = _fold_verdicts(conn, cfg, verdicts)
        # Bound the post-response write window to one judge batch.
        conn.commit()
        folded += fold_n
        skipped_missing += skip_n
        batches += 1
        total_cost += cost
        if total_cost >= cfg.judge.daily_usd_cap:
            break

    return {
        "action": "judge",
        "changed": folded,
        "status": "ok",
        "batches": batches,
        "skipped_missing": skipped_missing,
    }


def _supports_custom_temperature(model: str) -> bool:
    """Whether ``model`` accepts a non-default ``temperature``.

    GPT-5 and the o-series reasoning models only support the default
    ``temperature`` (1); sending ``temperature: 0`` returns HTTP 400
    ``unsupported_value``. For those we omit the field entirely.
    """
    name = model.lower()
    return not (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    )


def _build_payload(cfg: Any, model: str, included: list[dict[str, Any]]) -> dict[str, Any]:
    items = [{"id": item["id"], "text": item["text"]} for item in included]
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(items)},
        ],
    }
    if _supports_custom_temperature(model):
        payload["temperature"] = 0
    return payload


def _write_egress_audit(
    conn: sqlite3.Connection,
    included: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> str:
    payload_text = "\n\n".join(item["text"] for item in included)
    result = {
        "target": "hosted_teacher",
        "context": {},
        "query": None,
        "included": [
            {"id": item["id"], "scope": item["scope"], "body": item["text"]} for item in included
        ],
        "rejected": rejected,
        "payload_hash": sha256_text(payload_text),
    }
    return record_egress_audit(conn, result)


def _fold_verdicts(
    conn: sqlite3.Connection, cfg: Any, verdicts: list[dict[str, Any]]
) -> tuple[int, int]:
    """Fold verdicts into ``llm_judge`` signals. Returns ``(folded, skipped)``.

    A verdict id round-trips through a hosted model and may be *stale* (the row
    was archived/deleted between selection and fold) or *hallucinated* (an id we
    never sent). ``signal_events.knowledge_id`` carries a FK to ``knowledge(id)``
    and SQLite's ``INSERT OR IGNORE`` does NOT suppress FK violations, so folding
    an unknown id raises 'FOREIGN KEY constraint failed' and aborts the whole
    run. We skip-and-log those instead so real verdicts still land.
    """
    folded = 0
    skipped = 0
    for verdict in verdicts:
        polarity = _VERDICT_POLARITY.get(str(verdict.get("label")))
        knowledge_id = verdict.get("id")
        if polarity is None or not knowledge_id:
            continue
        if not _knowledge_exists(conn, knowledge_id):
            skipped += 1
            logger.warning(
                "judge: skipping verdict for unknown knowledge id "
                "(stale/archived/hallucinated); not folding"
            )
            continue
        record_signal(
            conn,
            Signal(
                kind="llm_judge",
                polarity=polarity,
                weight=cfg.judge.signal_weight,
                source="judge",
                source_ref=f"judge:{knowledge_id}",
                knowledge_id=knowledge_id,
                details={
                    "label": polarity,
                    "confidence": verdict.get("confidence"),
                },
            ),
        )
        folded += 1
    return folded, skipped
