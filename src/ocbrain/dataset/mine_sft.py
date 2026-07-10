"""SFT exchange mining (spec §7.2 SFT).

An SFT example is a context window (≤``sft_max_context_turns`` non-injected
turns, ≤``sft_max_context_chars`` chars, head-trimmed) ending in a final
assistant text turn (≥``sft_min_assistant_chars``). Rule labels: affirmation
follow-up (good 0.9), Hermes task-success ≥5 clean tool calls (good 0.7),
error-recovery arc (good 0.8); correction/refusal/terminal-failure/abandonment
(bad, retained for DPO, never exported to SFT); else neutral (0.5). Linked
``retrieval_uses`` good outcomes nudge +0.1; harmful/failed force bad.

Sessions whose only user turns are injected yield nothing (kills the
orchestrator→subagent lanes as SFT signal — correct per spec).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.dataset.batching import DatasetWriteBatch
from ocbrain.dataset.quality import _REFUSAL_RE, store_example
from ocbrain.dataset.transcripts import (
    Session,
    iter_unmined_transcripts,
    parse_transcript,
    record_source,
    resolve_transcript_evidence,
)
from ocbrain.fsutil import ParseCache, parse_cache_key
from ocbrain.text import AFFIRMATION_RE, correction_score

_GOOD_RETRIEVAL = {"improved", "helpful", "used"}
_BAD_RETRIEVAL = {"harmful", "failed"}


@dataclass(frozen=True)
class Exchange:
    context: tuple[dict[str, str], ...]  # [{role, content}, ...] before the target
    target_text: str
    trigger_idx: int
    target_idx: int
    n_tool_calls: int
    trailing_tool_error: bool
    had_tool_error: bool
    occurred_at: str | None
    reasons: list[str] = field(default_factory=list)


def _exported_context(session: Session, upto: int) -> list[dict[str, str]]:
    """Non-injected user + assistant text turns before ``upto`` (export order)."""
    out: list[dict[str, str]] = []
    for turn in session.turns[:upto]:
        text = turn.text.strip()
        if not text:
            continue
        if turn.role == "user" and turn.kind != "injected":
            out.append({"role": "user", "content": text})
        elif turn.role == "assistant":
            out.append({"role": "assistant", "content": text})
    return out


def _head_trim(
    context: list[dict[str, str]], max_turns: int, max_chars: int
) -> list[dict[str, str]]:
    trimmed = context[-max_turns:] if max_turns > 0 else list(context)
    while trimmed and sum(len(m["content"]) for m in trimmed) > max_chars:
        trimmed.pop(0)
    return trimmed


def segment_exchanges(session: Session, cfg: OcbrainConfig | None = None) -> list[Exchange]:
    cfg = cfg or load_config()
    ds = cfg.dataset
    turns = session.turns
    n = len(turns)
    user_idxs = [k for k, t in enumerate(turns) if t.role == "user"]
    exchanges: list[Exchange] = []
    for gi, u in enumerate(user_idxs):
        end = user_idxs[gi + 1] if gi + 1 < len(user_idxs) else n
        target_idx: int | None = None
        for k in range(u + 1, end):
            t = turns[k]
            if t.role == "assistant" and len(t.text.strip()) >= ds.sft_min_assistant_chars:
                target_idx = k
        if target_idx is None:
            continue
        # Tool-call accounting + error arc within the request→target span.
        n_tool_calls = 0
        had_error = False
        trailing_error = False
        for k in range(u + 1, target_idx + 1):
            t = turns[k]
            if t.role == "assistant":
                n_tool_calls += t.n_tool_calls
                trailing_error = False
            elif t.role == "tool":
                n_tool_calls += 1
                had_error = had_error or t.tool_error
                trailing_error = t.tool_error
        context = _head_trim(
            _exported_context(session, target_idx),
            ds.sft_max_context_turns,
            ds.sft_max_context_chars,
        )
        if not any(m["role"] == "user" for m in context):
            continue  # injected-only trigger: no human request survives
        exchanges.append(
            Exchange(
                context=tuple(context),
                target_text=turns[target_idx].text.strip(),
                trigger_idx=u,
                target_idx=target_idx,
                n_tool_calls=n_tool_calls,
                trailing_tool_error=trailing_error,
                had_tool_error=had_error,
                occurred_at=turns[target_idx].ts or session.occurred_at,
            )
        )
    return exchanges


def _next_user_text(session: Session, after: int) -> str | None:
    for turn in session.turns[after + 1 :]:
        if turn.role == "user" and turn.kind != "injected":
            return turn.text
    return None


def label_exchange(
    session: Session,
    exchange: Exchange,
    cfg: OcbrainConfig | None = None,
    *,
    retrieval_outcomes: Iterable[str] = (),
) -> tuple[str, float, list[str]]:
    cfg = cfg or load_config()
    reasons: list[str] = []
    outcomes = set(retrieval_outcomes)
    threshold = cfg.correction.threshold

    follow = _next_user_text(session, exchange.target_idx)
    corrected = bool(follow) and correction_score(follow or "") >= threshold
    affirmed = bool(follow) and not corrected and bool(AFFIRMATION_RE.search(follow or ""))
    refused = bool(_REFUSAL_RE.match(exchange.target_text.strip()))

    # Bad precedence (retained, feeds DPO, never exported to SFT).
    if outcomes & _BAD_RETRIEVAL:
        reasons.append("retrieval_bad")
        return "bad", 0.6, reasons
    if corrected:
        reasons.append("correction_followup")
        return "bad", min(0.9, correction_score(follow or "")), reasons
    if refused:
        reasons.append("refusal")
        return "bad", 0.6, reasons
    if exchange.trailing_tool_error:
        reasons.append("terminal_tool_failure")
        return "bad", 0.6, reasons

    # Good tiers.
    if affirmed:
        reasons.append("affirmation")
        conf = 0.9
        if outcomes & _GOOD_RETRIEVAL:
            reasons.append("retrieval_good")
            conf = min(0.95, conf + 0.1)
        return "good", conf, reasons
    if exchange.had_tool_error and not exchange.trailing_tool_error:
        reasons.append("error_recovery")
        conf = 0.8
        if outcomes & _GOOD_RETRIEVAL:
            reasons.append("retrieval_good")
            conf = min(0.95, conf + 0.1)
        return "good", conf, reasons
    if exchange.n_tool_calls >= cfg.review.min_tool_calls_success and not exchange.had_tool_error:
        reasons.append("task_success")
        conf = 0.7
        if outcomes & _GOOD_RETRIEVAL:
            reasons.append("retrieval_good")
            conf = min(0.95, conf + 0.1)
        return "good", conf, reasons

    if outcomes & _GOOD_RETRIEVAL:
        reasons.append("retrieval_good")
        return "good", 0.6, reasons
    reasons.append("neutral")
    return "neutral", 0.5, reasons


def sft_messages(exchange: Exchange) -> list[dict[str, str]]:
    messages = [dict(m) for m in exchange.context]
    messages.append({"role": "assistant", "content": exchange.target_text})
    return messages


def mine_sft(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    sessions: Iterable[Session] | None = None,
    roots: Iterable[str] | None = None,
    retrieval_by_session: dict[str, list[str]] | None = None,
    limit: int | None = None,
    time_budget_seconds: float | None = None,
    parse_cache: ParseCache | None = None,
    write_batch: DatasetWriteBatch | None = None,
) -> dict[str, Any]:
    """Mine SFT examples from ``sessions`` (or discovered transcripts).

    Returns a ``MaintenanceResult``-shaped dict of counts. When ``roots`` is
    given, only new/changed transcripts are parsed and each fully-processed file
    advances its ``dataset_sources`` fingerprint (spec §4.3).
    """
    cfg = cfg or load_config()
    retrieval_by_session = retrieval_by_session or {}
    started = time.monotonic()
    stored = 0
    excluded = 0
    examined = 0
    files_mined = 0
    batch = write_batch or DatasetWriteBatch(
        conn,
        max_operations=cfg.dataset.write_batch_size,
        max_seconds=cfg.dataset.write_batch_seconds,
    )

    def _emit(session: Session) -> int:
        nonlocal stored, excluded, examined
        count = 0
        evidence_id, scope = resolve_transcript_evidence(conn, session, write_batch=batch)
        for exchange in segment_exchanges(session, cfg):
            examined += 1
            label, confidence, reasons = label_exchange(
                session,
                exchange,
                cfg,
                retrieval_outcomes=retrieval_by_session.get(session.session_id, []),
            )
            result = store_example(
                conn,
                dataset="sft",
                source_kind=session.source_kind,
                source_uri=session.source_uri,
                source_span={"target_idx": exchange.target_idx},
                evidence_ids=[evidence_id],
                privacy_scope=scope,
                body={"messages": sft_messages(exchange)},
                metadata={
                    "n_tool_calls": exchange.n_tool_calls,
                    "sender_verified": False,
                    "session_id": session.session_id,
                    "source_kind": session.source_kind,
                },
                target_text=exchange.target_text,
                base_label=label,
                base_confidence=confidence,
                base_reasons=reasons,
                n_turns=len(exchange.context) + 1,
                session_id=session.session_id,
                occurred_at=exchange.occurred_at,
                write_batch=batch,
            )
            if result is None:
                continue
            if result["quality_label"] == "excluded":
                excluded += 1
            else:
                stored += 1
            count += 1
        return count

    if sessions is not None:
        for session in sessions:
            if limit is not None and examined >= limit:
                break
            _emit(session)
            batch.flush()
    elif roots is not None:
        ds = cfg.dataset
        # founder_ids=() here mirrors the SFT parse below; the tuple lets the
        # run-shared memo reuse a persona/DPO parse of the same file when no
        # founder is configured (identical Session), and stay distinct otherwise.
        params = (
            tuple(ds.persona_author_ids),
            tuple(ds.persona_direct_agents),
            ds.tool_result_truncate,
            (),
        )
        for path, fingerprint in iter_unmined_transcripts(conn, roots, "sft"):
            if time_budget_seconds is not None and time.monotonic() - started > time_budget_seconds:
                break

            def _load(p: object = path) -> Session | None:
                return parse_transcript(
                    p,  # type: ignore[arg-type]
                    author_ids=ds.persona_author_ids,
                    direct_agents=ds.persona_direct_agents,
                    tool_result_truncate=ds.tool_result_truncate,
                )

            if parse_cache is not None:
                session = parse_cache.get(parse_cache_key(fingerprint, params), _load)
            else:
                session = _load()
            emitted = 0 if session is None else _emit(session)
            batch.ensure()
            record_source(conn, str(path), "sft", fingerprint, emitted)
            batch.operation()
            # A source watermark closes its file boundary. Partial earlier
            # batches are safe: reruns deduplicate them if a later batch fails.
            batch.flush()
            files_mined += 1
            if limit is not None and files_mined >= limit:
                break

    batch.flush()
    return {
        "ok": True,
        "dataset": "sft",
        "examined": examined,
        "stored": stored,
        "excluded": excluded,
        "files_mined": files_mined,
        "writer_lock": batch.metrics(),
    }
