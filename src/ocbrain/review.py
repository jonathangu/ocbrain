"""Hermes-style post-turn background review of settled sessions (spec §6).

After a session file goes idle (``review.settle_minutes``), review walks its
turns and fires four triggers — task success, error recovery, user correction,
novel workflow — emitting session :class:`~ocbrain.autolabel.Signal` votes and,
for the productive triggers, :class:`~ocbrain.schema.Candidate` knowledge that is
upserted with ``origin='harvest'``, ``gate='auto'``, ``status='candidate'`` and
linked to the session's evidence rows. Admission to ``current`` happens later,
only once autolabel labels the candidate good (the no-clobber guard in
``upsert_knowledge`` protects any human row automatically).

Lane isolation: review does **not** parse transcripts. It consumes the frozen
``Session`` / ``Turn`` DTO shapes owned by ``ocbrain.dataset.transcripts`` (§7.1,
R2) via attribute access, so it develops against fixture sessions until lane 4
lands. The attribute contract it relies on:

* ``Session``: ``session_key: str``, ``path: str``, ``turns: list[Turn]``,
  optional ``agent``, ``occurred_at``, and either ``mtime_ns: int`` or
  ``last_activity_at: str`` for settle gating and ``fingerprint`` for the
  re-review watermark.
* ``Turn``: ``role: str`` ('user'|'assistant'|'tool'), ``text: str``, optional
  ``tool_calls: int``, ``is_error: bool``, ``occurred_at: str``.
"""

from __future__ import annotations

import re
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any

from ocbrain.autolabel import (
    Signal,
    get_watermark,
    record_signal,
    set_watermark,
)
from ocbrain.config import founder_weight
from ocbrain.db import link_knowledge_evidence, upsert_knowledge
from ocbrain.schema import Candidate, Risk, Scope, Target
from ocbrain.text import (
    AFFIRMATION_RE,
    claim_key,
    correction_score,
    summarize_text,
    title_from_text,
)

REVIEW_DOMAIN = "review"

APPROVAL_RE = re.compile(
    r"(?i)\b(yes|approved|lgtm|go ahead|do it|please do|sounds good|proceed)\b"
)
TEST_PASS_RE = re.compile(r"(\d+)\s+passed")
TEST_FAIL_RE = re.compile(r"FAILED|AssertionError|✕|\b\d+\s+failed\b")
DEPLOY_OK_RE = re.compile(
    r"(?i)(deploy(ment)?\s+(succeeded|successful|complete)|"
    r"fly deploy.*(success|complete)|CI (passed|green)|✓ Deployed)"
)
DEPLOY_FAIL_RE = re.compile(
    r"(?i)(deploy(ment)?\s+failed|fly deploy.*fail|CI (failed|red)|release_command failed)"
)
REVERT_RE = re.compile(r"(?i)(git revert|git reset --hard|rolling back|reverting)")


def is_settled(session: Any, cfg: Any, *, now_ns: int | None = None) -> bool:
    """True once the session has been idle >= ``review.settle_minutes`` (§6).

    Falls back to *settled* when the session exposes no activity timestamp — the
    caller (autopilot) is responsible for only handing over settled sessions in
    that case.
    """
    settle_seconds = cfg.review.settle_minutes * 60
    mtime_ns = getattr(session, "mtime_ns", None)
    if mtime_ns is not None:
        now_ns = now_ns if now_ns is not None else time.time_ns()
        return (now_ns - int(mtime_ns)) / 1e9 >= settle_seconds
    last_activity = getattr(session, "last_activity_at", None)
    if last_activity:
        parsed = _parse_ts(last_activity)
        if parsed is not None:
            idle = (datetime.now(UTC) - parsed).total_seconds()
            return idle >= settle_seconds
    return True


def review_sessions(
    conn: sqlite3.Connection,
    sessions: Any,
    cfg: Any,
    *,
    now: datetime | None = None,
    now_ns: int | None = None,
) -> dict[str, Any]:
    """Review every settled, changed session; return a MaintenanceResult summary."""
    signals = 0
    candidates = 0
    reviewed = 0
    for session in sessions:
        if not is_settled(session, cfg, now_ns=now_ns):
            continue
        fingerprint = str(getattr(session, "fingerprint", None) or _session_fingerprint(session))
        stream = str(getattr(session, "path", "") or getattr(session, "session_key", ""))
        if stream and get_watermark(conn, REVIEW_DOMAIN, stream) == fingerprint:
            continue
        result = review_session(conn, session, cfg, now=now)
        signals += result["signals"]
        candidates += result["candidates"]
        reviewed += 1
        if stream:
            set_watermark(conn, REVIEW_DOMAIN, stream, fingerprint)
    return {
        "action": "review",
        "changed": reviewed,
        "signals": signals,
        "candidates": candidates,
    }


def review_session(
    conn: sqlite3.Connection,
    session: Any,
    cfg: Any,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fire the four triggers over one session's turns (§6)."""
    del now
    turns = list(getattr(session, "turns", []) or [])
    session_key = str(getattr(session, "session_key", "") or getattr(session, "path", ""))
    path = str(getattr(session, "path", "") or session_key)
    threshold = cfg.correction.threshold

    signal_count = 0
    candidate_count = 0

    tool_turns = [t for t in turns if _role(t) == "tool"]
    tool_successes = [t for t in tool_turns if not _is_error(t)]
    tool_errors = [t for t in tool_turns if _is_error(t)]
    trailing_error = bool(tool_turns) and _is_error(tool_turns[-1])
    has_correction = False

    for index, turn in enumerate(turns):
        role = _role(turn)
        text = _text(turn)
        occurred = _occurred(turn)
        ref = f"{path}#{index}"

        if role == "user" and text:
            author = _author(turn)
            weight_mult = founder_weight(cfg, author)
            if correction_score(text) >= threshold and _assistant_within(turns, index, 3):
                signal_count += _emit(
                    conn, "user_correction", "bad", 0.8 * weight_mult, session_key, ref,
                    occurred,
                    _author_details(author, weight_mult, {"snippet": summarize_text(text, 200)}),
                )
                has_correction = True
            elif AFFIRMATION_RE.search(text):
                signal_count += _emit(
                    conn, "user_thanks", "good", 0.6 * weight_mult, session_key, ref, occurred,
                    _author_details(author, weight_mult, {}),
                )
            if (
                APPROVAL_RE.search(text)
                and index > 0
                and _role(turns[index - 1]) == "assistant"
            ):
                signal_count += _emit(
                    conn, "user_approval", "good", 0.5 * weight_mult, session_key, ref, occurred,
                    _author_details(author, weight_mult, {}),
                )

        if role == "tool" and text:
            if TEST_PASS_RE.search(text):
                signal_count += _emit(
                    conn, "test_pass", "good", 0.4, session_key, ref, occurred, {}
                )
            if TEST_FAIL_RE.search(text):
                signal_count += _emit(
                    conn, "test_fail", "bad", 0.6, session_key, ref, occurred, {}
                )
            if DEPLOY_OK_RE.search(text):
                signal_count += _emit(
                    conn, "deploy_success", "good", 0.6, session_key, ref, occurred, {}
                )
            if DEPLOY_FAIL_RE.search(text):
                signal_count += _emit(
                    conn, "deploy_failure", "bad", 0.6, session_key, ref, occurred, {}
                )

        if REVERT_RE.search(text):
            signal_count += _emit(
                conn, "revert", "bad", 0.7, session_key, ref, occurred, {}
            )

    # Error-recovery arc: an error followed later by a success, ending clean.
    recovered = bool(tool_errors) and bool(tool_successes) and not trailing_error and (
        _last_index(tool_turns, error=False) > _first_index(tool_turns, error=True)
    )
    if recovered:
        signal_count += _emit(
            conn, "error_recovery", "good", 0.6, session_key, f"{path}#recovery", None, {}
        )
        candidate_count += _upsert_candidate(
            conn,
            session,
            Candidate(
                target=Target.SKILL,
                title=_recovery_title(turns),
                body=_recovery_body(turns),
                confidence=0.55,
                scope=Scope.WORKSPACE,
                risk=Risk.LOW,
            ),
            marker="recovery",
        )

    # Task closeout success: enough clean tool work, no correction, no trailing error.
    if (
        len(tool_successes) >= cfg.review.min_tool_calls_success
        and not trailing_error
        and not has_correction
    ):
        signal_count += _emit(
            conn, "task_closeout_success", "good", 0.7, session_key, f"{path}#closeout", None,
            {"tool_successes": len(tool_successes)},
        )
        candidate_count += _upsert_candidate(
            conn,
            session,
            Candidate(
                target=Target.SKILL,
                title=_closeout_title(session, turns),
                body=_closeout_body(turns),
                confidence=0.6,
                scope=Scope.WORKSPACE,
                risk=Risk.LOW,
            ),
            marker="closeout",
        )

    # Novel workflow: a tool sequence not seen before for this session.
    if len(tool_successes) >= 3 and _is_novel_workflow(conn, session_key):
        signal_count += _emit(
            conn, "novel_workflow", "neutral", 0.2, session_key, f"{path}#novel", None, {}
        )
        candidate_count += _upsert_candidate(
            conn,
            session,
            Candidate(
                target=Target.WIKI,
                title=_closeout_title(session, turns),
                body=_closeout_body(turns),
                confidence=0.4,
                scope=Scope.WORKSPACE,
                risk=Risk.LOW,
            ),
            marker="novel",
        )

    return {"signals": signal_count, "candidates": candidate_count}


# --------------------------------------------------------------------------- #
# Emission helpers
# --------------------------------------------------------------------------- #
def _emit(
    conn: sqlite3.Connection,
    kind: str,
    polarity: str,
    weight: float,
    session_key: str,
    source_ref: str,
    occurred_at: str | None,
    details: dict[str, Any],
) -> int:
    record_signal(
        conn,
        Signal(
            kind=kind,
            polarity=polarity,
            weight=weight,
            source="session",
            source_ref=source_ref,
            session_key=session_key,
            details=details,
            occurred_at=occurred_at,
        ),
    )
    return 1


def _upsert_candidate(
    conn: sqlite3.Connection,
    session: Any,
    candidate: Candidate,
    *,
    marker: str,
) -> int:
    """Upsert a harvested Candidate as a knowledge row + link session evidence.

    Idempotent: the slug is derived from the session key + trigger marker, so a
    re-review updates in place rather than duplicating.
    """
    body = candidate.body.strip()
    if not body:
        return 0
    session_key = str(getattr(session, "session_key", "") or getattr(session, "path", ""))
    slug = f"rev:{marker}:{claim_key(session_key + body, limit=120)}"
    scope = candidate.scope.value

    if candidate.target == Target.SKILL:
        knowledge_id = upsert_knowledge(
            conn,
            knowledge_type="capability",
            gate="auto",
            slug=slug,
            title=candidate.title,
            value_text=None,
            status="candidate",
            risk=candidate.risk.value,
            confidence=candidate.confidence,
            privacy_scope=scope,
            origin="harvest",
            actor="ocbrain-review",
        )
    else:  # WIKI / POLICY -> doc
        knowledge_id = upsert_knowledge(
            conn,
            knowledge_type="doc",
            gate="auto",
            slug=slug,
            title=candidate.title,
            doc_kind="wiki",
            status="candidate",
            prescriptive=candidate.target == Target.POLICY,
            risk=candidate.risk.value,
            confidence=candidate.confidence,
            privacy_scope=scope,
            origin="harvest",
            actor="ocbrain-review",
        )

    _link_session_evidence(conn, knowledge_id, session)
    return 1


def _link_session_evidence(
    conn: sqlite3.Connection, knowledge_id: str, session: Any
) -> None:
    """Link the harvested candidate to the session's evidence rows if present.

    Evidence rows are created by the harvest stage keyed on ``source_uri`` = the
    transcript path; the privacy-scope ratchet fires automatically on link.
    """
    path = getattr(session, "path", None)
    if not path:
        return
    for row in conn.execute(
        "SELECT id FROM evidence WHERE source_uri = ?", (str(path),)
    ).fetchall():
        link_knowledge_evidence(conn, knowledge_id, row["id"], relation="derived_from")


def _is_novel_workflow(conn: sqlite3.Connection, session_key: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM signal_events WHERE session_key = ? AND kind = 'novel_workflow' LIMIT 1",
        (session_key,),
    ).fetchone()
    return row is None


# --------------------------------------------------------------------------- #
# Turn / session accessors (duck-typed against the §7.1 contract)
# --------------------------------------------------------------------------- #
def _role(turn: Any) -> str:
    return str(getattr(turn, "role", "") or "")


def _text(turn: Any) -> str:
    return str(getattr(turn, "text", "") or "")


def _author(turn: Any) -> str | None:
    value = getattr(turn, "authored_by", None)
    return str(value) if value else None


def _author_details(
    author: str | None, weight_mult: float, base: dict[str, Any]
) -> dict[str, Any]:
    """Fold author provenance into a signal's details when a founder authored it.

    A generic (weight 1.0) author adds nothing, keeping stable signal ids and the
    existing fixtures unchanged; a founder stamps ``authored_by`` + ``author_weight``
    so the label fold's weight is auditable to the person who spoke.
    """
    if author and weight_mult != 1.0:
        return {**base, "authored_by": author, "author_weight": weight_mult}
    return base


def _is_error(turn: Any) -> bool:
    return bool(getattr(turn, "is_error", False) or getattr(turn, "tool_error", False))


def _occurred(turn: Any) -> str | None:
    value = getattr(turn, "occurred_at", None)
    return str(value) if value else None


def _assistant_within(turns: list[Any], index: int, window: int) -> bool:
    for j in range(max(0, index - window), index):
        if _role(turns[j]) == "assistant":
            return True
    return False


def _first_index(tool_turns: list[Any], *, error: bool) -> int:
    for i, turn in enumerate(tool_turns):
        if _is_error(turn) == error:
            return i
    return len(tool_turns)


def _last_index(tool_turns: list[Any], *, error: bool) -> int:
    last = -1
    for i, turn in enumerate(tool_turns):
        if _is_error(turn) == error:
            last = i
    return last


def _session_fingerprint(session: Any) -> str:
    turns = getattr(session, "turns", []) or []
    mtime = getattr(session, "mtime_ns", None)
    return f"{len(turns)}:{mtime}"


def _closeout_title(session: Any, turns: list[Any]) -> str:
    agent = getattr(session, "agent", None)
    last_user = _last_text(turns, "user")
    base = last_user or f"session {getattr(session, 'session_key', '')}"
    prefix = f"{agent}: " if agent else ""
    return title_from_text(prefix + base, "Reviewed workflow")


def _closeout_body(turns: list[Any]) -> str:
    last_assistant = _last_text(turns, "assistant")
    return summarize_text(last_assistant or "Reviewed a successful task workflow.", 600)


def _recovery_title(turns: list[Any]) -> str:
    return title_from_text(_last_text(turns, "assistant") or "Error recovery", "Error recovery")


def _recovery_body(turns: list[Any]) -> str:
    return summarize_text(
        _last_text(turns, "assistant") or "Recovered from an error with a different approach.",
        600,
    )


def _last_text(turns: list[Any], role: str) -> str:
    for turn in reversed(turns):
        if _role(turn) == role and _text(turn).strip():
            return _text(turn)
    return ""


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
