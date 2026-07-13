"""DPO preference-pair mining (spec §7.2 DPO).

Two sources:

* **Transcript pairs** — ``A1 → U(correction ≥ correction.threshold) → … An``:
  prompt is the context through the ORIGINAL request (correction excluded),
  ``rejected`` is the first attempt ``A1``, ``chosen`` is the final accepted
  attempt (walk forward while corrections continue; accept on affirmation or
  clean end). Both sides must differ by ``claim_key``, fit
  ``dpo_side_chars``, and survive the secret/injection scrub. Under
  ``cfg.dataset.dpo_relaxed_gate`` (v0.3), an *additive* relaxed pass anchored on
  the correction turn also admits pairs the strict structure rejects — a
  correction that states the fix as the thread's last word (no accepted answer
  follows) and a correction that lands several turns after the answer it refers
  to. Every relaxed pair is tagged ``gate='relaxed'`` in metadata; strict pairs
  are tagged ``gate='strict'``. The relaxed pass never restates a strict pair.
* **Event-core pairs** (0 today, grow as autopilot runs) — ``compilation_decided``
  edits, ``correction_recorded`` edit/reframe ops, and ``heal_conflicts``
  value-type supersessions. ``mark_wrong``/``retract`` without a replacement is a
  hard-block, not a preference, so it yields no pair. Event-sourced scope maps
  ``ScopeTag`` → ``privacy_scope`` per spec R8.

The correction detector is the SHARED ``text.correction_score`` (spec R2).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.config import OcbrainConfig, founder_ids, founder_weight, load_config
from ocbrain.scope import ScopeTag
from ocbrain.text import (
    AFFIRMATION_RE,
    claim_key,
    correction_score,
    find_probable_injection,
    find_probable_secret_leaks,
)
from ocbrain.write_batch import DatasetWriteBatch

from ocbrain_training.dataset.quality import (
    is_sft_process_chatter,
    sanitize_sft_text,
    store_example,
)
from ocbrain_training.dataset.transcripts import (
    Session,
    iter_unmined_transcripts,
    parse_transcript,
    record_source,
)

# A correction that also states the fix ("should be X", "use Y instead") is
# higher-signal — bump confidence toward the cap.
_ANSWER_RE = re.compile(
    r"(?i)\b(should be|supposed to be|it'?s actually|the correct|"
    r"needs? to be|has to be|use .+ instead|change .+ to|set .+ to)\b"
)

# Public discriminator for rows minted after the deterministic structural gate.
# Selection can require this version and cannot accidentally reuse legacy v0.3
# rows whose metadata only says gate=strict|relaxed.
CONTRAST_GATE_VERSION = 2

# Transcript parsers intentionally tolerate several runtimes.  A few runtime
# control records can therefore arrive as ordinary user turns when an upstream
# format changes.  They contain correction-like words ("do not", "fix",
# "instead") and used to manufacture DPO pairs from heartbeats, cron prompts,
# compaction notices, and tool/process warnings.  DPO is fail-closed here: these
# are transport/control records, never preference context.
_RUNTIME_WRAPPER_RE = re.compile(
    r"(?is)^\s*(?:"
    r"<goal_context\b|</goal_context\s*>|"
    r"<<<\s*(?:BEGIN|END)_OPENCLAW_INTERNAL_CONTEXT\s*>>>|"
    r"<task-notification\b|<environment_context\b|<system-reminder\b|"
    r"pre-compaction memory flush\b|"
    r"\[cron:[^\]]+\]|"
    r"read\s+HEARTBEAT\.md\b|"
    r"you are running a boot check\b|"
    r"system\s*\(untrusted\)\s*:|"
    r"system:\s*\[[^\]]+\]\s*reminder\b|"
    r"a scheduled reminder has been triggered\b|"
    r"warning:\s*(?:the maximum number of unified exec processes|"
    r"apply_patch was requested|you have \d+ weighted tokens left)|"
    r"current time:\s*[^\n]+(?:\n|$)"
    r")"
)
_TRANSPORT_RESIDUE_RE = re.compile(
    r"(?i)(?:\[\[\s*reply_to_current\s*\]\]|"
    r"(?:Conversation info|Sender|Replied message) "
    r"\(untrusted(?: metadata)?(?:, for context)?\))"
)
_TOPIC_TOKEN_RE = re.compile(r"[a-z][a-z0-9_./:-]{2,}|\b\d+(?:\.\d+)?\b", re.I)
_TOPIC_STOP = {
    "about",
    "actually",
    "after",
    "again",
    "also",
    "and",
    "answer",
    "are",
    "before",
    "being",
    "but",
    "can",
    "change",
    "check",
    "correct",
    "could",
    "current",
    "did",
    "does",
    "doing",
    "done",
    "every",
    "everything",
    "first",
    "for",
    "from",
    "going",
    "good",
    "had",
    "has",
    "have",
    "help",
    "here",
    "how",
    "into",
    "its",
    "just",
    "keep",
    "more",
    "need",
    "next",
    "not",
    "please",
    "right",
    "should",
    "something",
    "still",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "this",
    "those",
    "through",
    "too",
    "update",
    "use",
    "used",
    "using",
    "want",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "work",
    "working",
    "would",
    "wrong",
    "you",
    "your",
}


def _is_runtime_wrapper(text: str) -> bool:
    return bool(_RUNTIME_WRAPPER_RE.search(text or ""))


def _topic_tokens(text: str) -> set[str]:
    """Small deterministic topic sketch used only as a structural guard."""
    return {
        token.lower().strip("./:-")
        for token in _TOPIC_TOKEN_RE.findall(text or "")
        if token.lower().strip("./:-") not in _TOPIC_STOP and len(token.strip("./:-")) >= 3
    }


def _sanitize_pair_text(text: str) -> str:
    """Apply the shared training scrub to every DPO text surface."""
    return sanitize_sft_text(text or "")


def _valid_pair_structure(
    prompt_messages: Iterable[dict[str, str]],
    chosen: str,
    rejected: str,
    *,
    correction_text: str | None = None,
    event_backed: bool = False,
) -> bool:
    """Reject structurally ambiguous preference pairs before storage.

    Event-core edits have explicit edit/reframe/supersession provenance, so
    their correction direction does not need to be inferred from transcript
    wording.  Transcript pairs must additionally show a grounded prompt and a
    meaningful same-topic contrast.  In particular, forward-intent narration
    can be a useful *rejected* side, but is never a useful preferred answer.
    """
    prompt = [dict(message) for message in prompt_messages]
    if not prompt or not any(
        message.get("role") == "user" and str(message.get("content") or "").strip()
        for message in prompt
    ):
        return False
    all_text = [str(message.get("content") or "") for message in prompt]
    all_text.extend((chosen, rejected))
    if correction_text is not None:
        all_text.append(correction_text)
    if any(_is_runtime_wrapper(text) for text in all_text):
        return False
    if any(_TRANSPORT_RESIDUE_RE.search(text or "") for text in all_text):
        return False

    chosen_tokens = _topic_tokens(chosen)
    rejected_tokens = _topic_tokens(rejected)
    if not chosen_tokens or not rejected_tokens:
        return False
    # A punctuation/order-only rewrite is not a preference lesson.
    union = chosen_tokens | rejected_tokens
    if len(chosen_tokens ^ rejected_tokens) < 2 or (
        union and len(chosen_tokens & rejected_tokens) / len(union) >= 0.9
    ):
        return False
    if event_backed:
        return True

    chosen_chatter = is_sft_process_chatter(chosen)
    rejected_chatter = is_sft_process_chatter(rejected)
    if chosen_chatter:
        return False

    # Ground the pair in the last human request.  If only the rejected side is
    # on-topic, the purported correction moved away from the user's request.
    request = next(
        (
            str(message.get("content") or "")
            for message in reversed(prompt)
            if message.get("role") == "user" and str(message.get("content") or "").strip()
        ),
        "",
    )
    request_tokens = _topic_tokens(request)
    if not request_tokens:
        return False
    chosen_grounding = request_tokens & chosen_tokens
    rejected_grounding = request_tokens & rejected_tokens
    if not chosen_grounding:
        return False

    # A substantive-vs-substantive pair must describe the same situation.  A
    # generic/process rejected side is allowed: replacing chatter with an actual
    # answer is itself a clear defect correction.
    if not rejected_chatter and rejected_grounding:
        shared = chosen_tokens & rejected_tokens
        similarity = len(shared) / len(union) if union else 0.0
        if len(shared) < 2 and similarity < 0.08:
            return False

    # The correction must either name the chosen topic or be an explicit generic
    # defect report ("wrong", "not what I asked").  This blocks coincidental
    # correction-score hits whose response pair concerns another topic.
    correction = correction_text or ""
    correction_tokens = _topic_tokens(correction)
    explicit_generic_defect = bool(
        re.search(
            r"(?i)\b(?:wrong|incorrect|not right|not correct|not what i "
            r"(?:asked|wanted)|try again|redo)\b",
            correction,
        )
    )
    if not explicit_generic_defect and not (correction_tokens & chosen_tokens):
        return False
    return True


@dataclass(frozen=True)
class DpoPair:
    prompt_messages: tuple[dict[str, str], ...]
    chosen: str
    rejected: str
    correction_kind: str  # 'transcript'|'event_edit'|'event_correction'|'supersedes'
    hard: bool
    confidence: float
    evidence_ids: tuple[str, ...]
    privacy_scope: str
    source_uri: str | None
    occurred_at: str | None
    session_id: str | None = None
    # Structural provenance of the pair: 'strict' == the canonical
    # answer→correction→accepted structure; 'relaxed' == admitted only by the
    # widened v0.3 gate (missing acceptance turn, delayed correction, or implicit
    # correction cue). Stamped into metadata so training can filter.
    gate: str = "strict"
    # Author provenance of the correction (transcript pairs only): the telegram
    # sender id who issued the correction and their founder weight (1.0 == generic).
    corrected_by: str | None = None
    corrector_weight: float = 1.0


def scope_tag_to_privacy(tag: ScopeTag) -> str:
    """Map an event-core ScopeTag to the relational privacy_scope (spec R8)."""
    if (
        tag.visibility in {"confidential", "secret"}
        or tag.scope_type == "client"
        or tag.egress_policy != "hosted_ok"
    ):
        return "private"
    return "workspace"


def _exported_prompt(session: Session, upto: int, max_chars: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for turn in session.turns[: upto + 1]:
        text = _sanitize_pair_text(turn.text)
        if not text:
            continue
        if turn.role == "user" and turn.kind != "injected":
            out.append({"role": "user", "content": text})
        elif turn.role == "assistant":
            out.append({"role": "assistant", "content": text})
    while out and sum(len(m["content"]) for m in out) > max_chars:
        out.pop(0)
    if out and out[-1]["role"] == "assistant":
        out.pop()  # drop the rejected attempt from the prompt tail
    return out


def _next_user(session: Session, after: int) -> int | None:
    for k in range(after + 1, len(session.turns)):
        if session.turns[k].role == "user" and session.turns[k].kind != "injected":
            return k
    return None


def _accepted_correction_answer(
    session: Session, correction_idx: int, threshold: float
) -> int | None:
    """Return the final corrected answer only on affirmation or clean end.

    The last assistant message before the next human turn is the candidate.  A
    topic change is deliberately not treated as acceptance: without an explicit
    affirmation, it is impossible to know whether the user accepted the answer
    or simply abandoned it.
    """
    cursor = correction_idx
    while True:
        follow = _next_user(session, cursor)
        stop = len(session.turns) if follow is None else follow
        candidates = [
            idx
            for idx in range(cursor + 1, stop)
            if session.turns[idx].role == "assistant" and session.turns[idx].text.strip()
        ]
        if not candidates:
            return None
        candidate = candidates[-1]
        if follow is None:
            return candidate
        follow_text = _sanitize_pair_text(session.turns[follow].text)
        if _is_runtime_wrapper(follow_text):
            return None
        if correction_score(follow_text) >= threshold:
            cursor = follow
            continue
        if AFFIRMATION_RE.search(follow_text):
            return candidate
        return None


def _prev_user(session: Session, before: int) -> int | None:
    for k in range(before - 1, -1, -1):
        if session.turns[k].role == "user" and session.turns[k].kind != "injected":
            return k
    return None


def _prev_assistant_within(session: Session, before: int, max_back: int) -> int | None:
    """Nearest assistant answer within ``max_back`` turn positions before ``before``.

    Used by the relaxed gate to find the antecedent answer a *delayed* correction
    refers to (spec: search back N turns). ``max_back`` counts turn positions, so
    an assistant two turns back (answer → intermediate user → correction) is found
    at ``max_back >= 2``.
    """
    lo = max(0, before - max_back)
    for k in range(before - 1, lo - 1, -1):
        if session.turns[k].role == "assistant" and session.turns[k].text.strip():
            return k
    return None


def _pair_key(rejected: str, chosen: str) -> tuple[str, str]:
    return (claim_key(rejected), claim_key(chosen))


def find_transcript_pairs(session: Session, cfg: OcbrainConfig | None = None) -> list[DpoPair]:
    cfg = cfg or load_config()
    threshold = cfg.correction.threshold
    lo, hi = cfg.dataset.dpo_side_chars
    turns = session.turns
    pairs: list[DpoPair] = []
    seen: set[tuple[str, str]] = set()
    for a1 in range(len(turns)):
        if turns[a1].role != "assistant" or not turns[a1].text.strip():
            continue
        req = _prev_user(session, a1)
        if req is None:
            continue
        request_text = _sanitize_pair_text(turns[req].text)
        if not request_text or _is_runtime_wrapper(request_text):
            continue
        # A1 must be a FIRST attempt: its own request is not itself a correction.
        if correction_score(request_text) >= threshold:
            continue
        u = _next_user(session, a1)
        if u is None:
            continue
        correction_text = _sanitize_pair_text(turns[u].text)
        if _is_runtime_wrapper(correction_text) or correction_score(correction_text) < threshold:
            continue
        # Walk the correction chain forward to the final accepted answer.
        chosen_idx = _accepted_correction_answer(session, u, threshold)
        if chosen_idx is None:
            continue
        rejected = _sanitize_pair_text(turns[a1].text)
        chosen = _sanitize_pair_text(turns[chosen_idx].text)
        if not (lo <= len(rejected) <= hi and lo <= len(chosen) <= hi):
            continue
        if claim_key(chosen) == claim_key(rejected):
            continue
        if (
            find_probable_secret_leaks(rejected)
            or find_probable_injection(rejected)
            or find_probable_secret_leaks(chosen)
            or find_probable_injection(chosen)
        ):
            continue
        prompt = _exported_prompt(session, req, cfg.dataset.sft_max_context_chars)
        if not _valid_pair_structure(
            prompt,
            chosen,
            rejected,
            correction_text=correction_text,
        ):
            continue
        confidence = min(0.9, correction_score(correction_text))
        if _ANSWER_RE.search(correction_text):
            confidence = 0.95
        corrected_by = turns[u].authored_by
        corrector_weight = founder_weight(cfg, corrected_by)
        seen.add(_pair_key(rejected, chosen))
        pairs.append(
            DpoPair(
                prompt_messages=tuple(prompt),
                chosen=chosen,
                rejected=rejected,
                correction_kind="transcript",
                hard=False,
                confidence=confidence,
                evidence_ids=(),  # filled by the caller from the transcript evidence
                privacy_scope="workspace",
                source_uri=session.source_uri,
                occurred_at=turns[chosen_idx].ts or session.occurred_at,
                session_id=session.session_id,
                corrected_by=corrected_by,
                corrector_weight=corrector_weight,
                gate="strict",
            )
        )
    if cfg.dataset.dpo_relaxed_gate:
        pairs.extend(_find_relaxed_transcript_pairs(session, cfg, seen))
    return pairs


# How many turn positions back a delayed correction may sit from its antecedent
# answer under the relaxed gate (spec: N=4).
_RELAXED_LOOKBACK = 4


def _find_relaxed_transcript_pairs(
    session: Session, cfg: OcbrainConfig, seen: set[tuple[str, str]]
) -> list[DpoPair]:
    """Additive v0.3 relaxed-gate pairs the strict structure rejects.

    Anchored on the *correction* turn rather than the first attempt, this admits:

    * (a) **missing acceptance** — the correction that states the fix is the
      thread's last word (no accepted assistant answer follows). The correction
      text itself becomes ``chosen``.
    * (b) **delayed correction** — the correction lands several turns after the
      answer it refers to; we search back up to ``_RELAXED_LOOKBACK`` turns for
      the antecedent assistant answer (``rejected``).

    (c) implicit-correction *detection* is shared via ``text.correction_score``.

    Every pair here is tagged ``gate='relaxed'``. Strict pairs already emitted are
    skipped via ``seen`` so the relaxed pass only *adds* — it never restates or
    overrides a strict pair.
    """
    threshold = cfg.correction.threshold
    lo, hi = cfg.dataset.dpo_side_chars
    turns = session.turns
    out: list[DpoPair] = []
    for u in range(len(turns)):
        t = turns[u]
        if t.role != "user" or t.kind == "injected":
            continue
        correction_text = _sanitize_pair_text(t.text)
        if _is_runtime_wrapper(correction_text) or correction_score(correction_text) < threshold:
            continue
        a1 = _prev_assistant_within(session, u, _RELAXED_LOOKBACK)
        if a1 is None:
            continue
        req = _prev_user(session, a1)
        if req is None:
            continue
        request_text = _sanitize_pair_text(turns[req].text)
        if not request_text or _is_runtime_wrapper(request_text):
            continue
        # The antecedent's own request must not itself be a correction — same
        # first-attempt guard the strict path applies.
        if correction_score(request_text) >= threshold:
            continue
        # Resolve the chosen answer: the accepted answer after the correction, or
        # (case a) the correction text itself when it is the thread's last word.
        chosen_idx = _accepted_correction_answer(session, u, threshold)
        if chosen_idx is not None:
            chosen_src = turns[chosen_idx].text
            occurred_at = turns[chosen_idx].ts or session.occurred_at
        elif _next_user(session, u) is None:
            # (a) missing acceptance turn: the corrected instruction is the last
            # word. The founder's stated fix is the preferred output.
            chosen_src = t.text
            occurred_at = t.ts or session.occurred_at
        else:
            continue
        rejected = _sanitize_pair_text(turns[a1].text)
        chosen = _sanitize_pair_text(chosen_src)
        if not (lo <= len(rejected) <= hi and lo <= len(chosen) <= hi):
            continue
        if claim_key(chosen) == claim_key(rejected):
            continue
        if find_probable_secret_leaks(rejected) or find_probable_injection(rejected):
            continue
        if find_probable_secret_leaks(chosen) or find_probable_injection(chosen):
            continue  # chosen can be a raw user turn under (a); scrub it too
        prompt = _exported_prompt(session, req, cfg.dataset.sft_max_context_chars)
        if not _valid_pair_structure(
            prompt,
            chosen,
            rejected,
            correction_text=correction_text,
        ):
            continue
        key = _pair_key(rejected, chosen)
        if key in seen:
            continue  # strict already emitted this pair
        seen.add(key)
        corrected_by = t.authored_by
        corrector_weight = founder_weight(cfg, corrected_by)
        # Relaxed structural evidence is softer than strict; cap confidence below
        # the strict ceiling so training can weight accordingly.
        confidence = min(0.8, correction_score(correction_text))
        out.append(
            DpoPair(
                prompt_messages=tuple(prompt),
                chosen=chosen,
                rejected=rejected,
                correction_kind="transcript",
                hard=False,
                confidence=confidence,
                evidence_ids=(),  # filled by the caller from the transcript evidence
                privacy_scope="workspace",
                source_uri=session.source_uri,
                occurred_at=occurred_at,
                session_id=session.session_id,
                corrected_by=corrected_by,
                corrector_weight=corrector_weight,
                gate="relaxed",
            )
        )
    return out


def _knowledge_evidence_ids(conn: sqlite3.Connection, knowledge_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT evidence_id FROM knowledge_evidence WHERE knowledge_id = ?",
        (knowledge_id,),
    ).fetchall()
    return [r["evidence_id"] for r in rows]


def find_event_pairs(conn: sqlite3.Connection, cfg: OcbrainConfig | None = None) -> list[DpoPair]:
    cfg = cfg or load_config()
    lo, hi = cfg.dataset.dpo_side_chars
    pairs: list[DpoPair] = []

    def _admit(prompt: str, rejected: str, chosen: str) -> bool:
        if not (lo <= len(rejected) <= hi and lo <= len(chosen) <= hi):
            return False
        if claim_key(chosen) == claim_key(rejected):
            return False
        if (
            find_probable_secret_leaks(rejected)
            or find_probable_injection(rejected)
            or find_probable_secret_leaks(chosen)
            or find_probable_injection(chosen)
        ):
            return False
        return _valid_pair_structure(
            ({"role": "user", "content": prompt},),
            chosen,
            rejected,
            correction_text=chosen,
            event_backed=True,
        )

    # (A) compilation_decided edits: proposal body -> edited body.
    proposals: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT id, body_json, ts FROM brain_events WHERE kind='compilation_proposed'"
    ):
        proposals[row["id"]] = {**json.loads(row["body_json"]), "_ts": row["ts"]}
    for row in conn.execute(
        "SELECT body_json, ts FROM brain_events WHERE kind='compilation_decided'"
    ):
        body = json.loads(row["body_json"])
        if body.get("decision") != "edit" or not body.get("edited_body"):
            continue
        proposal = proposals.get(body.get("proposal_event_id"))
        if proposal is None:
            continue
        rejected = _sanitize_pair_text(str(proposal.get("body") or ""))
        chosen = _sanitize_pair_text(str(body["edited_body"]))
        evidence_ids = [e for e in (proposal.get("evidence_ids") or []) if e]
        claims = _evidence_claims(conn, evidence_ids)
        prompt = f"Compile the belief supported by this evidence:\n{claims}".strip()
        if not evidence_ids or not _admit(prompt, rejected, chosen):
            continue
        scope = scope_tag_to_privacy(ScopeTag.from_dict(proposal.get("scope")))
        pairs.append(
            DpoPair(
                prompt_messages=({"role": "user", "content": prompt},),
                chosen=chosen,
                rejected=rejected,
                correction_kind="event_edit",
                hard=False,
                confidence=0.85,
                evidence_ids=tuple(evidence_ids),
                privacy_scope=scope,
                source_uri=None,
                occurred_at=proposal.get("_ts"),
            )
        )

    # (B) correction_recorded edit/reframe ops on knowledge rows.
    for row in conn.execute(
        "SELECT id, body_json, ts FROM brain_events WHERE kind='correction_recorded'"
    ):
        body = json.loads(row["body_json"])
        if body.get("op") not in {"edit", "reframe"} or not body.get("body"):
            continue  # mark_wrong/retract w/o replacement is a hard-block, not a pair
        if body.get("target_layer") != "knowledge":
            continue
        target = conn.execute(
            "SELECT value_text, title, subject, predicate, privacy_scope "
            "FROM knowledge WHERE id = ?",
            (body.get("target_id"),),
        ).fetchone()
        if target is None:
            continue
        rejected = _sanitize_pair_text(str(target["value_text"] or target["title"] or ""))
        chosen = _sanitize_pair_text(str(body["body"]))
        evidence_ids = _knowledge_evidence_ids(conn, body.get("target_id"))
        subject = " ".join(str(target[c] or "") for c in ("subject", "predicate")).strip()
        prompt = f"State the correct value for: {subject or target['title'] or 'this belief'}"
        if not evidence_ids or not _admit(prompt, rejected, chosen):
            continue
        pairs.append(
            DpoPair(
                prompt_messages=({"role": "user", "content": prompt},),
                chosen=chosen,
                rejected=rejected,
                correction_kind="event_correction",
                hard=bool(body.get("hard")),
                confidence=0.95 if body.get("hard") else 0.85,
                evidence_ids=tuple(evidence_ids),
                privacy_scope=target["privacy_scope"] or "workspace",
                source_uri=None,
                occurred_at=row["ts"],
            )
        )

    # (C) heal_conflicts value-type supersessions (loser -> winner).
    for row in conn.execute(
        """
        SELECT loser.value_text AS loser_text, loser.subject AS subject,
               loser.predicate AS predicate, winner.id AS winner_id,
               winner.value_text AS winner_text, winner.privacy_scope AS scope,
               winner.updated_at AS ts
        FROM knowledge loser
        JOIN knowledge winner ON loser.superseded_by = winner.id
        WHERE loser.type = 'value' AND winner.type = 'value'
        """
    ):
        rejected = _sanitize_pair_text(str(row["loser_text"] or ""))
        chosen = _sanitize_pair_text(str(row["winner_text"] or ""))
        evidence_ids = _knowledge_evidence_ids(conn, row["winner_id"])
        subject = " ".join(str(row[c] or "") for c in ("subject", "predicate")).strip()
        prompt = f"State the current value for: {subject or 'this belief'}"
        if not evidence_ids or not _admit(prompt, rejected, chosen):
            continue
        pairs.append(
            DpoPair(
                prompt_messages=({"role": "user", "content": prompt},),
                chosen=chosen,
                rejected=rejected,
                correction_kind="supersedes",
                hard=False,
                confidence=0.7,
                evidence_ids=tuple(evidence_ids),
                privacy_scope=row["scope"] or "workspace",
                source_uri=None,
                occurred_at=row["ts"],
            )
        )
    return pairs


def _evidence_claims(conn: sqlite3.Connection, evidence_ids: list[str], limit: int = 4) -> str:
    claims: list[str] = []
    for eid in evidence_ids[:limit]:
        row = conn.execute("SELECT claim FROM evidence WHERE id = ?", (eid,)).fetchone()
        if row and row["claim"]:
            claims.append(f"- {row['claim'][:400]}")
    return "\n".join(claims)


def _store_pair(
    conn: sqlite3.Connection,
    pair: DpoPair,
    *,
    write_batch: DatasetWriteBatch | None = None,
) -> dict[str, Any] | None:
    if not pair.evidence_ids:
        return None
    body = {
        "input": {"messages": [dict(m) for m in pair.prompt_messages]},
        "preferred_output": [{"role": "assistant", "content": pair.chosen}],
        "non_preferred_output": [{"role": "assistant", "content": pair.rejected}],
    }
    return store_example(
        conn,
        dataset="dpo",
        source_kind="correction_event"
        if pair.correction_kind != "transcript"
        else _transcript_source_kind(pair),
        source_uri=pair.source_uri,
        source_span={"correction_kind": pair.correction_kind},
        evidence_ids=list(pair.evidence_ids),
        privacy_scope=pair.privacy_scope,
        body=body,
        metadata=_pair_metadata(pair),
        target_text=pair.chosen,
        base_label="good",
        base_confidence=pair.confidence,
        n_turns=len(pair.prompt_messages) + 1,
        session_id=pair.session_id,
        occurred_at=pair.occurred_at,
        write_batch=write_batch,
    )


def _pair_metadata(pair: DpoPair) -> dict[str, Any]:
    """DPO example metadata, with founder author provenance when a founder issued
    the correction (``corrected=chosen`` tagged with who corrected it)."""
    meta: dict[str, Any] = {
        "correction_kind": pair.correction_kind,
        "hard": pair.hard,
        "confidence": pair.confidence,
        "session_id": pair.session_id,
        "gate": pair.gate,
        "contrast_gate_version": CONTRAST_GATE_VERSION,
    }
    if pair.corrected_by and pair.corrector_weight != 1.0:
        meta["corrected_by"] = pair.corrected_by
        meta["corrector_weight"] = pair.corrector_weight
        meta["founder_correction"] = True
    return meta


def _transcript_source_kind(pair: DpoPair) -> str:
    # Transcript pairs carry their originating session's source_kind via source_uri
    # extension; default to a generic session kind when unknown.
    uri = pair.source_uri or ""
    if uri.endswith(".jsonl") and "codex" in uri.lower():
        return "codex_session"
    if ".claude" in uri.lower():
        return "claude_session"
    return "openclaw_session"


def mine_dpo(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    sessions: Iterable[Session] | None = None,
    roots: Iterable[str] | None = None,
    include_events: bool = True,
    limit: int | None = None,
    time_budget_seconds: float | None = None,
    write_batch: DatasetWriteBatch | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    from ocbrain_training.dataset.transcripts import resolve_transcript_evidence

    started = time.monotonic()
    deadline = None if time_budget_seconds is None else started + time_budget_seconds

    def budget_exhausted() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    stored = 0
    excluded = 0
    examined = 0
    files_mined = 0
    batch = write_batch or DatasetWriteBatch(
        conn,
        max_operations=cfg.dataset.write_batch_size,
        max_seconds=cfg.dataset.write_batch_seconds,
    )

    def _emit_transcript(session: Session) -> int:
        nonlocal stored, excluded, examined
        evidence_id, scope = resolve_transcript_evidence(conn, session, write_batch=batch)
        count = 0
        for pair in find_transcript_pairs(session, cfg):
            examined += 1
            pair = DpoPair(
                **{**pair.__dict__, "evidence_ids": (evidence_id,), "privacy_scope": scope}
            )
            result = _store_pair(conn, pair, write_batch=batch)
            if result is None:
                continue
            count += 1
            if result["quality_label"] == "excluded":
                excluded += 1
            else:
                stored += 1
        return count

    if sessions is not None:
        for session in sessions:
            if budget_exhausted():
                break
            _emit_transcript(session)
            batch.flush()
    elif roots is not None:
        for path, fingerprint in iter_unmined_transcripts(conn, roots, "dpo"):
            if budget_exhausted():
                break
            session = parse_transcript(
                path,
                author_ids=cfg.dataset.persona_author_ids,
                direct_agents=cfg.dataset.persona_direct_agents,
                tool_result_truncate=cfg.dataset.tool_result_truncate,
                founder_ids=founder_ids(cfg),
            )
            emitted = 0 if session is None else _emit_transcript(session)
            batch.ensure()
            record_source(conn, str(path), "dpo", fingerprint, emitted)
            batch.operation()
            batch.flush()
            files_mined += 1
            if limit is not None and files_mined >= limit:
                break

    if include_events and not budget_exhausted():
        for pair in find_event_pairs(conn, cfg):
            if budget_exhausted():
                break
            examined += 1
            result = _store_pair(conn, pair, write_batch=batch)
            if result is None:
                continue
            if result["quality_label"] == "excluded":
                excluded += 1
            else:
                stored += 1

    batch.flush()
    return {
        "ok": True,
        "dataset": "dpo",
        "examined": examined,
        "stored": stored,
        "excluded": excluded,
        "files_mined": files_mined,
        "writer_lock": batch.metrics(),
    }


# --------------------------------------------------------------------------- #
# One-time founder re-mine (v0.3, Ruling 3)
# --------------------------------------------------------------------------- #
# Real founder corrections sat in sessions fingerprint-watermarked as mined under
# the OLD strict gate, so ``iter_unmined_transcripts`` skips them and the relaxed
# gate never re-examines them. This narrow, idempotent mechanism BYPASSES (never
# clears) the dpo watermark ledger for sessions that actually contain a configured
# founder id — read from LOCAL config only — and re-mines just those files under
# the relaxed gate. The ``dataset_sources`` watermark ledger is left untouched, so
# normal watermarking is preserved for every later run; content-hash + dedup-key
# in ``store_example`` make re-runs produce no duplicate examples.


def _session_has_founder_turn(session: Session, founder: set[str]) -> bool:
    """True if any non-injected user turn was authored by a configured founder."""
    return any(
        turn.role == "user" and turn.kind != "injected" and turn.authored_by in founder
        for turn in session.turns
    )


def remine_founder_sessions(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    roots: Iterable[str] | None = None,
    limit: int | None = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    """Re-mine watermarked transcripts that contain a founder correction.

    Additive and idempotent: bypasses the dpo watermark for founder-bearing
    sessions only, re-mines them under the relaxed gate, and stores any new pairs
    (dedup drops what was already mined). Returns counts plus the ids of the
    founder-provenance pairs that now exist. Founder identifiers are read from
    LOCAL config and are never returned or logged.
    """
    cfg = cfg or load_config()
    from ocbrain_training.dataset.transcripts import (
        iter_transcript_files,
        resolve_transcript_evidence,
    )

    founder = set(founder_ids(cfg))
    batch = DatasetWriteBatch(
        conn,
        max_operations=cfg.dataset.write_batch_size,
        max_seconds=cfg.dataset.write_batch_seconds,
    )
    base = {
        "ok": True,
        "dataset": "dpo",
        "mode": "founder_rescan",
        "files_scanned": 0,
        "files_remined": 0,
        "stored": 0,
        "excluded": 0,
        "founder_pairs": 0,
        "founder_pair_ids": [],
        "writer_lock": batch.metrics(),
    }
    if not founder:
        base["reason"] = "no_founder_ids"
        return base

    search_roots = list(roots) if roots is not None else list(cfg.review.session_roots)
    started = time.monotonic()
    files_scanned = files_remined = stored = excluded = founder_pairs = 0
    founder_pair_ids: list[str] = []

    for path in iter_transcript_files(search_roots):
        if time_budget_seconds is not None and time.monotonic() - started > time_budget_seconds:
            break
        files_scanned += 1
        # Cheap pre-filter: the raw file must mention a founder id before we pay to
        # parse it. Parse then confirms a genuine founder-authored turn.
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if not any(fid in raw for fid in founder):
            continue
        session = parse_transcript(
            path,
            author_ids=cfg.dataset.persona_author_ids,
            direct_agents=cfg.dataset.persona_direct_agents,
            tool_result_truncate=cfg.dataset.tool_result_truncate,
            founder_ids=list(founder),
        )
        if session is None or not _session_has_founder_turn(session, founder):
            continue
        files_remined += 1
        evidence_id, scope = resolve_transcript_evidence(conn, session, write_batch=batch)
        for pair in find_transcript_pairs(session, cfg):
            pair = DpoPair(
                **{**pair.__dict__, "evidence_ids": (evidence_id,), "privacy_scope": scope}
            )
            result = _store_pair(conn, pair, write_batch=batch)
            if result is None:
                continue
            if result["quality_label"] == "excluded":
                excluded += 1
            else:
                stored += 1
            # Founder-provenance pair: a relaxed-gate pair whose correction turn was
            # authored by a founder (corrected_by is stamped only for weight != 1.0).
            if (
                pair.gate == "relaxed"
                and pair.corrected_by in founder
                and result["quality_label"] != "excluded"
            ):
                founder_pairs += 1
                founder_pair_ids.append(result["id"])
        batch.flush()
        if limit is not None and files_remined >= limit:
            break

    base.update(
        {
            "files_scanned": files_scanned,
            "files_remined": files_remined,
            "stored": stored,
            "excluded": excluded,
            "founder_pairs": founder_pairs,
            "founder_pair_ids": founder_pair_ids,
            "writer_lock": batch.metrics(),
        }
    )
    return base


def _main(argv: list[str] | None = None) -> int:
    """``python -m ocbrain_training.dataset.mine_dpo --founder-rescan`` entry point.

    A deliberately narrow, one-time driver for the founder re-mine. Uses the live
    DB/config paths from ``$OCBRAIN_DB`` / ``$OCBRAIN_CONFIG`` (same env the
    autopilot wrapper exports). Never prints founder identifiers.
    """
    import argparse
    import os

    from ocbrain.db import connect, init_db

    parser = argparse.ArgumentParser(prog="ocbrain_training.dataset.mine_dpo")
    parser.add_argument(
        "--founder-rescan",
        action="store_true",
        help="Re-mine watermarked founder-bearing sessions under the relaxed gate",
    )
    parser.add_argument("--db", default=os.environ.get("OCBRAIN_DB", "data/ocbrain.sqlite"))
    parser.add_argument("--root", action="append", dest="roots", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-budget", type=float, default=None)
    args = parser.parse_args(argv)

    if not args.founder_rescan:
        parser.error("no action requested; pass --founder-rescan")

    cfg = load_config()
    conn = connect(Path(args.db))
    # The live brain is a busy, shared DB (launchd cycles + MCP servers write to
    # it). Wait politely for the write lock rather than failing fast so init_db's
    # migration and our writes don't lose a lock race under contention.
    conn.execute("PRAGMA busy_timeout=120000")
    init_db(conn)
    try:
        result = remine_founder_sessions(
            conn,
            cfg=cfg,
            roots=args.roots,
            limit=args.limit,
            time_budget_seconds=args.time_budget,
        )
        conn.commit()
    finally:
        conn.close()
    # Never emit founder_pair ids' authorship; ids here are synthetic dsx_* stable
    # ids, safe to print. corrected_by (a real id) is NOT included by the miner.
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(_main())
