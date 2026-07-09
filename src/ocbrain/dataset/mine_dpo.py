"""DPO preference-pair mining (spec §7.2 DPO).

Two sources:

* **Transcript pairs** — ``A1 → U(correction ≥ correction.threshold) → … An``:
  prompt is the context through the ORIGINAL request (correction excluded),
  ``rejected`` is the first attempt ``A1``, ``chosen`` is the final accepted
  attempt (walk forward while corrections continue; accept on affirmation /
  topic-change / clean end). Both sides must differ by ``claim_key``, fit
  ``dpo_side_chars``, and survive the secret/injection scrub.
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
from typing import Any

from ocbrain.config import OcbrainConfig, founder_ids, founder_weight, load_config
from ocbrain.dataset.quality import store_example
from ocbrain.dataset.transcripts import (
    Session,
    iter_unmined_transcripts,
    parse_transcript,
    record_source,
)
from ocbrain.scope import ScopeTag
from ocbrain.text import (
    claim_key,
    correction_score,
    find_probable_injection,
    find_probable_secret_leaks,
    redact_secrets,
)

# A correction that also states the fix ("should be X", "use Y instead") is
# higher-signal — bump confidence toward the cap.
_ANSWER_RE = re.compile(
    r"(?i)\b(should be|supposed to be|it'?s actually|the correct|"
    r"needs? to be|has to be|use .+ instead|change .+ to|set .+ to)\b"
)


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
        text = turn.text.strip()
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


def _next_assistant(session: Session, after: int) -> int | None:
    for k in range(after + 1, len(session.turns)):
        if session.turns[k].role == "assistant" and session.turns[k].text.strip():
            return k
    return None


def _next_user(session: Session, after: int) -> int | None:
    for k in range(after + 1, len(session.turns)):
        if session.turns[k].role == "user" and session.turns[k].kind != "injected":
            return k
    return None


def _prev_user(session: Session, before: int) -> int | None:
    for k in range(before - 1, -1, -1):
        if session.turns[k].role == "user" and session.turns[k].kind != "injected":
            return k
    return None


def find_transcript_pairs(session: Session, cfg: OcbrainConfig | None = None) -> list[DpoPair]:
    cfg = cfg or load_config()
    threshold = cfg.correction.threshold
    lo, hi = cfg.dataset.dpo_side_chars
    turns = session.turns
    pairs: list[DpoPair] = []
    for a1 in range(len(turns)):
        if turns[a1].role != "assistant" or not turns[a1].text.strip():
            continue
        req = _prev_user(session, a1)
        if req is None:
            continue
        # A1 must be a FIRST attempt: its own request is not itself a correction.
        if correction_score(turns[req].text) >= threshold:
            continue
        u = _next_user(session, a1)
        if u is None or correction_score(turns[u].text) < threshold:
            continue
        # Walk the correction chain forward to the final accepted answer.
        chosen_idx: int | None = None
        cursor = u
        while True:
            cand = _next_assistant(session, cursor)
            if cand is None:
                break
            chosen_idx = cand
            follow = _next_user(session, cand)
            if follow is None:
                break
            if correction_score(turns[follow].text) >= threshold:
                cursor = follow
                continue
            break  # affirmation / topic-change / clean end accepts this answer
        if chosen_idx is None:
            continue
        rejected = redact_secrets(turns[a1].text.strip())
        chosen = redact_secrets(turns[chosen_idx].text.strip())
        if not (lo <= len(rejected) <= hi and lo <= len(chosen) <= hi):
            continue
        if claim_key(chosen) == claim_key(rejected):
            continue
        if find_probable_secret_leaks(rejected) or find_probable_injection(rejected):
            continue
        correction_text = turns[u].text
        confidence = min(0.9, correction_score(correction_text))
        if _ANSWER_RE.search(correction_text):
            confidence = 0.95
        corrected_by = turns[u].authored_by
        corrector_weight = founder_weight(cfg, corrected_by)
        pairs.append(
            DpoPair(
                prompt_messages=tuple(
                    _exported_prompt(session, req, cfg.dataset.sft_max_context_chars)
                ),
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
            )
        )
    return pairs


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
        if find_probable_secret_leaks(rejected) or find_probable_injection(rejected):
            return False
        return True

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
        rejected = redact_secrets(str(proposal.get("body") or "").strip())
        chosen = redact_secrets(str(body["edited_body"]).strip())
        evidence_ids = [e for e in (proposal.get("evidence_ids") or []) if e]
        if not evidence_ids or not _admit("", rejected, chosen):
            continue
        claims = _evidence_claims(conn, evidence_ids)
        prompt = f"Compile the belief supported by this evidence:\n{claims}".strip()
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
        rejected = redact_secrets(str(target["value_text"] or target["title"] or "").strip())
        chosen = redact_secrets(str(body["body"]).strip())
        evidence_ids = _knowledge_evidence_ids(conn, body.get("target_id"))
        if not evidence_ids or not _admit("", rejected, chosen):
            continue
        subject = " ".join(str(target[c] or "") for c in ("subject", "predicate")).strip()
        prompt = f"State the correct value for: {subject or target['title'] or 'this belief'}"
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
        rejected = redact_secrets(str(row["loser_text"] or "").strip())
        chosen = redact_secrets(str(row["winner_text"] or "").strip())
        evidence_ids = _knowledge_evidence_ids(conn, row["winner_id"])
        if not evidence_ids or not _admit("", rejected, chosen):
            continue
        subject = " ".join(str(row[c] or "") for c in ("subject", "predicate")).strip()
        prompt = f"State the current value for: {subject or 'this belief'}"
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


def _store_pair(conn: sqlite3.Connection, pair: DpoPair) -> dict[str, Any] | None:
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
    )


def _pair_metadata(pair: DpoPair) -> dict[str, Any]:
    """DPO example metadata, with founder author provenance when a founder issued
    the correction (``corrected=chosen`` tagged with who corrected it)."""
    meta: dict[str, Any] = {
        "correction_kind": pair.correction_kind,
        "hard": pair.hard,
        "confidence": pair.confidence,
        "session_id": pair.session_id,
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
) -> dict[str, Any]:
    cfg = cfg or load_config()
    from ocbrain.dataset.transcripts import resolve_transcript_evidence

    started = time.monotonic()
    stored = 0
    excluded = 0
    examined = 0
    files_mined = 0

    def _emit_transcript(session: Session) -> int:
        nonlocal stored, excluded, examined
        evidence_id, scope = resolve_transcript_evidence(conn, session)
        count = 0
        for pair in find_transcript_pairs(session, cfg):
            examined += 1
            pair = DpoPair(
                **{**pair.__dict__, "evidence_ids": (evidence_id,), "privacy_scope": scope}
            )
            result = _store_pair(conn, pair)
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
            _emit_transcript(session)
    elif roots is not None:
        for path, fingerprint in iter_unmined_transcripts(conn, roots, "dpo"):
            if time_budget_seconds is not None and time.monotonic() - started > time_budget_seconds:
                break
            session = parse_transcript(
                path,
                author_ids=cfg.dataset.persona_author_ids,
                direct_agents=cfg.dataset.persona_direct_agents,
                tool_result_truncate=cfg.dataset.tool_result_truncate,
                founder_ids=founder_ids(cfg),
            )
            emitted = 0 if session is None else _emit_transcript(session)
            record_source(conn, str(path), "dpo", fingerprint, emitted)
            files_mined += 1
            if limit is not None and files_mined >= limit:
                break

    if include_events:
        for pair in find_event_pairs(conn, cfg):
            examined += 1
            result = _store_pair(conn, pair)
            if result is None:
                continue
            if result["quality_label"] == "excluded":
                excluded += 1
            else:
                stored += 1

    return {
        "ok": True,
        "dataset": "dpo",
        "examined": examined,
        "stored": stored,
        "excluded": excluded,
        "files_mined": files_mined,
    }
