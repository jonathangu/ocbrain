"""Dataset quality scrub + the shared example-storage gate (spec §7.3-7.4).

Every candidate example passes through :func:`store_example` before it lands in
``dataset_examples``. That gate: redacts secrets, runs the nine exclusion rules
(§7.4), assigns the near-dup key, computes the stable ``content_hash`` over the
messages/pair ONLY, and upserts idempotently. A rule hit downgrades the example
to ``quality_label='excluded'`` with the fired reason recorded — excluded rows
are kept (for stats) but never exported.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from ocbrain.db import now_iso
from ocbrain.events import canonical_json
from ocbrain.ids import content_hash, stable_id
from ocbrain.text import (
    claim_key,
    find_high_entropy_spans,
    find_probable_injection,
    find_probable_secret_leaks,
    redact_secrets,
)
from ocbrain.write_batch import DatasetWriteBatch

# Universal length bounds (spec §7.4 rule 3). The per-dataset floors (SFT 80,
# DPO side 40) are enforced by the miners; this is the outer guard.
MIN_TARGET_CHARS = 40
MAX_EXAMPLE_CHARS = 32000

_REFUSAL_RE = re.compile(
    r"(?i)^\W*(i'?m sorry|i am sorry|i apologi[sz]e|sorry[,.]|"
    r"i can'?t (?:help|assist|do|comply|provide)|i cannot (?:help|assist|do|comply|provide)|"
    r"i'?m (?:not able|unable) to|i am (?:not able|unable) to|"
    r"as an ai|unfortunately,? i (?:can'?t|cannot))"
)
_MANAGED_BLOCK_RE = re.compile(r"(?i)(BEGIN|END) OCBRAIN MANAGED BLOCK")
_ENVELOPE_RESIDUE_RE = re.compile(
    r"(?i)(?:Conversation info|Sender|Replied message) "
    r"\(untrusted(?: metadata)?(?:, for context)?\)"
)
_TRANSPORT_RESIDUE_RE = re.compile(
    r"(?is)(?:"
    r'"content"\s*:\s*"\s*\[\[\s*reply_to_current\s*\]\]|'
    r"system\s*\(untrusted\)\s*:|"
    r"<goal_context\b|</goal_context\s*>|"
    r"<<<\s*(?:BEGIN|END)_OPENCLAW_INTERNAL_CONTEXT\s*>>>|"
    r"<task-notification\b|<environment_context\b|"
    r"pre-compaction memory flush\b|"
    r"\[cron:[^\]]+\]|"
    r"read\s+HEARTBEAT\.md\b|"
    r"a new session was started via /new or /reset\b|"
    r"warning:\s*(?:the maximum number of unified exec processes|"
    r"apply_patch was requested|you have \d+ weighted tokens left)"
    r")"
)
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
_STACK_LINE_RE = re.compile(r'^\s*File ".*", line \d+', re.MULTILINE)

# Transport/UI residue is not model behavior.  OpenClaw prepends the routing
# token to some otherwise-useful assistant turns, and newer Telegram transcripts
# can carry one or more fenced, explicitly-untrusted metadata envelopes before
# the human text.  Both are removed at the final SFT storage boundary so parser
# version drift cannot put them back into weights.
_REPLY_ROUTING_PREFIX_RE = re.compile(r"\A(?:\s*\[\[\s*reply_to_current\s*\]\]\s*)+", re.IGNORECASE)
_UNTRUSTED_ENVELOPE_BLOCK_RE = re.compile(
    r"(?ims)^[ \t]*(?:Conversation info|Sender|Replied message)\s*"
    r"\(untrusted(?: metadata)?(?:,\s*for context)?\):[ \t]*\n"
    r"```(?:json)?[ \t]*\n.*?^```[ \t]*(?:\n|\Z)"
)

# Training-only identifier cleanup.  Secrets keep using the shared scanner;
# these additional patterns cover human identifiers that are not credentials.
# Labeled identifiers are deliberately narrow so useful run/build/commit ids in
# a verified status report remain intact.
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<![\w])(?:\+?1[ ./-]?)?\(?\d{3}\)?[ ./-]\d{3}[ ./-]\d{4}(?![\w])")
_LABELED_PRIVATE_ID_RE = re.compile(
    r"(?ix)\b("
    r"sender[_ -]?id|message[_ -]?id|chat[_ -]?id|conversation[_ -]?id|"
    r"context[_ -]?id|telegram[_ -]?id|"
    r"broker(?:age)?[_ -]?(?:account|acct)(?:[_ -]?(?:id|number|no\.?))?|"
    r"account[_ -]?(?:id|number|no\.?)"
    r")([\"']?\s*[:=#-]\s*[\"']?)([A-Z0-9][A-Z0-9_.@+-]{3,})"
)
_ACCOUNT_VALUE_RE = re.compile(
    r"(?ix)\b((?:broker(?:age)?\s+)?account)(\s+)`?"
    r"((?=[A-Z0-9-]{0,24}\d)[A-Z0-9][A-Z0-9-]{5,})`?"
)

# The SFT target gate separates useful status from progress narration.  It does
# not require every response to be a completion report: ordinary explanations
# never enter this branch.  It only scrutinizes turns shaped like acknowledg-
# ments, heartbeat residue, or first-person next-step narration.
_EXPLICIT_BLOCKED_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?BLOCKED\b")
_ACK_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:got it|understood|on it|will do|okay|ok(?:ay)?|"
    r"sounds good|agreed|acknowledged|thanks|thank you|yes)[.!,:;\s-]+"
)
_HEARTBEAT_PROCESS_RE = re.compile(
    r"(?i)\b(?:handled (?:the )?heartbeat|heartbeat (?:task )?(?:handled|recorded)|"
    r"notify\s*=\s*false|no user notification (?:is )?needed|"
    r"no visible heartbeat notification (?:was )?sent|"
    r"no unfinished tasks?|nothing needs (?:the )?user'?s attention)\b"
)
_FORWARD_INTENT_RE = re.compile(
    r"(?ix)(?:"
    r"\bi(?:'|’)m\s+(?:now\s+|actively\s+|still\s+)?"
    r"(?:checking|inspecting|implementing|adding|updating|running|rerunning|starting|"
    r"tracing|locating|patching|finishing|testing|applying|pulling|reviewing|"
    r"rebuilding|deploying|editing|doing|taking|working|staying|pushing|watching|waiting)|"
    r"\b(?:i\s+am|i(?:'|’)m)\s+going\s+to|\bi(?:'|’)ll\s+"
    r"(?:check|inspect|implement|add|update|run|start|trace|locate|patch|"
    r"finish|test|apply|pull|review|keep|continue|work|watch|wait|back\s+up|"
    r"rebuild|deploy|edit|surface)|"
    r"\blet\s+me\s+(?:check|inspect|run|trace|locate|patch|review)|"
    r"\bnext\s+(?:i(?:'|’)ll|i\s+will)|"
    r"(?m:^\s*(?:launching|checking|inspecting|implementing|updating|running|starting|"
    r"tracing|patching|testing|pulling)\b)"
    r")"
)
_VERIFICATION_RE = re.compile(
    r"(?i)\b(?:verified|confirmed|double-checked|fresh (?:live )?check|current state|"
    r"proof (?:shows|is|was)|(?:preflight|checks?) (?:is |are )?(?:clean|green|pass(?:ed)?)|"
    r"[a-z_-]+ path pass(?:es|ed)|"
    r"tests? passed|exit(?: code)?\s*0)\b"
)
_CONCRETE_RESULT_RE = re.compile(
    r"(?ix)(?:"
    r"\b\d+[\s,]*(?:/\s*\d+|tests?|checks?|files?|rows?|items?|calls?|"
    r"errors?|failures?|jobs?|processes?|submissions?|passed|failed|%)\b|"
    r"\b(?:run|job|build|task|ticket|commit|pid|status)\s*(?:id\s*)?"
    r"[:#=-]\s*[A-Z0-9][A-Z0-9_.-]{2,}\b|"
    r"\b[A-Z_][A-Z0-9_.-]*\s*=\s*\d+(?:\.\d+)?\b|"
    r"`[0-9a-f]{7,40}`|"
    r"\b(?:tests?\s+passed|[a-z_-]+\s+path\s+pass(?:es|ed)|deployed|shipped|merged|"
    r"committed|completed|resolved|"
    r"running|healthy|green|clean|failed|failure|no\s+live\s+submissions?|"
    r"zero\s+(?:errors?|failures?|submissions?))\b"
    r")"
)
_JUDGMENT_RE = re.compile(
    r"(?i)\b(?:because|therefore|which means|so (?:the|this|we|i)|"
    r"i (?:recommend|will|won'?t)|next\s*:|the (?:right|safe|honest) (?:move|step|call)|"
    r"should|must|safe|unsafe|honest|dishonest|fail(?:s)? closed|table stakes|wrong)\b"
)
_COMPLETED_CHANGE_RE = re.compile(
    r"(?i)\b(?:the (?:code-side )?fix is in|i (?:fixed|implemented|added|removed|updated)|"
    r"is implemented|now (?:prefers|uses|rejects|requires)|is set to)\b"
)


def sanitize_sft_text(text: str) -> str:
    """Remove deterministic transport residue and redact training-text PII.

    The historical name is retained for API compatibility, but this is the
    shared training sanitizer. It is intentionally idempotent and is applied to
    every dataset message content plus the quality-gate target.
    """
    cleaned = _REPLY_ROUTING_PREFIX_RE.sub("", text or "")
    cleaned = _UNTRUSTED_ENVELOPE_BLOCK_RE.sub("", cleaned)
    cleaned = redact_secrets(cleaned)
    cleaned = _EMAIL_RE.sub("[REDACTED_EMAIL]", cleaned)
    cleaned = _PHONE_RE.sub("[REDACTED_PHONE]", cleaned)
    cleaned = _LABELED_PRIVATE_ID_RE.sub(r"\1\2[REDACTED_ID]", cleaned)
    cleaned = _ACCOUNT_VALUE_RE.sub(r"\1\2[REDACTED_ID]", cleaned)
    return cleaned.strip()


def is_sft_process_chatter(target_text: str) -> bool:
    """Return true for hollow/process targets that should not train behavior.

    A concrete, verified status with an outcome plus a judgment/commitment is
    admissible.  Explicit BLOCKED reports are also admissible.  The conservative
    default for non-process-shaped prose is admissible.
    """
    target = sanitize_sft_text(target_text)
    if not target or _EXPLICIT_BLOCKED_RE.search(target):
        return False

    # These are transport acknowledgments even when they mention a task id.
    if _HEARTBEAT_PROCESS_RE.search(target):
        return True

    concrete_result = bool(_CONCRETE_RESULT_RE.search(target))
    verified_result = bool(_VERIFICATION_RE.search(target)) and concrete_result
    completed_change = bool(_COMPLETED_CHANGE_RE.search(target))
    judgment = bool(_JUDGMENT_RE.search(target))

    # Acknowledgment is fine when followed by an actual conclusion; otherwise
    # it is a hollow target rather than a reusable answer.
    if _ACK_PREFIX_RE.match(target) and not (verified_result or judgment):
        return True

    intents = list(_FORWARD_INTENT_RE.finditer(target))
    if not intents:
        return False

    # Concrete verified state + the agent's judgment or commitment is useful
    # status, even if it also names the next step.
    if (verified_result and judgment) or (completed_change and intents[0].start() > 0):
        return False

    # Reject intent-dominant turns, and turns whose final clause is merely what
    # the agent is about to do.  Earlier "I'll explain..." scaffolding followed
    # by a substantive explanation does not fire this final-clause check.
    after_last_intent = target[intents[-1].end() :].rstrip(" \t\r\n.!?;:")
    later_sentence = re.search(r"[.!?]\s+(?=\S)", after_last_intent)
    return len(intents) >= 2 or later_sentence is None


def scrub_reasons(target_text: str, example_text: str, *, dataset: str | None = None) -> list[str]:
    """Return the exclusion-rule slugs that fire (empty == clean).

    ``target_text`` is the assistant/chosen content (already secret-redacted by
    the caller); ``example_text`` is the full serialized record. ``near_dup`` is
    handled in :func:`store_example`, not here.
    """
    reasons: list[str] = []
    target = target_text or ""
    stripped = target.strip()

    # 1. secret_residue — leaks survive redaction.
    if find_probable_secret_leaks(target):
        reasons.append("secret_residue")
    # 2. entropy_blob — long high-entropy runs that redaction can't touch.
    if find_high_entropy_spans(target):
        reasons.append("entropy_blob")
    # 3. length — target too short or whole example too large.
    if len(stripped) < MIN_TARGET_CHARS or len(example_text) > MAX_EXAMPLE_CHARS:
        reasons.append("length")
    # 5. refusal_only — the target is nothing but an apology/refusal.
    if _REFUSAL_RE.match(stripped) and len(stripped) < 240:
        reasons.append("refusal_only")
    # 6. error_dump — target is mostly a stack trace / tool noise.
    if _TRACEBACK_RE.search(target) or len(_STACK_LINE_RE.findall(target)) >= 2:
        reasons.append("error_dump")
    # 7. managed_block_leak — never train on injected memory blocks.
    if _MANAGED_BLOCK_RE.search(target):
        reasons.append("managed_block_leak")
    # 8. envelope_residue — an unparsed telegram envelope fragment remains.
    if _ENVELOPE_RESIDUE_RE.search(target) or _ENVELOPE_RESIDUE_RE.search(example_text):
        reasons.append("envelope_residue")
    # Runtime/transport wrappers in any serialized message are hard failures.
    # This is separate from the legacy injection advisory below.
    if _TRANSPORT_RESIDUE_RE.search(example_text):
        reasons.append("transport_residue")
    # 9. injection_flagged — an injection pattern hides in the target.
    if find_probable_injection(target):
        reasons.append("injection_flagged")
    # SFT-specific terminality gate.  Persona and DPO have separate voice/pair
    # rubrics and must not inherit this behavioral selector.
    if dataset == "sft" and is_sft_process_chatter(target):
        reasons.append("process_chatter")
    return reasons


def _redact_body(value: Any) -> Any:
    """Deep-copy ``value`` redacting every ``content`` string (chat + DPO shapes)."""
    if isinstance(value, dict):
        return {
            key: redact_secrets(val)
            if key == "content" and isinstance(val, str)
            else _redact_body(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_body(item) for item in value]
    return value


def _sanitize_sft_body(value: Any) -> Any:
    """Deep-copy a training body cleaning every message ``content`` string."""
    if isinstance(value, dict):
        return {
            key: sanitize_sft_text(val)
            if key == "content" and isinstance(val, str)
            else _sanitize_sft_body(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_sft_body(item) for item in value]
    return value


def _existing_dedup(conn: sqlite3.Connection, dataset: str, dedup_key: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM dataset_examples
        WHERE dataset = ? AND dedup_key = ? AND quality_label != 'excluded'
        LIMIT 1
        """,
        (dataset, dedup_key),
    ).fetchone()
    return row is not None


def store_example(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    source_kind: str,
    source_uri: str | None,
    source_span: Any = None,
    evidence_ids: list[str],
    privacy_scope: str,
    body: dict[str, Any],
    metadata: dict[str, Any],
    target_text: str,
    base_label: str,
    base_confidence: float | None,
    base_reasons: list[str] | None = None,
    n_turns: int | None = None,
    session_id: str | None = None,
    occurred_at: str | None = None,
    write_batch: DatasetWriteBatch | None = None,
) -> dict[str, Any] | None:
    """Scrub, dedup, and idempotently upsert one example. Returns the stored dict.

    ``body`` is the JSONL record without metadata (``{"messages": [...]}`` for
    chat, the ``input``/``*_output`` triple for DPO). ``content_hash`` is taken
    over ``body`` ONLY so re-mines are stable (spec §7.3). Provenance is
    enforced: ``evidence_ids`` must be non-empty.
    """
    if not evidence_ids:
        raise ValueError("every dataset example needs >=1 evidence id (provenance)")
    if dataset not in ("sft", "dpo", "persona"):
        raise ValueError(f"unknown dataset: {dataset}")

    if write_batch is not None:
        # Evidence/source writers may enter with a short active transaction.
        # Close it before redaction/dedup. Prepared dataset INSERTs are buffered
        # separately and therefore do not own SQLite while this work runs.
        if conn.in_transaction:
            write_batch.flush()
        else:
            write_batch.flush_if_expired()

    # Final secret redaction pass over the target AND the stored body, so no raw
    # secret survives in the exported record (spec §7.4 rule 1). Redaction is
    # deterministic, keeping content_hash stable across re-mines.
    target_text = redact_secrets(target_text or "")
    body = _redact_body(body)
    # Repeat shared cleanup here even when a miner already performed it.
    # ``store_example`` is the final fail-safe used by every ingestion path.
    target_text = sanitize_sft_text(target_text)
    body = _sanitize_sft_body(body)
    canonical_body = canonical_json(body)
    dedup_key = claim_key(target_text)

    # base_reasons are the label rationale (e.g. "affirmation"); only scrub/near-dup
    # hits actually downgrade the row to excluded.
    reasons = list(base_reasons or [])
    label = base_label
    confidence = base_confidence
    scrub = scrub_reasons(target_text, canonical_body, dataset=dataset)
    # injection_flagged is ADVISORY (spec R2): a flagged example STAYS in the
    # dataset — knowledge-layer quarantine is the enforcement path, and the count
    # is surfaced in the export manifest. It is recorded in quality_reasons but
    # never excludes an example on its own. Every other scrub reason is hard.
    hard_scrub = [r for r in scrub if r != "injection_flagged"]
    if hard_scrub:
        label = "excluded"
        reasons.extend(hard_scrub)
    elif _existing_dedup(conn, dataset, dedup_key) or (
        write_batch is not None and write_batch.pending_dedup(dataset, dedup_key)
    ):
        label = "excluded"
        reasons.append("near_dup")
    if "injection_flagged" in scrub:
        reasons.append("injection_flagged")

    digest = content_hash(canonical_body)
    example_id = stable_id("dsx", dataset, digest)

    full_metadata = dict(metadata)
    full_metadata.update(
        {
            "id": example_id,
            "dataset": dataset,
            "content_hash": digest,
            "quality_label": label,
            "quality_confidence": confidence,
            "quality_reasons": reasons,
            "privacy_scope": privacy_scope,
            "evidence_ids": list(evidence_ids),
            "source_kind": source_kind,
            "source_uri": source_uri,
            "occurred_at": occurred_at,
        }
    )
    example_record = dict(body)
    example_record["metadata"] = full_metadata
    example_json = canonical_json(example_record)
    n_chars = len(example_json)
    ts = now_iso()

    # Redaction, serialization, scrub rules, and dedup lookup can be expensive
    # for long persona examples. Do all of that before acquiring SQLite's
    # single-writer slot; the transaction owns only the final INSERT.
    statement = """
        INSERT INTO dataset_examples (
          id, dataset, content_hash, dedup_key, source_kind, source_uri,
          source_span, evidence_ids, privacy_scope, quality_label,
          quality_confidence, quality_reasons, n_turns, n_chars, example_json,
          session_id, occurred_at, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset, content_hash) DO NOTHING
        """
    params = (
        example_id,
        dataset,
        digest,
        dedup_key,
        source_kind,
        source_uri,
        canonical_json(source_span) if source_span is not None else None,
        canonical_json(list(evidence_ids)),
        privacy_scope,
        label,
        confidence,
        canonical_json(reasons),
        n_turns,
        n_chars,
        example_json,
        session_id,
        occurred_at,
        ts,
        ts,
    )
    if write_batch is not None:
        write_batch.queue(
            statement,
            params,
            dedup=(dataset, dedup_key) if label != "excluded" else None,
        )
    else:
        conn.execute(statement, params)
    return {
        "id": example_id,
        "dataset": dataset,
        "content_hash": digest,
        "dedup_key": dedup_key,
        "quality_label": label,
        "quality_confidence": confidence,
        "quality_reasons": reasons,
        "privacy_scope": privacy_scope,
        "evidence_ids": list(evidence_ids),
        "example_json": example_json,
    }
