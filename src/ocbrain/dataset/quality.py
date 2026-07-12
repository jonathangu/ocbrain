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

from ocbrain.dataset.batching import DatasetWriteBatch
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
_ENVELOPE_RESIDUE_RE = re.compile(r"Conversation info \(untrusted metadata\)")
_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
_STACK_LINE_RE = re.compile(r'^\s*File ".*", line \d+', re.MULTILINE)


def scrub_reasons(target_text: str, example_text: str) -> list[str]:
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
    if _ENVELOPE_RESIDUE_RE.search(target):
        reasons.append("envelope_residue")
    # 9. injection_flagged — an injection pattern hides in the target.
    if find_probable_injection(target):
        reasons.append("injection_flagged")
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
    canonical_body = canonical_json(body)
    dedup_key = claim_key(target_text)

    # base_reasons are the label rationale (e.g. "affirmation"); only scrub/near-dup
    # hits actually downgrade the row to excluded.
    reasons = list(base_reasons or [])
    label = base_label
    confidence = base_confidence
    scrub = scrub_reasons(target_text, canonical_body)
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
