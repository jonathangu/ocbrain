from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from ocbrain.db import now_iso
from ocbrain.ids import stable_id
from ocbrain.scope import ScopeContext, ScopeTag, resolve_write_scope, scope_match

EVENT_KINDS = {
    "evidence_recorded",
    "compilation_proposed",
    "compilation_decided",
    "correction_recorded",
    "tombstone_recorded",
    "scope_promoted",
}
REWARD_BANDS = {"discard", "weak", "moderate", "strong"}
DECISIONS = ("approve", "reject", "edit", "shadow")


def append_event(
    conn: sqlite3.Connection,
    kind: str,
    body: dict[str, Any],
    *,
    writer: str = "ocbrain",
    session_id: str | None = None,
) -> str:
    if kind not in EVENT_KINDS:
        raise ValueError(f"invalid event kind: {kind}")
    ts = now_iso()
    body_json = canonical_json(body)
    body_hash = sha256_text(body_json)
    # The chain-tail read and the insert must be one atomic unit: two concurrent
    # writers that both read the same tail would insert sibling events claiming
    # the same prev_hash and silently fork the tamper-evidence chain. BEGIN
    # IMMEDIATE takes the write lock before the tail is read; if this connection
    # already has a transaction open, the caller owns the write window and the
    # read+insert stay inside it.
    started_transaction = not conn.in_transaction
    if started_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        prev_hash = conn.execute(
            "SELECT event_hash FROM brain_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        prev_hash_text = prev_hash["event_hash"] if prev_hash else None
        event_hash = sha256_text(
            canonical_json(
                {
                    "ts": ts,
                    "kind": kind,
                    "writer": writer,
                    "session_id": session_id,
                    "body_hash": body_hash,
                    "prev_hash": prev_hash_text,
                }
            )
        )
        event_id = stable_id("evt", kind, event_hash)
        conn.execute(
            """
            INSERT INTO brain_events (
              id, ts, kind, writer, session_id, body_json, body_hash, prev_hash, event_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                ts,
                kind,
                writer,
                session_id,
                body_json,
                body_hash,
                prev_hash_text,
                event_hash,
            ),
        )
    except BaseException:
        if started_transaction and conn.in_transaction:
            conn.rollback()
        raise
    return event_id


def record_evidence(
    conn: sqlite3.Connection,
    *,
    body: str,
    kind: str = "observation",
    context: ScopeContext | None = None,
    scope: ScopeTag | dict[str, Any] | None = None,
    writer: str = "ocbrain",
    session_id: str | None = None,
    artifact_ref: str | None = None,
) -> str:
    scope_tag = resolve_write_scope(context, explicit=scope)
    evidence_id = evidence_id_for(
        body=body,
        kind=kind,
        artifact_ref=artifact_ref,
        scope=scope_tag,
    )
    return append_event(
        conn,
        "evidence_recorded",
        {
            "evidence_id": evidence_id,
            "kind": kind,
            "body": body,
            "artifact_ref": artifact_ref,
            "scope": scope_tag.to_dict(),
        },
        writer=writer,
        session_id=session_id,
    )


def evidence_id_for(
    *,
    body: str,
    kind: str,
    artifact_ref: str | None,
    scope: ScopeTag | dict[str, Any],
) -> str:
    scope_tag = scope if isinstance(scope, ScopeTag) else ScopeTag.from_dict(scope)
    return stable_id("evd", body, kind, artifact_ref or "", scope_tag.scope_id)


def propose_compilation(
    conn: sqlite3.Connection,
    *,
    belief_id: str,
    body: str,
    evidence_ids: list[str],
    scope: ScopeTag | dict[str, Any],
    confidence: float | None = None,
    teacher_model: str | None = None,
    teacher_rationale: str | None = None,
    reward_band: str | None = None,
    writer: str = "teacher",
    session_id: str | None = None,
    check_hard_block: bool = True,
) -> str:
    if not evidence_ids:
        raise ValueError("compiled beliefs require at least one evidence id")
    if check_hard_block and hard_blocked_belief(conn, belief_id):
        raise PermissionError(f"belief is blocked by a hard correction: {belief_id}")
    if reward_band is not None and reward_band not in REWARD_BANDS:
        allowed = ", ".join(sorted(REWARD_BANDS))
        raise ValueError(f"reward_band must be one of: {allowed}")
    scope_tag = scope if isinstance(scope, ScopeTag) else ScopeTag.from_dict(scope)
    return append_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "body": body,
            "evidence_ids": evidence_ids,
            "scope": scope_tag.to_dict(),
            "confidence": confidence,
            "teacher_model": teacher_model,
            "teacher_rationale": teacher_rationale,
            "reward_band": reward_band,
        },
        writer=writer,
        session_id=session_id,
    )


def decide_compilation(
    conn: sqlite3.Connection,
    *,
    proposal_event_id: str,
    decision: str,
    actor: str = "human:jonathan",
    edited_body: str | None = None,
    reason: str | None = None,
    rebuild: bool = True,
    check_existing: bool = True,
) -> str:
    if decision not in DECISIONS:
        raise ValueError("decision must be approve, reject, edit, or shadow")
    if check_existing:
        existing = proposal_decision_event(conn, proposal_event_id)
        if existing is not None:
            raise ValueError(f"proposal already decided: {proposal_event_id}")
    if decision in {"approve", "edit"}:
        belief_id = proposal_belief_id(conn, proposal_event_id)
        if belief_id is not None:
            if hard_blocked_belief(conn, belief_id):
                raise PermissionError(
                    f"cannot {decision}: belief is blocked by a hard correction: {belief_id}"
                )
            if tombstoned_belief(conn, belief_id):
                raise PermissionError(
                    f"cannot {decision}: belief is tombstoned: {belief_id}"
                )
    event_id = append_event(
        conn,
        "compilation_decided",
        {
            "proposal_event_id": proposal_event_id,
            "decision": decision,
            "actor": actor,
            "edited_body": edited_body,
            "reason": reason,
        },
        writer=actor,
    )
    if rebuild:
        rebuild_projection(conn)
    return event_id


def record_correction(
    conn: sqlite3.Connection,
    *,
    target_layer: str,
    target_id: str,
    op: str,
    body: str | None,
    author: str = "human:jonathan",
    hard: bool = False,
) -> str:
    if target_layer == "evidence":
        raise ValueError(
            "evidence-layer corrections are not applied by the projection; "
            "correct the derived belief instead"
        )
    if target_layer not in {"knowledge", "belief"}:
        raise ValueError("target_layer must be knowledge or belief")
    event_id = append_event(
        conn,
        "correction_recorded",
        {
            "target_layer": target_layer,
            "target_id": target_id,
            "op": op,
            "body": body,
            "author": author,
            "hard": hard,
        },
        writer=author,
    )
    rebuild_projection(conn)
    return event_id


def record_tombstone(
    conn: sqlite3.Connection,
    *,
    target: str,
    mode: str,
    reason: str | None,
    approved_by: str = "human:jonathan",
) -> str:
    if mode not in {"soft", "shred"}:
        raise ValueError("mode must be soft or shred")
    body: dict[str, Any] = {
        "target": target,
        "target_hash": sha256_text(target),
        "mode": mode,
        "reason": reason,
        "approved_by": approved_by,
    }
    if mode == "shred":
        body["serving_policy"] = "redact_projection_body_and_evidence_ids"
    event_id = append_event(
        conn,
        "tombstone_recorded",
        body,
        writer=approved_by,
    )
    rebuild_projection(conn)
    return event_id


def rebuild_projection(conn: sqlite3.Connection, *, at_ts: str | None = None) -> None:
    projected = fold_projection(conn, at_ts=at_ts)
    replace_projection(conn, projected)


def fold_projection(
    conn: sqlite3.Connection, *, at_ts: str | None = None
) -> dict[str, dict[str, Any]]:
    events = list(iter_events(conn, at_ts=at_ts))
    proposals: dict[str, dict[str, Any]] = {}
    projected: dict[str, dict[str, Any]] = {}
    tombstoned_targets: dict[str, dict[str, Any]] = {}

    for event in events:
        body = json.loads(event["body_json"])
        kind = event["kind"]
        if kind == "compilation_proposed":
            proposals[event["id"]] = {"event": event, "body": body}
        elif kind == "compilation_decided":
            apply_decision(event, body, proposals, projected)
        elif kind == "correction_recorded":
            apply_correction(event, body, projected)
        elif kind == "tombstone_recorded":
            target = body["target"]
            tombstone = tombstoned_targets.setdefault(target, {"mode": "soft"})
            if body.get("mode") == "shred":
                tombstone["mode"] = "shred"
            tombstone["event_id"] = event["id"]
        elif kind == "scope_promoted":
            apply_scope_promotion(event, body, projected)

    # Tombstones win regardless of event order at fold time: a compilation
    # decision (or any other event) replaying after a tombstone for the same
    # target must never resurrect a tombstoned/shredded belief, so tombstones
    # are applied last over whatever the fold projected.
    for target, tombstone in tombstoned_targets.items():
        belief = projected.get(target)
        if belief is None:
            continue
        belief["status"] = "tombstoned"
        if tombstone["mode"] == "shred":
            belief["body"] = "[shredded by tombstone]"
            belief["evidence_ids"] = []
        belief["last_event_id"] = tombstone["event_id"]

    return projected


def replace_projection(conn: sqlite3.Connection, projected: dict[str, dict[str, Any]]) -> None:
    conn.execute("DELETE FROM current_beliefs")
    for belief_id, row in projected.items():
        conn.execute(
            """
            INSERT INTO current_beliefs (
              belief_id, body, scope_type, scope_id, visibility, egress_policy,
              confidence, confidence_band, evidence_ids, status, pinned,
              approved_event_id, last_event_id, last_compiled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                belief_id,
                row["body"],
                row["scope"]["scope_type"],
                row["scope"]["scope_id"],
                row["scope"]["visibility"],
                row["scope"]["egress_policy"],
                row["confidence"],
                confidence_band(row["confidence"]),
                canonical_json(row["evidence_ids"]),
                row["status"],
                1 if row["pinned"] else 0,
                row["approved_event_id"],
                row["last_event_id"],
                row["last_compiled_at"],
            ),
        )


def projected_rows_as_of(conn: sqlite3.Connection, *, at_ts: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for belief_id, row in fold_projection(conn, at_ts=at_ts).items():
        rows.append(
            {
                "belief_id": belief_id,
                "body": row["body"],
                "scope_type": row["scope"]["scope_type"],
                "scope_id": row["scope"]["scope_id"],
                "visibility": row["scope"]["visibility"],
                "egress_policy": row["scope"]["egress_policy"],
                "confidence": row["confidence"],
                "confidence_band": confidence_band(row["confidence"]),
                "evidence_ids": canonical_json(row["evidence_ids"]),
                "status": row["status"],
                "pinned": 1 if row["pinned"] else 0,
                "approved_event_id": row["approved_event_id"],
                "last_event_id": row["last_event_id"],
                "last_compiled_at": row["last_compiled_at"],
            }
        )
    return rows


def iter_events(conn: sqlite3.Connection, *, at_ts: str | None = None):
    if at_ts is None:
        yield from conn.execute("SELECT * FROM brain_events ORDER BY rowid ASC")
    else:
        yield from conn.execute(
            "SELECT * FROM brain_events WHERE ts <= ? ORDER BY rowid ASC", (at_ts,)
        )


def proposal_belief_id(conn: sqlite3.Connection, proposal_event_id: str) -> str | None:
    row = conn.execute(
        "SELECT body_json FROM brain_events WHERE id = ? AND kind = 'compilation_proposed'",
        (proposal_event_id,),
    ).fetchone()
    if row is None:
        return None
    belief_id = json.loads(row["body_json"]).get("belief_id")
    return str(belief_id) if belief_id else None


def tombstoned_belief(conn: sqlite3.Connection, belief_id: str) -> bool:
    for row in iter_events(conn):
        if row["kind"] != "tombstone_recorded":
            continue
        body = json.loads(row["body_json"])
        if body.get("target") == belief_id:
            return True
    return False


def hard_blocked_belief(conn: sqlite3.Connection, belief_id: str) -> bool:
    for row in iter_events(conn):
        if row["kind"] != "correction_recorded":
            continue
        body = json.loads(row["body_json"])
        if body.get("target_layer") not in {"knowledge", "belief"}:
            continue
        if body.get("target_id") != belief_id:
            continue
        if not body.get("hard"):
            continue
        if body.get("op") in {"mark_wrong", "retract", "demote"}:
            return True
    return False


def get_current_belief(conn: sqlite3.Connection, belief_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM current_beliefs WHERE belief_id = ? LIMIT 1", (belief_id,)
    ).fetchone()
    if row is None:
        return None
    evidence_ids = json.loads(row["evidence_ids"])
    tombstoned = row["status"] == "tombstoned"
    shredded = tombstoned and row["body"] == "[shredded by tombstone]"
    provenance = []
    for event in iter_events(conn):
        body = json.loads(event["body_json"])
        if event["kind"] == "compilation_proposed" and body.get("belief_id") == belief_id:
            provenance.append(event_summary(event, body))
        elif event["kind"] == "compilation_decided" and row["approved_event_id"] == event["id"]:
            provenance.append(event_summary(event, body))
        elif event["kind"] in {"correction_recorded", "tombstone_recorded"} and (
            body.get("target_id") == belief_id or body.get("target") == belief_id
        ):
            provenance.append(event_summary(event, body))
    if tombstoned:
        # Never re-serve tombstoned/shredded content through provenance: the
        # projection row is the redaction source of truth, so event bodies must
        # not leak what the tombstone removed.
        provenance = [
            redact_provenance_entry(entry, shredded=shredded) for entry in provenance
        ]
    return {
        "object_kind": "belief",
        "belief_id": row["belief_id"],
        "body": row["body"],
        "scope": {
            "scope_type": row["scope_type"],
            "scope_id": row["scope_id"],
            "visibility": row["visibility"],
            "egress_policy": row["egress_policy"],
        },
        "confidence": row["confidence"],
        "confidence_band": row["confidence_band"],
        "evidence_ids": evidence_ids,
        "status": row["status"],
        "pinned": bool(row["pinned"]),
        "approved_event_id": row["approved_event_id"],
        "last_event_id": row["last_event_id"],
        "last_compiled_at": row["last_compiled_at"],
        "provenance": provenance,
        "evidence_provenance": evidence_provenance(conn, evidence_ids),
    }


def evidence_provenance(
    conn: sqlite3.Connection, evidence_ids: list[str]
) -> list[dict[str, Any]]:
    if not evidence_ids:
        return []
    wanted = set(evidence_ids)
    rows: list[dict[str, Any]] = []
    for event in iter_events(conn):
        if event["kind"] != "evidence_recorded":
            continue
        body = json.loads(event["body_json"])
        if body.get("evidence_id") not in wanted:
            continue
        source = body.get("artifact_ref")
        rows.append(
            {
                "evidence_id": body.get("evidence_id"),
                "event_id": event["id"],
                "ts": event["ts"],
                "kind": body.get("kind"),
                "source": source,
                "source_kind": source_kind(str(source or "")),
                "body_hash": sha256_text(str(body.get("body") or "")),
                "scope": ScopeTag.from_dict(body.get("scope")).to_dict(),
            }
        )
    return rows


def source_kind(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return "web"
    if source:
        return "local"
    return "inline"


def event_summary(event: sqlite3.Row, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["id"],
        "ts": event["ts"],
        "kind": event["kind"],
        "writer": event["writer"],
        "body": body,
    }


def redact_provenance_entry(entry: dict[str, Any], *, shredded: bool) -> dict[str, Any]:
    """Strip belief content from a provenance entry for a tombstoned belief.

    Soft tombstones hide proposal/decision bodies; shred tombstones redact every
    body-bearing field plus evidence ids, matching the projection serving policy.
    """
    kind = entry["kind"]
    if not shredded and kind not in {"compilation_proposed", "compilation_decided"}:
        return entry
    marker = "[shredded by tombstone]" if shredded else "[redacted by tombstone]"
    body = dict(entry["body"])
    if body.get("body"):
        body["body"] = marker
    if body.get("edited_body"):
        body["edited_body"] = marker
    if shredded and body.get("evidence_ids"):
        body["evidence_ids"] = []
    return entry | {"body": body}


def proposal_decision_event(
    conn: sqlite3.Connection, proposal_event_id: str
) -> dict[str, Any] | None:
    for event in iter_events(conn):
        if event["kind"] != "compilation_decided":
            continue
        body = json.loads(event["body_json"])
        if body.get("proposal_event_id") == proposal_event_id:
            return event_summary(event, body)
    return None


def scope_visible(scope: ScopeTag, context: ScopeContext) -> bool:
    """Mirror retrieve()'s hard exclusion for scoped read surfaces.

    An empty/absent context is most-restrictive for confidential data, never a
    bypass: without a matching context, confidential/secret rows stay hidden.
    """
    if context.to_dict():
        return scope_match(scope, context) != 0
    return not scope.confidential


def list_compilation_proposals(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext | None = None,
    include_decided: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    context = context or ScopeContext()
    proposals: list[dict[str, Any]] = []
    decisions = proposal_decisions(conn)
    for event in iter_events(conn):
        if event["kind"] != "compilation_proposed":
            continue
        body = json.loads(event["body_json"])
        scope = ScopeTag.from_dict(body.get("scope"))
        if not scope_visible(scope, context):
            continue
        decision = decisions.get(event["id"])
        if decision is not None and not include_decided:
            continue
        proposals.append(
            {
                "proposal_event_id": event["id"],
                "ts": event["ts"],
                "writer": event["writer"],
                "status": "decided" if decision else "pending",
                "decision": decision,
                "belief_id": body.get("belief_id"),
                "body": body.get("body"),
                "scope": scope.to_dict(),
                "confidence": body.get("confidence"),
                "confidence_band": confidence_band(body.get("confidence")),
                "evidence_ids": body.get("evidence_ids") or [],
                "teacher_model": body.get("teacher_model"),
                "teacher_rationale": body.get("teacher_rationale"),
                "reward_band": body.get("reward_band"),
            }
        )
    return proposals[-limit:]


def proposal_decisions(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for event in iter_events(conn):
        if event["kind"] != "compilation_decided":
            continue
        body = json.loads(event["body_json"])
        proposal_event_id = body.get("proposal_event_id")
        if proposal_event_id:
            decisions[str(proposal_event_id)] = event_summary(event, body)
    return decisions


def approval_packet(
    proposals: list[dict[str, Any]],
    *,
    context: ScopeContext | None = None,
    actor: str = "human:jonathan",
    cli_prefix: list[str] | None = None,
) -> dict[str, Any]:
    """Return a local, send-ready approval packet without contacting a messenger."""
    context = context or ScopeContext()
    prefix = cli_prefix or ["ocbrain"]
    pending = [proposal for proposal in proposals if proposal["status"] == "pending"]
    items = [
        approval_packet_item(proposal, actor=actor, cli_prefix=prefix)
        for proposal in pending
    ]
    return {
        "channel": "telegram",
        "send_performed": False,
        "context": context.to_dict(),
        "actor": actor,
        "summary": {
            "pending": len(pending),
            "total_proposals": len(proposals),
        },
        "text": telegram_approval_text(items),
        "items": items,
    }


def approval_packet_item(
    proposal: dict[str, Any], *, actor: str, cli_prefix: list[str]
) -> dict[str, Any]:
    proposal_id = proposal["proposal_event_id"]
    actions = {
        decision: {
            "decision": decision,
            "mcp_tool": "brain.feedback",
            "mcp_arguments": {
                "proposal_event_id": proposal_id,
                "decision": decision,
                "actor": actor,
            },
            "cli_argv": [
                *cli_prefix,
                "event-decide",
                "--proposal-event-id",
                proposal_id,
                "--decision",
                decision,
                "--actor",
                actor,
            ],
        }
        for decision in DECISIONS
    }
    return {
        "proposal_event_id": proposal_id,
        "belief_id": proposal["belief_id"],
        "scope": proposal["scope"],
        "confidence_band": proposal["confidence_band"],
        "reward_band": proposal["reward_band"],
        "body": proposal["body"],
        "actions": actions,
    }


def telegram_approval_text(items: list[dict[str, Any]]) -> str:
    if not items:
        return "OCBrain gate: no pending compilation proposals."
    lines = [f"OCBrain gate: {len(items)} pending compilation proposal(s)."]
    for index, item in enumerate(items, start=1):
        scope_id = item["scope"]["scope_id"]
        body = item["body"] or ""
        if len(body) > 160:
            body = f"{body[:157]}..."
        lines.extend(
            [
                "",
                f"{index}. {item['belief_id']} [{scope_id}]",
                body,
                f"Approve: /ocbrain_gate approve {item['proposal_event_id']}",
                f"Reject: /ocbrain_gate reject {item['proposal_event_id']}",
                f"Shadow: /ocbrain_gate shadow {item['proposal_event_id']}",
            ]
        )
    return "\n".join(lines)


def event_core_digest(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext | None = None,
    since_ts: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    context = context or ScopeContext()
    event_counts: dict[str, int] = {}
    recent_events: list[dict[str, Any]] = []
    for event in iter_events(conn):
        if since_ts and event["ts"] <= since_ts:
            continue
        body = json.loads(event["body_json"])
        scope_data = body.get("scope")
        if scope_data and not scope_visible(ScopeTag.from_dict(scope_data), context):
            continue
        event_counts[event["kind"]] = event_counts.get(event["kind"], 0) + 1
        recent_events.append(
            {
                "event_id": event["id"],
                "ts": event["ts"],
                "kind": event["kind"],
                "writer": event["writer"],
            }
        )
    current_beliefs = scoped_current_beliefs(conn, context=context, limit=limit)
    pending = list_compilation_proposals(conn, context=context, limit=limit)
    return {
        "context": context.to_dict(),
        "since_ts": since_ts,
        "event_counts": event_counts,
        "pending_compilations": pending,
        "current_beliefs": current_beliefs,
        "recent_events": recent_events[-limit:],
        "runtime_health": runtime_health(conn, context=context),
        "quiet_loop": quiet_loop_surface(
            conn,
            context=context,
            event_counts=event_counts,
            pending=pending,
            current_beliefs=current_beliefs,
        ),
        "summary": {
            "events": sum(event_counts.values()),
            "pending_compilations": len(pending),
            "current_beliefs": len(current_beliefs),
        },
    }


def quiet_loop_surface(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    event_counts: dict[str, int],
    pending: list[dict[str, Any]],
    current_beliefs: list[dict[str, Any]],
) -> dict[str, Any]:
    total_current = count_scoped_current(conn, context=context)
    checks = [
        {
            "name": "no_pending_compilations",
            "passed": len(pending) == 0,
            "observed": len(pending),
            "expected": 0,
        },
        {
            "name": "has_current_projection",
            "passed": total_current > 0,
            "observed": total_current,
            "expected": ">0",
        },
        {
            "name": "has_useful_write",
            "passed": any(
                event_counts.get(kind, 0)
                for kind in ("evidence_recorded", "compilation_decided")
            ),
            "observed": {
                "evidence_recorded": event_counts.get("evidence_recorded", 0),
                "compilation_decided": event_counts.get("compilation_decided", 0),
            },
            "expected": "one_or_more",
        },
    ]
    state = "quiet" if all(check["passed"] for check in checks) else "attention"
    return {
        "state": state,
        "claim": (
            "quiet means there is a scoped projection, no visible pending gate work, "
            "and at least one useful ledger write in the selected event window"
        ),
        "falsifiable_checks": checks,
        "sample_current_beliefs": [row["belief_id"] for row in current_beliefs[:5]],
    }


def count_scoped_current(conn: sqlite3.Connection, *, context: ScopeContext) -> int:
    total = 0
    for row in conn.execute(
        """
        SELECT scope_type, scope_id, visibility, egress_policy
        FROM current_beliefs
        WHERE status = 'current'
        """
    ):
        scope = ScopeTag(
            row["scope_type"],
            row["scope_id"],
            visibility=row["visibility"],
            egress_policy=row["egress_policy"],
        )
        if not scope_visible(scope, context):
            continue
        total += 1
    return total


def runtime_health(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext | None = None,
) -> list[dict[str, Any]]:
    context = context or ScopeContext()
    useful_kinds = {
        "evidence_recorded",
        "compilation_decided",
        "correction_recorded",
        "tombstone_recorded",
    }
    by_writer: dict[str, dict[str, Any]] = {}
    for event in iter_events(conn):
        body = json.loads(event["body_json"])
        scope_data = body.get("scope")
        if scope_data and not scope_visible(ScopeTag.from_dict(scope_data), context):
            continue
        writer = event["writer"]
        row = by_writer.setdefault(
            writer,
            {
                "writer": writer,
                "last_write_at": None,
                "last_useful_write_at": None,
                "last_useful_kind": None,
                "sessions": set(),
                "writes": 0,
                "useful_writes": 0,
            },
        )
        row["last_write_at"] = event["ts"]
        row["writes"] += 1
        if event["session_id"]:
            row["sessions"].add(event["session_id"])
        if event["kind"] in useful_kinds:
            row["last_useful_write_at"] = event["ts"]
            row["last_useful_kind"] = event["kind"]
            row["useful_writes"] += 1
    rows = []
    for row in by_writer.values():
        rows.append(row | {"sessions": sorted(row["sessions"])})
    return sorted(
        rows,
        key=lambda item: (item["last_useful_write_at"] or "", item["writer"]),
        reverse=True,
    )


def scoped_current_beliefs(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in conn.execute(
        """
        SELECT *
        FROM current_beliefs
        ORDER BY pinned DESC, last_compiled_at DESC, belief_id ASC
        """
    ):
        scope = ScopeTag(
            row["scope_type"],
            row["scope_id"],
            visibility=row["visibility"],
            egress_policy=row["egress_policy"],
        )
        if not scope_visible(scope, context):
            continue
        rows.append(
            {
                "belief_id": row["belief_id"],
                "body": row["body"],
                "scope": scope.to_dict(),
                "confidence_band": row["confidence_band"],
                "status": row["status"],
                "last_compiled_at": row["last_compiled_at"],
                "pinned": bool(row["pinned"]),
            }
        )
    return rows[:limit]


def apply_decision(
    event: sqlite3.Row,
    decision_body: dict[str, Any],
    proposals: dict[str, dict[str, Any]],
    projected: dict[str, dict[str, Any]],
) -> None:
    proposal = proposals.get(decision_body["proposal_event_id"])
    if proposal is None:
        return
    decision = decision_body["decision"]
    if decision not in {"approve", "edit"}:
        return
    proposal_body = proposal["body"]
    belief_id = proposal_body["belief_id"]
    body = decision_body.get("edited_body") or proposal_body["body"]
    projected[belief_id] = {
        "body": body,
        "scope": proposal_body["scope"],
        "confidence": proposal_body.get("confidence"),
        "evidence_ids": proposal_body["evidence_ids"],
        "status": "current",
        "pinned": False,
        "approved_event_id": event["id"],
        "last_event_id": event["id"],
        "last_compiled_at": event["ts"],
    }


def apply_correction(
    event: sqlite3.Row,
    correction: dict[str, Any],
    projected: dict[str, dict[str, Any]],
) -> None:
    if correction["target_layer"] not in {"knowledge", "belief"}:
        return
    belief = projected.get(correction["target_id"])
    if belief is None:
        return
    op = correction["op"]
    if op in {"edit", "reframe"} and correction.get("body"):
        belief["body"] = correction["body"]
    elif op == "pin":
        belief["pinned"] = True
    elif op == "demote":
        confidence = belief.get("confidence")
        belief["confidence"] = min(confidence or 0.5, 0.4)
    elif op in {"mark_wrong", "retract"}:
        belief["status"] = "retracted"
    belief["last_event_id"] = event["id"]


def apply_scope_promotion(
    event: sqlite3.Row,
    body: dict[str, Any],
    projected: dict[str, dict[str, Any]],
) -> None:
    belief = projected.get(body.get("belief_id"))
    if belief is None:
        return
    if body.get("approved_by"):
        belief["scope"] = ScopeTag.from_dict(body["scope"]).to_dict()
        belief["last_event_id"] = event["id"]


def confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.45:
        return "moderate"
    return "weak"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
