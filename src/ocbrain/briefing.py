"""Session-start reorientation: the briefing contract, goals, and the ledger.

Three objects live here, and they exist for one reason: a harness that starts a
fresh context window every iteration has to re-acquire its bearings the same way
every time, or it drifts.

``brain.context`` cannot do that job. It is a *query*: it ranks, it takes free
text, and two calls a minute apart can legitimately return different material.
That is correct for "what do I know about X" and wrong for "where was I". A loop
that reorients through a ranked query re-reads a different brain each iteration.

So this module adds three things that are deliberately *not* queries.

``briefing``
    A contract. Same scope plus same corpus state gives byte-identical output.
    No similarity ranking appears anywhere in it -- order and selection are
    rules. It is hard-bounded in characters, and truncation is counted out loud
    rather than happening silently, because a payload that quietly grows is a
    payload that quietly poisons the window it is injected into.

``goals``
    An objective, its executable finish line, and a *pointer* to the spec. The
    spec itself stays in the repo, git-versioned and human-reviewable. The brain
    pins the pointer and never becomes the editable home of a requirement.

``ledger``
    A read-only projection over ``task_closeouts``: which task refs reached a
    verified done, which were attempted and failed, which are still in flight.
    Negative results are first-class here. The specific failure this prevents is
    an agent re-implementing something it already built because a search missed
    it -- "attempted and failed" has to be exactly as retrievable as "done".

There is no loop in here, and there must not be one. The brain owns no
scheduler, no queue, and no execution state; it answers three questions and the
harness decides what to do about the answers.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocbrain.closeout import normalize_task_ref
from ocbrain.core_v1 import (
    CORE_V1_SCHEMA_VERSION,
    GOAL_BELIEF_TYPE,
    append_core_event,
    compilation_block_reason,
    now_iso,
    record_core_v1_evidence,
    resolve_object_id,
)
from ocbrain.ids import stable_id
from ocbrain.provenance import EMPTY_PROVENANCE, Provenance
from ocbrain.scope import ScopeContext, ScopeTag, matching_stored_scope_ids
from ocbrain.text import compact_whitespace

BRIEFING_SCHEMA_VERSION = "ocbrain.briefing.v1"
LEDGER_SCHEMA_VERSION = "ocbrain.ledger.v1"
GOAL_ATTRIBUTE_KIND = "goal.v1"
GOAL_EVIDENCE_KIND = "goal_declaration"

# ~1500 characters is ~300-400 tokens. Practitioner guidance for a SessionStart
# hook is 200-500 tokens, and Chroma's context-rot work (2025-07-14) is the
# reason for the ceiling rather than a larger one: a single distractor
# measurably degrades output, and coherent-but-irrelevant text hurts *more* than
# random filler. A comprehensive briefing is a worse briefing.
DEFAULT_BRIEFING_BUDGET_CHARS = 1500
MIN_BRIEFING_BUDGET_CHARS = 400
MAX_BRIEFING_BUDGET_CHARS = 8000

# Per-section item ceilings, applied before the character budget. Two bounds,
# not one: the item cap keeps the *shape* stable so a scope with 200 closeouts
# and a scope with 4 produce recognisably the same briefing, and the character
# budget is the hard backstop.
MAX_GOALS = 5
MAX_LEDGER_DONE = 3
MAX_LEDGER_FAILED = 3
MAX_CHAIN_ROWS = 3
MAX_GOTCHAS = 3

# One rendered line never exceeds this. A single 4KB closeout summary must not
# be able to consume the whole budget and push every other section out.
MAX_LINE_CHARS = 200
# The free-text excerpt inside a line. Bounded separately and more tightly than
# the line, so a summary can never crowd out the identifiers beside it.
MAX_SUMMARY_CHARS = 90

GOAL_STATUSES = ("open", "done", "abandoned")
GOAL_CLOSED_STATUSES = ("done", "abandoned")

# Closeout statuses that count as an attempt that did not land. ``cancelled`` is
# in here with ``failed`` and ``blocked``: for the purpose of "did anyone
# already try this", a cancelled attempt is still an attempt, and the reason it
# was cancelled is the thing the next iteration needs to see.
FAILED_CLOSEOUT_STATUSES = frozenset({"failed", "blocked", "cancelled"})

# Fixed section order. This tuple is the contract: sections never reorder, and
# an empty one renders a marker rather than disappearing, so a reader can tell
# "no open goals" from "the goals section was dropped".
SECTION_ORDER: tuple[tuple[str, str], ...] = (
    ("goals", "A. OPEN GOALS"),
    ("ledger", "B. DONE LEDGER"),
    ("chain", "C. LATEST CLOSEOUT CHAIN"),
    ("gotchas", "D. GOTCHAS"),
)

EMPTY_MARKERS: dict[str, str] = {
    "goals": "(none open in this scope)",
    "ledger": "(no closeouts recorded for this scope)",
    "chain": "(no chained closeout for this scope)",
    "gotchas": "(none recorded for this scope)",
}

NOTHING_KNOWN_LINE = "E. NOTHING KNOWN FOR THIS SCOPE"
TRUNCATION_PREFIX = "-- truncated:"
BRIEFING_TITLE_PREFIX = "OCBRAIN BRIEFING · "
MIN_RENDERED_SCOPE_CHARS = 32


class GoalError(ValueError):
    """A goal could not be opened or closed as asked."""


# --------------------------------------------------------------------------- #
# Goals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SourcePointer:
    """Where the real spec lives. The brain pins this and never owns the spec.

    ``path`` is repo-relative or absolute; ``git_ref`` is whatever the caller
    can name the revision by (a sha, a tag, a branch). Both are recorded
    verbatim. Resolution is checked at read time and reported as a typed
    warning, never repaired here -- a pointer that stopped resolving is
    information about the repo, and rewriting it would destroy that.
    """

    path: str
    git_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"path": self.path}
        if self.git_ref:
            payload["git_ref"] = self.git_ref
        return payload


def goal_scope(context: ScopeContext) -> ScopeTag:
    """File a goal at the broadest shared scope, never the narrowest.

    ``resolve_write_scope`` prefers ``task:`` over ``project:``, which is right
    for evidence and wrong for a goal: a goal filed under the task scope of the
    session that opened it is invisible to the next session, which is the entire
    thing goals exist to fix. Mirrors ``mcp_v1.shared_continuity_scope``.
    """
    for scope_type, value in (
        ("project", context.project),
        ("repo", context.repo),
        ("client", context.client),
    ):
        if value:
            return ScopeTag(
                scope_type,
                f"{scope_type}:{value}",
                visibility="internal",
                egress_policy="local_only",
                provenance="inferred",
            )
    raise GoalError(
        "a goal needs a shared scope: pass context.project, context.repo, or "
        "context.client. A goal scoped to one task or session cannot be found "
        "by the next session, which is the reason goals exist."
    )


def open_goal(
    conn: sqlite3.Connection,
    *,
    objective: str,
    finish_line: str,
    source_path: str,
    context: ScopeContext,
    source_git_ref: str | None = None,
    actor: str = "agent",
    provenance: Provenance | None = None,
    session_id: str | None = None,
    opened_at: str | None = None,
) -> dict[str, Any]:
    """Open one goal as a serving belief of type ``goal``.

    Rides the existing belief machinery on purpose. ``CORE_V1_TABLES`` is a
    closed allow-list asserted by ``assert_core_v1_inventory``, so a goals table
    would be a schema change with an inventory assertion to renegotiate; the
    ``belief_type`` column is free text and already carries ``procedure``,
    ``gotcha``, and ``wiki_fact``. Goals join that vocabulary.

    The write is the same evidence -> proposal -> decision triple every compiled
    belief goes through, so a goal is auditable and replayable exactly like any
    other belief, and it is one transaction: a goal that exists as a proposal
    nobody approved is a goal that silently does not exist.
    """
    from ocbrain.mcp_v1 import _one_transaction, decide_proposal_v1

    objective = compact_whitespace(objective or "")
    finish_line = compact_whitespace(finish_line or "")
    source_path = (source_path or "").strip()
    if not objective:
        raise GoalError("objective is required: state what done looks like")
    if not finish_line:
        raise GoalError(
            "finish_line is required and must be executable: a command or a "
            "test path a later session can run without asking anyone"
        )
    if not source_path:
        raise GoalError(
            "source_path is required: goals point at a spec in the repo. "
            "OCBrain pins the pointer; it is never the editable home of a spec."
        )
    provenance = provenance or EMPTY_PROVENANCE
    scope = goal_scope(context)
    pointer = SourcePointer(source_path, (source_git_ref or "").strip() or None)
    opened = opened_at or now_iso()

    # Content-and-scope addressed like a supersede successor: re-opening the
    # same objective against the same spec in the same scope converges on one
    # goal instead of minting a second one beside it.
    goal_id = stable_id("belief", "goal", objective, source_path, scope.scope_id)
    existing = conn.execute(
        "SELECT belief_id, attributes_json, status, serve FROM current_beliefs WHERE belief_id=?",
        (goal_id,),
    ).fetchone()
    if existing is not None:
        attributes = json.loads(existing["attributes_json"] or "{}")
        if attributes.get("status") == "open" and int(existing["serve"] or 0) == 1:
            return {
                "kind": "goal_already_open",
                "goal_id": goal_id,
                "status": "open",
                "scope": scope.to_dict(),
                "opened_at": attributes.get("opened_at"),
            }

    head_seq = int(
        conn.execute("SELECT COALESCE(MAX(event_seq), 0) FROM brain_events").fetchone()[0]
    )
    blocked = compilation_block_reason(conn, goal_id, proposal_event_seq=head_seq)
    if blocked is not None:
        raise GoalError(f"blocked: this goal was previously {blocked}")

    attributes: dict[str, Any] = {
        "kind": GOAL_ATTRIBUTE_KIND,
        "objective": objective,
        "finish_line": finish_line,
        "source_pointer": pointer.to_dict(),
        "status": "open",
        "opened_at": opened,
        # Not a knowledge claim, and labelled so. A goal is task state that
        # happens to be stored as a belief; nothing should promote it, curate
        # it, or serve it as a fact.
        "lifecycle": "current",
        "category": "project",
        "title": objective[:120],
    }
    body = f"Goal: {objective} Finish line: {finish_line} Spec: {_pointer_text(pointer)}"

    with _one_transaction(conn):
        evidence_id, _evidence_event = record_core_v1_evidence(
            conn,
            body=(
                f"Goal opened.\nObjective: {objective}\n"
                f"Finish line: {finish_line}\nSpec pointer: {_pointer_text(pointer)}"
            ),
            kind=GOAL_EVIDENCE_KIND,
            scope=scope,
            writer=actor,
            session_id=provenance.client_session_hint or session_id,
            artifact_ref=f"goal:{goal_id}",
        )
        proposal_event_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "schema_version": "ocbrain.compilation.v1",
                "subject": {"kind": "belief", "id": goal_id},
                "belief_id": goal_id,
                "belief_type": GOAL_BELIEF_TYPE,
                "body": body,
                "evidence_ids": [evidence_id],
                "scope": scope.to_dict(),
                # Goals are not knowledge and carry no epistemic confidence. The
                # column is NOT NULL-free but a number here would be read as one,
                # so it stays absent.
                "confidence": None,
                "attributes": attributes,
            },
            writer=actor,
            session_id=provenance.client_session_hint or session_id,
        )
        decision = decide_proposal_v1(
            conn,
            proposal_event_id=proposal_event_id,
            decision="approve",
            actor=actor,
            edited_body=None,
            reason="goal opened through brain.goal_open",
            provenance=provenance,
        )
    return {
        "kind": "goal_opened",
        "goal_id": goal_id,
        "status": "open",
        "scope": scope.to_dict(),
        "objective": objective,
        "finish_line": finish_line,
        "source_pointer": pointer.to_dict(),
        "opened_at": opened,
        "evidence_id": evidence_id,
        "proposal_event_id": proposal_event_id,
        "decision_event_id": decision["event_id"],
    }


def close_goal(
    conn: sqlite3.Connection,
    *,
    goal_id: str,
    status: str,
    verifier_uri: str,
    verifier_status: str | None = None,
    note: str | None = None,
    actor: str = "agent",
    provenance: Provenance | None = None,
    session_id: str | None = None,
    closed_at: str | None = None,
) -> dict[str, Any]:
    """Close one goal by appending a correction, never by editing a row.

    The status transition is an ``annotate`` correction event carrying an
    attributes patch. That op already exists, is metadata-only by construction
    (it can never touch body, status, serve, or confidence), and is replay
    stable. So a closed goal is a *new event* folded over the old belief, and a
    full rebuild reaches the same place -- which is what makes "when did this
    close, and who said so" answerable at all.

    Closing requires naming the verifier evidence and its status explicitly. A
    goal whose finish line was an executable command and whose closure cites
    nothing is indistinguishable from a goal someone got bored of. ``done``
    requires ``passed``; ``abandoned`` preserves the other truthful states.
    """
    from ocbrain.mcp_v1 import correct_v1

    if status not in GOAL_CLOSED_STATUSES:
        raise GoalError(f"status must be one of: {', '.join(GOAL_CLOSED_STATUSES)}")
    verifier_uri = (verifier_uri or "").strip()
    if not verifier_uri:
        raise GoalError(
            "verifier_uri is required: name the evidence (a log path, a receipt, "
            "or a stable locator like repo://<name>/pytest). A verifier nobody "
            "can go and check is not evidence."
        )
    if not isinstance(verifier_status, str) or not verifier_status.strip():
        raise GoalError(
            "verifier_status is required: pass the verifier result explicitly; "
            "it is never inferred"
        )
    if verifier_status not in {"passed", "failed", "unknown", "not_required"}:
        raise GoalError("verifier_status must be passed, failed, unknown, or not_required")
    if status == "done" and verifier_status != "passed":
        raise GoalError("status='done' requires verifier_status='passed'")

    resolved_id = resolve_object_id(conn, (goal_id or "").strip())
    row = conn.execute(
        "SELECT belief_id, belief_type, attributes_json, status, serve "
        "FROM current_beliefs WHERE belief_id=?",
        (resolved_id,),
    ).fetchone()
    if row is None:
        raise GoalError(f"goal not found: {goal_id}")
    if str(row["belief_type"] or "") != GOAL_BELIEF_TYPE:
        raise GoalError(f"not a goal: {goal_id}")
    attributes = json.loads(row["attributes_json"] or "{}")
    if attributes.get("status") in GOAL_CLOSED_STATUSES:
        return {
            "kind": "goal_already_closed",
            "goal_id": resolved_id,
            "status": attributes.get("status"),
            "closed_at": attributes.get("closed_at"),
        }

    stamp = closed_at or now_iso()
    patch: dict[str, Any] = {
        "status": status,
        "closed_at": stamp,
        "closed_by": actor,
        "verifier": {"uri": verifier_uri, "status": verifier_status},
    }
    if note and note.strip():
        patch["close_note"] = compact_whitespace(note)
    result = correct_v1(
        conn,
        layer="belief",
        target=resolved_id,
        op="annotate",
        body=None,
        actor=actor,
        hard=False,
        attributes_patch=patch,
        provenance=provenance,
        session_id=session_id,
    )
    return {
        "kind": "goal_closed",
        "goal_id": resolved_id,
        "status": status,
        "closed_at": stamp,
        "verifier": patch["verifier"],
        "correction_event_id": result["event_id"],
    }


def list_goals(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    status: str = "open",
    limit: int = MAX_GOALS,
    check_source_pointers: bool = True,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return goals for a scope, selected by scope and status only.

    **There is no similarity path here and there must never be one.** Selection
    is a scope IN-list and an attribute equality test; ordering is
    ``opened_at`` then ``belief_id``, both stored values. Deterministic
    selection beats a similarity or LLM judgement by +10.8pp at short context
    and +21pp at long context (arXiv 2606.01435), and a goal you can only find
    when your phrasing happens to match it is not a goal.

    ``tests/test_briefing.py`` mutation-tests this: swapping in a ranked path
    must fail the suite.
    """
    if status not in GOAL_STATUSES:
        raise GoalError(f"status must be one of: {', '.join(GOAL_STATUSES)}")
    scope_ids = matching_stored_scope_ids(
        conn, "current_beliefs", sorted(context.compatible_scope_ids())
    )
    if not scope_ids:
        return []
    pointer_repo_root = _usable_local_directory(repo_root) or _usable_local_directory(
        context.repo
    )
    placeholders = ",".join("?" for _ in scope_ids)
    rows = conn.execute(
        f"""
        SELECT belief_id, body, attributes_json, scope_type, scope_id
        FROM current_beliefs
        WHERE serve=1 AND status='current' AND belief_type=?
          AND scope_id IN ({placeholders})
        ORDER BY belief_id
        """,  # noqa: S608 - placeholders are generated, values are bound
        (GOAL_BELIEF_TYPE, *scope_ids),
    ).fetchall()
    goals: list[dict[str, Any]] = []
    for row in rows:
        attributes = json.loads(row["attributes_json"] or "{}")
        if str(attributes.get("status") or "open") != status:
            continue
        pointer = attributes.get("source_pointer")
        pointer = pointer if isinstance(pointer, dict) else {}
        entry: dict[str, Any] = {
            "goal_id": str(row["belief_id"]),
            "objective": str(attributes.get("objective") or row["body"]),
            "finish_line": str(attributes.get("finish_line") or ""),
            "source_pointer": pointer,
            "status": status,
            "opened_at": str(attributes.get("opened_at") or ""),
            "scope": {"scope_type": str(row["scope_type"]), "scope_id": str(row["scope_id"])},
        }
        if attributes.get("closed_at"):
            entry["closed_at"] = str(attributes["closed_at"])
        if isinstance(attributes.get("verifier"), dict):
            entry["verifier"] = attributes["verifier"]
        if check_source_pointers:
            warning = _source_pointer_warning(pointer, repo_root=pointer_repo_root)
            if warning is not None:
                entry["warning"] = warning
        goals.append(entry)
    # Oldest goal first: the one that has been open longest is the one most
    # likely to be drifting. `belief_id` breaks ties so the order is total.
    goals.sort(key=lambda item: (item["opened_at"], item["goal_id"]))
    return goals[:limit] if limit > 0 else goals


def _source_pointer_warning(
    pointer: dict[str, Any], *, repo_root: Path | None
) -> dict[str, Any] | None:
    """Type a spec pointer that no longer resolves. Never silently drop the goal.

    A goal whose spec vanished is the most interesting goal in the corpus: it
    means the repo moved and the objective did not. Hiding it would turn a loud
    problem into a quiet one.
    """
    raw = str(pointer.get("path") or "").strip()
    if not raw:
        return {"type": "source_pointer_absent", "path": None}
    try:
        candidate = Path(raw).expanduser()
    except (OSError, RuntimeError):
        return {"type": "source_pointer_unresolved", "path": raw}
    is_absolute = candidate.is_absolute()
    if not is_absolute:
        if repo_root is None:
            return {"type": "source_pointer_unresolved", "path": raw}
        candidate = repo_root / candidate
    try:
        exists = candidate.exists()
    except OSError:
        exists = False
    if not exists:
        return {"type": "source_pointer_unresolved", "path": raw}

    git_ref = str(pointer.get("git_ref") or "").strip()
    if not git_ref:
        return None
    if is_absolute:
        # An explicit path owns its repository identity; repo_root is only a
        # resolution hint for pointers that need one.
        git_root = _git_repository_root(candidate if candidate.is_dir() else candidate.parent)
    else:
        git_root = _git_repository_root(repo_root)
        if git_root is None:
            git_root = _git_repository_root(
                candidate if candidate.is_dir() else candidate.parent
            )
    if git_root is None or not _git_ref_resolves(git_root, git_ref):
        return {
            "type": "source_git_ref_unresolved",
            "path": raw,
            "git_ref": git_ref,
        }
    return None


def _usable_local_directory(value: str | Path | None) -> Path | None:
    """Return an existing local directory, never a guessed repository identifier."""
    if value is None:
        return None
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_dir():
            return None
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _git_repository_root(start: Path | None) -> Path | None:
    """Discover a local Git root without invoking a shell or repository hooks."""
    local_start = _usable_local_directory(start)
    if local_start is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(local_start), "rev-parse", "--show-toplevel"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _usable_local_directory(completed.stdout.strip())


def _git_ref_resolves(repo_root: Path, git_ref: str) -> bool:
    """Resolve one commit-ish literally after Git's end-of-options marker."""
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                "--quiet",
                "--end-of-options",
                f"{git_ref}^{{commit}}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


# --------------------------------------------------------------------------- #
# The done / attempt ledger
# --------------------------------------------------------------------------- #


def build_ledger(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    task_ref: str | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Project ``task_closeouts`` into per-task state. Read-only, no new writes.

    There is no ledger table and no ledger write path. Everything here is
    derived from receipts that already exist plus the chain columns
    (``parent_closeout_id``, ``task_ref_norm``) already on main. A second write
    path would be a second thing to keep true.

    Grouping folds ``task_ref`` at read time rather than trusting the stored
    ``task_ref_norm``: the column is written at closeout time and history is
    deliberately never rewritten, so on a real corpus it is NULL for nearly
    every historical row (1161 of 1171 on the install this was built against).
    A ledger that grouped on the column alone would report that a task with
    fourteen attempts has none -- exactly the false negative this object exists
    to prevent.
    """
    wanted_norm = normalize_task_ref(task_ref) if task_ref else None
    rows = conn.execute(
        "SELECT id, closed_at, task_ref, task_ref_norm, status, summary, "
        "parent_closeout_id, receipt_json FROM task_closeouts "
        "ORDER BY closed_at, id"
    ).fetchall()

    groups: dict[str, list[dict[str, Any]]] = {}
    scanned = 0
    for row in rows:
        receipt = json.loads(row["receipt_json"])
        norm = str(row["task_ref_norm"] or "") or normalize_task_ref(row["task_ref"])
        if not norm:
            continue
        if wanted_norm is not None:
            if norm != wanted_norm:
                continue
        elif not _closeout_in_scope(receipt, context):
            continue
        scanned += 1
        groups.setdefault(norm, []).append(
            {
                "id": str(row["id"]),
                "closed_at": str(row["closed_at"]),
                "task_ref": str(row["task_ref"]),
                "status": str(row["status"]),
                "summary": _clip(str(row["summary"] or ""), MAX_LINE_CHARS),
                "parent_closeout_id": row["parent_closeout_id"] or None,
                "previous_in_chain": (receipt.get("chain") or {}).get("previous_in_chain"),
                "verification_status": str(receipt.get("verification_status") or ""),
                "verifier_refs": [
                    {"uri": ref.get("uri"), "status": ref.get("status")}
                    for ref in receipt.get("verifier_refs") or []
                    if isinstance(ref, dict) and ref.get("uri")
                ],
                "awaiting": receipt.get("awaiting"),
                # What the caller said did not work. `awaiting` is who unblocks
                # me; this is what is still broken, and it is the only field the
                # write-time gate charges a non-clean closeout for. A required
                # field no reader serves is a toll, not a gate -- so it is read
                # here, on the object whose description promises to read it.
                # NULL on every row written before 2026-08-28: the table is
                # append-only under a trigger and is never backfilled.
                "unresolved": receipt.get("unresolved"),
            }
        )

    entries = [_ledger_entry(norm, chain) for norm, chain in groups.items()]
    # Newest activity first, id as the total-order tiebreak. Recency is a stored
    # timestamp, not a score: nothing here is ranked.
    entries.sort(key=lambda item: (item["last_closed_at"], item["task_ref_norm"]), reverse=True)
    counts = {
        "verified_done": sum(1 for e in entries if e["state"] == "verified_done"),
        "attempted_failed": sum(1 for e in entries if e["state"] == "attempted_failed"),
        "in_flight": sum(1 for e in entries if e["state"] == "in_flight"),
        "tasks": len(entries),
        "closeouts_scanned": scanned,
    }
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "resolved_context": context.to_dict(),
        "task_ref": task_ref,
        "counts": counts,
        "entries": entries[:limit] if limit > 0 else entries,
        "truncated_entries": max(len(entries) - limit, 0) if limit > 0 else 0,
    }


def _ledger_entry(norm: str, chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one task's closeout chain into a single state plus its evidence."""
    latest = chain[-1]
    failed_attempts = [
        row
        for row in chain
        if row["status"] in FAILED_CLOSEOUT_STATUSES or row["verification_status"] == "failed"
    ]
    if latest["status"] == "completed" and latest["verification_status"] == "verified":
        state = "verified_done"
    elif latest["status"] in FAILED_CLOSEOUT_STATUSES or latest["verification_status"] == "failed":
        state = "attempted_failed"
    else:
        # `completed` with no passing verifier is deliberately not `verified_done`.
        # An agent-reported completion is a claim; the ledger reports claims as
        # claims. Calling it done is how a loop stops checking.
        state = "in_flight"
    verifiers: list[dict[str, Any]] = []
    for row in chain:
        for ref in row["verifier_refs"]:
            if ref not in verifiers:
                verifiers.append(ref)
    return {
        "task_ref_norm": norm,
        "task_ref": latest["task_ref"],
        "state": state,
        "closeout_count": len(chain),
        "first_closed_at": chain[0]["closed_at"],
        "last_closed_at": latest["closed_at"],
        "latest_status": latest["status"],
        "latest_summary": latest["summary"],
        # The summary says what the session did; this says what is still not
        # working. `None` on pre-gate rows, and every consumer degrades to the
        # summary rather than printing an empty failure.
        "latest_unresolved": latest["unresolved"],
        "latest_closeout_id": latest["id"],
        "failed_attempt_count": len(failed_attempts),
        "failed_attempts": [
            {
                "id": row["id"],
                "closed_at": row["closed_at"],
                "status": row["status"],
                "summary": row["summary"],
                "awaiting": row["awaiting"],
                "unresolved": row["unresolved"],
            }
            for row in failed_attempts
        ],
        "verifiers": verifiers[:8],
        "chain": [
            {
                "id": row["id"],
                "closed_at": row["closed_at"],
                "status": row["status"],
                "parent_closeout_id": row["parent_closeout_id"],
                "previous_in_chain": row["previous_in_chain"],
            }
            for row in chain
        ],
        "chain_linked": any(
            row["parent_closeout_id"] or row["previous_in_chain"] for row in chain
        ),
    }


def _closeout_in_scope(receipt: dict[str, Any], context: ScopeContext) -> bool:
    """Whether a receipt belongs to the caller's scope.

    Same predicate ``digest_v1`` uses, reimplemented here rather than imported
    so the ledger does not take a dependency on a private digest helper. An
    unscoped context matches everything, which is correct: a caller that named
    no scope asked for the whole ledger.
    """
    receipt_context = receipt.get("context")
    receipt_context = receipt_context if isinstance(receipt_context, dict) else {}
    scoped = False
    for key in ("project", "repo", "client"):
        expected = getattr(context, key)
        if not expected:
            continue
        scoped = True
        if str(receipt_context.get(key) or "") != expected:
            return False
    if not scoped and context.task:
        return (
            str(receipt.get("task_ref") or "") == context.task
            or str(receipt_context.get("task") or "") == context.task
        )
    return True


# --------------------------------------------------------------------------- #
# Gotchas
# --------------------------------------------------------------------------- #


def list_gotchas(
    conn: sqlite3.Connection, *, context: ScopeContext, limit: int = MAX_GOTCHAS
) -> list[dict[str, Any]]:
    """Standing cautions for a scope, selected by type and pin, never by score.

    Two deterministic rules, in this order:

    1. beliefs of type ``gotcha`` -- the ``procmine`` mint's output, a repair
       somebody already paid for once;
    2. pinned beliefs -- an operator's standing decision that this must not be
       forgotten.

    Confidence orders within each rule. Confidence is a stored scalar the
    compiler wrote, not a similarity between this belief and a query; there is
    no query here to be similar to.
    """
    scope_ids = matching_stored_scope_ids(
        conn, "current_beliefs", sorted(context.compatible_scope_ids())
    )
    if not scope_ids:
        return []
    placeholders = ",".join("?" for _ in scope_ids)
    rows = conn.execute(
        f"""
        SELECT belief_id, body, belief_type, pinned, confidence, attributes_json
        FROM current_beliefs
        WHERE serve=1 AND status='current' AND scope_id IN ({placeholders})
          AND (belief_type='gotcha' OR pinned=1)
        ORDER BY
          CASE WHEN belief_type='gotcha' THEN 0 ELSE 1 END,
          COALESCE(confidence, 0) DESC,
          belief_id
        LIMIT ?
        """,  # noqa: S608 - placeholders are generated, values are bound
        (*scope_ids, max(limit, 0)),
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        attributes = json.loads(row["attributes_json"] or "{}")
        result.append(
            {
                "id": str(row["belief_id"]),
                "title": str(attributes.get("title") or "").strip() or None,
                "body": _clip(compact_whitespace(str(row["body"])), MAX_LINE_CHARS),
                "reason": "gotcha" if str(row["belief_type"] or "") == "gotcha" else "pinned",
            }
        )
    return result


# --------------------------------------------------------------------------- #
# The briefing itself
# --------------------------------------------------------------------------- #


def build_briefing(
    conn: sqlite3.Connection,
    *,
    context: ScopeContext,
    budget_chars: int = DEFAULT_BRIEFING_BUDGET_CHARS,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic, bounded session-start briefing.

    Determinism is over two inputs and only two: the resolved scope, and the
    state of the corpus (plus, for the ``source_pointer`` warnings, the state of
    the filesystem those pointers name -- checking them is the point, and it is
    named here rather than hidden). Nothing in this payload is a timestamp of
    *now*, nothing is a random sample, and nothing is a score. Two calls a day
    apart against an unchanged corpus are byte-identical.

    There is no ``query`` parameter, and that absence is the contract. A free
    text argument would make this a query with extra steps, and a loop that
    reorients through a query reorients differently every iteration.
    """
    budget = _clamp_budget(budget_chars)
    scope_id = _briefing_scope_id(context)

    goals = list_goals(conn, context=context, status="open", limit=MAX_GOALS, repo_root=repo_root)
    ledger = build_ledger(conn, context=context, limit=64)
    gotchas = list_gotchas(conn, context=context, limit=MAX_GOTCHAS)

    done = [e for e in ledger["entries"] if e["state"] == "verified_done"][:MAX_LEDGER_DONE]
    failed = [e for e in ledger["entries"] if e["state"] == "attempted_failed"][:MAX_LEDGER_FAILED]
    chain_entry = _latest_chain(ledger["entries"])

    items: dict[str, list[str]] = {
        "goals": [_goal_line(goal) for goal in goals],
        "ledger": [_ledger_line(entry) for entry in (*done, *failed)],
        "chain": _chain_lines(chain_entry),
        "gotchas": [_gotcha_line(gotcha) for gotcha in gotchas],
    }
    warnings = [
        {"goal_id": goal["goal_id"], **goal["warning"]} for goal in goals if goal.get("warning")
    ]
    empty = not any(items.values())
    text, accounting, sections = _render(scope_id, items, budget=budget, empty=empty)
    return {
        "schema_version": BRIEFING_SCHEMA_VERSION,
        "core_schema": CORE_V1_SCHEMA_VERSION,
        "resolved_context": context.to_dict(),
        "scope_id": scope_id,
        "budget_chars": budget,
        "used_chars": len(text),
        "truncation": accounting,
        "sections": sections,
        "warnings": warnings,
        "counts": {
            "open_goals": len(goals),
            "verified_done": ledger["counts"]["verified_done"],
            "attempted_failed": ledger["counts"]["attempted_failed"],
            "in_flight": ledger["counts"]["in_flight"],
            "gotchas": len(gotchas),
        },
        "text": text,
    }


def _clamp_budget(value: Any) -> int:
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("budget_chars must be an integer") from exc
    if budget < MIN_BRIEFING_BUDGET_CHARS or budget > MAX_BRIEFING_BUDGET_CHARS:
        raise ValueError(
            f"budget_chars must be between {MIN_BRIEFING_BUDGET_CHARS} and "
            f"{MAX_BRIEFING_BUDGET_CHARS}; a briefing that can grow without "
            "bound is the failure mode this object exists to prevent"
        )
    return budget


def _briefing_scope_id(context: ScopeContext) -> str:
    for scope_type, value in (
        ("project", context.project),
        ("repo", context.repo),
        ("client", context.client),
        ("task", context.task),
    ):
        if value:
            return f"{scope_type}:{value}"
    return "unscoped"


def _latest_chain(entries: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recent multi-closeout task for the scope, or None.

    A chain of one is not a chain. Entries arrive newest-first from
    ``build_ledger``, so the first multi-row group is the answer and no
    comparison is needed.
    """
    for entry in entries:
        if entry["closeout_count"] > 1:
            return entry
    return None


def _goal_line(goal: dict[str, Any]) -> str:
    pointer = goal.get("source_pointer") or {}
    head = f"- {goal['goal_id']}"
    # The warning goes second, before anything that can be clipped away. A typed
    # warning that only survives when the objective happens to be short is not a
    # warning.
    if goal.get("warning"):
        head += f" [{goal['warning']['type']}]"
    parts = [f"{head}: {_clip(goal['objective'], MAX_SUMMARY_CHARS)}"]
    if goal.get("finish_line"):
        parts.append(f"verify: {_clip(goal['finish_line'], MAX_SUMMARY_CHARS)}")
    spec = SourcePointer(str(pointer.get("path") or "?"), pointer.get("git_ref"))
    parts.append(f"spec: {_clip(_pointer_text(spec), MAX_SUMMARY_CHARS)}")
    if goal.get("opened_at"):
        parts.append(f"open since {goal['opened_at'][:10]}")
    return _clip(" | ".join(parts), MAX_LINE_CHARS)


def _ledger_line(entry: dict[str, Any]) -> str:
    mark = {"verified_done": "done", "attempted_failed": "FAILED", "in_flight": "open"}[
        entry["state"]
    ]
    line = (
        f"- {mark} {entry['task_ref']} "
        f"({entry['closeout_count']} closeouts, last {entry['last_closed_at'][:10]})"
    )
    if entry["failed_attempt_count"]:
        line += f" {entry['failed_attempt_count']} failed attempt(s)"
    if entry["state"] == "attempted_failed":
        # The failure text, not just the count. "TASK-2 failed" tells the next
        # iteration to avoid the task; "TASK-2 failed: the trainer import is
        # circular" tells it what not to try again, which is the difference
        # between skipping work and not repeating it.
        #
        # `unresolved` first, because that is the sentence about what is still
        # broken; the summary is what the session did, which is a weaker answer
        # to the same question. Falls back to the summary for every row written
        # before the field existed -- which is all 1,238 of them in the live
        # core, and nothing there may lose its line.
        because = entry["latest_unresolved"] or entry["latest_summary"]
        line += f": {_clip(because, MAX_SUMMARY_CHARS)}"
    elif entry["verifiers"]:
        line += f" verifier: {entry['verifiers'][0].get('uri')}"
    return _clip(line, MAX_LINE_CHARS)


def _chain_lines(entry: dict[str, Any] | None) -> list[str]:
    if entry is None:
        return []
    lines = [_clip(f"- {entry['task_ref']} ({entry['closeout_count']} closeouts)", MAX_LINE_CHARS)]
    for row in entry["chain"][-MAX_CHAIN_ROWS:]:
        parent = row["parent_closeout_id"] or row["previous_in_chain"]
        suffix = f" <- {parent}" if parent else ""
        row_text = f"  - {row['closed_at'][:10]} {row['status']} {row['id']}{suffix}"
        lines.append(_clip(row_text, MAX_LINE_CHARS))
    return lines


def _gotcha_line(gotcha: dict[str, Any]) -> str:
    label = gotcha.get("title") or gotcha["body"]
    return _clip(f"- ({gotcha['reason']}) {label}", MAX_LINE_CHARS)


def _pointer_text(pointer: SourcePointer) -> str:
    return f"{pointer.path}@{pointer.git_ref}" if pointer.git_ref else pointer.path


def _clip(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _render(
    scope_id: str,
    items: dict[str, list[str]],
    *,
    budget: int,
    empty: bool,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Render within budget, reserving the skeleton before spending on items.

    The skeleton -- title, every section header, one marker line per section,
    and the truncation notice -- is costed first. Only user-supplied identifier
    text and items can be compacted. That is what keeps the *shape* fixed: a
    briefing that ran out of room still shows all four headers in order and
    says how many items it dropped, instead of silently ending after section B.

    The original reservation algorithm remains the fast path so an ordinary
    briefing that already fits stays byte-identical. A long scope identifier or
    any other over-budget result takes the exact-render path below, where every
    candidate is measured after its omission markers and notice are present.
    """
    safe_scope_id = _safe_render_line(scope_id)
    max_scope_chars = MAX_LINE_CHARS - len(BRIEFING_TITLE_PREFIX)
    title_scope_id = _clip(safe_scope_id, max_scope_chars)
    scope_id_compacted = title_scope_id != scope_id
    title = f"{BRIEFING_TITLE_PREFIX}{title_scope_id}"
    safe_items = {
        key: [_bounded_render_line(line) for line in items[key]] for key, _ in SECTION_ORDER
    }

    notice_reserve = len(TRUNCATION_PREFIX) + 64
    skeleton = len(title) + 1
    for key, header in SECTION_ORDER:
        skeleton += len(header) + 1
        skeleton += len(EMPTY_MARKERS[key]) + 3
    if empty:
        skeleton += len(NOTHING_KNOWN_LINE) + 1
    remaining = budget - skeleton - notice_reserve

    kept: dict[str, list[str]] = {}
    dropped: dict[str, int] = {}
    for key, _header in SECTION_ORDER:
        kept[key] = []
        dropped[key] = 0
        for line in safe_items[key]:
            cost = len(line) + 1
            if cost <= remaining:
                kept[key].append(line)
                remaining -= cost
            else:
                dropped[key] += 1

    rendered = _assemble_briefing(title, kept, dropped, empty=empty)
    if len(rendered[0]) <= budget and not scope_id_compacted:
        return rendered

    return _render_exactly_bounded(safe_scope_id, safe_items, budget=budget, empty=empty)


def _render_exactly_bounded(
    scope_id: str,
    items: dict[str, list[str]],
    *,
    budget: int,
    empty: bool,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Render a compact title and admit items only after exact measurement."""
    max_scope_chars = MAX_LINE_CHARS - len(BRIEFING_TITLE_PREFIX)
    bounded_scope_id = _clip(scope_id, max_scope_chars)
    initial_scope_chars = min(len(bounded_scope_id), MIN_RENDERED_SCOPE_CHARS)
    rendered_scope_id = _clip(bounded_scope_id, initial_scope_chars)
    title = f"{BRIEFING_TITLE_PREFIX}{rendered_scope_id}"

    kept = {key: [] for key, _ in SECTION_ORDER}
    dropped = {key: len(items[key]) for key, _ in SECTION_ORDER}
    rendered = _assemble_briefing(title, kept, dropped, empty=empty)

    # MIN_BRIEFING_BUDGET_CHARS is deliberately large enough for this complete
    # skeleton, including all four headers and omission accounting. Items are
    # then admitted in the same deterministic section/item order as the legacy
    # path, but against their fully rendered cost rather than an estimate.
    for key, _header in SECTION_ORDER:
        for line in items[key]:
            kept[key].append(line)
            dropped[key] -= 1
            candidate = _assemble_briefing(title, kept, dropped, empty=empty)
            if len(candidate[0]) <= budget:
                rendered = candidate
            else:
                kept[key].pop()
                dropped[key] += 1

    # Item content has priority over a verbose identifier. Spend any exact
    # remainder on the scope after the useful section content has been chosen.
    spare = budget - len(rendered[0])
    if spare and len(rendered_scope_id) < len(bounded_scope_id):
        expanded_chars = min(len(bounded_scope_id), len(rendered_scope_id) + spare)
        rendered_scope_id = _clip(bounded_scope_id, expanded_chars)
        title = f"{BRIEFING_TITLE_PREFIX}{rendered_scope_id}"
        rendered = _assemble_briefing(title, kept, dropped, empty=empty)

    text, accounting, sections = rendered
    if rendered_scope_id != scope_id:
        accounting["truncated"] = True
        accounting["scope_id_truncated"] = True
    return text, accounting, sections


def _assemble_briefing(
    title: str,
    kept: dict[str, list[str]],
    dropped: dict[str, int],
    *,
    empty: bool,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Assemble one measured rendering from already-selected item lines."""
    lines = [title]
    sections: list[dict[str, Any]] = []
    for key, header in SECTION_ORDER:
        lines.append(header)
        section: dict[str, Any] = {
            "id": key,
            "header": header,
            "item_count": len(kept[key]),
            "omitted_count": dropped[key],
        }
        if kept[key]:
            lines.extend(kept[key])
            section["present"] = True
        elif dropped[key]:
            marker = f"  (omitted: {dropped[key]} item(s) over budget)"
            lines.append(marker)
            section["present"] = False
            section["marker"] = marker.strip()
        else:
            marker = f"  {EMPTY_MARKERS[key]}"
            lines.append(marker)
            section["present"] = False
            section["marker"] = EMPTY_MARKERS[key]
        if kept[key] and dropped[key]:
            note = f"  (omitted: {dropped[key]} item(s) over budget)"
            lines.append(note)
        sections.append(section)
    if empty:
        lines.append(NOTHING_KNOWN_LINE)

    total_dropped = sum(dropped.values())
    accounting = {
        "truncated": bool(total_dropped),
        "items_omitted": total_dropped,
        "omitted_by_section": {key: count for key, count in dropped.items() if count},
    }
    if total_dropped:
        lines.append(f"{TRUNCATION_PREFIX} {total_dropped} item(s) dropped to stay within budget")
    text = "\n".join(lines)
    accounting["chars_rendered"] = len(text)
    return text, accounting, sections


def _safe_render_line(value: Any) -> str:
    """Return one valid UTF-8 line without changing ordinary text."""
    text = str(value)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", errors="replace").decode("utf-8")
    return " ".join(text.splitlines())


def _bounded_render_line(value: Any) -> str:
    """Bound an arbitrary item line while preserving intentional indentation."""
    line = _safe_render_line(value)
    if len(line) <= MAX_LINE_CHARS:
        return line
    return line[: MAX_LINE_CHARS - 1].rstrip() + "…"


__all__ = [
    "BRIEFING_SCHEMA_VERSION",
    "DEFAULT_BRIEFING_BUDGET_CHARS",
    "GOAL_ATTRIBUTE_KIND",
    "GOAL_CLOSED_STATUSES",
    "GOAL_EVIDENCE_KIND",
    "GOAL_STATUSES",
    "LEDGER_SCHEMA_VERSION",
    "MAX_BRIEFING_BUDGET_CHARS",
    "MIN_BRIEFING_BUDGET_CHARS",
    "SECTION_ORDER",
    "GoalError",
    "SourcePointer",
    "build_briefing",
    "build_ledger",
    "close_goal",
    "goal_scope",
    "list_goals",
    "list_gotchas",
    "open_goal",
]
