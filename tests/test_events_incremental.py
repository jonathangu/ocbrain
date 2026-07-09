"""Incremental projection: equivalence with the full fold + corruption fallback.

``rebuild_projection`` defaults to an incremental fold that consumes only events
with ``rowid`` past ``projection_cursor.last_event_rowid`` and updates
``current_beliefs`` in place, advancing the cursor in the same transaction. These
tests pin the two guarantees that make that safe to run on every decide/correct:

1. Over a randomized (seeded) event log with rebuilds interleaved at arbitrary
   points, the incrementally maintained projection is byte-identical to a full
   DELETE/INSERT rebuild of the same log.
2. A missing / ahead / hash-chain-broken cursor is detected and transparently
   falls back to a full rebuild, and the cursor is repaired to the log head.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import ocbrain.events as events
from ocbrain.db import connect, init_db
from ocbrain.events import (
    decide_compilation,
    propose_compilation,
    rebuild_projection,
    record_correction,
    record_evidence,
    record_tombstone,
)
from ocbrain.scope import ScopeTag, global_scope

_BELIEF_COLUMNS = (
    "belief_id",
    "body",
    "scope_type",
    "scope_id",
    "visibility",
    "egress_policy",
    "confidence",
    "confidence_band",
    "evidence_ids",
    "status",
    "pinned",
    "approved_event_id",
    "last_event_id",
    "last_compiled_at",
)

_SCOPES = [
    global_scope(),
    ScopeTag("project", "project:bountiful", visibility="internal", egress_policy="hosted_ok"),
    ScopeTag(
        "personal_finance",
        "personal_finance:pelican",
        visibility="confidential",
        egress_policy="local_only",
    ),
]

_CORRECTION_OPS = ("edit", "reframe", "pin", "demote", "mark_wrong", "retract")


def _snapshot(conn) -> list[tuple[Any, ...]]:
    """Ordered, column-explicit snapshot of current_beliefs (rowid-independent)."""
    cols = ", ".join(_BELIEF_COLUMNS)
    rows = conn.execute(
        f"SELECT {cols} FROM current_beliefs ORDER BY belief_id"  # noqa: S608 - fixed cols
    ).fetchall()
    return [tuple(row[col] for col in _BELIEF_COLUMNS) for row in rows]


def _read_cursor(conn) -> int | None:
    row = conn.execute(
        "SELECT last_event_rowid FROM projection_cursor WHERE id = 1"
    ).fetchone()
    return None if row is None else row["last_event_rowid"]


def _max_rowid(conn) -> int:
    row = conn.execute("SELECT MAX(rowid) AS m FROM brain_events").fetchone()
    return row["m"] or 0


def _drive_random_log(conn, rng: random.Random, *, steps: int) -> None:
    """Append a diverse, randomized event log, rebuilding at random points.

    Uses the real event writers so proposals, decisions (approve/edit/reject/
    shadow), corrections (every op), and tombstones (soft/shred) all appear, plus
    scope promotions written directly to the ledger. ``rebuild=False`` on decisions
    lets several events batch behind a single incremental fold.
    """
    belief_ids = [f"belief:rand-{i}" for i in range(6)]
    pending: list[str] = []  # proposal_event_ids not yet decided
    evidence_ids: list[str] = []

    for step in range(steps):
        choice = rng.random()
        scope = rng.choice(_SCOPES)
        if choice < 0.2 or not evidence_ids:
            eid = record_evidence(
                conn,
                body=f"evidence body {step} {rng.random()}",
                scope=scope,
                writer=rng.choice(["codex", "ocbrain"]),
            )
            # evidence_id lives in the event body; re-derive from the event we just wrote.
            body = conn.execute(
                "SELECT body_json FROM brain_events WHERE id = ?", (eid,)
            ).fetchone()["body_json"]
            import json as _json

            evidence_ids.append(_json.loads(body)["evidence_id"])
        elif choice < 0.5:
            belief_id = rng.choice(belief_ids)
            proposal_id = propose_compilation(
                conn,
                belief_id=belief_id,
                body=f"proposed belief {step} for {belief_id}",
                evidence_ids=rng.sample(evidence_ids, k=min(2, len(evidence_ids))),
                scope=scope,
                confidence=rng.choice([None, 0.2, 0.5, 0.8, 0.95]),
                check_hard_block=False,
            )
            pending.append(proposal_id)
        elif choice < 0.75 and pending:
            proposal_id = pending.pop(rng.randrange(len(pending)))
            decision = rng.choice(["approve", "edit", "reject", "shadow"])
            decide_compilation(
                conn,
                proposal_event_id=proposal_id,
                decision=decision,
                edited_body=f"edited {step}" if decision == "edit" else None,
                rebuild=False,
                check_existing=False,
            )
        elif choice < 0.9:
            belief_id = rng.choice(belief_ids)
            op = rng.choice(_CORRECTION_OPS)
            record_correction(
                conn,
                target_layer=rng.choice(["belief", "knowledge"]),
                target_id=belief_id,
                op=op,
                body=f"correction {step}" if op in {"edit", "reframe"} else None,
                hard=False,
            )
        else:
            belief_id = rng.choice(belief_ids)
            record_tombstone(
                conn,
                target=belief_id,
                mode=rng.choice(["soft", "shred"]),
                reason=f"tombstone {step}",
            )

        # Rebuild at random points (default = incremental) to advance the cursor.
        if rng.random() < 0.4:
            rebuild_projection(conn)


def test_incremental_matches_full_over_randomized_log(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)

    full_calls = 0
    incr_success = 0
    real_full = events._full_rebuild
    real_incr = events._incremental_projection

    def counting_full(c):
        nonlocal full_calls
        full_calls += 1
        return real_full(c)

    def counting_incr(c):
        nonlocal incr_success
        result = real_incr(c)
        if result:
            incr_success += 1
        return result

    events._full_rebuild = counting_full
    events._incremental_projection = counting_incr
    try:
        _drive_random_log(conn, random.Random(1337), steps=200)
        rebuild_projection(conn)  # settle any trailing un-folded events, incrementally
        incremental_snapshot = _snapshot(conn)
        # The cursor must sit exactly at the log head after an incremental settle.
        assert _read_cursor(conn) == _max_rowid(conn)
    finally:
        events._full_rebuild = real_full
        events._incremental_projection = real_incr

    # The incremental path must have actually run (not merely fallen back to full).
    assert incr_success >= 1
    # After the initial cursor-less fallback, steady state should be incremental:
    # far fewer full rebuilds than successful incremental folds.
    assert full_calls < incr_success

    # Force a full rebuild of the identical log; it must reproduce the projection
    # the incremental path maintained, byte for byte.
    rebuild_projection(conn, full=True)
    full_snapshot = _snapshot(conn)
    assert incremental_snapshot == full_snapshot
    assert len(full_snapshot) > 0  # the randomized log must have produced beliefs


def test_incremental_seed_variants_match_full(tmp_path: Path) -> None:
    """Repeat the equivalence check across several seeds for broader coverage."""
    for seed in (1, 7, 42, 99, 2026):
        conn = connect(tmp_path / f"seed-{seed}.sqlite")
        init_db(conn)
        _drive_random_log(conn, random.Random(seed), steps=120)
        rebuild_projection(conn)
        incremental_snapshot = _snapshot(conn)
        assert _read_cursor(conn) == _max_rowid(conn)
        rebuild_projection(conn, full=True)
        assert incremental_snapshot == _snapshot(conn), f"mismatch for seed {seed}"


def test_incremental_consumes_proposal_recorded_before_cursor(tmp_path: Path) -> None:
    """A decision folded after the cursor whose proposal predates it still lands.

    The proposal is recorded and folded (cursor advances past it), then the decision
    is recorded separately — so the incremental fold must hydrate the pre-cursor
    proposal from the ledger rather than an in-memory map.
    """
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = _seed_evidence(conn)

    proposal_id = propose_compilation(
        conn,
        belief_id="belief:split",
        body="belief proposed in one batch",
        evidence_ids=[evidence_id],
        scope=global_scope(),
        confidence=0.8,
    )
    rebuild_projection(conn)  # cursor now past the proposal; no belief yet (undecided)
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM current_beliefs WHERE belief_id = 'belief:split'"
    ).fetchone()["n"] == 0

    decide_compilation(
        conn, proposal_event_id=proposal_id, decision="approve", rebuild=False
    )
    rebuild_projection(conn)  # incremental fold must hydrate the pre-cursor proposal

    incremental = _snapshot(conn)
    rebuild_projection(conn, full=True)
    assert incremental == _snapshot(conn)
    row = conn.execute(
        "SELECT body, status FROM current_beliefs WHERE belief_id = 'belief:split'"
    ).fetchone()
    assert row["status"] == "current"
    assert row["body"] == "belief proposed in one batch"


def test_missing_cursor_falls_back_to_full(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    _seed_decided_belief(conn)
    # Fresh DB has never written the cursor row.
    assert _read_cursor(conn) is None

    rebuild_projection(conn)  # must fall back to full and populate the cursor
    assert _read_cursor(conn) == _max_rowid(conn)
    assert conn.execute("SELECT COUNT(*) AS n FROM current_beliefs").fetchone()["n"] == 1


def test_cursor_ahead_of_log_falls_back_to_full(tmp_path: Path) -> None:
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    _seed_decided_belief(conn)
    rebuild_projection(conn, full=True)

    # Corrupt the cursor to point past the log head.
    conn.execute(
        "UPDATE projection_cursor SET last_event_rowid = ? WHERE id = 1",
        (_max_rowid(conn) + 10_000,),
    )
    expected = _snapshot(conn)

    rebuild_projection(conn)  # ahead cursor -> full rebuild
    assert _read_cursor(conn) == _max_rowid(conn)
    assert _snapshot(conn) == expected


def test_broken_hash_chain_falls_back_to_full(tmp_path: Path) -> None:
    """Tampering an event's chain hash forces the consumed range to fail verify.

    Only the hash columns are corrupted (body_json is untouched), so a full fold
    reproduces the correct projection; the incremental verify must reject the range
    and hand off to that full rebuild, repairing the cursor.
    """
    conn = connect(tmp_path / "ocbrain.sqlite")
    init_db(conn)
    evidence_id = _seed_evidence(conn)
    rebuild_projection(conn, full=True)
    cursor_before = _read_cursor(conn)

    proposal_id = propose_compilation(
        conn,
        belief_id="belief:chain",
        body="belief behind a tampered event",
        evidence_ids=[evidence_id],
        scope=global_scope(),
        confidence=0.9,
    )
    decide_compilation(
        conn, proposal_event_id=proposal_id, decision="approve", rebuild=False
    )

    # Tamper the event_hash of a consumed event (rowid just past the old cursor).
    victim = conn.execute(
        "SELECT rowid AS rid, event_hash FROM brain_events WHERE rowid > ? "
        "ORDER BY rowid ASC LIMIT 1",
        (cursor_before,),
    ).fetchone()
    conn.execute(
        "UPDATE brain_events SET event_hash = ? WHERE rowid = ?",
        ("deadbeef" * 8, victim["rid"]),
    )

    # The correct projection is the full fold of the (hash-tampered, body-intact) log.
    reference = connect(tmp_path / "reference.sqlite")
    init_db(reference)
    for row in conn.execute("SELECT * FROM brain_events ORDER BY rowid ASC"):
        reference.execute(
            "INSERT INTO brain_events (id, ts, kind, writer, session_id, body_json, "
            "body_hash, prev_hash, event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["ts"],
                row["kind"],
                row["writer"],
                row["session_id"],
                row["body_json"],
                row["body_hash"],
                row["prev_hash"],
                row["event_hash"],
            ),
        )
    rebuild_projection(reference, full=True)
    expected = _snapshot(reference)

    rebuild_projection(conn)  # broken chain -> full rebuild fallback
    assert _read_cursor(conn) == _max_rowid(conn)
    assert _snapshot(conn) == expected
    assert conn.execute(
        "SELECT status FROM current_beliefs WHERE belief_id = 'belief:chain'"
    ).fetchone()["status"] == "current"


def _seed_evidence(conn) -> str:
    import json

    eid = record_evidence(conn, body="seed evidence", scope=global_scope())
    body = conn.execute(
        "SELECT body_json FROM brain_events WHERE id = ?", (eid,)
    ).fetchone()["body_json"]
    return json.loads(body)["evidence_id"]


def _seed_decided_belief(conn) -> None:
    evidence_id = _seed_evidence(conn)
    proposal_id = propose_compilation(
        conn,
        belief_id="belief:seed",
        body="seed belief",
        evidence_ids=[evidence_id],
        scope=global_scope(),
        confidence=0.9,
    )
    decide_compilation(
        conn, proposal_event_id=proposal_id, decision="approve", rebuild=False
    )
