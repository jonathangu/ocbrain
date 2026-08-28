"""Write-time closeout discipline: session identity, runtime family, failure.

Every number asserted here was measured against ONE read-only backup of the
live core, taken at **2026-08-28 12:30:51 PDT** and holding 1,239 closeouts from
2026-07-15 to 2026-08-28. A backup rather than the live file on purpose: the
corpus gained rows while the earlier draft of this file was being measured, and
a census assembled from several reads of a moving table is not a census. The
numbers are frozen here rather than recomputed, so a test failure means the rule
changed and not that the corpus grew.

One snapshot, one timestamp, one set of numbers. The earlier draft mixed a
1,236-row morning read with a fixture described as the top thirteen spellings
that was in fact ranks 1-9 and 12-15 -- the two omissions being exactly the two
that would have added families to the assertion below. A census quietly pruned
until its assertion holds is an assertion about the census.
"""

from __future__ import annotations

import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

from ocbrain.briefing import build_briefing, build_ledger
from ocbrain.closeout import (
    RUNTIME_FAMILIES,
    RUNTIME_SESSION_SHAPES,
    SERVER_CONNECTION_SESSION_PREFIX,
    SESSION_ID_SOURCES,
    _requires_unresolved,
    classify_session_id,
    is_runtime_session_id,
    record_closeout,
    resolve_session_identity,
    runtime_family,
)
from ocbrain.core_v1 import (
    CORE_V1_SCHEMA,
    init_core_v1,
    migrate_core_v1_columns,
    record_core_v1_retrieval,
)
from ocbrain.db import SCHEMA as LEGACY_SCHEMA
from ocbrain.db import connect, init_db, log_retrieval_use
from ocbrain.provenance import Provenance
from ocbrain.scope import ScopeContext

# `scripts/` is not on pytest's pythonpath (pyproject sets only `src`), and the
# reconciliation tests below compare this repo's three runtime folders. Inserted
# here rather than relied on from another test module: an import that only works
# because some other file ran first is an instrument that can silently stop
# working, which is the class of defect this whole file exists to close.
_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# --------------------------------------------------------------------------- #
# Frozen live-corpus censuses
# --------------------------------------------------------------------------- #

# `task_closeouts.session_id` on the live core, 2026-08-28: one synthetic,
# shape-equivalent literal per class, with the measured number of rows it covers.
# Public fixtures retain the distribution and classifier coverage, not caller ids.
LIVE_SESSION_CENSUS: tuple[tuple[str | None, str, int], ...] = (
    (None, "absent", 431),
    ("2026-07-22", "date_like", 296),
    ("example_cleanup_audit", "slug", 239),
    ("018f27db-3a4c-7b19-92ef-123456789abc", "runtime_uuid", 211),
    ("2026-07-21 release checklist", "contains_space", 35),
    ("/srv/example/receipt", "filesystem_path", 27),
)
LIVE_CLOSEOUTS = 1239
# Closeouts whose session id is byte-identical to a Claude Code transcript
# filename -- the join this whole gate exists to protect. Counted against the
# 1,223 transcript stems under ~/.claude/projects at the same moment.
LIVE_TRANSCRIPT_JOINS = 94

# `task_closeouts.runtime`: the fifteen most-used spellings by row count,
# ranks 1-15 with nothing skipped -- 917 of 1,239 rows. Five spell "local mac",
# four spell "codex desktop", and one has an environment description welded onto
# the client name. Truncating a ranked census is legitimate; skipping a rank in
# the middle of one is not, which is what the ranks are written out for.
LIVE_RUNTIME_CENSUS: tuple[tuple[str, int], ...] = (
    ("codex-desktop", 171),  # 1
    ("mcp", 96),  # 2
    ("local", 95),  # 3
    ("codex", 92),  # 4
    ("claude-code", 79),  # 5
    ("desktop", 67),  # 6
    ("codex-desktop-heartbeat", 60),  # 7
    ("local-mac", 50),  # 8
    ("Codex desktop", 49),  # 9
    ("cursor", 47),  # 10
    ("hermes", 45),  # 11
    ("local macOS", 23),  # 12
    ("local-macos", 18),  # 13
    ("local macOS + analytics ClickHouse", 13),  # 14
    ("macos", 12),  # 15
)
LIVE_RUNTIME_CENSUS_ROWS = 917
LIVE_RUNTIME_SPELLINGS = 160

# A real `GROUP BY status, verification_status` over all 1,239 closeouts: nine
# distinct pairs, each appearing once. An earlier draft listed eleven tuples of
# this shape by flattening a third dimension into it, so `completed/
# agent_reported` appeared twice with two different counts -- a fixture that
# cannot be what its own type says it is.
LIVE_STATUS_CENSUS: tuple[tuple[str, str, int], ...] = (
    ("completed", "verified", 890),
    ("partial", "verified", 116),
    ("completed", "failed", 95),
    ("completed", "agent_reported", 67),
    ("blocked", "failed", 35),
    ("partial", "failed", 19),
    ("partial", "agent_reported", 13),
    ("blocked", "verified", 3),
    ("failed", "failed", 1),
)

# The third dimension, given its own fixture: (status, verification_status,
# rows) for the closeouts carrying NO verifier_ref at all. These are the rows
# that had nothing to run, and the reason `completed` with no verifier is not
# charged for an `unresolved`.
LIVE_VERIFIERLESS_CENSUS: tuple[tuple[str, str, int], ...] = (
    ("completed", "agent_reported", 19),
    ("partial", "agent_reported", 3),
)


def _core(tmp_path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _create_legacy_table(
    conn: sqlite3.Connection,
    schema: str,
    table: str,
    omitted_columns: set[str],
) -> None:
    """Create the pre-gate table directly, without reverse-migrating new DDL.

    SQLite has to rewrite and reparse a table's stored SQL for ``DROP COLUMN``.
    That is not a valid way to model an older append-only table with dependent
    indexes/triggers. This fixture instead starts from the repository's canonical
    table definition and omits only the columns that did not exist before the
    closeout-discipline migration.
    """
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(table)}\s*\(.*?^\);",
        schema,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, table
    lines = [
        line
        for line in match.group(0).splitlines()
        if not any(re.match(rf"\s*{re.escape(column)}\s+", line) for column in omitted_columns)
    ]
    for index in range(len(lines) - 2, 0, -1):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("--"):
            continue
        lines[index] = lines[index].rstrip().removesuffix(",")
        break
    conn.execute("\n".join(lines))


def _init_pre_closeout_discipline_schema(
    conn: sqlite3.Connection, schema: str, *, core_v1: bool
) -> None:
    """Build the two real pre-migration tables plus their dependencies."""
    _create_legacy_table(conn, schema, "retrieval_uses", {"session_id_source"})
    _create_legacy_table(
        conn,
        schema,
        "task_closeouts",
        {"session_id_source", "runtime_family", "unresolved"},
    )
    retrieval_index = (
        "CREATE INDEX idx_retrieval_uses_outcome_served "
        "ON retrieval_uses(outcome, served_at);"
        if core_v1
        else "CREATE INDEX idx_retrieval_uses_knowledge_outcome_served "
        "ON retrieval_uses(knowledge_id, outcome, served_at);"
    )
    conn.executescript(
        f"""
        {retrieval_index}
        CREATE INDEX idx_task_closeouts_chain
          ON task_closeouts(task_ref_norm, closed_at);
        CREATE TRIGGER task_closeouts_no_update
        BEFORE UPDATE ON task_closeouts BEGIN
          SELECT RAISE(ABORT, 'task_closeouts is append-only');
        END;
        CREATE TRIGGER task_closeouts_no_delete
        BEFORE DELETE ON task_closeouts BEGIN
          SELECT RAISE(ABORT, 'task_closeouts is append-only');
        END;
        """
    )


def _close(conn, **kwargs):
    payload = {
        "task_ref": "TASK-292",
        "status": "completed",
        "summary": "Closed the receipt discipline defect at the write path.",
    }
    payload.update(kwargs)
    return record_closeout(conn, **payload)


# --------------------------------------------------------------------------- #
# Defect 1 -- session identity
# --------------------------------------------------------------------------- #


def test_the_shape_gate_reproduces_the_live_session_id_census():
    tally = collections.Counter()
    for value, expected, rows in LIVE_SESSION_CENSUS:
        assert classify_session_id(value) == expected, value
        tally[expected] += rows
    assert sum(tally.values()) == LIVE_CLOSEOUTS
    # 211 of 1,239 (17.0%) are runtime-shaped. The other 1,028 are absent or
    # hand-built, and 597 of those are a human typing something descriptive.
    assert tally["runtime_uuid"] == 211
    assert tally["absent"] == 431
    hand_written = sum(
        rows for shape, rows in tally.items() if shape not in {"absent", "runtime_uuid"}
    )
    assert hand_written == 597
    assert tally["runtime_uuid"] + tally["absent"] + hand_written == LIVE_CLOSEOUTS


def test_the_gate_admits_every_id_that_joins_a_transcript_and_no_others():
    """The refusal cannot cost a joinable row, because none of them is refused.

    All 94 closeouts that join a Claude Code transcript are ``runtime_uuid``;
    zero of the 597 hand-written ids join one. So admitting exactly the
    runtime-minted shapes keeps 94/94 and drops 0/94 -- the gate is not a
    trade-off between strictness and coverage, and that is why it is a shape
    question rather than a taste question.
    """
    joinable_shape = "runtime_uuid"
    admitted = [
        rows for value, shape, rows in LIVE_SESSION_CENSUS if classify_session_id(value) == shape
    ]
    assert sum(admitted) == LIVE_CLOSEOUTS
    by_shape = {shape: rows for _v, shape, rows in LIVE_SESSION_CENSUS}
    assert LIVE_TRANSCRIPT_JOINS <= by_shape[joinable_shape]
    for value, shape, _rows in LIVE_SESSION_CENSUS:
        if shape == joinable_shape:
            assert resolve_session_identity(value, Provenance())["session_id"] == value
        elif value is not None:
            with pytest.raises(ValueError):
                resolve_session_identity(value, Provenance())


@pytest.mark.parametrize(
    "value",
    [
        "chat-operator-2026-08-25",
        "2026-07-21 release checklist",
        "/srv/example/receipt",
        "example_cleanup_audit",
        "2026-07-22",
        "20260801_075651_56e87d0b",
        "current Codex thread",
    ],
)
def test_a_hand_written_session_id_is_refused_and_the_error_says_where_to_get_one(
    tmp_path, value
):
    """Every literal preserves an observed class without exposing a caller id."""
    conn = _core(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        _close(conn, context=ScopeContext(project="ocbrain", session=value))
    message = str(excinfo.value)
    assert "CLAUDE_CODE_SESSION_ID" in message
    assert "OCBRAIN_SESSION_ID" in message
    assert "omit context.session" in message
    assert repr(value) in message


def test_omitting_the_session_is_legal_and_the_server_fills_it_from_its_own_connection(
    tmp_path,
):
    """The gate has to be satisfiable by a client that has no session id at all.

    Otherwise it is a gate that refuses work nobody can do differently, which is
    worse than the free-text column it replaces.
    """
    conn = _core(tmp_path)
    receipt = _close(
        conn,
        context=ScopeContext(project="ocbrain", runtime="hermes-cron"),
        provenance=Provenance(server_connection_id="cafe" * 8),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'cafe' * 8}"
    assert row["session_id_source"] == "server_connection"
    # Prefixed, so a later transcript join can never mistake it for one.
    assert not row["session_id"].startswith("cafe")


def test_a_caller_with_neither_a_session_nor_a_connection_still_files_a_closeout(tmp_path):
    conn = _core(tmp_path)
    receipt = _close(conn, context=ScopeContext(project="ocbrain"))
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] is None
    assert row["session_id_source"] == "none"


def test_the_harness_attested_hint_outranks_the_model_and_the_disagreement_is_kept(
    tmp_path,
):
    conn = _core(tmp_path)
    observed = "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7"
    claimed = "018f27db-3a4c-7b19-92ef-123456789abc"
    receipt = _close(
        conn,
        context=ScopeContext(project="ocbrain", session=claimed),
        provenance=Provenance(
            server_connection_id="beef" * 8, client_session_hint=observed
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == observed
    assert row["session_id_source"] == "harness_attested"
    identity = receipt["provenance"]["session_identity"]
    assert identity["session_id_claim"] == claimed
    assert identity["session_id_conflict"] is True
    # The model's claim is still in the receipt under its historical key.
    assert receipt["provenance"]["session_id"] == claimed


def test_quarantine_keeps_the_claim_out_of_the_column_without_refusing(tmp_path):
    resolved = resolve_session_identity(
        "example_cleanup_audit",
        Provenance(server_connection_id="feed" * 8),
        policy="quarantine",
    )
    assert resolved["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'feed' * 8}"
    assert resolved["session_id_source"] == "server_connection"
    assert resolved["session_id_claim"] == "example_cleanup_audit"
    assert "session_id_conflict" not in resolved


def test_every_source_a_closeout_can_carry_is_a_declared_one():
    """``SESSION_ID_SOURCES`` is the documented vocabulary of that column.

    A constant nothing checks drifts away from the code, and then a consumer
    filtering on it silently drops rows. Every path through
    ``resolve_session_identity`` is exercised here.
    """
    uuid = "018f27db-3a4c-7b19-92ef-123456789abc"
    other = "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7"
    produced = {
        resolve_session_identity(None, Provenance())["session_id_source"],
        resolve_session_identity(uuid, Provenance())["session_id_source"],
        resolve_session_identity(
            None, Provenance(server_connection_id="ab" * 16)
        )["session_id_source"],
        resolve_session_identity(
            uuid, Provenance(client_session_hint=other)
        )["session_id_source"],
        resolve_session_identity("a-slug", Provenance(), policy="off")["session_id_source"],
        resolve_session_identity("a-slug", Provenance(), policy="quarantine")[
            "session_id_source"
        ],
    }
    assert produced == SESSION_ID_SOURCES


def test_policy_off_restores_the_pre_gate_behaviour_exactly(tmp_path):
    resolved = resolve_session_identity("example_cleanup_audit", Provenance(), policy="off")
    assert resolved["session_id"] == "example_cleanup_audit"
    assert resolved["session_id_source"] == "agent_reported"


# --------------------------------------------------------------------------- #
# Defect 2 -- runtime family
# --------------------------------------------------------------------------- #


def test_the_fifteen_top_runtime_spellings_collapse_to_five_families_and_unknown():
    """Ranks 1-15, nothing skipped, and the answer the whole census gives.

    The count in the name is the count in the assertion. The previous version
    of this test named three families and asserted four, over a census that had
    quietly dropped ranks 10 and 11 -- which are exactly the two rows that add
    `cursor` and `hermes`. A census pruned until the assertion holds is an
    assertion about the census.
    """
    tally = collections.Counter()
    for spelling, rows in LIVE_RUNTIME_CENSUS:
        tally[runtime_family(spelling)] += rows
    assert sum(tally.values()) == LIVE_RUNTIME_CENSUS_ROWS == 917
    # Four spellings of "codex desktop" plus bare "codex" -- 372 rows, one family.
    assert tally["codex"] == 171 + 92 + 60 + 49
    assert tally["mcp"] == 96
    assert tally["claude-code"] == 79
    assert tally["cursor"] == 47
    assert tally["hermes"] == 45
    # "local", "desktop", "macOS" and friends name the machine, not the client.
    # 278 rows across seven spellings, and `unknown` is the honest answer for all
    # of them: inventing a client here would be guessing.
    assert tally["unknown"] == 95 + 67 + 50 + 23 + 18 + 13 + 12
    assert set(tally) == {"codex", "mcp", "claude-code", "cursor", "hermes", "unknown"}
    # Descending by rows, so a later edit cannot reintroduce a skipped rank
    # without also breaking the order.
    counts = [rows for _spelling, rows in LIVE_RUNTIME_CENSUS]
    assert counts == sorted(counts, reverse=True)
    assert LIVE_RUNTIME_SPELLINGS == 160


def test_a_normaliser_matching_substrings_invents_data():
    """Regression: "ClickHouse" contains "cli".

    Matching family tokens as substrings put the 13 live rows spelled
    'local macOS + analytics ClickHouse' in the `cli` family, and 16 more
    besides. Segment matching is what stops a normaliser being confidently
    wrong about a third of a family.
    """
    assert runtime_family("local macOS + analytics ClickHouse") == "unknown"
    assert runtime_family("local Mac; Dagster localhost; analytics lake") == "cli"
    assert runtime_family("gcloud-cli") == "cli"
    # Path- and profile-separated spellings still resolve, which is what the
    # wider separator set buys.
    assert runtime_family("hermes@example-profile") == "hermes"
    assert runtime_family("~/.local/share/hermes-runtimes/example-profile") == "hermes"
    assert runtime_family("hermes:example-worker") == "hermes"


def test_the_server_observed_key_outranks_the_model_and_detail_gets_its_own_field(
    tmp_path,
):
    """The value that smuggled an environment into the client name, fixed.

    'local macOS + analytics ClickHouse' appeared 13 times because there was
    nowhere else to put the second half.
    """
    conn = _core(tmp_path)
    receipt = _close(
        conn,
        context=ScopeContext(project="ocbrain", runtime="local macOS"),
        runtime_detail="analytics ClickHouse",
        provenance=Provenance(
            server_connection_id="0" * 32, client_runtime_key="hermes:example-worker"
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT runtime, runtime_family FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    # The model said "local macOS"; the process saw Hermes. The observed one wins.
    assert row["runtime_family"] == "hermes"
    assert row["runtime"] == "local macOS"
    assert receipt["provenance"]["runtime_detail"] == "analytics ClickHouse"


def test_an_unrecognised_server_key_falls_through_to_the_model_rather_than_unknown():
    """64 live rows carry a client key the shipped rules do not know.

    Their model claim does say `claude-code`. Falling through preserves that
    rather than throwing it away for the sake of precedence.
    """
    assert runtime_family("local-agent-mode-ocbrain", "claude-code") == "claude-code"
    assert runtime_family("local-agent-mode-ocbrain", "local") == "unknown"


def test_an_operator_alias_can_name_an_install_specific_label():
    aliases = {"local-agent-mode-ocbrain": "claude-code", "example-profile": "hermes"}
    assert runtime_family("local-agent-mode-ocbrain", aliases=aliases) == "claude-code"
    assert runtime_family("example-profile", aliases=aliases) == "hermes"
    # An alias may not invent an eighth family.
    assert runtime_family("whatever", aliases={"whatever": "teapot"}) == "unknown"
    assert "teapot" not in RUNTIME_FAMILIES


def test_runtime_family_is_pure_so_history_stays_analysable():
    """``task_closeouts`` is append-only; the 160 historical spellings can never
    be rewritten in place. A pure function is what keeps them groupable."""
    for spelling, _rows in LIVE_RUNTIME_CENSUS:
        assert runtime_family(spelling) == runtime_family(spelling)
        assert runtime_family(spelling) in RUNTIME_FAMILIES


# --------------------------------------------------------------------------- #
# Defect 3 -- failure reporting
# --------------------------------------------------------------------------- #


def test_the_unresolved_gate_catches_282_of_the_1239_live_closeouts():
    """282 rows (22.8%) carry evidence something did not work and no field for it.

    95 of those claim `completed`. Gating on `status` alone would have missed
    every one of them, which is the whole reason the verifier evidence is a
    second, independent trigger.
    """
    # A GROUP BY has one row per key. Asserted, because the previous fixture
    # looked like one and was not.
    pairs = [(status, verification) for status, verification, _r in LIVE_STATUS_CENSUS]
    assert len(pairs) == len(set(pairs)) == 9
    assert sum(rows for _s, _v, rows in LIVE_STATUS_CENSUS) == LIVE_CLOSEOUTS
    caught = sum(
        rows for status, verification, rows in LIVE_STATUS_CENSUS
        if _requires_unresolved(status, verification)
    )
    assert caught == 282
    by_status_alone = sum(
        rows for status, _v, rows in LIVE_STATUS_CENSUS if status != "completed"
    )
    assert by_status_alone == 187
    assert caught - by_status_alone == 95
    clean = sum(
        rows for status, verification, rows in LIVE_STATUS_CENSUS
        if not _requires_unresolved(status, verification)
    )
    assert clean == LIVE_CLOSEOUTS - 282 == 957
    # The verifierless rows are a strict subset of the pairs above, not a
    # separate population: 22 of the 67 `completed/agent_reported` and 13
    # `partial/agent_reported` rows had no verifier to run at all.
    for status, verification, rows in LIVE_VERIFIERLESS_CENSUS:
        total = next(
            r for s, v, r in LIVE_STATUS_CENSUS if (s, v) == (status, verification)
        )
        assert rows < total
    assert sum(rows for _s, _v, rows in LIVE_VERIFIERLESS_CENSUS) == 22


def test_a_completed_closeout_with_a_failed_verifier_must_say_what_failed(tmp_path):
    conn = _core(tmp_path)
    verifiers = [
        {"uri": "repo://ocbrain/pytest", "status": "passed"},
        {"uri": "repo://ocbrain/ruff", "status": "failed"},
    ]
    with pytest.raises(ValueError, match="unresolved is required"):
        _close(conn, status="completed", verifier_refs=verifiers)
    receipt = _close(
        conn,
        status="completed",
        verifier_refs=verifiers,
        unresolved="ruff still reports two E501s in closeout.py.",
    )
    conn.commit()
    row = conn.execute(
        "SELECT unresolved FROM task_closeouts WHERE id=?", (receipt["id"],)
    ).fetchone()
    assert row["unresolved"] == "ruff still reports two E501s in closeout.py."
    assert receipt["unresolved"] == "ruff still reports two E501s in closeout.py."


def test_both_gates_report_together_so_one_retry_fixes_both(tmp_path):
    """A caller with two problems learns both at once.

    Refusing one at a time costs an unattended agent two retries for one
    closeout, and the second refusal arrives only after it has already
    rewritten something.
    """
    conn = _core(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        _close(
            conn,
            status="partial",
            context=ScopeContext(project="ocbrain", session="example_cleanup_audit"),
        )
    message = str(excinfo.value)
    assert "CLAUDE_CODE_SESSION_ID" in message
    assert "unresolved is required" in message


def test_a_clean_success_is_not_asked_for_an_explanation(tmp_path):
    conn = _core(tmp_path)
    receipt = _close(
        conn, verifier_refs=[{"uri": "repo://ocbrain/pytest", "status": "passed"}]
    )
    conn.commit()
    assert receipt["unresolved"] is None
    # And a `completed` with no verifiers at all is still clean: the ledger
    # already reports that as `in_flight` rather than done, and charging it for
    # an explanation would tax the 22 live rows that simply had nothing to run
    # (LIVE_VERIFIERLESS_CENSUS).
    assert _close(conn, task_ref="no-verifier")["unresolved"] is None


def test_an_audit_whose_verifiers_all_failed_may_still_be_completed(tmp_path):
    """Do not derive the status from the evidence. Seven live closeouts claim
    `completed` with every verifier failed, and all seven are read-only audits
    where the FAIL verdict IS the deliverable -- "Read-only re-review found
    remaining blockers; verdict FAIL". Relabelling those `failed` would call
    successful work a failure. The caller keeps the verdict and owes a sentence.
    """
    conn = _core(tmp_path)
    receipt = _close(
        conn,
        status="completed",
        summary="Read-only re-review found remaining blockers; verdict FAIL.",
        verifier_refs=[
            {"uri": "audit://autoresearch/lineage", "status": "failed"},
            {"uri": "audit://autoresearch/capacity", "status": "failed"},
        ],
        unresolved="The four blocking defects are reported, not fixed; nobody owns them yet.",
    )
    conn.commit()
    assert receipt["status"] == "completed"
    assert receipt["verification_status"] == "failed"


def test_every_non_completion_status_owes_an_explanation(tmp_path):
    conn = _core(tmp_path)
    for status in ("partial", "failed", "cancelled"):
        with pytest.raises(ValueError, match="unresolved is required"):
            _close(conn, task_ref=f"t-{status}", status=status)
    # `blocked` already required `awaiting`; it now owes both, because "what
    # unblocks me" and "what did not work" are different sentences.
    with pytest.raises(ValueError, match="unresolved is required"):
        _close(conn, task_ref="t-blocked", status="blocked", awaiting="a human")
    receipt = _close(
        conn,
        task_ref="t-blocked",
        status="blocked",
        awaiting="An operator to approve the analytics credential",
        unresolved="The source refresh has never run against production.",
    )
    assert receipt["awaiting"] != receipt["unresolved"]


def test_an_existing_core_gains_the_columns_before_the_first_closeout_lands(tmp_path):
    """The live core is 208 MB and predates all three columns.

    ``CREATE TABLE IF NOT EXISTS`` means a fresh core gets them from the schema
    and proves nothing about an existing one. The write path names all three, so
    without the additive migration the first ``brain.closeout`` after deploy
    fails on an unknown column -- and a fresh-core test cannot see that.
    """
    path = tmp_path / "legacy-core.sqlite"
    conn = connect(path)
    _init_pre_closeout_discipline_schema(conn, CORE_V1_SCHEMA, core_v1=True)
    conn.commit()
    present = {row[1] for row in conn.execute("PRAGMA table_info(task_closeouts)")}
    assert not present & {"session_id_source", "runtime_family", "unresolved"}
    with pytest.raises(sqlite3.OperationalError):
        _close(conn, task_ref="before-migration")

    assert migrate_core_v1_columns(conn)
    conn.commit()
    receipt = _close(conn, task_ref="after-migration")
    conn.commit()
    row = conn.execute(
        "SELECT session_id_source, runtime_family, unresolved FROM task_closeouts "
        "WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id_source"] == "none"
    assert row["runtime_family"] == "unknown"
    assert row["unresolved"] is None


def test_the_legacy_initializer_also_migrates_an_existing_database(tmp_path):
    """``db.init_db`` carries its own copy of the schema and its own migration
    list. Both have to add the columns, or a legacy store is broken by a deploy
    the v1 core survived."""
    conn = connect(tmp_path / "legacy.sqlite")
    _init_pre_closeout_discipline_schema(conn, LEGACY_SCHEMA, core_v1=False)
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        _close(conn, task_ref="before-migration")

    init_db(conn)
    conn.commit()
    receipt = _close(conn, task_ref="after-migration")
    conn.commit()
    assert conn.execute(
        "SELECT runtime_family FROM task_closeouts WHERE id=?", (receipt["id"],)
    ).fetchone()["runtime_family"] == "unknown"


def test_a_misspelled_policy_falls_back_instead_of_taking_the_write_path_down(
    tmp_path, monkeypatch
):
    """A typo in a config file must not refuse every closeout on the install."""
    monkeypatch.setenv("OCBRAIN_CLOSEOUT_SESSION_ID_POLICY", "enfroce")
    conn = _core(tmp_path)
    receipt = _close(conn, context=ScopeContext(project="ocbrain"))
    conn.commit()
    assert receipt["provenance"]["session_identity"]["session_id_source"] == "none"
    # Still the shipped default, not "anything goes".
    with pytest.raises(ValueError, match="CLAUDE_CODE_SESSION_ID"):
        _close(
            conn,
            task_ref="typo-policy",
            context=ScopeContext(project="ocbrain", session="a-slug"),
        )


def test_the_gates_are_configurable_and_off_reproduces_the_old_behaviour(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OCBRAIN_CLOSEOUT_SESSION_ID_POLICY", "off")
    monkeypatch.setenv("OCBRAIN_CLOSEOUT_REQUIRE_UNRESOLVED", "false")
    conn = _core(tmp_path)
    receipt = _close(
        conn,
        status="partial",
        context=ScopeContext(project="ocbrain", session="example_cleanup_audit"),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, unresolved FROM task_closeouts WHERE id=?", (receipt["id"],)
    ).fetchone()
    assert row["session_id"] == "example_cleanup_audit"
    assert row["unresolved"] is None


# --------------------------------------------------------------------------- #
# Remediation -- the instrument has to back the claim on every path
# --------------------------------------------------------------------------- #

# The sentence a blocked closeout files. Distinctive on purpose: every assertion
# below looks for this exact string, so a surface that merely reports "blocked"
# or echoes the summary cannot pass by accident.
_UNRESOLVED_SENTENCE = "THE-SENTENCE-THAT-MATTERS: refresh never ran against prod."


def _blocked_with_unresolved(conn, task_ref="LEDGER-1"):
    return _close(
        conn,
        task_ref=task_ref,
        status="blocked",
        summary="Tried the refresh; could not get a credential.",
        awaiting="An operator to approve the analytics credential",
        unresolved=_UNRESOLVED_SENTENCE,
        context=ScopeContext(project="ocbrain"),
    )


def test_the_ledger_serves_the_unresolved_sentence_the_gate_charges_for(tmp_path):
    """The gate's whole justification, made true.

    ``brain.closeout``'s schema, the refusal text, AGENT_USE_GUIDE and
    THRESHOLDS all tell the caller that ``brain.ledger`` reads ``unresolved`` to
    stop the next session repeating the attempt. Before this test, the ledger's
    row projection stopped at ``awaiting`` and the sentence appeared nowhere in
    its output: the field was required at write time and read by nothing, which
    is a gate charging for a column no reader serves.
    """
    conn = _core(tmp_path)
    _blocked_with_unresolved(conn)
    conn.commit()

    ledger = build_ledger(conn, context=ScopeContext(project="ocbrain"))
    entry = ledger["entries"][0]
    assert entry["state"] == "attempted_failed"
    assert entry["latest_unresolved"] == _UNRESOLVED_SENTENCE
    assert entry["failed_attempts"][0]["unresolved"] == _UNRESOLVED_SENTENCE
    # And nowhere-in-the-payload is the failure the reviewer actually reproduced.
    assert _UNRESOLVED_SENTENCE in json.dumps(ledger)


def test_the_briefings_failed_line_carries_what_did_not_work_not_just_the_summary(
    tmp_path,
):
    """The briefing is the other surface that claims to stop a repeat.

    ``_ledger_line`` already argued in a comment that the failure *text* is what
    distinguishes skipping work from not repeating it -- and then printed the
    summary, which is what the agent did, not what is still broken. Now that a
    non-clean closeout is charged for the second sentence, the briefing serves
    it.
    """
    conn = _core(tmp_path)
    _blocked_with_unresolved(conn)
    conn.commit()

    briefing = build_briefing(conn, context=ScopeContext(project="ocbrain"))
    assert _UNRESOLVED_SENTENCE[:40] in briefing["text"]


def test_a_failed_attempt_with_no_unresolved_still_reports_its_summary(
    tmp_path, monkeypatch
):
    """Every historical row has ``unresolved`` NULL and must not lose its line.

    The column is never backfilled -- ``task_closeouts`` is append-only under a
    trigger -- so the read path has to degrade to the summary rather than
    printing an empty failure. Written with the gate off, which is exactly the
    shape of the 1,238 rows already in the live core.
    """
    monkeypatch.setenv("OCBRAIN_CLOSEOUT_REQUIRE_UNRESOLVED", "false")
    conn = _core(tmp_path)
    _close(
        conn,
        task_ref="LEGACY-1",
        status="failed",
        summary="Legacy row written before the gate existed.",
        context=ScopeContext(project="ocbrain"),
    )
    conn.commit()
    monkeypatch.delenv("OCBRAIN_CLOSEOUT_REQUIRE_UNRESOLVED")
    ledger = build_ledger(conn, context=ScopeContext(project="ocbrain"))
    entry = ledger["entries"][0]
    assert entry["latest_unresolved"] is None
    briefing = build_briefing(conn, context=ScopeContext(project="ocbrain"))
    assert "Legacy row written before the gate existed." in briefing["text"]


@pytest.mark.parametrize(
    "hint",
    [
        "example_cleanup_audit",
        "2026-07-22",
        "2026-07-21 release checklist",
        "/srv/example/receipt",
        "current",
        "ocbrain",
    ],
)
def test_a_junk_harness_hint_never_wins_the_identity_column(tmp_path, hint):
    """The highest-trust path is the one door the junk can walk through.

    ``client_session_hint`` is read from ``$OCBRAIN_SESSION_ID`` -- an
    operator-settable environment variable -- and outranks everything. The shape
    check on it had no test at all: mutating
    ``classify_session_id(value) in RUNTIME_SESSION_SHAPES`` to
    ``classify_session_id(value) != "absent"`` left the entire suite green while
    ``example_cleanup_audit`` landed in the identity column wearing the
    ``harness_attested`` label. A guard nothing can make fail is the defect.
    """
    resolved = resolve_session_identity(
        None, Provenance(server_connection_id="ab" * 16, client_session_hint=hint)
    )
    assert resolved["session_id"] != hint
    assert resolved["session_id_source"] == "server_connection"
    assert resolved["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'ab' * 16}"

    conn = _core(tmp_path)
    receipt = _close(
        conn,
        task_ref=f"hint-{abs(hash(hint))}",
        context=ScopeContext(project="ocbrain"),
        provenance=Provenance(server_connection_id="ab" * 16, client_session_hint=hint),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id_source"] == "server_connection"
    assert row["session_id"] != hint


def test_a_junk_hint_does_not_rescue_a_caller_who_omitted_everything(tmp_path):
    """With no connection id either, a junk hint must leave the column empty
    rather than filling it with the label it happens to carry."""
    resolved = resolve_session_identity(None, Provenance(client_session_hint="current"))
    assert resolved["session_id"] is None
    assert resolved["session_id_source"] == "none"


def test_is_runtime_session_id_admits_exactly_the_two_machine_shapes():
    """The membership predicate itself, not only its caller.

    Every shape ``classify_session_id`` can return is asserted here, so widening
    the predicate to anything-not-absent fails on six cases rather than passing
    silently.
    """
    admitted = {
        "018f27db-3a4c-7b19-92ef-123456789abc": True,
        "0123456789abcdef0123456789abcdef01234567": True,
        "ab" * 16: True,
        "example_cleanup_audit": False,
        "2026-07-22": False,
        "2026-07-21 release checklist": False,
        "/srv/example/receipt": False,
        "current": False,
        "": False,
        None: False,
    }
    for value, expected in admitted.items():
        assert is_runtime_session_id(value) is expected, value
    assert {
        classify_session_id(value) for value, ok in admitted.items() if ok
    } <= RUNTIME_SESSION_SHAPES
    assert not {
        classify_session_id(value) for value, ok in admitted.items() if not ok
    } & RUNTIME_SESSION_SHAPES


# --------------------------------------------------------------------------- #
# Remediation -- three normalisers, reconciled by an instrument
# --------------------------------------------------------------------------- #

# The two families that exist only in procmine's historical-mining taxonomy,
# and their name in the write-time enum. `mcp-direct` and `mcp` are the same
# answer; `host-batch` and `cli` are the same answer. Declared here so the
# reconciliation test below can compare the two mappers instead of asserting
# that they look similar.
PROCMINE_FAMILY_OF: dict[str, str] = {
    "claude-code": "claude-code",
    "codex": "codex",
    "cursor": "cursor",
    "hermes": "hermes",
    "mcp": "mcp-direct",
    "cli": "host-batch",
}


def test_this_repos_own_runtime_is_not_unknown_to_its_own_normaliser():
    """`ocbrain-runtime-call` is set by src/ocbrain/runtime_call.py.

    The sibling mapper in scripts/procmine has placed it since it was written;
    the new column called this repo's own runtime `unknown` on the 2 live rows
    that carry it. An exact spelling, not a token, because the token `ocbrain`
    also appears in `local-agent-mode-ocbrain` -- 66 live rows of a Claude Code
    client key -- and folding those into `mcp` is exactly the invent-a-family
    failure the segment rule exists to prevent.
    """
    assert runtime_family("ocbrain-runtime-call") == "mcp"
    assert runtime_family("local-agent-mode-ocbrain") == "unknown"
    assert runtime_family("local-agent-mode-ocbrain", "claude-code") == "claude-code"


def test_an_alias_key_with_a_space_is_reachable():
    """Operator alias keys are folded the same way candidates are.

    The keys were lowercased and the candidate was fully folded, so any key
    containing a space or punctuation could never match. The shipped table is
    empty, so nothing was broken -- but a table whose entries silently do
    nothing is a trap with a config file in front of it. The literal here is
    chosen to share no token with any shipped rule, so it can only resolve
    through the alias.
    """
    aliases = {"Example Worker": "hermes"}
    assert runtime_family("example worker", aliases=aliases) == "hermes"
    assert runtime_family("example-worker", aliases=aliases) == "hermes"
    assert runtime_family("example worker") == "unknown"


def test_the_write_time_enum_and_the_mining_taxonomy_never_contradict_each_other():
    """Three normalisers existed; this is the instrument that keeps them one.

    ``procmine.episodes.normalize_runtime`` now asks
    ``closeout.runtime_family`` first and only falls through to its own
    install-specific rules when the shared mapper abstains, so the two can
    disagree only by one of them saying "I don't know". This test is what makes
    that structural rather than a comment: it fails the moment either mapper
    grows a rule the other contradicts.

    The check is fail-closed -- every spelling in the frozen live census is
    compared, and there is no exemption list. Abstention is not a contradiction:
    procmine carries operator-specific tokens (`telegram`, `kanban`, a Hermes
    profile hash) that the shipped rules deliberately do not, because this repo
    is public and those belong in ``closeout.runtime_aliases``.
    """
    from procmine.episodes import normalize_runtime

    for spelling, _rows in LIVE_RUNTIME_CENSUS:
        shared = runtime_family(spelling)
        mined = normalize_runtime(spelling)
        if shared == "unknown":
            continue
        assert mined == PROCMINE_FAMILY_OF[shared], (spelling, shared, mined)
    # The concrete row the reviewer named, from both ends.
    assert runtime_family("ocbrain-runtime-call") == "mcp"
    assert normalize_runtime("ocbrain-runtime-call") == "mcp-direct"


def test_the_mining_taxonomy_no_longer_matches_family_tokens_as_substrings():
    """The sibling carried the defect the new code documents, and kept it.

    ``_RUNTIME_RULES`` was ``re.search`` over the raw string for every family,
    including the six the shipped folder knows -- so "hermeneutics" read as
    Hermes and "Codexterity" as Codex, the same class as "ClickHouse" reading as
    a CLI. Those six now resolve through the segment matcher. The probes below
    are synthetic, because the live corpus happens not to contain a word that
    trips it: the point is the class, and a defect you can only demonstrate
    after it has cost you a row is one you fixed too late.

    What remains in procmine IS substring-matched, deliberately:
    `hermes-runtimes` is a path fragment and the profile portion is synthetic.
    """
    from procmine.episodes import normalize_runtime

    for word, was in (
        ("hermeneutics", "hermes"),
        ("Codexterity", "codex"),
        ("claudette", "claude-code"),
        ("precursory", "cursor"),
    ):
        assert normalize_runtime(word) == "unknown", (word, was)
    # Install-specific tokens still place, and `hermeswork` is now named rather
    # than riding on a substring of the family name: 3 live rows.
    assert normalize_runtime("HermesWork") == "hermes"
    assert normalize_runtime("telegram") == "hermes"
    assert normalize_runtime("~/.local/share/hermes-runtimes/example-profile") == "hermes"
    # 13 live rows. procmine still places them, by "local" -- never by "cli".
    assert normalize_runtime("local macOS + analytics ClickHouse") == (
        "unattributed-local"
    )
    assert normalize_runtime("analytics ClickHouse") == "unknown"
    # 8 live rows across closeouts and retrieval receipts that the shared folder
    # places and procmine could not: bare `cli`.
    assert normalize_runtime("cli") == "host-batch"


def test_the_legible_slug_folder_never_names_a_different_client():
    """``procmine.runtimes.canonical_runtime`` is the third mapper.

    It answers a different question -- keep an unrecognized runtime legible
    rather than bucketing it -- and it is left alone. What is asserted is the
    only thing that would be a defect: where both name a client, they must name
    the same one.
    """
    from procmine.runtimes import canonical_runtime

    for spelling, _rows in LIVE_RUNTIME_CENSUS:
        slug = canonical_runtime(spelling)
        shared = runtime_family(spelling)
        if slug in RUNTIME_FAMILIES and shared != "unknown":
            assert slug == shared, (spelling, slug, shared)


def test_the_miner_reads_the_authoritative_source_column_not_its_own_guess(tmp_path):
    """A server-minted `conn:` id is server-observed, whatever the miner infers.

    ``procmine.episodes`` decided `server_observed` vs `model_reported` from
    whether a ``client_session_hint`` was present. Rows that used to be NULL now
    carry `conn:<32hex>` -- minted by the server, with no hint -- so every one of
    them would have been labelled `model_reported`, which is the opposite of the
    truth. The write path already records who filled the column; the miner now
    reads it instead of guessing.

    A mislabel, not a bad join: `conn:` + 32 hex has no hyphens, so it cannot
    match a UUID transcript stem. That is why this is minor and still wrong.
    """
    from procmine.episodes import load_episodes

    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    uuid = "018f27db-3a4c-7b19-92ef-123456789abc"
    _close(
        conn,
        task_ref="conn-only",
        context=ScopeContext(project="ocbrain"),
        provenance=Provenance(server_connection_id="ab" * 16),
    )
    _close(
        conn,
        task_ref="harness",
        context=ScopeContext(project="ocbrain"),
        provenance=Provenance(server_connection_id="cd" * 16, client_session_hint=uuid),
    )
    _close(
        conn,
        task_ref="model",
        context=ScopeContext(project="ocbrain", session=uuid),
    )
    conn.commit()
    conn.close()

    by_ref = {ep.task_ref: ep for ep in load_episodes(path)}
    assert by_ref["conn-only"].session_id.startswith(SERVER_CONNECTION_SESSION_PREFIX)
    assert by_ref["conn-only"].session_source == "server_observed"
    assert by_ref["harness"].session_source == "server_observed"
    # The one path where the value really is the model's word.
    assert by_ref["model"].session_source == "model_reported"


# --------------------------------------------------------------------------- #
# Remediation -- the sibling column, at four times the scale
# --------------------------------------------------------------------------- #

_RETRIEVAL_SLUG = "2026-07-16-release-checklist"


def _v1_retrieval(conn, session, provenance=None):
    return record_core_v1_retrieval(
        conn,
        query="what do I know about the closeout gate",
        context={"project": "ocbrain", "session": session},
        items=[],
        runtime="codex-desktop",
        task_ref="TASK-292",
        session_id=session,
        provenance=provenance,
    )


def test_the_retrieval_receipt_gets_the_same_identity_discipline(tmp_path):
    """`retrieval_uses.session_id` carried the identical defect, larger.

    Measured on the same backup: 2,048 rows, 1,115 with a session id, of which
    **967 are hand-written and zero join a transcript**; 148 are machine-shaped
    and 18 join. That is the closeout finding again at four times the scale, and
    a fix that lands on one of a pair is this repository's most-repeated defect.

    The policy is `quarantine`, not `enforce`, and the difference is deliberate:
    ``brain.closeout`` is a write the agent chose to make and can retry, while
    this receipt is a side effect of ``brain.context``. Refusing a *read*
    because the session label is wrong would break retrieval to fix a join.
    """
    conn = _core(tmp_path)
    slug = _v1_retrieval(
        conn, _RETRIEVAL_SLUG, Provenance(server_connection_id="ab" * 16)
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source, context_json FROM retrieval_uses "
        "WHERE id=?",
        (slug,),
    ).fetchone()
    assert row["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'ab' * 16}"
    assert row["session_id_source"] == "server_connection"
    # Nothing is destroyed: the caller's own word is still in the row, verbatim,
    # where all 1,115 live rows already keep it.
    assert json.loads(row["context_json"])["session"] == _RETRIEVAL_SLUG


def test_a_retrieval_is_never_refused_for_its_session_label(tmp_path):
    """The read path must not raise. Whatever arrives, a receipt is written."""
    conn = _core(tmp_path)
    for session in (_RETRIEVAL_SLUG, "/srv/example/receipt", "current", None, ""):
        rid = _v1_retrieval(conn, session)
        conn.commit()
        row = conn.execute(
            "SELECT session_id, session_id_source FROM retrieval_uses WHERE id=?",
            (rid,),
        ).fetchone()
        assert row["session_id"] is None
        assert row["session_id_source"] == "none"


def test_a_runtime_shaped_retrieval_session_is_kept_exactly(tmp_path):
    """The 18 rows that do join must keep joining."""
    conn = _core(tmp_path)
    uuid = "018f27db-3a4c-7b19-92ef-123456789abc"
    rid = _v1_retrieval(conn, uuid)
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM retrieval_uses WHERE id=?", (rid,)
    ).fetchone()
    assert row["session_id"] == uuid
    assert row["session_id_source"] == "agent_reported"


def test_the_legacy_retrieval_writer_is_gated_too(tmp_path):
    """``db.log_retrieval_use`` is the v0 twin, and a pair is where a fix stops.

    Two writers, one rule. This one is reached by ``brain.feedback`` and the
    resource-read receipt, neither of which goes anywhere near the v1 path.
    """
    conn = connect(tmp_path / "legacy.sqlite")
    init_db(conn)
    rid = log_retrieval_use(
        conn,
        None,
        runtime="codex-desktop",
        task_ref="TASK-292",
        outcome="served",
        session_id=_RETRIEVAL_SLUG,
        provenance=Provenance(server_connection_id="cd" * 16),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM retrieval_uses WHERE id=?", (rid,)
    ).fetchone()
    assert row["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'cd' * 16}"
    assert row["session_id_source"] == "server_connection"


def test_a_junk_harness_hint_never_wins_the_retrieval_column_either(tmp_path):
    """The highest-trust door, on the second table."""
    conn = _core(tmp_path)
    rid = _v1_retrieval(
        conn,
        None,
        Provenance(server_connection_id="ef" * 16, client_session_hint="current"),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM retrieval_uses WHERE id=?", (rid,)
    ).fetchone()
    assert row["session_id"] == f"{SERVER_CONNECTION_SESSION_PREFIX}{'ef' * 16}"
    assert row["session_id_source"] == "server_connection"


def test_an_existing_core_gains_the_retrieval_column_before_the_first_read(tmp_path):
    """Same argument as the closeout columns: the live core predates it."""
    path = tmp_path / "legacy-core.sqlite"
    conn = connect(path)
    _init_pre_closeout_discipline_schema(conn, CORE_V1_SCHEMA, core_v1=True)
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        _v1_retrieval(conn, None)

    assert migrate_core_v1_columns(conn)
    conn.commit()
    rid = _v1_retrieval(conn, None)
    conn.commit()
    assert conn.execute(
        "SELECT session_id_source FROM retrieval_uses WHERE id=?", (rid,)
    ).fetchone()["session_id_source"] == "none"


def test_the_legacy_initializer_also_adds_the_retrieval_column(tmp_path):
    conn = connect(tmp_path / "legacy.sqlite")
    _init_pre_closeout_discipline_schema(conn, LEGACY_SCHEMA, core_v1=False)
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        log_retrieval_use(conn, None, runtime="mcp", task_ref="t", outcome="served")

    init_db(conn)
    conn.commit()
    rid = log_retrieval_use(conn, None, runtime="mcp", task_ref="t", outcome="served")
    conn.commit()
    assert conn.execute(
        "SELECT session_id_source FROM retrieval_uses WHERE id=?", (rid,)
    ).fetchone()["session_id_source"] == "none"
