"""Selftest metrics, measured against a fixture whose every number is known.

Each metric here asserts the exact value the seeded corpus must produce, not
merely that the command ran. A scorecard that reports numbers nobody has checked
is the thing this module exists to replace.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ocbrain.briefing import close_goal, open_goal
from ocbrain.cli import main
from ocbrain.core_v1 import append_core_event, init_core_v1
from ocbrain.db import connect
from ocbrain.scope import ScopeContext, ScopeTag
from ocbrain.selftest import (
    ALARM,
    NOT_MEASURED,
    OK,
    THRESHOLDS,
    WATCH,
    SelftestError,
    Threshold,
    _briefing_scopes,
    diff_scorecards,
    exit_code,
    open_readonly,
    render_pretty,
    run_selftest,
)

NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=UTC)

SCOPE = ScopeTag(
    "project",
    "project:bountiful",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)
# A serving global belief, so that "does the caller's scope reach the corpus"
# is a real question. Every caller's compatible set contains global:doctrine, so
# without an occupied global scope the metric cannot tell a correct
# implementation from one that counts global and reports 100% forever.
GLOBAL_SCOPE = ScopeTag(
    "global",
    "global:doctrine",
    visibility="internal",
    egress_policy="local_only",
    provenance="test",
)


def _ts(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _evidence(conn, *, writer: str, days_ago: float) -> str:
    # The body must be unique: evidence ids are content-addressed, so two rows
    # with the same text and scope collapse into one and the harvest stream
    # silently loses its second observation.
    return append_core_event(
        conn,
        "evidence_recorded",
        {
            "body": f"observed by {writer} at {_ts(days_ago)}",
            "kind": "observation",
            "scope": SCOPE.to_dict(),
        },
        writer=writer,
        ts=_ts(days_ago),
        project=True,
    )


def _belief(
    conn,
    *,
    belief_id: str,
    body: str,
    days_ago: float,
    confidence: float = 0.8,
    attributes: dict | None = None,
    scope: ScopeTag = SCOPE,
) -> str:
    proposal = append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": belief_id,
            "belief_type": "curated_fact",
            "body": body,
            "evidence_ids": [],
            "scope": scope.to_dict(),
            "confidence": confidence,
            "attributes": attributes or {},
        },
        writer="test",
        ts=_ts(days_ago),
    )
    append_core_event(
        conn,
        "compilation_decided",
        {"proposal_event_id": proposal, "decision": "approve", "actor": "test"},
        writer="test",
        ts=_ts(days_ago),
        project=True,
    )
    return belief_id


def _correct(
    conn, *, target: str, op: str, days_ago: float, writer: str, successor: str = ""
) -> None:
    body: dict = {
        "schema_version": "ocbrain.correction.v1",
        "target_layer": "belief",
        "target_id": target,
        "subject": {"kind": "belief", "id": target},
        "op": op,
        "body": f"{op} {target}",
        "author": writer,
    }
    if successor:
        body["successor_id"] = successor
    append_core_event(
        conn, "correction_recorded", body, writer=writer, ts=_ts(days_ago), project=True
    )


def _retrieval(
    conn,
    *,
    rid: str,
    days_ago: float,
    project: str | None,
    served: list[str],
    connection_id: str | None = None,
    runtime_key: str | None = None,
) -> None:
    context = {"delivery_target": "local_model"}
    if project:
        context["project"] = project
    conn.execute(
        "INSERT INTO retrieval_uses(id, outcome, query_text, served_ids_json, context_json, "
        "packet_schema, served_at, server_connection_id, client_runtime_key) "
        "VALUES (?, 'served', 'q', ?, ?, 'ocbrain.context.v1', ?, ?, ?)",
        (
            rid,
            json.dumps(served),
            json.dumps(context),
            _ts(days_ago),
            connection_id,
            runtime_key,
        ),
    )


def _closeout(
    conn,
    *,
    cid: str,
    days_ago: float,
    session_id: str | None,
    connection_id: str | None = None,
    context: dict | None = None,
) -> None:
    conn.execute(
        "INSERT INTO task_closeouts(id, schema_version, closed_at, task_ref, status, summary, "
        "decision_impact, context_json, artifact_refs_json, verifier_refs_json, "
        "provenance_json, receipt_json, content_hash, session_id, server_connection_id) "
        "VALUES (?, 'ocbrain.closeout.v1', ?, 'task', 'completed', 's', 'informed', "
        "?, '[]', '[]', '{}', '{}', ?, ?, ?)",
        (cid, _ts(days_ago), json.dumps(context or {}), cid, session_id, connection_id),
    )


@pytest.fixture
def core(tmp_path: Path) -> Path:
    """A corpus whose every measured quantity is arithmetic, not accident.

    Deliberate contents:

    * 6 beliefs. ``keep-a``/``keep-b`` serve and share ``attributes.key`` = one
      duplicate-key cluster. ``doctrine`` serves in ``global:doctrine``, so
      scope reachability has to distinguish a named scope from the global one
      every caller carries. ``rot`` is minted 20d ago and retracted 2d later
      (inside the 14d pollution horizon); ``slow-rot`` is minted 28d ago and
      retracted 23 days later (outside it). ``old`` is superseded by ``keep-b``.
    * 5 retrievals: 3 answered, 2 empty; 4 name a reachable scope, 1 names a
      scope no serving belief occupies.
    * 3 agent corrections, one structured -- adoption is exactly 1/3 -- plus one
      machine correction that must not count.
    * Two harvest streams, one fresh and one silent for 5 days.
    """
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)

    shared = {"key": "shared"}
    _belief(conn, belief_id="belief:keep-a", body="A is true.", days_ago=40, attributes=shared)
    _belief(conn, belief_id="belief:keep-b", body="B is true.", days_ago=40, attributes=shared)
    _belief(conn, belief_id="belief:rot", body="Rot.", days_ago=20, confidence=0.9)
    _belief(conn, belief_id="belief:old", body="Old.", days_ago=40, confidence=0.6)
    _belief(
        conn,
        belief_id="belief:doctrine",
        body="Doctrine holds.",
        days_ago=40,
        scope=GLOBAL_SCOPE,
    )
    # Minted inside the window but removed 23 days later: outside the 14-day
    # horizon, so it is NOT pollution. Without this row the horizon check has
    # nothing to be wrong about.
    _belief(conn, belief_id="belief:slow-rot", body="Slow rot.", days_ago=28, confidence=0.9)

    # Minted 20d ago, gone 18d ago: inside the 14-day pollution horizon.
    _correct(conn, target="belief:rot", op="retract", days_ago=18, writer="agent-one")
    # A structured supersession, and the only conflict in the corpus.
    _correct(
        conn,
        target="belief:old",
        op="supersede",
        days_ago=10,
        writer="agent-two",
        successor="belief:keep-b",
    )
    _correct(conn, target="belief:slow-rot", op="retract", days_ago=5, writer="agent-three")
    # Machine-issued, so it must not count toward agent adoption.
    _correct(
        conn, target="belief:rot", op="retract", days_ago=5, writer="maintenance:belief-hygiene-v1"
    )

    for days in (30, 20, 10):
        _evidence(conn, writer="live-stream", days_ago=days)
    for index in range(3):
        _evidence(conn, writer="stale-stream", days_ago=5 + index * 0.1)
    for days in (2.0, 1.0, 0.5):
        _evidence(conn, writer="live-stream", days_ago=days)

    _retrieval(conn, rid="ret_1", days_ago=1, project="bountiful", served=["belief:keep-a"])
    _retrieval(
        conn,
        rid="ret_2",
        days_ago=2,
        project="bountiful",
        served=["belief:keep-a", "belief:keep-b"],
    )
    _retrieval(conn, rid="ret_3", days_ago=3, project="bountiful", served=["belief:keep-b"])
    _retrieval(conn, rid="ret_4", days_ago=4, project="bountiful", served=[])
    _retrieval(conn, rid="ret_5", days_ago=5, project="nowhere-at-all", served=[])

    _closeout(conn, cid="close_1", days_ago=1, session_id="not-a-transcript")
    conn.commit()
    conn.close()
    return path


def _score(path: Path, **kwargs) -> dict:
    conn = open_readonly(path)
    try:
        return run_selftest(conn, now=NOW, **kwargs)
    finally:
        conn.close()


def _metric(scorecard: dict, key: str) -> dict:
    return next(item for item in scorecard["metrics"] if item["key"] == key)


# --------------------------------------------------------------------------- #
# Section A
# --------------------------------------------------------------------------- #


def test_answer_rate_counts_packets_that_served_something(core: Path) -> None:
    metric = _metric(_score(core), "answer_rate")
    assert metric["value"] == pytest.approx(3 / 5)
    assert metric["detail"]["retrievals"] == 5
    assert metric["detail"]["answered"] == 3
    assert metric["status"] == OK


def test_answer_rate_splits_by_whether_the_scope_could_reach_the_corpus(core: Path) -> None:
    detail = _metric(_score(core), "answer_rate")["detail"]
    assert detail["reachable_scope"] == {"retrievals": 4, "answered": 3, "rate": 0.75}
    assert detail["unreachable_scope"] == {"retrievals": 1, "answered": 0, "rate": 0.0}


def test_scope_reachability_excludes_the_always_present_global_scope(core: Path) -> None:
    metric = _metric(_score(core), "scope_reachability")
    # 4 of 5 name project:bountiful, which serving beliefs occupy; one names a
    # scope nothing occupies. Were global:doctrine counted this would be 100%.
    assert metric["value"] == pytest.approx(0.8)
    assert metric["detail"]["reachable"] == 4
    assert metric["detail"]["unreachable"] == 1
    assert metric["detail"]["serving_scopes"] == ["global:doctrine", "project:bountiful"]


def test_zero_result_census_names_the_projects_that_came_back_empty(core: Path) -> None:
    metric = _metric(_score(core), "zero_result_rate")
    assert metric["value"] == pytest.approx(2 / 5)
    assert metric["detail"]["zero_result_count"] == 2
    assert metric["detail"]["top_passed_projects"] == [
        {"project": "bountiful", "count": 1},
        {"project": "nowhere-at-all", "count": 1},
    ]


# --------------------------------------------------------------------------- #
# Section B
# --------------------------------------------------------------------------- #


def test_pollution_rate_counts_only_beliefs_removed_inside_the_horizon(core: Path) -> None:
    metric = _metric(_score(core), "pollution_rate")
    # Two beliefs minted in the 30d window. rot (20d ago) was removed 2 days
    # later and counts; slow-rot (28d ago) was removed 23 days later and must
    # not, because it outlived the horizon.
    assert metric["detail"]["minted_in_window"] == 2
    assert metric["detail"]["removed_within_horizon"] == 1
    assert metric["value"] == pytest.approx(0.5)


def test_structured_removal_share_sees_a_bare_retract_as_unstructured(core: Path) -> None:
    metric = _metric(_score(core), "structured_removal_share")
    assert metric["value"] == pytest.approx(0.0)
    assert metric["detail"] == {"structured": 0, "removals": 1}
    assert metric["status"] == ALARM


def test_conflict_preservation_requires_a_validity_window_on_the_loser(core: Path) -> None:
    metric = _metric(_score(core), "conflict_preservation")
    assert metric["value"] == pytest.approx(1.0)
    assert metric["detail"]["conflicts"] == 1
    assert metric["detail"]["preserved"] == 1
    assert metric["status"] == OK


def test_conflict_preservation_alarms_when_the_losing_side_is_unreachable(
    core: Path, tmp_path: Path
) -> None:
    """A supersession whose loser lost its era stamp is exactly what must alarm."""
    conn = connect(core)
    conn.execute(
        "UPDATE current_beliefs SET attributes_json='{}' WHERE belief_id='belief:old'"
    )
    conn.commit()
    conn.close()
    metric = _metric(_score(core), "conflict_preservation")
    assert metric["value"] == pytest.approx(0.0)
    assert metric["status"] == ALARM
    assert metric["detail"]["unreachable_sample"][0]["belief_id"] == "belief:old"


def test_calibration_reports_the_widest_band_and_why_beliefs_left(core: Path) -> None:
    metric = _metric(_score(core), "calibration_gap")
    bands = {entry["band"]: entry for entry in metric["detail"]["bands"]}
    # belief:old (0.6) is moderate and was superseded within the horizon.
    assert bands["moderate"]["beliefs"] == 1
    assert bands["moderate"]["survival_rate"] == pytest.approx(0.0)
    assert bands["moderate"]["removed_by"] == {"supersede": 1}
    # keep-a, keep-b and doctrine are strong and survived; the cohort excludes
    # rot and slow-rot, neither of which has had a full 30-day horizon.
    assert bands["strong"]["beliefs"] == 3
    assert bands["strong"]["survival_rate"] == pytest.approx(1.0)


def test_duplicate_key_clusters_counts_shared_attribute_keys(core: Path) -> None:
    metric = _metric(_score(core), "duplicate_key_clusters")
    assert metric["value"] == 1.0
    assert metric["detail"]["clusters"] == [
        {"key": "shared", "belief_ids": ["belief:keep-a", "belief:keep-b"]}
    ]
    assert metric["status"] == WATCH


def test_near_duplicate_clusters_is_not_measured_without_a_sidecar(core: Path) -> None:
    metric = _metric(_score(core), "near_duplicate_clusters")
    assert metric["status"] == NOT_MEASURED
    assert "no vector sidecar" in metric["reason"]
    assert metric["value"] is None


def test_near_duplicate_clusters_reads_a_sidecar_when_one_exists(core: Path) -> None:
    from ocbrain.vector import encode_embedding

    sidecar = sqlite3.connect(core.with_name(f"{core.stem}-vectors.sqlite"))
    sidecar.executescript(
        "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE belief_vectors(belief_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL,"
        " model TEXT NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL,"
        " scope_type TEXT NOT NULL, scope_id TEXT NOT NULL, visibility TEXT NOT NULL,"
        " egress_policy TEXT NOT NULL, last_compiled_at TEXT NOT NULL);"
    )
    sidecar.execute(
        "INSERT INTO meta VALUES ('schema_version', 'ocbrain.vectors.v2')",
    )
    for belief_id, vector in (
        ("belief:keep-a", [1.0, 0.0, 0.0]),
        ("belief:keep-b", [0.97, 0.24, 0.0]),
    ):
        sidecar.execute(
            "INSERT INTO belief_vectors VALUES (?, 'h', 'm', 3, ?, 'project', 'project:bountiful',"
            " 'internal', 'local_only', 'now')",
            (belief_id, encode_embedding(vector)),
        )
    sidecar.commit()
    sidecar.close()

    metric = _metric(_score(core), "near_duplicate_clusters")
    # cosine(keep-a, keep-b) = 0.97, above the 0.88 corpus threshold.
    assert metric["value"] == 1.0
    assert metric["detail"]["clusters"] == [{"belief_ids": ["belief:keep-a", "belief:keep-b"]}]
    # keep-a, keep-b and doctrine serve; only two are embedded.
    assert metric["detail"]["embedding_coverage"] == 0.6667


# --------------------------------------------------------------------------- #
# Section C
# --------------------------------------------------------------------------- #


def test_correction_adoption_ignores_machine_writers(core: Path) -> None:
    metric = _metric(_score(core), "correction_adoption")
    # agent-one retracted, agent-two superseded; the hygiene retraction is not an
    # agent correction and must not dilute the denominator.
    assert metric["detail"]["agent_corrections"] == 3
    assert metric["detail"]["machine_corrections"] == 1
    assert metric["value"] == pytest.approx(1 / 3)
    assert metric["detail"]["shapes"] == {"retract": 2, "supersede": 1}


def test_pending_queue_is_empty_when_no_proposal_awaits_a_decision(core: Path) -> None:
    scorecard = _score(core)
    assert _metric(scorecard, "pending_supersede_depth")["value"] == 0.0
    age = _metric(scorecard, "pending_supersede_age_hours")
    assert age["status"] == NOT_MEASURED
    assert "empty" in age["reason"]


def test_pending_queue_measures_the_age_of_the_oldest_undecided_supersede(core: Path) -> None:
    conn = connect(core)
    append_core_event(
        conn,
        "compilation_proposed",
        {
            "belief_id": "belief:successor",
            "body": "Replacement.",
            "evidence_ids": [],
            "scope": SCOPE.to_dict(),
            "confidence": 0.7,
            "attributes": {"supersedes": "belief:keep-a"},
        },
        writer="agent-two",
        ts=_ts(2),
    )
    conn.commit()
    conn.close()
    scorecard = _score(core)
    assert _metric(scorecard, "pending_supersede_depth")["value"] == 1.0
    age = _metric(scorecard, "pending_supersede_age_hours")
    assert age["value"] == pytest.approx(48.0)
    assert age["status"] == OK


def test_pending_queue_headlines_distinct_targets_not_raw_proposal_count(core: Path) -> None:
    """Raw depth alone hid an unbounded proposal loop behind a plausible number.

    Three proposals against two beliefs: an operator has two decisions to make,
    and needs to see that the queue is duplicated rather than deep.
    """
    conn = connect(core)
    for index, (target, body) in enumerate(
        [
            ("belief:keep-a", "Replacement one."),
            ("belief:keep-a", "Replacement one."),
            ("belief:keep-b", "Replacement two."),
        ]
    ):
        append_core_event(
            conn,
            "compilation_proposed",
            {
                "belief_id": f"belief:successor-{index}",
                "body": body,
                "evidence_ids": [],
                "scope": SCOPE.to_dict(),
                "confidence": 0.7,
                "attributes": {"supersedes": target},
            },
            writer="agent-two",
            ts=_ts(2),
        )
    conn.commit()
    conn.close()

    depth = _metric(_score(core), "pending_supersede_depth")
    assert depth["value"] == 2.0
    assert depth["display"] == "2 distinct (3 proposals)"
    assert depth["detail"]["distinct_targets"] == 2
    assert depth["detail"]["proposals"] == 3
    # The age metric still measures the whole queue, unchanged.
    assert _metric(_score(core), "pending_supersede_age_hours")["detail"]["pending"] == 3


def test_contradictions_rate_finds_the_duplicate_key_pair_in_a_served_packet(core: Path) -> None:
    metric = _metric(_score(core), "contradictions_nonempty_rate")
    # Only ret_2 served both halves of the shared-key pair.
    assert metric["detail"]["packets"] == 5
    assert metric["detail"]["packets_with_contradictions"] == 1
    assert metric["value"] == pytest.approx(0.2)
    assert metric["detail"]["by_reason"] == {"duplicate_key": 1}
    assert metric["basis"].startswith("RECONSTRUCTED")


# --------------------------------------------------------------------------- #
# Section D
# --------------------------------------------------------------------------- #


def test_provenance_is_not_measured_before_capture_has_ever_happened(core: Path) -> None:
    metric = _metric(_score(core), "provenance_coverage")
    assert metric["status"] == NOT_MEASURED
    assert "has ever carried a server_connection_id" in metric["reason"]


def test_provenance_measures_only_rows_written_since_capture_began(core: Path) -> None:
    """Rows that predate capture cannot dilute the rate; that is the whole point."""
    conn = connect(core)
    for index in range(20):
        _retrieval(
            conn,
            rid=f"ret_p{index}",
            days_ago=0.5,
            project="bountiful",
            served=["belief:keep-a"],
            connection_id=f"conn-{index}" if index < 15 else None,
            runtime_key="claude-code",
        )
    conn.commit()
    conn.close()
    metric = _metric(_score(core), "provenance_coverage")
    assert metric["detail"]["rows_since_capture"] == 20
    assert metric["detail"]["covered"] == 15
    assert metric["value"] == pytest.approx(0.75)
    assert metric["detail"]["by_client_runtime_key"]["claude-code"]["retrievals"] == 20
    assert metric["status"] == WATCH


def test_harvest_alarms_on_a_live_stream_that_went_quiet(core: Path) -> None:
    metric = _metric(_score(core), "harvest_silence_hours")
    # stale-stream wrote 3 rows within the 7-day liveness window and has been
    # silent 5 days; live-stream wrote 12 hours ago.
    assert metric["detail"]["live_streams"] == 2
    assert metric["value"] == pytest.approx(120.0)
    assert metric["status"] == ALARM
    assert [item["runtime"] for item in metric["detail"]["silent_streams"]] == ["stale-stream"]


def test_harvest_ignores_one_shot_runtime_labels(core: Path) -> None:
    """Ninety historical lane labels must not make this row permanently red."""
    conn = connect(core)
    _evidence(conn, writer="one-shot-lane", days_ago=6)
    conn.commit()
    conn.close()
    metric = _metric(_score(core), "harvest_silence_hours")
    assert metric["detail"]["live_streams"] == 2
    assert "one-shot-lane" not in {item["runtime"] for item in metric["detail"]["streams"]}


def test_growth_reports_rows_by_table(core: Path) -> None:
    metric = _metric(_score(core), "rows_added_in_window")
    tables = metric["detail"]["tables"]
    assert tables["current_beliefs"]["added_30d"] == 2
    assert tables["retrieval_uses"]["added_30d"] == 5
    assert tables["task_closeouts"]["added_30d"] == 1
    assert metric["status"] == OK


def test_integrity_passes_on_a_sound_core(core: Path) -> None:
    metric = _metric(_score(core), "integrity")
    assert metric["value"] == 1.0
    assert metric["detail"]["foreign_key_violations"] == 0


def test_sidecar_freshness_is_not_measured_when_absent(core: Path) -> None:
    metric = _metric(_score(core), "vector_sidecar_lag_events")
    assert metric["status"] == NOT_MEASURED
    assert "no vector sidecar" in metric["reason"]


def test_closeout_join_is_not_measured_without_a_transcript_root(
    core: Path, tmp_path: Path
) -> None:
    metric = _metric(_score(core, transcript_root=tmp_path / "absent"), "closeout_trace_join_rate")
    assert metric["status"] == NOT_MEASURED
    assert "no transcripts found" in metric["reason"]


# --------------------------------------------------------------------------- #
# Section E
# --------------------------------------------------------------------------- #


def test_briefing_scopes_do_not_let_closed_goals_crowd_out_active_work(tmp_path: Path) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    spec = tmp_path / "SPEC.md"
    spec.write_text("# acceptance\n", encoding="utf-8")

    for index in range(5):
        opened = open_goal(
            conn,
            objective=f"Historical goal {index}",
            finish_line="pytest -q tests/test_selftest.py",
            source_path=str(spec),
            context=ScopeContext(project=f"a-closed-{index}"),
        )
        close_goal(
            conn,
            goal_id=opened["goal_id"],
            status="done",
            verifier_uri="repo://selftest/passed",
            verifier_status="passed",
        )
    open_goal(
        conn,
        objective="The sole active sampled goal",
        finish_line="pytest -q tests/test_selftest.py",
        source_path=str(spec),
        context=ScopeContext(project="zz-active"),
    )
    conn.commit()

    assert _briefing_scopes(conn, limit=5) == [
        "zz-active",
        "a-closed-0",
        "a-closed-1",
        "a-closed-2",
        "a-closed-3",
    ]
    conn.close()


def test_briefing_scopes_order_and_deduplicate_project_repo_and_task_contexts(
    tmp_path: Path,
) -> None:
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    spec = tmp_path / "SPEC.md"
    spec.write_text("# acceptance\n", encoding="utf-8")
    repo = tmp_path / "active-repo"
    repo.mkdir()

    for name, context in (
        ("Beta active", ScopeContext(project="beta-active")),
        ("Alpha active", ScopeContext(project="alpha-active")),
        ("Repo active", ScopeContext(repo=str(repo))),
    ):
        open_goal(
            conn,
            objective=name,
            finish_line="pytest -q tests/test_selftest.py",
            source_path=str(spec),
            context=context,
        )
    closed = open_goal(
        conn,
        objective="Closed-only fallback",
        finish_line="pytest -q tests/test_selftest.py",
        source_path=str(spec),
        context=ScopeContext(project="closed-only"),
    )
    close_goal(
        conn,
        goal_id=closed["goal_id"],
        status="done",
        verifier_uri="repo://selftest/passed",
        verifier_status="passed",
    )

    for index in range(4):
        _closeout(
            conn,
            cid=f"close_active_{index}",
            days_ago=1,
            session_id=None,
            context={"project": "alpha-active", "task": "must-not-split-the-scope"},
        )
    for index in range(3):
        _closeout(
            conn,
            cid=f"close_task_{index}",
            days_ago=1,
            session_id=None,
            context={"task": "task-only"},
        )
    for index in range(2):
        _closeout(
            conn,
            cid=f"close_history_{index}",
            days_ago=1,
            session_id=None,
            context={"project": "historical-project"},
        )
    conn.commit()

    expected = [
        "alpha-active",
        "beta-active",
        f"repo:{repo}",
        "task:task-only",
        "historical-project",
        "closed-only",
    ]
    assert _briefing_scopes(conn, limit=6) == expected
    assert _briefing_scopes(conn, limit=6) == expected
    assert len(expected) == len(set(expected))
    scorecard = run_selftest(conn, now=NOW)
    assert _metric(scorecard, "briefing_determinism")["detail"]["scopes"] == 5
    assert _metric(scorecard, "goal_pointer_resolution")["detail"]["open_goals"] == 3
    conn.close()


# --------------------------------------------------------------------------- #
# Read-only enforcement
# --------------------------------------------------------------------------- #


def test_the_selftest_connection_cannot_write(core: Path) -> None:
    conn = open_readonly(core)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
            conn.execute("DELETE FROM current_beliefs")
    finally:
        conn.close()


def test_open_readonly_refuses_a_missing_core(tmp_path: Path) -> None:
    with pytest.raises(SelftestError, match="does not exist"):
        open_readonly(tmp_path / "nope.sqlite")


def test_running_the_scorecard_leaves_the_core_byte_identical(core: Path) -> None:
    before = core.read_bytes()
    _score(core)
    assert core.read_bytes() == before


# --------------------------------------------------------------------------- #
# Verdicts, exit codes, rendering, diffing
# --------------------------------------------------------------------------- #


def test_thresholds_all_carry_a_source() -> None:
    """A threshold nobody can trace is a threshold people learn to mute."""
    for key, threshold in THRESHOLDS.items():
        assert threshold.source.strip(), f"{key} has no documented source"
        assert len(threshold.source) > 40, f"{key}'s source is too thin to be provenance"


def test_every_metric_has_a_threshold_entry(core: Path) -> None:
    for metric in _score(core)["metrics"]:
        assert metric.get("threshold") is not None, f"{metric['key']} has no threshold"


def test_exit_code_is_non_zero_when_anything_alarms(core: Path) -> None:
    scorecard = _score(core)
    assert scorecard["tally"][ALARM] > 0
    assert scorecard["status"] == ALARM
    assert exit_code(scorecard) == 1


def test_exit_code_is_zero_when_nothing_alarms() -> None:
    clean = {"tally": {OK: 4, WATCH: 1, ALARM: 0, NOT_MEASURED: 2}}
    assert exit_code(clean) == 0


def test_watch_alone_never_fails_the_gate() -> None:
    """The middle band exists so the gate does not fire before a human looks."""
    assert exit_code({"tally": {OK: 0, WATCH: 9, ALARM: 0, NOT_MEASURED: 0}}) == 0


def test_not_measured_alone_never_fails_the_gate() -> None:
    assert exit_code({"tally": {OK: 0, WATCH: 0, ALARM: 0, NOT_MEASURED: 9}}) == 0


@pytest.mark.parametrize(
    ("direction", "ok", "watch", "value", "expected"),
    [
        ("higher_better", 0.8, 0.5, 0.9, OK),
        ("higher_better", 0.8, 0.5, 0.6, WATCH),
        ("higher_better", 0.8, 0.5, 0.4, ALARM),
        ("lower_better", 0.1, 0.3, 0.05, OK),
        ("lower_better", 0.1, 0.3, 0.2, WATCH),
        ("lower_better", 0.1, 0.3, 0.9, ALARM),
        ("info", None, None, 999.0, OK),
        ("higher_better", 0.8, 0.5, None, NOT_MEASURED),
    ],
)
def test_threshold_classification(direction, ok, watch, value, expected) -> None:
    assert Threshold(direction, ok, watch, "test source").classify(value) == expected


def test_pretty_output_marks_alarms_and_explains_them(core: Path) -> None:
    rendered = render_pretty(_score(core))
    assert "OCBrain selftest" in rendered
    assert "verdict ALARM" in rendered
    assert "!!" in rendered
    assert "FLAGGED" in rendered
    assert "NOT MEASURED" in rendered
    # Every section renders even when a section is entirely healthy.
    for heading in ("RETRIEVAL HEALTH", "CORPUS QUALITY", "CORRECTION PATHWAY", "PLUMBING"):
        assert heading in rendered


def test_diff_reports_movement_and_flags_regressions() -> None:
    baseline = {
        "generated_at": "2026-08-01T00:00:00+00:00",
        "metrics": [
            {"key": "answer_rate", "label": "Answer rate", "value": 0.9, "status": OK},
            {"key": "gone", "label": "Gone", "value": 1.0, "status": OK},
        ],
    }
    current = {
        "metrics": [
            {"key": "answer_rate", "label": "Answer rate", "value": 0.3, "status": ALARM},
            {"key": "fresh", "label": "Fresh", "value": 2.0, "status": OK},
        ]
    }
    diff = diff_scorecards(baseline, current)
    assert diff["baseline_generated_at"] == "2026-08-01T00:00:00+00:00"
    assert diff["added"] == ["fresh"]
    assert diff["removed"] == ["gone"]
    assert diff["regressions"] == 1
    change = diff["changes"][0]
    assert change["key"] == "answer_rate"
    assert change["delta"] == pytest.approx(-0.6)
    assert change["regressed"] is True


def test_diff_stays_silent_when_nothing_moved() -> None:
    card = {"metrics": [{"key": "a", "label": "A", "value": 1.0, "status": OK}]}
    assert diff_scorecards(card, card)["changes"] == []


def test_scorecard_reports_its_own_wall_clock(core: Path) -> None:
    scorecard = _score(core)
    assert scorecard["elapsed_seconds"] >= 0.0
    assert scorecard["schema_version"] == "ocbrain.selftest.v1"
    assert scorecard["window"]["since_days"] == 30


def test_since_days_moves_the_window(core: Path) -> None:
    narrow = _metric(_score(core, since_days=3), "answer_rate")
    assert narrow["detail"]["retrievals"] == 3


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_emits_json_and_propagates_the_alarm_exit_code(core: Path, capsys) -> None:
    code = main(["--db", str(core), "selftest"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "ocbrain.selftest.v1"
    assert code == 1


def test_cli_pretty_flag_works_after_the_subcommand(core: Path, capsys) -> None:
    main(["--db", str(core), "selftest", "--pretty"])
    assert "OCBrain selftest" in capsys.readouterr().out


def test_cli_pretty_flag_still_works_before_the_subcommand(core: Path, capsys) -> None:
    """The global --pretty must survive the subparser rather than be reset."""
    main(["--pretty", "--db", str(core), "selftest"])
    assert "OCBrain selftest" in capsys.readouterr().out


def test_cli_writes_and_then_diffs_a_saved_scorecard(core: Path, tmp_path: Path, capsys) -> None:
    saved = tmp_path / "baseline.json"
    main(["--db", str(core), "selftest", "--out", str(saved)])
    capsys.readouterr()
    assert json.loads(saved.read_text(encoding="utf-8"))["schema_version"] == "ocbrain.selftest.v1"

    main(["--db", str(core), "selftest", "--baseline", str(saved)])
    payload = json.loads(capsys.readouterr().out)
    assert payload["baseline_diff"]["regressions"] == 0
    assert payload["baseline_diff"]["added"] == []


def test_cli_refuses_a_missing_core(tmp_path: Path, capsys) -> None:
    assert main(["--db", str(tmp_path / "absent.sqlite"), "selftest"]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_refuses_an_unreadable_baseline(core: Path, tmp_path: Path, capsys) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert main(["--db", str(core), "selftest", "--baseline", str(broken)]) == 2
    assert "cannot read baseline" in capsys.readouterr().err


def test_lossy_supersession_flags_a_machine_rewording_that_drops_checkable_tokens(core):
    """The 2026-08-26 finding: curator rewordings frequently destroy specifics."""
    conn = connect(core)
    _belief(
        conn,
        belief_id="belief_lossy_old",
        body="PR #3495 fixed the anyOf-null break; verify with `wiki-lint` and the A/A pass.",
        days_ago=10,
    )
    _belief(
        conn,
        belief_id="belief_lossy_new",
        body="A schema break was repaired across all client integrations.",
        days_ago=2,
    )
    _correct(
        conn,
        target="belief_lossy_old",
        op="supersede",
        days_ago=2,
        writer="operator-approved:wiki-curator-v2",
        successor="belief_lossy_new",
    )
    conn.commit()

    metric = _metric(_score(core), "lossy_supersession_share")
    assert metric["value"] == 1.0
    assert metric["status"] == "alarm"
    assert metric["detail"]["sample_lossy_targets"] == ["belief_lossy_old"]
    assert (
        metric["detail"]["by_writer"]["operator-approved:wiki-curator-v2"]["lossy"] == 1
    )
    conn.close()


def test_lossy_supersession_passes_a_rewording_that_keeps_every_token(core):
    conn = connect(core)
    _belief(
        conn,
        belief_id="belief_kept_old",
        body="PR #3495 fixed the anyOf-null break; verify with `wiki-lint`.",
        days_ago=10,
    )
    _belief(
        conn,
        belief_id="belief_kept_new",
        body="The anyOf-null break was fixed by PR #3495 -- `wiki-lint` verifies it.",
        days_ago=2,
    )
    _correct(
        conn,
        target="belief_kept_old",
        op="supersede",
        days_ago=2,
        writer="operator-approved:wiki-curator-v2",
        successor="belief_kept_new",
    )
    conn.commit()

    metric = _metric(_score(core), "lossy_supersession_share")
    assert metric["value"] == 0.0
    assert metric["status"] == "ok"
    conn.close()


def test_lossy_supersession_excludes_agent_corrections(core):
    """An agent correction is SUPPOSED to drop the tokens of the fact it refutes."""
    conn = connect(core)
    _belief(conn, belief_id="belief_agent_old", body="The live VM is asa1 (#100).", days_ago=10)
    _belief(conn, belief_id="belief_agent_new", body="The live VM is asa2.", days_ago=2)
    _correct(
        conn,
        target="belief_agent_old",
        op="supersede",
        days_ago=2,
        writer="hermes:fixture",
        successor="belief_agent_new",
    )
    conn.commit()

    metric = _metric(_score(core), "lossy_supersession_share")
    assert metric["status"] == "not_measured"
    # Mine plus the agent pair the shared core fixture seeds; the point is that
    # neither reaches the machine population, so the metric stays unmeasured.
    assert metric["detail"]["agent_supersessions"] >= 1
    conn.close()


def test_lossy_supersession_is_unmeasured_on_a_quiet_core(core):
    metric = _metric(_score(core), "lossy_supersession_share")
    assert metric["status"] == "not_measured"
