"""Tests for the harness surface: briefing, goals, and the done/attempt ledger.

The three properties under test are the three that make these objects usable by
a loop at all, and each has a mutation test beside it, because a determinism
guard that cannot fail is not a guard:

* the briefing is byte-identical across calls and across client identities;
* goals are selected by scope and status, never by similarity;
* the ledger reports failures as loudly as successes.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from ocbrain.briefing import (
    DEFAULT_BRIEFING_BUDGET_CHARS,
    MAX_BRIEFING_BUDGET_CHARS,
    MAX_LINE_CHARS,
    MIN_BRIEFING_BUDGET_CHARS,
    SECTION_ORDER,
    GoalError,
    build_briefing,
    build_ledger,
    close_goal,
    list_goals,
    open_goal,
)
from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import GOAL_BELIEF_TYPE, init_core_v1
from ocbrain.db import connect
from ocbrain.mcp import handle_request
from ocbrain.scope import ScopeContext

PROJECT = "ocbrain"


@pytest.fixture
def ctx() -> ScopeContext:
    """Build the context inside a test, never at import time.

    ``ScopeContext.__post_init__`` folds the project through the operator
    alias table, and the autouse fixture that isolates operator config has
    not run when a module body executes. A module-level context therefore
    picks up the real ``~/.ocbrain`` aliases and silently becomes a
    different scope than the one the test names.
    """
    return ScopeContext(project=PROJECT)


def _core(tmp_path, name="harness.sqlite"):
    conn = connect(tmp_path / name)
    init_core_v1(conn)
    return conn


def _spec(tmp_path, name="SPEC.md"):
    path = tmp_path / name
    path.write_text("# spec\n", encoding="utf-8")
    return str(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _git_repo_with_tag(root: Path, tag: str) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    spec = root / "SPEC.md"
    spec.write_text(f"# {tag}\n", encoding="utf-8")
    _git(root, "add", "--", "SPEC.md")
    _git(root, "commit", "-q", "-m", tag)
    _git(root, "tag", tag)
    return spec


def _closeout(conn, ctx, task_ref, status, *, verifier=None, summary="did a thing", **kwargs):
    return record_closeout(
        conn,
        task_ref=task_ref,
        status=status,
        summary=summary,
        context=ctx,
        retrieval_use_ids=[],
        decision_impact="informed",
        decision_note=None,
        artifact_refs=[],
        verifier_refs=[{"uri": verifier, "status": "passed"}] if verifier else [],
        actions=[],
        outcomes=[],
        awaiting=kwargs.pop("awaiting", "a human" if status == "blocked" else None),
        # Anything that is not a clean success owes an `unresolved`; the fixture
        # supplies a default so these tests stay about the briefing.
        unresolved=kwargs.pop(
            "unresolved", None if status == "completed" else "the thing did not work"
        ),
        actor="test",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_briefing_is_byte_identical_across_two_calls(tmp_path, ctx):
    conn = _core(tmp_path)
    open_goal(
        conn,
        objective="Ship the harness track",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    _closeout(conn, ctx, "TASK-1", "completed", verifier="repo://ocbrain/pytest")
    _closeout(conn, ctx, "TASK-2", "failed")
    conn.commit()

    first = build_briefing(conn, context=ctx)
    second = build_briefing(conn, context=ctx)
    assert first["text"] == second["text"]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # Nothing in the payload may be a clock reading of *now*: that is the one
    # thing that would make two calls differ without the corpus changing.
    assert "generated_at" not in first


def test_briefing_is_identical_across_client_identities(tmp_path, ctx):
    """One contract, no per-dialect divergence.

    A briefing that differs between Claude Code and Codex is a briefing two
    agents cannot hand to each other, which defeats the point of a shared brain.
    """
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-1", "completed", verifier="repo://ocbrain/pytest")
    conn.commit()

    payloads = []
    for client in ("claude-code", "codex", "cursor", None):
        session_state = {}
        if client is not None:
            handle_request(
                conn,
                {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": client, "version": "0"},
                    },
                },
                session_state=session_state,
            )
        response = handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "brain.briefing",
                    "arguments": {"context": {"project": "ocbrain"}},
                },
            },
            session_state=session_state,
        )
        payloads.append(response["result"]["content"][0]["text"])
    assert len(set(payloads)) == 1


def test_briefing_does_not_rank_and_a_ranked_order_would_be_visible(tmp_path, ctx):
    """Mutation guard on the determinism contract.

    Reversing the documented order must change the bytes. If it does not, the
    determinism test above is passing on a payload whose order nothing controls,
    and it would keep passing after someone swapped in a ranker.
    """
    conn = _core(tmp_path)
    for index in range(3):
        _closeout(
            conn,
            ctx,
            f"TASK-{index}",
            "completed",
            verifier="repo://ocbrain/pytest",
            summary=f"attempt {index}",
        )
    conn.commit()

    baseline = build_briefing(conn, context=ctx)["text"]
    entries = build_ledger(conn, context=ctx)["entries"]
    assert [e["task_ref"] for e in entries] == sorted(
        (e["task_ref"] for e in entries), reverse=True
    ), "ledger order must be the documented rule, not incidental"
    mutated = "\n".join(reversed(baseline.splitlines()))
    assert mutated != baseline


# --------------------------------------------------------------------------- #
# Budget and section order
# --------------------------------------------------------------------------- #


def test_briefing_stays_within_budget_and_counts_what_it_dropped(tmp_path, ctx):
    conn = _core(tmp_path)
    for index in range(40):
        _closeout(
            conn,
            ctx,
            f"TASK-{index:03d}",
            "failed",
            summary="a deliberately long closeout summary " * 6,
        )
    conn.commit()

    payload = build_briefing(conn, context=ctx, budget_chars=400)
    assert len(payload["text"]) <= 400
    assert payload["used_chars"] == len(payload["text"])
    assert payload["truncation"]["truncated"] is True
    assert payload["truncation"]["items_omitted"] > 0
    assert "-- truncated:" in payload["text"]
    # Counted, not silent: the notice and the accounting must agree.
    assert str(payload["truncation"]["items_omitted"]) in payload["text"]


def test_briefing_compacts_a_1000_character_scope_at_the_minimum_budget(tmp_path):
    conn = _core(tmp_path)
    context = ScopeContext(project="x" * 1000)

    first = build_briefing(
        conn,
        context=context,
        budget_chars=MIN_BRIEFING_BUDGET_CHARS,
    )
    second = build_briefing(
        conn,
        context=context,
        budget_chars=MIN_BRIEFING_BUDGET_CHARS,
    )

    assert first == second
    assert first["used_chars"] == len(first["text"]) <= MIN_BRIEFING_BUDGET_CHARS
    assert first["truncation"]["scope_id_truncated"] is True
    assert first["truncation"]["items_omitted"] == 0
    assert first["text"].splitlines()[0].startswith("OCBRAIN BRIEFING · project:")
    assert first["text"].splitlines()[0].endswith("…")
    section_headers = [
        line for line in first["text"].splitlines() if line[:2] in {"A.", "B.", "C.", "D."}
    ]
    assert section_headers == [header for _, header in SECTION_ORDER]


@pytest.mark.parametrize(
    "budget",
    [
        MIN_BRIEFING_BUDGET_CHARS,
        MIN_BRIEFING_BUDGET_CHARS + 1,
        DEFAULT_BRIEFING_BUDGET_CHARS,
        MAX_BRIEFING_BUDGET_CHARS,
    ],
)
@pytest.mark.parametrize("identifier", ["界" * 1000, "🙂\n" * 600])
def test_briefing_budget_holds_for_unicode_identifier_boundaries(tmp_path, budget, identifier):
    conn = _core(tmp_path)
    payload = build_briefing(
        conn,
        context=ScopeContext(task=identifier),
        budget_chars=budget,
    )

    assert payload["used_chars"] == len(payload["text"])
    assert payload["used_chars"] <= payload["budget_chars"] == budget
    assert all(len(line) <= MAX_LINE_CHARS for line in payload["text"].splitlines())
    assert payload["text"].encode("utf-8").decode("utf-8") == payload["text"]


def test_briefing_budget_holds_for_long_scope_and_item_text(tmp_path):
    conn = _core(tmp_path)
    context = ScopeContext(project="scope" * 200)
    _closeout(
        conn,
        context,
        "任务🙂" * 400,
        "failed",
        summary="failure detail " * 400,
    )
    conn.commit()

    for budget in (
        MIN_BRIEFING_BUDGET_CHARS,
        MIN_BRIEFING_BUDGET_CHARS + 1,
        DEFAULT_BRIEFING_BUDGET_CHARS,
        MAX_BRIEFING_BUDGET_CHARS,
    ):
        payload = build_briefing(conn, context=context, budget_chars=budget)
        assert payload["used_chars"] == len(payload["text"]) <= budget
        assert payload["truncation"]["items_omitted"] == sum(
            section["omitted_count"] for section in payload["sections"]
        )
        assert all(len(line) <= MAX_LINE_CHARS for line in payload["text"].splitlines())


@pytest.mark.parametrize(
    "budget",
    [MIN_BRIEFING_BUDGET_CHARS - 1, MAX_BRIEFING_BUDGET_CHARS + 1],
)
def test_briefing_rejects_budgets_just_outside_the_public_boundaries(tmp_path, ctx, budget):
    conn = _core(tmp_path)
    with pytest.raises(ValueError, match="budget_chars"):
        build_briefing(conn, context=ctx, budget_chars=budget)


def test_briefing_line_length_is_bounded_so_one_summary_cannot_eat_the_budget(tmp_path, ctx):
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-HUGE", "failed", summary="x" * 5000)
    conn.commit()
    payload = build_briefing(conn, context=ctx)
    assert all(len(line) <= MAX_LINE_CHARS for line in payload["text"].splitlines())
    assert len(payload["text"]) <= DEFAULT_BRIEFING_BUDGET_CHARS


def test_briefing_rejects_an_unbounded_budget(tmp_path, ctx):
    conn = _core(tmp_path)
    with pytest.raises(ValueError, match="budget_chars"):
        build_briefing(conn, context=ctx, budget_chars=10**9)


def test_section_order_is_stable_and_empty_sections_are_marked(tmp_path, ctx):
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-1", "completed", verifier="repo://ocbrain/pytest")
    conn.commit()

    payload = build_briefing(conn, context=ctx)
    assert [section["id"] for section in payload["sections"]] == [key for key, _ in SECTION_ORDER]
    marks = {"A.", "B.", "C.", "D."}
    headers = [line for line in payload["text"].splitlines() if line[:2] in marks]
    assert headers == [header for _, header in SECTION_ORDER]
    goals_section = next(s for s in payload["sections"] if s["id"] == "goals")
    assert goals_section["present"] is False
    assert goals_section["marker"] == "(none open in this scope)"
    assert "(none open in this scope)" in payload["text"]


def test_empty_scope_says_so_explicitly(tmp_path):
    conn = _core(tmp_path)
    conn.commit()
    payload = build_briefing(conn, context=ScopeContext(project="nothing-here"))
    assert "E. NOTHING KNOWN FOR THIS SCOPE" in payload["text"]
    assert [section["id"] for section in payload["sections"]] == [key for key, _ in SECTION_ORDER]
    assert payload["counts"] == {
        "open_goals": 0,
        "verified_done": 0,
        "attempted_failed": 0,
        "in_flight": 0,
        "gotchas": 0,
    }


# --------------------------------------------------------------------------- #
# Goals
# --------------------------------------------------------------------------- #


def test_goal_open_close_is_an_event_not_an_edit(tmp_path, ctx):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the harness track",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        source_git_ref="abc1234",
        context=ctx,
    )
    conn.commit()
    assert opened["status"] == "open"
    assert list_goals(conn, context=ctx, status="open")

    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    closed = close_goal(
        conn,
        goal_id=opened["goal_id"],
        status="done",
        verifier_uri="repo://ocbrain/pytest",
        verifier_status="passed",
    )
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    assert after > before, "a status transition must append an event"
    assert closed["status"] == "done"
    assert not list_goals(conn, context=ctx, status="open")
    done = list_goals(conn, context=ctx, status="done")
    assert done[0]["verifier"] == {"uri": "repo://ocbrain/pytest", "status": "passed"}
    # The belief itself is untouched by the transition: annotate is metadata-only.
    row = conn.execute(
        "SELECT status, serve FROM current_beliefs WHERE belief_id=?", (opened["goal_id"],)
    ).fetchone()
    assert row["status"] == "current" and row["serve"] == 1


def test_goal_close_requires_naming_the_verifier(tmp_path, ctx):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the harness track",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()
    with pytest.raises(GoalError, match="verifier_uri is required"):
        close_goal(conn, goal_id=opened["goal_id"], status="done", verifier_uri="  ")


def test_goal_done_requires_an_explicit_passing_verifier_without_appending(tmp_path, ctx):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship only after the finish line passes",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]

    with pytest.raises(GoalError, match="verifier_status is required"):
        close_goal(
            conn,
            goal_id=opened["goal_id"],
            status="done",
            verifier_uri="repo://ocbrain/pytest",
        )
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before
    assert [goal["goal_id"] for goal in list_goals(conn, context=ctx, status="open")] == [
        opened["goal_id"]
    ]


@pytest.mark.parametrize("verifier_status", ["failed", "unknown", "not_required"])
def test_goal_done_rejects_nonpassing_verifier_without_appending(
    tmp_path, ctx, verifier_status
):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Keep a failed verification open",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]

    with pytest.raises(GoalError, match="status='done' requires verifier_status='passed'"):
        close_goal(
            conn,
            goal_id=opened["goal_id"],
            status="done",
            verifier_uri="repo://ocbrain/pytest",
            verifier_status=verifier_status,
        )
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before
    assert [goal["goal_id"] for goal in list_goals(conn, context=ctx, status="open")] == [
        opened["goal_id"]
    ]
    assert list_goals(conn, context=ctx, status="done") == []


@pytest.mark.parametrize("verifier_status", ["failed", "unknown", "not_required"])
def test_goal_abandoned_preserves_nonpassing_verifier_state(tmp_path, ctx, verifier_status):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Record an attempt that did not land",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()

    closed = close_goal(
        conn,
        goal_id=opened["goal_id"],
        status="abandoned",
        verifier_uri="repo://ocbrain/pytest",
        verifier_status=verifier_status,
    )
    conn.commit()

    assert closed["verifier"] == {
        "uri": "repo://ocbrain/pytest",
        "status": verifier_status,
    }
    abandoned = list_goals(conn, context=ctx, status="abandoned")
    assert abandoned[0]["verifier"] == closed["verifier"]


def test_goal_requires_a_finish_line_and_a_spec_pointer(tmp_path, ctx):
    conn = _core(tmp_path)
    with pytest.raises(GoalError, match="finish_line is required"):
        open_goal(
            conn,
            objective="Ship it",
            finish_line="",
            source_path=_spec(tmp_path),
            context=ctx,
        )
    with pytest.raises(GoalError, match="source_path is required"):
        open_goal(conn, objective="Ship it", finish_line="pytest -q", source_path="", context=ctx)


def test_goal_needs_a_shared_scope(tmp_path):
    conn = _core(tmp_path)
    with pytest.raises(GoalError, match="shared scope"):
        open_goal(
            conn,
            objective="Ship it",
            finish_line="pytest -q",
            source_path=_spec(tmp_path),
            context=ScopeContext(task="one-off", session="s1"),
        )


def test_missing_source_pointer_surfaces_typed_and_the_goal_does_not_vanish(tmp_path, ctx):
    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the harness track",
        finish_line="pytest -q",
        source_path=str(tmp_path / "gone" / "SPEC.md"),
        context=ctx,
    )
    conn.commit()
    goals = list_goals(conn, context=ctx, status="open")
    assert len(goals) == 1
    assert goals[0]["goal_id"] == opened["goal_id"]
    assert goals[0]["warning"]["type"] == "source_pointer_unresolved"

    payload = build_briefing(conn, context=ctx)
    assert payload["warnings"][0]["type"] == "source_pointer_unresolved"
    assert "source_pointer_unresolved" in payload["text"]


def test_repo_relative_source_pointer_resolves_from_local_repo_context(tmp_path):
    conn = _core(tmp_path)
    repo = tmp_path / "target-repo"
    spec = repo / "docs" / "TARGET-SPEC.md"
    spec.parent.mkdir(parents=True)
    spec.write_text("# target spec\n", encoding="utf-8")
    context = ScopeContext(repo=str(repo))
    opened = open_goal(
        conn,
        objective="Resolve the repository spec",
        finish_line="pytest -q",
        source_path="docs/TARGET-SPEC.md",
        context=context,
    )
    conn.commit()

    goal = list_goals(conn, context=context)[0]
    assert goal["goal_id"] == opened["goal_id"]
    assert "warning" not in goal
    assert build_briefing(conn, context=context)["warnings"] == []


@pytest.mark.parametrize("repo_value", [None, "named-repository", "https://example.test/repo"])
def test_relative_source_pointer_without_a_local_repo_root_stays_unresolved(
    tmp_path, monkeypatch, repo_value
):
    conn = _core(tmp_path)
    spec = tmp_path / "docs" / "TARGET-SPEC.md"
    spec.parent.mkdir()
    spec.write_text("# target spec\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    context = (
        ScopeContext(repo=repo_value)
        if repo_value is not None
        else ScopeContext(project="no-repo-context")
    )
    open_goal(
        conn,
        objective="Do not guess a repository root",
        finish_line="pytest -q",
        source_path="docs/TARGET-SPEC.md",
        context=context,
    )
    conn.commit()

    warning = list_goals(conn, context=context)[0]["warning"]
    assert warning == {"type": "source_pointer_unresolved", "path": "docs/TARGET-SPEC.md"}


def test_git_ref_warning_is_checked_in_the_local_repository(tmp_path, ctx):
    conn = _core(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    spec = repo / "SPEC.md"
    spec.write_text("# spec\n", encoding="utf-8")
    _git(repo, "add", "SPEC.md")
    _git(repo, "commit", "-q", "-m", "base")

    invalid = open_goal(
        conn,
        objective="Reject an unresolvable source revision",
        finish_line="pytest -q",
        source_path=str(spec),
        source_git_ref="definitely-not-a-real-git-ref",
        context=ctx,
    )
    valid = open_goal(
        conn,
        objective="Accept a resolvable source revision",
        finish_line="pytest -q",
        source_path=str(spec),
        source_git_ref="HEAD",
        context=ctx,
    )
    conn.commit()

    goals = {goal["goal_id"]: goal for goal in list_goals(conn, context=ctx)}
    assert goals[invalid["goal_id"]]["warning"] == {
        "type": "source_git_ref_unresolved",
        "path": str(spec),
        "git_ref": "definitely-not-a-real-git-ref",
    }
    assert "warning" not in goals[valid["goal_id"]]
    assert build_briefing(conn, context=ctx)["warnings"] == [
        {"goal_id": invalid["goal_id"], **goals[invalid["goal_id"]]["warning"]}
    ]


def test_git_ref_without_a_local_repository_is_explicitly_unresolved(tmp_path, ctx):
    conn = _core(tmp_path)
    spec = Path(_spec(tmp_path))
    opened = open_goal(
        conn,
        objective="Report a source revision that cannot be checked",
        finish_line="pytest -q",
        source_path=str(spec),
        source_git_ref="definitely-not-a-real-git-ref",
        context=ctx,
    )
    conn.commit()

    warning = list_goals(conn, context=ctx)[0]["warning"]
    assert warning == {
        "type": "source_git_ref_unresolved",
        "path": str(spec),
        "git_ref": "definitely-not-a-real-git-ref",
    }
    assert build_briefing(conn, context=ctx)["warnings"] == [
        {"goal_id": opened["goal_id"], **warning}
    ]


def test_goals_are_selected_by_scope_and_status_never_by_similarity(tmp_path, ctx):
    """The similarity ban, stated as a test.

    A goal must be reachable with no query at all, and must not be reachable
    through the ranked path. Both halves matter: the first is what makes a loop
    able to find its own goals, the second is what stops a goal displacing
    knowledge in a packet that asked for knowledge.
    """
    from ocbrain.core_v1 import search_core_v1

    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the deterministic harness briefing contract",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()

    # Found with no query.
    assert [g["goal_id"] for g in list_goals(conn, context=ctx)] == [opened["goal_id"]]
    # Not found by the ranked path, even when the query quotes the objective.
    packet = search_core_v1(
        conn,
        "Ship the deterministic harness briefing contract",
        context=ctx,
        limit=12,
    )
    assert opened["goal_id"] not in {item["belief_id"] for item in packet["items"]}
    # And the belief really is there to be found -- so the absence above is the
    # exclusion doing its job, not an empty corpus.
    assert conn.execute(
        "SELECT COUNT(*) FROM current_beliefs WHERE belief_id=? AND serve=1",
        (opened["goal_id"],),
    ).fetchone()[0] == 1


def test_mutation_removing_the_similarity_ban_is_caught(tmp_path, ctx, monkeypatch):
    """If the goal exclusion is removed, the ban test above must start failing.

    Proving the guard can fail is the whole point: a filter nobody can break is
    usually a filter that is not running.
    """
    from ocbrain import core_v1

    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the deterministic harness briefing contract",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()

    monkeypatch.setattr(core_v1, "_servable_knowledge_sql", core_v1._delivery_sql)
    packet = core_v1.search_core_v1(
        conn,
        "Ship the deterministic harness briefing contract",
        context=ctx,
        limit=12,
    )
    assert opened["goal_id"] in {item["belief_id"] for item in packet["items"]}


def test_goals_do_not_appear_in_the_digest(tmp_path, ctx):
    from ocbrain.mcp_v1 import digest_v1

    conn = _core(tmp_path)
    opened = open_goal(
        conn,
        objective="Ship the harness track",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()
    payload = digest_v1(conn, context=ctx, limit=12)
    assert opened["goal_id"] not in {item["id"] for item in payload["current"]}


def test_reopening_the_same_goal_converges_instead_of_duplicating(tmp_path, ctx):
    conn = _core(tmp_path)
    spec = _spec(tmp_path)
    first = open_goal(
        conn, objective="Ship it", finish_line="pytest -q", source_path=spec, context=ctx
    )
    conn.commit()
    second = open_goal(
        conn, objective="Ship it", finish_line="pytest -q", source_path=spec, context=ctx
    )
    conn.commit()
    assert second["kind"] == "goal_already_open"
    assert second["goal_id"] == first["goal_id"]
    assert len(list_goals(conn, context=ctx)) == 1


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_ledger_includes_failures_as_first_class_entries(tmp_path, ctx):
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-OK", "completed", verifier="repo://ocbrain/pytest")
    _closeout(conn, ctx, "TASK-BAD", "failed", summary="the import blew up")
    _closeout(conn, ctx, "TASK-STUCK", "blocked", summary="waiting on infra")
    conn.commit()

    ledger = build_ledger(conn, context=ctx)
    states = {entry["task_ref"]: entry["state"] for entry in ledger["entries"]}
    assert states == {
        "TASK-OK": "verified_done",
        "TASK-BAD": "attempted_failed",
        "TASK-STUCK": "attempted_failed",
    }
    bad = next(e for e in ledger["entries"] if e["task_ref"] == "TASK-BAD")
    assert bad["failed_attempt_count"] == 1
    assert bad["failed_attempts"][0]["summary"] == "the import blew up"
    assert ledger["counts"]["attempted_failed"] == 2


def test_a_failure_before_a_success_stays_retrievable(tmp_path, ctx):
    """The ralph double-implementation guard.

    A task that failed twice and then landed must still report the two failures.
    An agent that only sees "done" cannot tell whether the approach it is about
    to take is the one that already did not work.
    """
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-1", "failed", summary="first approach did not work")
    _closeout(conn, ctx, "TASK-1", "failed", summary="second approach did not work either")
    _closeout(conn, ctx, "TASK-1", "completed", verifier="repo://ocbrain/pytest", summary="landed")
    conn.commit()

    entry = build_ledger(conn, context=ctx)["entries"][0]
    assert entry["state"] == "verified_done"
    assert entry["closeout_count"] == 3
    assert entry["failed_attempt_count"] == 2
    assert [a["summary"] for a in entry["failed_attempts"]] == [
        "first approach did not work",
        "second approach did not work either",
    ]


def test_completed_without_a_passing_verifier_is_not_verified_done(tmp_path, ctx):
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-CLAIM", "completed", summary="agent says it is done")
    conn.commit()
    entry = build_ledger(conn, context=ctx)["entries"][0]
    assert entry["state"] == "in_flight"


def test_ledger_reflects_chain_parentage(tmp_path, ctx):
    conn = _core(tmp_path)
    first = _closeout(conn, ctx, "TASK-CHAIN", "partial", summary="part one")
    second = _closeout(
        conn,
        ctx,
        "TASK-CHAIN",
        "completed",
        verifier="repo://ocbrain/pytest",
        summary="part two",
        parent_closeout_id=first["id"],
    )
    conn.commit()

    entry = build_ledger(conn, context=ctx, task_ref="TASK-CHAIN")["entries"][0]
    assert entry["chain_linked"] is True
    assert entry["chain"][1]["id"] == second["id"]
    assert entry["chain"][1]["parent_closeout_id"] == first["id"]
    assert entry["chain"][1]["previous_in_chain"] == first["id"]


def test_ledger_groups_historical_rows_whose_task_ref_norm_is_null(tmp_path, ctx):
    """History is never rewritten, so the fold has to happen at read time.

    On a real core 1161 of 1171 closeouts predate ``task_ref_norm`` and carry
    NULL. A ledger that grouped on the column alone would report zero attempts
    for a task with fourteen -- the exact false negative it exists to prevent.
    """
    conn = _core(tmp_path)
    _closeout(conn, ctx, "COFASC-292", "partial", summary="one")
    _closeout(conn, ctx, "COFASC-292", "partial", summary="two")
    conn.execute("UPDATE current_beliefs SET belief_id=belief_id")  # no-op, keeps the conn warm
    # Simulate pre-column history by clearing the derived key on the stored rows.
    conn.execute("PRAGMA writable_schema=OFF")
    try:
        conn.execute("DROP TRIGGER task_closeouts_no_update")
        conn.execute("UPDATE task_closeouts SET task_ref_norm=NULL")
    finally:
        conn.execute(
            "CREATE TRIGGER IF NOT EXISTS task_closeouts_no_update "
            "BEFORE UPDATE ON task_closeouts BEGIN "
            "SELECT RAISE(ABORT, 'task_closeouts is append-only'); END"
        )
    conn.commit()

    entries = build_ledger(conn, context=ctx)["entries"]
    assert len(entries) == 1
    assert entries[0]["closeout_count"] == 2
    assert entries[0]["task_ref_norm"] == "COFASC-292"


def test_ledger_scopes_to_the_caller_and_a_task_ref_query_ignores_scope(tmp_path, ctx):
    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-MINE", "completed", verifier="repo://ocbrain/pytest")
    record_closeout(
        conn,
        task_ref="TASK-THEIRS",
        status="completed",
        summary="different project",
        context=ScopeContext(project="somewhere-else"),
        retrieval_use_ids=[],
        decision_impact="informed",
        decision_note=None,
        artifact_refs=[],
        verifier_refs=[],
        actions=[],
        outcomes=[],
        awaiting=None,
        actor="test",
    )
    conn.commit()

    scoped = build_ledger(conn, context=ctx)
    assert [e["task_ref"] for e in scoped["entries"]] == ["TASK-MINE"]
    # A named task_ref is an exact lookup, not a scoped browse: an agent asking
    # "has anyone ever tried this" must not be answered "not in your project".
    named = build_ledger(conn, context=ctx, task_ref="TASK-THEIRS")
    assert [e["task_ref"] for e in named["entries"]] == ["TASK-THEIRS"]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_harness_tools_are_reachable_over_mcp(tmp_path):
    conn = _core(tmp_path)
    spec = _spec(tmp_path)

    def call(name, arguments):
        response = handle_request(
            conn,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        assert "error" not in response, response
        return json.loads(response["result"]["content"][0]["text"])

    opened = call(
        "brain.goal_open",
        {
            "objective": "Ship the harness track",
            "finish_line": "pytest -q",
            "source_path": spec,
            "context": {"project": "ocbrain"},
        },
    )
    assert opened["status"] == "open"
    briefing = call("brain.briefing", {"context": {"project": "ocbrain"}})
    assert opened["goal_id"] in briefing["text"]
    ledger = call("brain.ledger", {"context": {"project": "ocbrain"}})
    assert ledger["schema_version"] == "ocbrain.ledger.v1"
    closed = call(
        "brain.goal_close",
        {
            "goal_id": opened["goal_id"],
            "status": "done",
            "verifier_uri": "repo://ocbrain/pytest",
            "verifier_status": "passed",
            "context": {"project": "ocbrain"},
        },
    )
    assert closed["status"] == "done"
    after = call("brain.briefing", {"context": {"project": "ocbrain"}})
    assert opened["goal_id"] not in after["text"]


@pytest.mark.parametrize(
    ("extra_arguments", "expected_message"),
    [
        ({}, "missing argument: verifier_status"),
        (
            {"verifier_status": "failed"},
            "status='done' requires verifier_status='passed'",
        ),
    ],
)
def test_mcp_goal_done_rejects_missing_or_failed_verifier_without_appending(
    tmp_path, extra_arguments, expected_message
):
    conn = _core(tmp_path)
    context = ScopeContext(project="ocbrain")
    opened = open_goal(
        conn,
        objective="Keep MCP goal closure truthful",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=context,
    )
    conn.commit()
    before = conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0]
    arguments = {
        "goal_id": opened["goal_id"],
        "status": "done",
        "verifier_uri": "repo://ocbrain/pytest",
        "context": {"project": "ocbrain"},
        **extra_arguments,
    }

    response = handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.goal_close", "arguments": arguments},
        },
    )

    assert response["error"] == {"code": -32602, "message": expected_message}
    assert conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0] == before
    assert [goal["goal_id"] for goal in list_goals(conn, context=context, status="open")] == [
        opened["goal_id"]
    ]


def test_briefing_refuses_hosted_delivery(tmp_path):
    from ocbrain.mcp import call_tool_v1

    conn = _core(tmp_path)
    with pytest.raises(PermissionError):
        call_tool_v1(
            conn,
            "brain.briefing",
            {"context": {"project": "ocbrain"}},
            profile="runtime",
            delivery_target="hosted_model",
        )


def test_goal_belief_type_is_the_shared_constant():
    assert GOAL_BELIEF_TYPE == "goal"


def test_briefing_does_not_add_a_table(tmp_path, ctx):
    """``CORE_V1_TABLES`` is a closed allow-list; goals ride the belief machinery."""
    from ocbrain.core_v1 import assert_core_v1_inventory

    conn = _core(tmp_path)
    open_goal(
        conn,
        objective="Ship it",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
    )
    conn.commit()
    assert_core_v1_inventory(conn)
    assert isinstance(conn, sqlite3.Connection)


# --------------------------------------------------------------------------- #
# Selftest section E
# --------------------------------------------------------------------------- #


def test_selftest_measures_briefing_determinism_and_goal_hygiene(tmp_path, ctx):
    from datetime import UTC, datetime, timedelta

    from ocbrain.selftest import run_selftest

    conn = _core(tmp_path)
    _closeout(conn, ctx, "TASK-1", "completed", verifier="repo://ocbrain/pytest")
    open_goal(
        conn,
        objective="Resolvable goal",
        finish_line="pytest -q",
        source_path=_spec(tmp_path),
        context=ctx,
        opened_at=(datetime.now(UTC) - timedelta(days=90)).isoformat(),
    )
    open_goal(
        conn,
        objective="Goal whose spec moved",
        finish_line="pytest -q",
        source_path=str(tmp_path / "gone.md"),
        context=ctx,
    )
    conn.commit()

    metrics = {m["key"]: m for m in run_selftest(conn)["metrics"] if m["section"] == "E"}
    assert metrics["briefing_determinism"]["value"] == 1.0
    assert metrics["briefing_determinism"]["status"] == "ok"
    assert metrics["briefing_budget_compliance"]["value"] == 1.0
    assert metrics["goal_pointer_resolution"]["value"] == 0.5
    assert metrics["goal_pointer_resolution"]["status"] == "alarm"
    # A goal open for 90 days is the observable form of goal drift.
    assert metrics["goal_open_age_days"]["value"] > 45
    assert metrics["goal_open_age_days"]["status"] == "alarm"


def test_selftest_section_e_abstains_rather_than_inventing_a_zero(tmp_path):
    from ocbrain.selftest import run_selftest

    conn = _core(tmp_path)
    conn.commit()
    metrics = {m["key"]: m for m in run_selftest(conn)["metrics"] if m["section"] == "E"}
    assert set(metrics) == {
        "briefing_determinism",
        "briefing_budget_compliance",
        "goal_pointer_resolution",
        "goal_open_age_days",
    }
    for metric in metrics.values():
        assert metric["status"] == "not_measured"
        assert metric["reason"]


# --------------------------------------------------------------------------- #
# CLI routes
# --------------------------------------------------------------------------- #


def test_cli_briefing_and_ledger_routes(tmp_path, ctx, capsys):
    from ocbrain.cli import main

    db = tmp_path / "cli.sqlite"
    conn = connect(db)
    init_core_v1(conn)
    _closeout(
        conn,
        ctx,
        "TASK-1",
        "failed",
        summary="tried three import orders",
        unresolved="the trainer import is circular",
    )
    conn.commit()
    conn.close()

    assert main(["--db", str(db), "briefing", "--project", "ocbrain", "--text"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("OCBRAIN BRIEFING")
    # The FAILED line carries what is still broken, not what the session did.
    # Both are in the receipt; only one of them stops the next iteration
    # repeating the afternoon, and the briefing has 1,500 characters.
    assert "the trainer import is circular" in text

    assert main(["--db", str(db), "ledger", "--task-ref", "TASK-1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["entries"][0]["state"] == "attempted_failed"
    assert payload["entries"][0]["latest_unresolved"] == "the trainer import is circular"


def test_cli_repo_root_resolves_a_pointer_without_narrowing_the_ledger(tmp_path, capsys):
    """``--repo-root`` is a filesystem hint; ``--repo`` is part of the scope.

    The SessionStart hook reached for ``--repo`` first, because it was the only
    flag that looked like "where the code is". That silently re-scoped the
    briefing: closeouts recorded without that repo dropped out of the ledger and
    the chain section emptied, trading a cosmetic warning for missing history.
    These two must stay distinguishable from the command line.
    """
    from ocbrain.cli import main

    db = tmp_path / "cli-repo-root.sqlite"
    conn = connect(db)
    init_core_v1(conn)
    context = ScopeContext(project="pointer-scope")
    spec = tmp_path / "docs" / "HARNESS.md"
    spec.parent.mkdir()
    spec.write_text("# harness\n", encoding="utf-8")
    open_goal(
        conn,
        objective="Resolve the pointer from the hook",
        finish_line="pytest -q",
        source_path="docs/HARNESS.md",
        context=context,
    )
    _closeout(conn, context, "TASK-LEDGER", "completed", verifier="repo://ocbrain/pytest")
    conn.commit()
    conn.close()

    base = ["--db", str(db), "briefing", "--project", "pointer-scope", "--text"]

    assert main(base) == 0
    without = capsys.readouterr().out
    assert "source_pointer_unresolved" in without
    assert "TASK-LEDGER" in without

    assert main([*base, "--repo-root", str(tmp_path)]) == 0
    with_root = capsys.readouterr().out
    # The warning is gone because the spec was found, and nothing else moved:
    # the ledger still reports the same task.
    assert "source_pointer_unresolved" not in with_root
    assert "TASK-LEDGER" in with_root
    assert with_root == without.replace(" [source_pointer_unresolved]", "")

    # --repo is the contrast: it joins the retrieval scope, so the closeout
    # recorded without it is no longer this scope's history.
    assert main([*base, "--repo", str(tmp_path)]) == 0
    with_repo = capsys.readouterr().out
    assert "TASK-LEDGER" not in with_repo


@pytest.mark.parametrize(
    ("git_ref", "expect_warning"),
    [
        pytest.param("repo-root-only", True, id="reject-ref-only-in-repo-root"),
        pytest.param("pointer-only", False, id="accept-ref-only-in-pointer-repo"),
    ],
)
def test_cli_absolute_pointer_checks_its_own_repo(
    tmp_path, capsys, git_ref, expect_warning
):
    from ocbrain.cli import main

    repo_root = tmp_path / "repo-root"
    pointer_repo = tmp_path / "pointer-repo"
    _git_repo_with_tag(repo_root, "repo-root-only")
    spec = _git_repo_with_tag(pointer_repo, "pointer-only")

    db = tmp_path / "absolute-pointer.sqlite"
    conn = connect(db)
    init_core_v1(conn)
    context = ScopeContext(project="absolute-pointer-scope")
    opened = open_goal(
        conn,
        objective=f"Validate {git_ref} in the pointer repository",
        finish_line="pytest -q",
        source_path=str(spec),
        source_git_ref=git_ref,
        context=context,
    )
    conn.commit()
    conn.close()

    assert (
        main(
            [
                "--db",
                str(db),
                "briefing",
                "--project",
                "absolute-pointer-scope",
                "--repo-root",
                str(repo_root),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    expected = (
        [
            {
                "goal_id": opened["goal_id"],
                "type": "source_git_ref_unresolved",
                "path": str(spec),
                "git_ref": git_ref,
            }
        ]
        if expect_warning
        else []
    )
    assert payload["warnings"] == expected
