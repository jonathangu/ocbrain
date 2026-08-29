"""Tests for the procedural-memory miner.

Four properties matter enough to pin: a signature must be stable across
irrelevant argument changes, a signature must never carry a secret, the grade
ladder must be ordered, and the miner must abstain rather than emit a procedure
it cannot support.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from procmine.adapters import hermes as hermes_adapter  # noqa: E402
from procmine.dag import (  # noqa: E402
    MIN_FAMILY_EPISODES,
    Family,
    collapse_runs,
    first_occurrence_order,
    is_subsequence,
    mine_family,
    mine_repairs,
)
from procmine.episodes import (  # noqa: E402
    GRADE_ORDER,
    GRADE_RANK,
    Episode,
    grade_closeout,
    normalize_runtime,
)
from procmine.normalize import (  # noqa: E402
    arg_signature,
    classify_path,
    error_fingerprint,
    result_class,
    shell_signature,
    step_class,
)

HOME = str(Path.home())


# --- arg signature stability ----------------------------------------------


def test_shell_signature_is_stable_across_different_paths():
    """Two runs of the same command on different files share a signature."""
    left = shell_signature(f"rg -n needle {HOME}/coframe/a/one.py")
    right = shell_signature(f"rg -n haystack {HOME}/coframe/b/two.py")
    assert left == right
    assert "one.py" not in left


def test_shell_signature_separates_different_verbs():
    assert shell_signature("git status") != shell_signature("git commit")


def test_shell_signature_peels_env_wrapper():
    """`env -u VAR python3 x.py` is a python run, not a run of "-u"."""
    assert shell_signature("env -u PYTHONPATH python3 script.py").startswith("bash:python3")


def test_shell_signature_peels_runner_and_its_valued_flags():
    """`uv run --python 3.13 pytest` is a pytest run."""
    signature = shell_signature("uv run --python 3.13 pytest -q tests/")
    assert signature.startswith("bash:pytest")
    assert "3.13" not in signature


def test_shell_signature_promotes_python_dash_m():
    assert shell_signature("python -m pytest -q tests/").startswith("bash:python -m pytest")


def test_shell_signature_marks_chains_without_expanding_them():
    signature = shell_signature("cd /tmp && ls && grep -r x . | head -5")
    assert signature.startswith("bash:cd")
    assert "<chain>" in signature


def test_arg_signature_handles_json_string_arguments():
    """Hermes stores arguments as a JSON string; codex sometimes as a dict."""
    as_string = arg_signature("terminal", '{"command": "git status"}')
    as_dict = arg_signature("Bash", {"command": "git status"})
    assert as_string == as_dict == "bash:git status"


def test_arg_signature_names_mcp_server_and_tool():
    assert arg_signature("mcp__ocbrain__brain_closeout", {}) == "mcp:ocbrain.brain_closeout"


def test_arg_signature_keeps_file_class_not_file_name():
    signature = arg_signature("Read", {"file_path": f"{HOME}/coframe/deep/secret_plan.py"})
    assert signature == "read:<path:repo:coframe:py>"
    assert "secret_plan" not in signature


def test_classify_path_orders_specific_before_generic():
    assert classify_path(f"{HOME}/HermesWork/x") == "<path:hermeswork>"
    assert classify_path(f"{HOME}/other/x") == "<path:home>"
    assert classify_path("relative/path.txt") == "<path:rel:txt>"
    assert classify_path("https://example.com/x") == "<url:https>"


# --- redaction -------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "curl -H 'Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345' https://api.example.com",
        "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        "gh auth login --with-token ghp_abcdefghijklmnopqrstuvwxyz01",
        'psql "password=hunter2hunter2hunter2"',
    ],
)
def test_shell_signature_never_emits_a_secret(command):
    signature = shell_signature(command)
    for leak in ("abcdefghijklmnopqrstuvwxyz", "hunter2"):
        assert leak not in signature


def test_error_fingerprint_redacts_and_strips_ids():
    text = "Error: token=sk-abcdefghijklmnopqrstuvwxyz rejected for run 1234567\nsecond line"
    fingerprint = error_fingerprint(text)
    assert fingerprint is not None
    assert "abcdefghijklmnopqrstuvwxyz" not in fingerprint
    assert "1234567" not in fingerprint
    assert "second line" not in fingerprint


def test_error_fingerprint_skips_codex_boilerplate():
    """Codex prefixes every result; fingerprinting the preamble groups nothing."""
    text = "Script completed\nWall time 7.2 seconds\nOutput:\nfatal: not a git repository"
    assert error_fingerprint(text) == "fatal: not a git repository"


def test_error_fingerprint_is_bounded():
    assert len(error_fingerprint("error: " + "x" * 5000) or "") <= 160


# --- result classes --------------------------------------------------------


def test_explicit_signal_beats_text_sniffing():
    """A successful find whose output mentions a denial is not a refusal."""
    assert result_class(exit_code=0, text="./a\nfind: /x: Permission denied") == "ok"
    assert result_class(is_error=True, text="Permission denied") == "refused"
    assert result_class(is_error=True, text="the request timed out") == "timeout"
    assert result_class(is_error=True, text="boom") == "error"


def test_text_sniffing_only_applies_without_a_signal():
    assert result_class(text="fatal: not a git repository") == "error"
    assert result_class(text="all good") == "ok"
    assert result_class(text="") == "empty"
    assert result_class(exit_code=0, text="") == "empty"


def test_error_marker_deep_in_a_long_success_is_ignored():
    body = "ok\n" * 400 + "error: something buried far below"
    assert result_class(text=body) == "ok"


# --- abstract step classes -------------------------------------------------


def test_step_class_merges_equivalent_steps_across_runtimes():
    """A hermes read and a claude read must land on the same abstract node."""
    assert step_class("toolonly:read_file") == step_class("read:<path:hermeswork:py>") == "read"
    assert step_class("bash:rg -n <path:rel>") == "search"
    assert step_class("bash:git status") == "git:status"
    assert step_class("mcp:ocbrain.brain.context") == "mcp:ocbrain.brain.context"


# --- label grades ----------------------------------------------------------


def test_grade_order_is_monotone():
    assert GRADE_RANK["verifier-receipted"] > GRADE_RANK["verifier-claimed"]
    assert GRADE_RANK["verifier-claimed"] > GRADE_RANK["artifact-linked"]
    assert GRADE_RANK["artifact-linked"] > GRADE_RANK["self-reported-completed"]
    assert GRADE_RANK["self-reported-completed"] > GRADE_RANK["partial"]
    assert GRADE_RANK["partial"] > GRADE_RANK["blocked-or-failed"]
    assert len(GRADE_RANK) == len(GRADE_ORDER)


def test_grade_prefers_a_receipt_that_still_resolves(tmp_path):
    receipt = tmp_path / "pytest.log"
    receipt.write_text("ok")
    grade, resolvable, _ = grade_closeout(
        "completed", [{"kind": "pytest", "status": "passed", "uri": str(receipt)}], []
    )
    assert grade == "verifier-receipted"
    assert resolvable == 1


def test_grade_demotes_a_verifier_whose_file_is_gone(tmp_path):
    grade, resolvable, _ = grade_closeout(
        "completed",
        [{"kind": "pytest", "status": "passed", "uri": str(tmp_path / "gone.log")}],
        [],
    )
    assert grade == "verifier-claimed"
    assert resolvable == 0


def test_grade_falls_back_to_artifacts_then_self_report(tmp_path):
    artifact = tmp_path / "report.md"
    artifact.write_text("x")
    grade, _, _ = grade_closeout(
        "completed", [{"kind": "prose", "status": "passed"}], [{"uri": str(artifact)}]
    )
    assert grade == "artifact-linked"
    assert grade_closeout("completed", [], [])[0] == "self-reported-completed"


def test_grade_takes_failure_states_straight_from_status(tmp_path):
    receipt = tmp_path / "r.log"
    receipt.write_text("x")
    strong = [{"kind": "pytest", "status": "passed", "uri": str(receipt)}]
    assert grade_closeout("partial", strong, [])[0] == "partial"
    assert grade_closeout("blocked", strong, [])[0] == "blocked-or-failed"
    assert grade_closeout("failed", strong, [])[0] == "blocked-or-failed"


def test_normalize_runtime_folds_free_text_and_admits_ignorance():
    assert normalize_runtime("Codex desktop local Mac") == "codex"
    assert normalize_runtime("hermes-f15a38ee") == "hermes"
    assert normalize_runtime("cursor-opus5") == "cursor"
    assert normalize_runtime("claude-code on Jonathan's Mac") == "claude-code"
    assert normalize_runtime("") == "unknown"
    assert normalize_runtime("wombat") == "unknown"


def test_hermes_adapter_includes_default_state_db(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        create table sessions (
          id text primary key, started_at real, ended_at real, cwd text
        );
        create table messages (
          id integer primary key, session_id text, role text, content text,
          tool_call_id text, tool_calls text, tool_name text, timestamp real
        );
        """
    )
    conn.execute(
        "insert into sessions values (?, ?, ?, ?)",
        ("session-1", 1.0, 2.0, str(tmp_path)),
    )
    conn.execute(
        "insert into messages values (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            1,
            "session-1",
            "assistant",
            None,
            None,
            json.dumps(
                [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command":"git status"}',
                        },
                    }
                ]
            ),
            None,
            1.0,
        ),
    )
    conn.execute(
        "insert into messages values (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            2,
            "session-1",
            "tool",
            '{"output":"clean","exit_code":0}',
            "call-1",
            None,
            "terminal",
            2.0,
        ),
    )
    conn.commit()
    conn.close()

    profiles = tmp_path / "profiles"
    profiles.mkdir()
    monkeypatch.setattr(hermes_adapter, "HERMES_DEFAULT_DB", db_path)
    monkeypatch.setattr(hermes_adapter, "HERMES_PROFILES", profiles)

    traces = list(hermes_adapter.iter_hermes_profile_traces())
    assert len(traces) == 1
    assert traces[0].runtime == "hermes:default"
    assert traces[0].events[0].arg_signature == "bash:git status"
    assert traces[0].events[0].result_class == "ok"


# --- sequence machinery ----------------------------------------------------


def test_collapse_runs_folds_repeats():
    assert collapse_runs(["a", "a", "a", "b", "a"]) == [("a", 3), ("b", 1), ("a", 1)]


def test_first_occurrence_order_keeps_order_and_drops_repeats():
    assert first_occurrence_order(["read", "edit", "read", "test"]) == ["read", "edit", "test"]


def test_is_subsequence_is_order_preserving_not_contiguous():
    assert is_subsequence(("a", "c"), ["a", "b", "c"])
    assert not is_subsequence(("c", "a"), ["a", "b", "c"])


# --- abstention ------------------------------------------------------------


def _episode(closeout_id: str, steps: list[str], grade: str = "verifier-receipted") -> Episode:
    return Episode(
        closeout_id=closeout_id,
        closed_at="2026-08-01T00:00:00+00:00",
        task_ref="t",
        status="completed",
        summary="s",
        runtime_raw="codex",
        runtime="codex",
        runtime_slug="codex",
        session_id=None,
        session_hint=None,
        project=None,
        repo=None,
        grade=grade,
        grade_rank=GRADE_RANK[grade],
        n_verifiers=1,
        n_verifiers_failed=0,
        n_artifacts=0,
        resolvable_verifier_uris=1,
        resolvable_artifact_uris=0,
        events=[
            {"step_index": i, "tool": "Bash", "arg_signature": s, "result_class": "ok", "at": None}
            for i, s in enumerate(steps)
        ],
    )


def test_mine_family_abstains_below_the_episode_floor():
    steps = ["bash:git status", "read:<path:rel:py>", "bash:pytest -q <path:rel>"]
    family = Family("fam_000", "tiny", [_episode(f"c{i}", steps) for i in range(2)])
    result = mine_family(family)
    assert "abstained" in result
    assert str(MIN_FAMILY_EPISODES) in result["abstained"]
    assert "patterns" not in result


def test_mine_family_abstains_when_no_pattern_is_shared():
    """Enough episodes, but every one did something different."""
    family = Family(
        "fam_001",
        "scattered",
        [
            _episode(f"c{i}", [f"bash:cmd{i}", f"read:<path:rel:{i}>", f"bash:other{i}"])
            for i in range(6)
        ],
    )
    result = mine_family(family)
    assert "abstained" in result
    assert "subsequence" in result["abstained"]


def test_mine_family_emits_a_procedure_when_the_order_recurs():
    """The steps interleave differently but the relative order is stable."""
    shared = ["bash:git status", "read:<path:rel:py>", "bash:pytest -q <path:rel>"]
    episodes = [
        _episode("c0", shared),
        _episode("c1", [shared[0], "bash:ls <path:rel>", shared[1], shared[2]]),
        _episode("c2", [shared[0], shared[1], "read:<path:rel:md>", shared[2]]),
        _episode("c3", shared),
        _episode("c4", [shared[0], shared[1], shared[2], "bash:git commit"]),
    ]
    result = mine_family(Family("fam_002", "shared", episodes))
    assert "abstained" not in result
    assert result["patterns"]
    assert result["patterns"][0]["support"] >= 4
    assert result["stats"]["n"] == 5
    assert len(result["receipts"]) == 5


def test_mine_family_reports_failure_branches():
    good = ["bash:git status", "read:<path:rel:py>", "bash:pytest -q <path:rel>"]
    bad = [*good, "bash:ssh -o <path:rel>"]
    episodes = [_episode(f"g{i}", good) for i in range(4)]
    episodes += [_episode(f"b{i}", bad, grade="blocked-or-failed") for i in range(3)]
    result = mine_family(Family("fam_003", "mixed", episodes))
    assert "abstained" not in result
    branch_steps = {branch["step"] for branch in result["failure_branches"]}
    assert "remote" in branch_steps  # ssh abstracts to the remote step class


def _event(index: int, signature: str, outcome: str) -> dict[str, object]:
    return {
        "step_index": index,
        "tool": "Bash",
        "arg_signature": signature,
        "result_class": outcome,
        "at": None,
    }


def _repair_session(pairs: int) -> list[dict[str, object]]:
    """Background noise, then the failure/repair pair, so lift is meaningful."""
    events = [_event(i, f"bash:noise{i % 7}", "ok") for i in range(14)]
    for _ in range(pairs):
        events.append(_event(len(events), "bash:flaky", "error"))
        events.append(_event(len(events), "bash:fixit", "ok"))
    return events


def test_mine_repairs_rejects_a_pair_seen_in_only_one_session():
    """Volume without spread is one bad afternoon, not a repair."""
    single = [{"trace_id": "one", "runtime": "codex", "events": _repair_session(30)}]
    assert mine_repairs(single)["repairs"] == []
    assert mine_repairs(single)["rejected_below_threshold"] > 0


def test_mine_repairs_finds_a_pair_spread_across_sessions():
    spread = [
        {"trace_id": f"s{i}", "runtime": "codex", "events": _repair_session(2)} for i in range(6)
    ]
    mined = mine_repairs(spread)["repairs"]
    assert mined
    assert mined[0]["failing_step"] == "bash:flaky"
    assert mined[0]["repair_step"] == "bash:fixit"
    assert mined[0]["distinct_sessions"] == 6
    assert mined[0]["repair_success_rate"] == 1.0


def test_mine_repairs_separates_retries_from_repairs():
    # Enough distinct background steps that `bash:flaky` is not itself the most
    # common successor in the corpus; otherwise the lift gate has nothing to bite.
    events = [_event(i, f"bash:noise{i % 20}", "ok") for i in range(60)]
    for _ in range(12):
        events.append(_event(len(events), "bash:flaky", "error"))
        events.append(_event(len(events), "bash:flaky", "ok"))
    mined = mine_repairs(
        [{"trace_id": f"s{i}", "runtime": "codex", "events": events} for i in range(5)]
    )
    assert mined["repairs"] == []
    assert mined["retries"]
    assert mined["retries"][0]["failing_step"] == "bash:flaky"


def test_free_text_paths_are_classed_not_leaked() -> None:
    """A captured error message is prose with paths inside it, not a path token.

    `classify_path` only ever saw whole tokens, so raw absolute paths rode
    through error fingerprints into a committed artifact and tripped the repo's
    own public-safety scan. Path classing now happens in `_safe`, the single
    choke point every free-text field passes through.
    """
    from procmine.normalize import _safe, error_fingerprint

    leaky_path = "/".join(
        ("", "Users", "someone", "HermesWork", "repos", "secret-project", "tool.py")
    )
    leaky = f'{{"output": "{leaky_path}"}}'
    assert "/Users/" not in (error_fingerprint(leaky) or "")
    assert "secret-project" not in (error_fingerprint(leaky) or "")

    local_core = "/".join(("", "Users", "someone", ".ocbrain", "ocbrain.sqlite"))
    cleaned = _safe(f"error: cannot open {local_core}")
    assert cleaned is not None
    assert "/Users/" not in cleaned
    assert "<path:" in cleaned
