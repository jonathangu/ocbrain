"""The procedural pipeline: incremental extract, episodes, gotcha mint, edges.

These pin the properties that make a *scheduled* miner safe to leave running.
Nothing here touches the operator's real stores: extraction state, cached
segments, and the episodes artifact all go to ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import init_core_v1
from ocbrain.db import connect
from ocbrain.scope import ScopeContext

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from procmine import extract as extract_mod  # noqa: E402
from procmine import mint as mint_mod  # noqa: E402
from procmine import normalize as normalize_mod  # noqa: E402
from procmine.atlas import build, edge_admission, write_episodes  # noqa: E402
from procmine.dag import MIN_GOTCHA_CALLS, MIN_GOTCHA_FAILURE_RATE  # noqa: E402
from procmine.extract import SourceSpec, read_cache, write_cache  # noqa: E402
from procmine.normalize import Trace, TraceEvent, admit_value_edges  # noqa: E402

# --- incremental extraction ------------------------------------------------


def _fake_source(root: Path, parses: list[Path]) -> SourceSpec:
    """A source whose files are JSON blobs and which records what it parsed."""

    def files() -> list[Path]:
        return sorted(root.glob("*.json"))

    def parse(path: Path):
        parses.append(path)
        payload = json.loads(path.read_text())
        yield Trace(
            trace_id=payload["id"],
            runtime="fake",
            source_uri=str(path),
            started_at="2026-08-25T00:00:00+00:00",
            ended_at="2026-08-25T00:10:00+00:00",
            cwd=None,
            events=[
                TraceEvent(
                    step_index=index,
                    tool="bash",
                    arg_signature=signature,
                    result_class="ok",
                )
                for index, signature in enumerate(payload["signatures"])
            ],
        )

    return SourceSpec(files, parse)


@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    root = tmp_path / "src"
    root.mkdir()
    parses: list[Path] = []
    monkeypatch.setitem(extract_mod.SOURCES, "fake", _fake_source(root, parses))
    monkeypatch.delitem(extract_mod.SOURCES, "claude-code")
    monkeypatch.delitem(extract_mod.SOURCES, "codex")
    monkeypatch.delitem(extract_mod.SOURCES, "hermes")
    monkeypatch.delitem(extract_mod.SOURCES, "hermes-legacy")
    return root, parses


def _write_source(root: Path, name: str, signatures: list[str]) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps({"id": name, "signatures": signatures}))
    return path


def test_incremental_extract_skips_unchanged_sources(tmp_path, fake_sources):
    root, parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    _write_source(root, "two", ["bash:pytest"])
    state = tmp_path / "state"

    first = write_cache(tmp_path / "traces.jsonl", state_dir=state)
    assert first["sources"]["fake"]["files_parsed"] == 2
    assert first["sources"]["fake"]["files_reused"] == 0
    assert len(parses) == 2

    parses.clear()
    second = write_cache(tmp_path / "traces2.jsonl", state_dir=state)

    assert second["sources"]["fake"]["files_parsed"] == 0
    assert second["sources"]["fake"]["files_reused"] == 2
    assert parses == []
    # Same corpus out, whether it was parsed or replayed.
    assert read_cache(tmp_path / "traces.jsonl") == read_cache(tmp_path / "traces2.jsonl")


def test_a_changed_source_is_reparsed_and_a_new_one_is_added(tmp_path, fake_sources):
    root, parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    _write_source(root, "two", ["bash:pytest"])
    state = tmp_path / "state"
    write_cache(tmp_path / "traces.jsonl", state_dir=state)

    parses.clear()
    _write_source(root, "two", ["bash:pytest", "bash:ruff"])
    _write_source(root, "three", ["bash:git commit"])
    summary = write_cache(tmp_path / "traces2.jsonl", state_dir=state)

    assert {path.stem for path in parses} == {"two", "three"}
    assert summary["sources"]["fake"]["files_parsed"] == 2
    assert summary["sources"]["fake"]["files_reused"] == 1
    traces = {trace["trace_id"]: trace for trace in read_cache(tmp_path / "traces2.jsonl")}
    assert set(traces) == {"one", "two", "three"}
    assert traces["two"]["n_events"] == 2


def test_no_incremental_reparses_everything(tmp_path, fake_sources):
    root, parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    state = tmp_path / "state"
    write_cache(tmp_path / "traces.jsonl", state_dir=state)

    parses.clear()
    summary = write_cache(
        tmp_path / "traces2.jsonl", state_dir=state, incremental=False
    )
    assert len(parses) == 1
    assert summary["sources"]["fake"]["files_reused"] == 0


def test_a_deleted_source_drops_out_and_its_segment_is_pruned(tmp_path, fake_sources):
    root, _parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    gone = _write_source(root, "two", ["bash:pytest"])
    state = tmp_path / "state"
    write_cache(tmp_path / "traces.jsonl", state_dir=state)
    assert len(list((state / "cache").glob("*.jsonl"))) == 2

    gone.unlink()
    write_cache(tmp_path / "traces2.jsonl", state_dir=state)

    assert len(list((state / "cache").glob("*.jsonl"))) == 1
    assert {t["trace_id"] for t in read_cache(tmp_path / "traces2.jsonl")} == {"one"}


def test_a_corrupt_segment_falls_back_to_parsing(tmp_path, fake_sources):
    root, parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    state = tmp_path / "state"
    write_cache(tmp_path / "traces.jsonl", state_dir=state)

    segment = next((state / "cache").glob("*.jsonl"))
    segment.write_text("{not json\n")
    parses.clear()
    summary = write_cache(tmp_path / "traces2.jsonl", state_dir=state)

    assert len(parses) == 1
    assert summary["sources"]["fake"]["files_parsed"] == 1
    assert {t["trace_id"] for t in read_cache(tmp_path / "traces2.jsonl")} == {"one"}


# --- episodes artifact -----------------------------------------------------


# Runtime-shaped, so ``record_closeout`` accepts it, and deliberately matching
# no trace in these fixtures: the tests that use it are about the mint gates,
# not the session join.
_UNJOINED_SESSION = "99999999-8888-7777-6666-555555555555"


def _core_with_closeout(tmp_path, session_id: str, task_ref: str = "mine-me") -> Path:
    path = tmp_path / "core.sqlite"
    conn = connect(path)
    init_core_v1(conn)
    record_closeout(
        conn,
        task_ref=task_ref,
        status="completed",
        summary="Mined a procedure from the trajectory corpus.",
        context=ScopeContext(project="ocbrain", session=session_id),
        verifier_refs=[{"uri": "repo://ocbrain/pytest", "status": "passed", "kind": "pytest"}],
    )
    conn.commit()
    conn.close()
    return path


def _trace_cache(tmp_path, session_id: str) -> Path:
    cache = tmp_path / "traces.jsonl"
    trace = {
        "trace_id": session_id,
        "runtime": "claude-code",
        "source_uri": "test",
        "started_at": "2026-08-25T00:00:00+00:00",
        "ended_at": "2026-08-25T01:00:00+00:00",
        "cwd": None,
        "cwd_tail": None,
        "truncated": False,
        "n_events": 3,
        "events": [
            {
                "step_index": index,
                "tool": "Bash",
                "arg_signature": signature,
                "result_class": "ok",
                "at": "2026-08-25T00:30:00+00:00",
            }
            for index, signature in enumerate(
                ["bash:git status", "bash:git status", "bash:pytest"]
            )
        ],
        "edges": [],
    }
    cache.write_text(json.dumps(trace) + "\n")
    return cache


def test_atlas_writes_the_episode_sequences_it_used_to_discard(tmp_path):
    session = "11111111-2222-3333-4444-555555555555"
    brain = _core_with_closeout(tmp_path, session)
    data = build(_trace_cache(tmp_path, session), brain_db=brain)

    episodes_path = tmp_path / "artifacts" / "episodes.json"
    write_episodes(data, episodes_path)
    payload = json.loads(episodes_path.read_text())

    assert payload["schema_version"] == "ocbrain.procmine.episodes.v1"
    assert payload["n_episodes"] == 1
    episode = payload["episodes"][0]
    assert episode["join_tier"] == "exact"
    assert episode["in_mining_set"] is True
    assert episode["signatures"] == ["bash:git status", "bash:git status", "bash:pytest"]
    # Run-length collapsed abstract classes: two reads of the same shape are one
    # step, which is the level the DAG miner aligns on.
    assert episode["steps"] == ["git:status", "test"]
    assert episode["result_classes"] == ["ok", "ok", "ok"]


def test_episodes_artifact_is_refreshed_in_place(tmp_path):
    session = "11111111-2222-3333-4444-555555555555"
    brain = _core_with_closeout(tmp_path, session)
    data = build(_trace_cache(tmp_path, session), brain_db=brain)
    episodes_path = tmp_path / "episodes.json"

    write_episodes(data, episodes_path)
    first = episodes_path.read_text()
    write_episodes(data, episodes_path)

    assert episodes_path.read_text() == first
    assert not list(tmp_path.glob("*.tmp"))


# --- gotcha mint -----------------------------------------------------------


def _flaky_traces(
    signature: str = "bash:flaky-step",
    repair: str = "bash:repair-step",
    *,
    sessions: int = 5,
    cycles: int = 5,
) -> list[dict]:
    """Sessions where one step fails a third of the time and one move fixes it."""
    traces = []
    for session in range(sessions):
        events = []
        for _cycle in range(cycles):
            for result in ("ok", "ok", "error"):
                events.append(
                    {
                        "step_index": len(events),
                        "tool": "Bash",
                        "arg_signature": signature,
                        "result_class": result,
                        "at": None,
                    }
                )
            events.append(
                {
                    "step_index": len(events),
                    "tool": "Bash",
                    "arg_signature": repair,
                    "result_class": "ok",
                    "at": None,
                }
            )
        traces.append(
            {
                "trace_id": f"session-{session}",
                "runtime": "claude-code",
                "events": events,
                "edges": [],
            }
        )
    return traces


def test_mint_reports_before_it_writes(tmp_path):
    brain = _core_with_closeout(tmp_path, _UNJOINED_SESSION)
    result = mint_mod.mint_gotchas(_flaky_traces(), brain_db=brain)

    assert result["applied_mode"] == "report_only"
    assert result["applied"] == []
    assert len(result["candidates"]) == 1
    conn = connect(brain)
    assert conn.execute("SELECT COUNT(*) FROM current_beliefs").fetchone()[0] == 0
    conn.close()


def test_mint_converges_on_one_belief_id_across_runs(tmp_path):
    """Recompute and replace. A re-mint must not add a second row."""
    brain = _core_with_closeout(tmp_path, _UNJOINED_SESSION)
    traces = _flaky_traces()

    first = mint_mod.mint_gotchas(traces, brain_db=brain, apply=True)
    # More evidence next cycle: the counts move, the identity must not.
    second = mint_mod.mint_gotchas(
        _flaky_traces(cycles=8), brain_db=brain, apply=True
    )

    assert len(first["applied"]) == 1
    assert len(second["applied"]) == 1
    belief_id = first["applied"][0]["belief_id"]
    assert second["applied"][0]["belief_id"] == belief_id

    conn = connect(brain)
    rows = conn.execute(
        "SELECT belief_id, belief_type, body, attributes_json, status, serve "
        "FROM current_beliefs"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["belief_id"] == belief_id
    assert rows[0]["belief_type"] == "gotcha"
    assert rows[0]["status"] == "current"
    attributes = json.loads(rows[0]["attributes_json"])
    assert attributes["kind"] == "gotcha.v1"
    assert attributes["lifecycle"] == "current"
    assert attributes["signature"] == "bash:flaky-step"
    assert attributes["valid_until"] > attributes["mined_at"]
    assert "reward_band" not in attributes
    # The second run's larger corpus is what is now published.
    assert attributes["support"]["calls"] == 120


def test_mint_is_capped_at_twelve_per_run(tmp_path):
    brain = _core_with_closeout(tmp_path, _UNJOINED_SESSION)
    traces: list[dict] = []
    for index in range(20):
        traces.extend(
            _flaky_traces(
                signature=f"bash:flaky-{index:02d}",
                repair=f"bash:repair-{index:02d}",
            )
        )
    result = mint_mod.mint_gotchas(traces, brain_db=brain, apply=True)

    assert mint_mod.MINT_LIMIT == 12
    assert len(result["candidates"]) == 12
    assert len(result["applied"]) == 12


def test_mint_refuses_a_step_below_either_threshold(tmp_path):
    brain = _core_with_closeout(tmp_path, _UNJOINED_SESSION)
    # Loud but reliable: plenty of calls, hardly any failures.
    reliable = [
        {
            "trace_id": f"session-{index}",
            "runtime": "claude-code",
            "events": [
                {
                    "step_index": step,
                    "tool": "Bash",
                    "arg_signature": "bash:reliable-step",
                    "result_class": "error" if step == 0 else "ok",
                    "at": None,
                }
                for step in range(40)
            ],
            "edges": [],
        }
        for index in range(5)
    ]
    # Flaky but rare: fails half the time, nowhere near the call floor.
    rare = [
        {
            "trace_id": "rare-session",
            "runtime": "claude-code",
            "events": [
                {
                    "step_index": step,
                    "tool": "Bash",
                    "arg_signature": "bash:rare-step",
                    "result_class": "error" if step % 2 else "ok",
                    "at": None,
                }
                for step in range(10)
            ],
            "edges": [],
        }
    ]
    result = mint_mod.mint_gotchas(reliable + rare, brain_db=brain)

    assert result["candidates"] == []
    assert MIN_GOTCHA_CALLS == 50
    assert MIN_GOTCHA_FAILURE_RATE == 0.20


def test_confidence_shrinks_toward_a_coin_flip_without_a_repair():
    assert mint_mod.shrunk_confidence(0, 0) == 0.5
    weak = mint_mod.shrunk_confidence(2, 2)
    strong = mint_mod.shrunk_confidence(200, 200)
    assert 0.5 < weak < strong < 1.0


def test_belief_id_depends_on_signature_and_scope_only():
    left = mint_mod.gotcha_belief_id("bash:flaky", "project:ocbrain")
    assert left == mint_mod.gotcha_belief_id("bash:flaky", "project:ocbrain")
    assert left != mint_mod.gotcha_belief_id("bash:flaky", "project:workspace")
    assert left != mint_mod.gotcha_belief_id("bash:other", "project:ocbrain")


def test_gotcha_scope_folds_through_the_alias_table(monkeypatch):
    """A gotcha must mint into the scope callers can actually reach.

    Historical closeouts carry whatever project string the writing agent typed
    (the live core attributes signatures to 'personalization-headroom' and
    friends), and retrieval only reaches a stored scope after folding the
    caller's string through the alias table. Minting into the raw variant
    parks the gotcha where no folded caller matches it.
    """
    monkeypatch.setenv(
        "OCBRAIN_SCOPES_ALIASES",
        json.dumps({"project:personalization-headroom": "project:coframe-personalization"}),
    )
    attribution = {
        "bash:flaky": {
            "qualities": [1.0],
            "projects": {"personalization-headroom": 3, "coframe": 1},
        }
    }
    scope, quality = mint_mod._scope_for("bash:flaky", attribution)
    assert scope.scope_id == "project:coframe-personalization"
    assert quality == 1.0

    # Without an alias entry the fold still normalizes case and spacing, and
    # an unmapped variant stays itself rather than being guessed at.
    monkeypatch.setenv("OCBRAIN_SCOPES_ALIASES", json.dumps({}))
    attribution["bash:flaky"]["projects"] = {"Personalization Headroom": 2}
    scope, _ = mint_mod._scope_for("bash:flaky", attribution)
    assert scope.scope_id == "project:personalization-headroom"


def test_a_gotcha_with_no_joined_episode_falls_back_to_the_workspace_scope(tmp_path):
    brain = _core_with_closeout(tmp_path, _UNJOINED_SESSION)
    candidates = mint_mod.build_candidates(_flaky_traces(), brain_db=brain)
    assert candidates[0]["scope"].scope_id == mint_mod.FALLBACK_SCOPE_ID
    assert candidates[0]["attributes"]["source_quality"] == mint_mod.LABEL_FREE_QUALITY


# --- value-provenance edge admission ---------------------------------------


def test_hard_edge_requires_a_value_the_previous_call_produced():
    edges = admit_value_edges(
        [
            ("list the directory", "report-2026-08-25-summary.md\n"),
            ("open report-2026-08-25-summary.md", "contents"),
        ]
    )
    assert [edge["admission"] for edge in edges] == ["hard"]
    assert edges[0]["producer_idx"] == 0
    assert edges[0]["consumer_idx"] == 1
    assert len(edges[0]["token_sha"]) == 16


def test_adjacency_without_a_shared_value_is_only_suspected():
    edges = admit_value_edges(
        [
            ("run the tests", "everything passed"),
            ("open a completely unrelated document", "contents"),
        ]
    )
    assert [edge["admission"] for edge in edges] == ["suspected"]
    assert "token_sha" not in edges[0]


def test_an_ambient_token_does_not_admit_an_edge():
    """A path every call mentions proves nothing about who produced what."""
    ambient = "/repo/very/long/shared/path.py"
    edges = admit_value_edges(
        [
            (f"read {ambient}", f"contents of {ambient}"),
            (f"edit {ambient}", f"wrote {ambient}"),
            (f"test {ambient}", "ok"),
        ]
    )
    assert [edge["admission"] for edge in edges] == ["suspected", "suspected"]


def test_a_short_shared_token_is_not_provenance():
    edges = admit_value_edges([("ls", "out.txt"), ("cat out.txt", "")])
    assert edges[0]["admission"] == "suspected"


def test_no_token_survives_into_the_edge():
    secret_ish = "artifact-9f3c1d2e4b5a6789.tar.gz"
    edges = admit_value_edges([("build", secret_ish), (f"upload {secret_ish}", "done")])
    assert secret_ish not in json.dumps(edges)


def test_atlas_names_a_runtime_that_can_only_ever_be_suspected():
    traces = [
        {
            "trace_id": "rich",
            "runtime": "claude-code",
            "edges": [
                {"producer_idx": 0, "consumer_idx": 1, "admission": "hard"},
                {"producer_idx": 1, "consumer_idx": 2, "admission": "suspected"},
            ],
        },
        {
            "trace_id": "thin",
            "runtime": "hermes-legacy",
            "edges": [{"producer_idx": 0, "consumer_idx": 1, "admission": "suspected"}],
        },
    ]
    report = edge_admission(traces)

    assert report["by_runtime"]["claude-code"]["hard"] == 1
    assert report["by_runtime"]["claude-code"]["hard_rate"] == 0.5
    assert report["suspected_only_runtimes"] == ["hermes-legacy"]


def test_a_normalizer_change_invalidates_every_cached_segment(tmp_path, fake_sources):
    """The fingerprint gate cannot see a code change, so the version must."""
    root, parses = fake_sources
    _write_source(root, "one", ["bash:git status"])
    state = tmp_path / "state"
    write_cache(tmp_path / "traces.jsonl", state_dir=state)

    parses.clear()
    stored = json.loads((state / extract_mod.EXTRACT_STATE_NAME).read_text())
    assert stored["normalizer_version"] == normalize_mod.NORMALIZER_VERSION
    stored["normalizer_version"] = "0-older-redaction"
    (state / extract_mod.EXTRACT_STATE_NAME).write_text(json.dumps(stored))

    summary = write_cache(tmp_path / "traces2.jsonl", state_dir=state)
    assert len(parses) == 1
    assert summary["sources"]["fake"]["files_reused"] == 0


# --- artifact scrubbing ----------------------------------------------------


def test_a_private_host_and_account_never_reach_an_artifact():
    """These are not secrets, so the leak detector passes them; publish them anyway
    and a public repository carries an internal address forever.

    The fixture address is RFC 5737 documentation space, not a real one — a test
    that guards against publishing an internal host must not publish one itself.
    """
    leaky = 'host="192.0.2.10"; user="first_last_com"; run as ' + Path.home().name
    scrubbed = normalize_mod.scrub_artifact_text(leaky)
    assert "192.0.2.10" not in scrubbed
    assert "first_last_com" not in scrubbed
    assert Path.home().name not in scrubbed
    assert "<ip>" in scrubbed and "<user>" in scrubbed


def test_error_fingerprints_class_a_host_before_they_are_stored():
    fingerprint = normalize_mod.error_fingerprint(
        "error: ssh 192.0.2.10 as first_last_com refused"
    )
    assert fingerprint is not None
    assert "192.0.2.10" not in fingerprint
    assert "first_last_com" not in fingerprint


def test_the_committed_atlas_artifacts_carry_no_private_identifier():
    """A regression guard on the two files that already leaked one."""
    repo = Path(__file__).resolve().parents[1]
    for name in ("docs/procedures.json", "docs/PROCEDURE-ATLAS-20260824.md"):
        text = (repo / name).read_text()
        assert normalize_mod.scrub_artifact_text(text) == text, name
