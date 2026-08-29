"""Per-cadence model profiles and the independent critic: plumbing, defaulting to off.

Everything here ships inert. `nightly_provider`/`nightly_model` are empty, so both
cadences resolve to the one configured pair; `critic_provider` is empty, so no
second opinion is asked for and no second call is billed. What is pinned is that
the defaults are genuinely unchanged, and that the critic -- when an operator does
turn it on -- refuses to run against the curator's own provider family, which is
the one configuration that would make it useless while looking configured.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ocbrain.curator import (
    CURATION_CADENCES,
    PROVIDER_DEFAULTS,
    critic_settings,
    critic_verdict,
    high_impact_change,
    resolve_model_profile,
)


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, curator: dict) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"curator": curator}), encoding="utf-8")
    monkeypatch.setenv("OCBRAIN_CONFIG", str(path))


def test_both_cadences_resolve_to_one_model_out_of_the_box() -> None:
    """No config: nothing observable changes on any install."""
    assert CURATION_CADENCES == ("hourly", "nightly")
    for cadence in CURATION_CADENCES:
        assert resolve_model_profile(cadence=cadence) == ("anthropic", "claude-sonnet-5")


def test_a_nightly_profile_moves_only_the_nightly_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(tmp_path, monkeypatch, {"nightly_model": "claude-opus-5"})
    assert resolve_model_profile(cadence="hourly") == ("anthropic", "claude-sonnet-5")
    assert resolve_model_profile(cadence="nightly") == ("anthropic", "claude-opus-5")


def test_a_cadence_that_names_a_provider_does_not_inherit_the_other_ones_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model id belongs to a provider; crossing them posts one to the other's API."""
    _config(tmp_path, monkeypatch, {"model": "claude-opus-5", "nightly_provider": "openai"})
    assert resolve_model_profile(cadence="hourly") == ("anthropic", "claude-opus-5")
    provider, model = resolve_model_profile(cadence="nightly")
    assert provider == "openai"
    assert model == PROVIDER_DEFAULTS["openai"]["model"]


def test_an_explicit_argument_outranks_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _config(tmp_path, monkeypatch, {"nightly_model": "claude-opus-5"})
    assert resolve_model_profile(cadence="nightly", model="claude-haiku-4-5") == (
        "anthropic",
        "claude-haiku-4-5",
    )


def test_an_unknown_cadence_is_refused() -> None:
    with pytest.raises(ValueError, match="cadence must be one of"):
        resolve_model_profile(cadence="weekly")


def test_no_provider_default_points_at_a_sunset_model() -> None:
    """A default is a dated fact about somebody else's catalogue.

    `moonshot-v1-32k` named a series that sunsets 2026-08-31: a shipped default
    that stops answering three days after this was written.
    """
    models = {entry["model"] for entry in PROVIDER_DEFAULTS.values()}
    assert not any(model.startswith("moonshot-v1") for model in models)
    assert PROVIDER_DEFAULTS["anthropic"]["model"] == "claude-sonnet-5"


# Model ids this repo has retired, and the paths allowed to still name one. The
# default is that a hit is a finding; every exemption is declared here with the
# reason it is one. Enumerating the *findings* instead is the shape that let the
# first version of this guard pass while `docs/V2_AUTONOMY_SPEC.md` still showed
# an operator a config example naming a retired model -- it only ever looked at
# PROVIDER_DEFAULTS.
RETIRED_MODEL_IDS = ("gpt-5-mini", "moonshot-v1")
RETIRED_ID_EXEMPTIONS = {
    "CHANGELOG.md": "the dated historical record; entries are never rewritten",
    "src/ocbrain/curator.py": "the comment recording why the defaults moved",
    "tests/test_curator_model_profile.py": "this guard has to spell the ids it forbids",
}
SCANNED_SUFFIXES = {".py", ".md", ".sh", ".json", ".toml", ".txt", ".cfg", ".yml", ".yaml"}


def _retired_id_findings(files: dict[str, str]) -> list[str]:
    """Every ``path:id`` where a retired model id appears outside an exemption."""
    findings = []
    for path, text in sorted(files.items()):
        if path in RETIRED_ID_EXEMPTIONS:
            continue
        for retired in RETIRED_MODEL_IDS:
            if retired in text:
                findings.append(f"{path}:{retired}")
    return findings


def _repo_text_files() -> dict[str, str]:
    root = Path(__file__).parents[1]
    files = {}
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root
    ).decode("utf-8").split("\0")
    for relative in tracked:
        if not relative:
            continue
        path = root / relative
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        files[relative] = path.read_text(encoding="utf-8", errors="replace")
    return files


def test_retired_model_repo_scan_is_bounded_to_tracked_files() -> None:
    root = Path(__file__).parents[1]
    tracked = set(
        subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        .decode("utf-8")
        .split("\0")
    )
    assert set(_repo_text_files()) <= tracked


def test_the_retired_model_scan_can_report_dirty() -> None:
    """An audit is not a pass until it has been shown it can fail.

    Fed a planted hit in a path with no exemption, the scan must name it; fed
    the same hit in a declared exemption, it must stay quiet. Without this the
    clean result below is indistinguishable from a scanner that reads nothing.
    """
    planted = {"docs/EXAMPLE.md": "model = gpt-5-mini", "src/ocbrain/other.py": "moonshot-v1-32k"}
    assert _retired_id_findings(planted) == [
        "docs/EXAMPLE.md:gpt-5-mini",
        "src/ocbrain/other.py:moonshot-v1",
    ]
    assert _retired_id_findings({"CHANGELOG.md": "gpt-5-mini"}) == []
    assert _retired_id_findings({"docs/EXAMPLE.md": "claude-sonnet-5"}) == []


def test_no_file_outside_a_declared_exemption_still_names_a_retired_model() -> None:
    """The sibling the first fix left standing.

    Correcting `PROVIDER_DEFAULTS` left `model = gpt-5-mini` as a live config
    example in `docs/V2_AUTONOMY_SPEC.md` -- the line an operator would copy --
    plus both retired ids as parameters in `tests/test_wiki_curator.py`. The
    guard that shipped with the correction only inspected the dict it corrected.
    """
    scanned = _repo_text_files()
    assert "src/ocbrain/curator.py" in scanned, "the scan reached nothing; check the walk"
    assert _retired_id_findings(scanned) == []


# --------------------------------------------------------------------------- #
# the critic
# --------------------------------------------------------------------------- #


def test_the_critic_is_off_unless_a_provider_is_named() -> None:
    assert critic_settings() == ("", "")


def test_high_impact_is_doctrine_and_pins_and_nothing_else() -> None:
    doctrine = {"scope": {"scope_id": "global:doctrine"}, "pinned": False}
    pinned = {"scope": {"scope_id": "project:fixture"}, "pinned": True}
    ordinary = {"scope": {"scope_id": "project:fixture"}, "pinned": False}
    assert high_impact_change(doctrine) is True
    assert high_impact_change(pinned) is True
    assert high_impact_change(ordinary) is False


def test_a_critic_of_the_curators_own_family_is_refused_without_calling_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two calls to one family is one opinion counted twice.

    The credential is deliberately present: this must be refused on the family
    check, before anything reaches a network.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    approved, reason = critic_verdict(
        old_body="The live VM is asa1.",
        new_body="The live VM is asa2.",
        provider="anthropic",
        model="claude-opus-5",
        curator_provider="anthropic",
    )
    assert approved is False
    assert "curator's own family" in reason


def test_a_critic_without_a_credential_is_a_non_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable second opinion is not a silent yes."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    approved, reason = critic_verdict(
        old_body="The live VM is asa1.",
        new_body="The live VM is asa2.",
        provider="openai",
        model="",
        curator_provider="anthropic",
    )
    assert approved is False
    assert "OPENAI_API_KEY" in reason


def test_a_critic_refusal_defers_a_doctrine_supersession(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turned on, the critic's "no" routes the change to the pending ledger.

    Deferred, never dropped: the correction is recorded for an operator, and the
    standing belief keeps serving until somebody decides.
    """
    import ocbrain.curator
    from ocbrain.core_v1 import init_core_v1
    from ocbrain.curator import apply_claims
    from ocbrain.db import connect

    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)

    def claim(body: str) -> dict:
        return {
            "key": "operator-comms-preference",
            "title": "operator comms preference",
            "body": body,
            "category": "preference",
            "lifecycle": "durable",
            "confidence": 0.9,
            "evidence_ids": [],
        }

    # A durable preference compiles into global:doctrine, which is high impact.
    first = apply_claims(
        conn,
        [claim("Jonathan prefers updates to travel through the work artifact.")],
        model="test",
        project="test",
    )
    assert len(first["applied"]) == 1

    monkeypatch.setattr(ocbrain.curator, "critic_settings", lambda: ("openai", "gpt-5.6-luna"))
    monkeypatch.setattr(
        ocbrain.curator,
        "critic_verdict",
        lambda **kwargs: (False, "the replacement drops the artifact qualifier"),
    )
    second = apply_claims(
        conn,
        [claim("Jonathan prefers updates to be short.")],
        model="test",
        project="test",
    )

    assert second["superseded"] == []
    assert len(second["deferred"]) == 1
    proposal = json.loads(
        conn.execute(
            "SELECT body_json FROM brain_events WHERE kind='compilation_proposed' "
            "ORDER BY event_seq DESC LIMIT 1"
        ).fetchone()[0]
    )
    assert "independent critic (openai) did not approve" in json.dumps(proposal)
