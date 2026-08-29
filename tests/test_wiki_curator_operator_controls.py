"""What the curator's operator-facing flags actually reach.

Two of this repo's own recurring defects meet in this file. A rule was re-keyed
and the flag for the old rule was left standing, still parsed, still documented,
and completely inert -- `--current-ttl-days 0` read 0 and stamped 14 days. And a
resolver was added with four config fields, six tests and no call site at all,
while the summary said the policy was "plumbed". Both are only visible from the
command line inward, which is where these tests look.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import ocbrain.curator
from ocbrain.config import CuratorConfig
from ocbrain.core_v1 import init_core_v1, record_core_v1_evidence
from ocbrain.curator import PROVIDER_DEFAULTS
from ocbrain.db import connect
from ocbrain.scope import ScopeTag


def _curator_module():
    path = Path(__file__).parents[1] / "scripts" / "wiki-curator.py"
    spec = importlib.util.spec_from_file_location("wiki_curator_controls", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse(module, argv: list[str]):
    return module.build_parser().parse_args(["--db", "core.sqlite", *argv])


# --------------------------------------------------------------------------- #
# --current-ttl-days and the TTL scheme it belongs to
# --------------------------------------------------------------------------- #


def test_the_ttl_number_and_the_ttl_scheme_are_both_reachable_from_the_cli() -> None:
    """`--current-ttl-days N` has to be able to *be* the number again.

    Under volatility keying the class TTLs decide, so the only way an operator
    can still set the number themselves is to name the scheme it belongs to.
    Without `--no-volatility-ttl` the flag is decoration.
    """
    module = _curator_module()
    shipped = CuratorConfig()

    args = _parse(module, ["--no-volatility-ttl", "--current-ttl-days", "30"])
    assert module.resolve_ttl_policy(args, shipped) == (30, False)

    args = _parse(module, ["--volatility-ttl"])
    assert module.resolve_ttl_policy(args, shipped) == (shipped.current_ttl_days, True)


def test_unset_ttl_flags_defer_to_config_rather_than_to_an_argparse_default() -> None:
    """`curator.current_ttl_days` is the scheduled curator's only voice.

    An argparse `default=90` silently outranks a config file that says 45, and
    the operator has no way to see which one won.
    """
    module = _curator_module()
    configured = replace(CuratorConfig(), current_ttl_days=45, volatility_ttl=False)
    args = _parse(module, [])
    assert module.resolve_ttl_policy(args, configured) == (45, False)


def test_zero_reaches_apply_claims_as_zero_and_not_as_a_clamped_default() -> None:
    module = _curator_module()
    args = _parse(module, ["--current-ttl-days", "0"])
    assert module.resolve_ttl_policy(args, CuratorConfig()) == (0, True)
    # Negative input is an operator typo, not a request for negative expiry.
    args = _parse(module, ["--current-ttl-days", "-5"])
    assert module.resolve_ttl_policy(args, CuratorConfig())[0] == 0


def test_the_ttl_flag_help_points_at_the_scheme_that_makes_it_apply() -> None:
    """Prose must not outrun the instrument on the one surface an operator reads.

    The old help said "0 disables expiry. Durable claims never expire". Under
    volatility keying the second sentence is false -- a durable body naming a
    rotating host expires in 14 days -- and the first was false too until the
    zero case was restored.
    """
    module = _curator_module()
    parser = module.build_parser()
    ttl_help = parser._option_string_actions["--current-ttl-days"].help or ""
    assert "0 disables expiry entirely" in ttl_help
    assert "--no-volatility-ttl" in ttl_help
    # The false half of the old sentence. Under volatility keying a durable body
    # naming a rotating host expires in 14 days.
    assert "Durable claims never expire" not in ttl_help


# --------------------------------------------------------------------------- #
# --cadence: the model profile resolver's only caller
# --------------------------------------------------------------------------- #


def test_the_cadence_flag_exists_and_selects_the_nightly_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resolve_model_profile` had no caller and no flag; this is both."""
    module = _curator_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"curator": {"nightly_model": "claude-opus-5"}}), encoding="utf-8"
    )
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))

    assert _parse(module, []).cadence == "hourly"
    assert _parse(module, ["--cadence", "nightly"]).cadence == "nightly"
    hourly = module.resolve_model_profile(cadence=_parse(module, []).cadence)
    nightly = module.resolve_model_profile(
        cadence=_parse(module, ["--cadence", "nightly"]).cadence
    )
    assert hourly == ("anthropic", "claude-sonnet-5")
    assert nightly == ("anthropic", "claude-opus-5")


def test_provider_is_not_argparse_defaulted_over_the_configured_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--provider` defaulting to "anthropic" made `curator.provider` unreachable."""
    module = _curator_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"curator": {"provider": "openai"}}), encoding="utf-8")
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))

    args = _parse(module, [])
    assert args.provider is None
    provider, model = module.resolve_model_profile(
        cadence=args.cadence, provider=args.provider, model=args.model
    )
    assert provider == "openai"
    assert model == PROVIDER_DEFAULTS["openai"]["model"]

    explicit = _parse(module, ["--provider", "moonshot"])
    assert module.resolve_model_profile(
        cadence=explicit.cadence, provider=explicit.provider, model=explicit.model
    )[0] == "moonshot"


# --------------------------------------------------------------------------- #
# the call site itself
# --------------------------------------------------------------------------- #


def _rollup(capsys) -> dict:
    for line in capsys.readouterr().out.splitlines():
        payload = json.loads(line)
        if payload.get("action") == "wiki-curate-rollup":
            return payload
    raise AssertionError("no wiki-curate-rollup line was emitted")


def test_a_real_run_reports_the_policy_it_actually_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """End-to-end through `main`, with no eligible evidence and no network call.

    This is the test the flags were missing: whatever `build_parser`,
    `resolve_model_profile` and `resolve_ttl_policy` agree on has to be what the
    run announces. A resolver nothing calls cannot show up here.
    """
    module = _curator_module()
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    conn.commit()
    conn.close()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"curator": {"nightly_model": "claude-opus-5"}}), encoding="utf-8"
    )
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))

    def run(*extra: str) -> dict:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "wiki-curator.py",
                "--db",
                str(db_path),
                "--wiki-dir",
                str(tmp_path / "wiki"),
                "--project",
                "test",
                *extra,
            ],
        )
        assert module.main() == 0
        return _rollup(capsys)

    hourly = run()
    assert hourly["cadence"] == "hourly"
    assert (hourly["provider"], hourly["model"]) == ("anthropic", "claude-sonnet-5")
    assert hourly["current_ttl_days"] == CuratorConfig().current_ttl_days
    assert hourly["volatility_ttl"] is True

    nightly = run("--cadence", "nightly", "--no-volatility-ttl", "--current-ttl-days", "30")
    assert nightly["cadence"] == "nightly"
    assert nightly["model"] == "claude-opus-5"
    assert nightly["current_ttl_days"] == 30
    assert nightly["volatility_ttl"] is False

    off = run("--current-ttl-days", "0")
    assert off["current_ttl_days"] == 0


HOST_EVIDENCE = (
    "The analytics ClickHouse host rotates (as of 2026-07-24, host-one died and "
    "host-two is live) - confirm the current host before querying."
)


def _stub_durable_host_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer the hosted call locally with one durable claim naming a rotating host."""

    def fake_urlopen(request, timeout):
        del timeout
        prompt = json.loads(request.data)["messages"][1]["content"]
        match = re.search(r'<evidence id="([^"]+)"[^>]*>\n(.*?)\n</evidence>', prompt, re.DOTALL)
        assert match is not None, "curation prompt carried no evidence"
        evidence_id, body = match.group(1), match.group(2)
        belief = {
            "key": "clickhouse-host-rotation",
            "title": "ClickHouse host rotation",
            "body": body[:400],
            "category": "system",
            "lifecycle": "durable",
            "confidence": 0.9,
            "supports": [{"evidence_id": evidence_id, "quote": body[:60]}],
        }
        message = {"content": json.dumps({"beliefs": [belief]})}
        response = {"choices": [{"finish_reason": "stop", "message": message}]}
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(ocbrain.curator.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("KIMI_API_KEY", "test-key-never-sent")


def _compiled_valid_until(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str
) -> str | None:
    db_path = tmp_path / "core.sqlite"
    conn = connect(db_path)
    init_core_v1(conn)
    record_core_v1_evidence(
        conn,
        body=HOST_EVIDENCE,
        kind="task_closeout_summary",
        scope=ScopeTag("project", "project:test", visibility="internal", egress_policy="hosted_ok"),
        writer="test",
    )
    conn.commit()
    conn.close()

    _stub_durable_host_claim(monkeypatch)
    module = _curator_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wiki-curator.py",
            "--provider",
            "moonshot",
            "--db",
            str(db_path),
            "--wiki-dir",
            str(tmp_path / "wiki"),
            "--project",
            "test",
            "--min-evidence-per-project",
            "1",
            "--apply",
            "--force",
            *extra,
        ],
    )
    assert module.main() == 0
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT attributes_json FROM current_beliefs "
            "WHERE belief_type='wiki_fact' AND serve=1 AND status='current'"
        ).fetchone()
        assert row is not None, "the run compiled nothing"
        return json.loads(row[0] or "{}").get("valid_until")
    finally:
        conn.close()


def test_the_resolved_ttl_policy_reaches_the_compiled_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole chain: flag -> resolve_ttl_policy -> apply_claims -> stored expiry.

    A rollup line that reports the policy proves the resolver ran, not that its
    answer was used. These three runs differ only in the flags, and the belief
    they store has to differ with them.
    """
    default = _compiled_valid_until(tmp_path / "a", monkeypatch)
    assert default is not None  # volatile body: 14 days, even though it is durable

    lifecycle = _compiled_valid_until(tmp_path / "b", monkeypatch, "--no-volatility-ttl")
    assert lifecycle is None  # the old rule: durable never expires

    disabled = _compiled_valid_until(tmp_path / "c", monkeypatch, "--current-ttl-days", "0")
    assert disabled is None  # the off switch, under the default scheme
