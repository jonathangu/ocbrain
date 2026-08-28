"""TTL by volatility class, not by lifecycle.

Lifecycle answers "is this meant to outlive its evidence". It does not answer
"does the thing this names change weekly", and the corpus shows the gap: 158
`current` beliefs all carried the same ~90-day expiry and 185 `durable` beliefs
carried none at all, so a belief naming which ClickHouse host was live "as of
2026-07-24" was still serving 35 days later, with a valid_until running to
2026-11-02. The body below is that shape with the host ids replaced.

Re-dating what is already stored is a separate, opt-in decision: on a copy of the
live corpus the plan re-dates 173 beliefs and 7 of them are already expired, so it
is a command an operator runs after reading the plan, never a background sweep.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ocbrain.cli import main
from ocbrain.core_v1 import get_core_v1_belief, init_core_v1
from ocbrain.curator import (
    DEFAULT_CURRENT_TTL_DAYS,
    DEFAULT_MEASURED_TTL_DAYS,
    DEFAULT_VOLATILE_TTL_DAYS,
    apply_claims,
    apply_volatility_ttl,
    claim_ttl_days,
    claim_valid_until,
    claim_volatility,
    plan_volatility_ttl,
)
from ocbrain.db import connect

PROJECT = "test"

HOST_BODY = (
    "The analytics ClickHouse host rotates (as of 2026-07-24, host-one died and "
    "host-two is live) - confirm the current host before querying."
)
MEASUREMENT_BODY = (
    "The T1 header feature set beat the T1+T2 set out of sample by 0.0166 on the "
    "held-out split, so the extra features were net-negative."
)
DOCTRINE_BODY = (
    "Jonathan prefers outward communication to travel through the work artifact "
    "rather than a separate heads-up message."
)


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def _claim(key: str, body: str, *, lifecycle: str = "durable", volatility: str | None = None):
    claim = {
        "key": key,
        "title": key.replace("-", " ")[:80],
        "body": body,
        "category": "system",
        "lifecycle": lifecycle,
        "confidence": 0.9,
        "evidence_ids": [],
    }
    if volatility is not None:
        claim["volatility"] = volatility
    return claim


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #


def test_a_rotating_host_is_volatile_whatever_lifecycle_it_was_filed_under() -> None:
    """The judgement being taken away from the model is exactly this one."""
    volatility, markers = claim_volatility(_claim("clickhouse-host", HOST_BODY))
    assert volatility == "volatile"
    assert set(markers) == {"as_of", "host", "live_state"}
    assert (
        claim_ttl_days(
            _claim("clickhouse-host", HOST_BODY),
            current_ttl_days=DEFAULT_CURRENT_TTL_DAYS,
            volatility_ttl=True,
        )
        == DEFAULT_VOLATILE_TTL_DAYS
        == 14
    )


def test_a_measurement_gets_weeks_and_doctrine_gets_no_expiry() -> None:
    measurement = _claim("t1-vs-t1t2", MEASUREMENT_BODY, lifecycle="current")
    doctrine = _claim("comms-preference", DOCTRINE_BODY)
    assert claim_volatility(measurement)[0] == "measured"
    assert claim_volatility(doctrine)[0] == "doctrine"
    assert (
        claim_ttl_days(measurement, current_ttl_days=90, volatility_ttl=True)
        == DEFAULT_MEASURED_TTL_DAYS
        == 45
    )
    assert claim_ttl_days(doctrine, current_ttl_days=90, volatility_ttl=True) is None


def test_zero_current_ttl_days_disables_expiry_under_either_scheme() -> None:
    """The operator's off switch has to keep working after the rule was re-keyed.

    `--current-ttl-days 0` is documented as "disables expiry", and under the old
    lifecycle rule it did. Re-keying TTL on volatility left that number read but
    ignored: a curator run started with 0 still stamped 14 days on a claim naming
    a rotating host. An operator-facing control that reads its input and does
    something else is worse than not having the control.
    """
    host = _claim("clickhouse-host", HOST_BODY)
    measurement = _claim("t1-vs-t1t2", MEASUREMENT_BODY, lifecycle="current")
    assert claim_ttl_days(host, current_ttl_days=0, volatility_ttl=True) is None
    assert claim_ttl_days(measurement, current_ttl_days=0, volatility_ttl=True) is None
    assert claim_ttl_days(host, current_ttl_days=0, volatility_ttl=False) is None
    assert (
        claim_valid_until(
            host, current_ttl_days=0, now=datetime(2026, 8, 28, tzinfo=UTC), volatility_ttl=True
        )
        is None
    )
    # And a positive number still buys the class TTL, so this is an off switch
    # rather than a second way to spell "no expiry".
    assert claim_ttl_days(host, current_ttl_days=90, volatility_ttl=True) == 14


def test_the_old_rule_is_reproduced_exactly_when_the_new_one_is_off() -> None:
    """`volatility_ttl=False` is the historical behaviour, not an approximation."""
    volatile_current = _claim("clickhouse-host", HOST_BODY, lifecycle="current")
    durable_host = _claim("clickhouse-host", HOST_BODY)
    assert claim_ttl_days(volatile_current, current_ttl_days=90, volatility_ttl=False) == 90
    assert claim_ttl_days(durable_host, current_ttl_days=90, volatility_ttl=False) is None


def test_a_declaration_may_shorten_a_claim_but_never_lengthen_it() -> None:
    """Otherwise `durable` becomes a way for the model to opt out of expiry."""
    shortened = _claim("some-state", "A statement with no mechanical markers at all.",
                       lifecycle="current", volatility="volatile")
    assert claim_volatility(shortened)[0] == "volatile"

    lengthened = _claim("some-state", "A statement with no mechanical markers at all.",
                        lifecycle="current", volatility="doctrine")
    assert claim_volatility(lengthened)[0] == "measured"

    # And a mechanical marker outranks any declaration.
    overridden = _claim("clickhouse-host", HOST_BODY, volatility="doctrine")
    assert claim_volatility(overridden)[0] == "volatile"


# --------------------------------------------------------------------------- #
# what a compiled claim gets
# --------------------------------------------------------------------------- #


def test_a_compiled_host_fact_expires_in_days_not_never(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    now = datetime(2026, 8, 28, tzinfo=UTC)
    result = apply_claims(
        conn,
        [_claim("clickhouse-host-rotation", HOST_BODY)],
        model="test",
        project=PROJECT,
        now=now,
    )
    belief = get_core_v1_belief(conn, result["applied"][0])
    assert belief["attributes"]["volatility"] == "volatile"
    assert sorted(belief["attributes"]["volatility_markers"]) == [
        "as_of",
        "host",
        "live_state",
    ]
    expiry = datetime.fromisoformat(belief["attributes"]["valid_until"])
    assert (expiry - now).days == 14
    # The belief this reproduces carried no expiry at all, and the one beside it
    # carried 90 days.
    assert (expiry - now).days < DEFAULT_CURRENT_TTL_DAYS


def test_turning_the_new_rule_off_restores_the_ninety_day_expiry(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    now = datetime(2026, 8, 28, tzinfo=UTC)
    result = apply_claims(
        conn,
        [_claim("some-current-state", "A current statement with no volatile markers.",
                lifecycle="current")],
        model="test",
        project=PROJECT,
        now=now,
        volatility_ttl=False,
    )
    belief = get_core_v1_belief(conn, result["applied"][0])
    expiry = datetime.fromisoformat(belief["attributes"]["valid_until"])
    assert (expiry - now).days == 90


# --------------------------------------------------------------------------- #
# the opt-in sweep
# --------------------------------------------------------------------------- #


def _seed_for_sweep(tmp_path: Path):
    conn = _core(tmp_path)
    old = datetime.now(UTC) - timedelta(days=40)
    apply_claims(
        conn,
        [_claim("clickhouse-host-rotation", HOST_BODY)],
        model="test",
        project=PROJECT,
        now=old,
        volatility_ttl=False,
    )
    apply_claims(
        conn,
        [_claim("t1-vs-t1t2-eval", MEASUREMENT_BODY, lifecycle="current")],
        model="test",
        project=PROJECT,
        now=old,
        volatility_ttl=False,
    )
    apply_claims(
        conn,
        [_claim("comms-preference", DOCTRINE_BODY)],
        model="test",
        project=PROJECT,
        now=old,
        volatility_ttl=False,
    )
    return conn


def test_the_plan_counts_what_it_would_change_and_what_it_would_expire(
    tmp_path: Path,
) -> None:
    conn = _seed_for_sweep(tmp_path)
    plan = plan_volatility_ttl(conn)

    assert plan["serving"] == 3
    assert plan["by_class"] == {"volatile": 1, "measured": 1, "doctrine": 1}
    # The host fact was filed `durable` and so had no expiry at all.
    assert plan["gains_a_ttl"] == 1
    # The measurement had 90 days and moves to 45.
    assert plan["shortened"] == 1
    assert plan["unchanged"] == 1
    # Nothing is expired the day it is compiled...
    assert plan["already_expired"] == 0
    # ...and three weeks later the host fact is, which is the number that makes
    # this an opt-in command: those beliefs stop serving when the sweep lands.
    later = plan_volatility_ttl(conn, now=datetime.now(UTC) + timedelta(days=21))
    assert later["already_expired"] == 1
    assert later["sample"]["already_expired"][0]["volatility"] == "volatile"


def test_the_plan_writes_nothing(tmp_path: Path) -> None:
    conn = _seed_for_sweep(tmp_path)
    before = json.dumps(
        sorted(
            str(row[0])
            for row in conn.execute("SELECT attributes_json FROM current_beliefs")
        )
    )
    plan_volatility_ttl(conn)
    after = json.dumps(
        sorted(
            str(row[0])
            for row in conn.execute("SELECT attributes_json FROM current_beliefs")
        )
    )
    assert before == after


def test_applying_the_sweep_rewrites_exactly_the_planned_beliefs(tmp_path: Path) -> None:
    conn = _seed_for_sweep(tmp_path)
    plan = plan_volatility_ttl(conn)
    result = apply_volatility_ttl(conn, actor="operator")

    assert result["rewritten"] == plan["gains_a_ttl"] + plan["shortened"] == 2
    # Doctrine is untouched: no expiry before, no expiry after.
    doctrine = next(
        row
        for row in conn.execute(
            "SELECT attributes_json FROM current_beliefs WHERE serve=1 AND status='current'"
        )
        if json.loads(row[0]).get("key") == "comms-preference"
    )
    assert "valid_until" not in json.loads(doctrine[0])
    # A second run is a no-op: the expiries are already the shorter ones.
    assert apply_volatility_ttl(conn, actor="operator")["rewritten"] == 0


# --------------------------------------------------------------------------- #
# the CLI route
# --------------------------------------------------------------------------- #


def test_cli_prints_a_plan_and_writes_nothing_by_default(tmp_path: Path, capsys) -> None:
    conn = _seed_for_sweep(tmp_path)
    before = _attribute_snapshot(conn)
    conn.close()

    assert main(["--db", str(tmp_path / "core.sqlite"), "wiki-volatility"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "wiki-volatility"
    assert payload["applied"] is False
    assert payload["shortened"] + payload["gains_a_ttl"] == 2

    assert _attribute_snapshot(connect(tmp_path / "core.sqlite")) == before


def test_cli_apply_is_refused_without_yes(tmp_path: Path, capsys) -> None:
    conn = _seed_for_sweep(tmp_path)
    before = _attribute_snapshot(conn)
    conn.close()

    assert main(["--db", str(tmp_path / "core.sqlite"), "wiki-volatility", "--apply"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert payload["refused"] == "--apply needs --yes; nothing was written"
    assert _attribute_snapshot(connect(tmp_path / "core.sqlite")) == before


def test_cli_apply_with_yes_rewrites_the_planned_expiries(tmp_path: Path, capsys) -> None:
    conn = _seed_for_sweep(tmp_path)
    before = _attribute_snapshot(conn)
    conn.close()

    assert (
        main(["--db", str(tmp_path / "core.sqlite"), "wiki-volatility", "--apply", "--yes"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["rewritten"] == 2
    assert _attribute_snapshot(connect(tmp_path / "core.sqlite")) != before


def _attribute_snapshot(conn) -> str:
    return json.dumps(
        sorted(
            (str(row[0]), json.loads(row[1] or "{}").get("valid_until"))
            for row in conn.execute(
                "SELECT belief_id, attributes_json FROM current_beliefs "
                "WHERE serve=1 AND status='current'"
            )
        )
    )


def test_the_expiry_off_switch_reaches_the_volatility_sweep_too(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The sibling the compile-path fix left standing.

    `curator.current_ttl_days = 0` means "this brain does not expire beliefs".
    Honouring it only where claims are compiled, while `ocbrain wiki-volatility`
    went on re-dating the serving corpus from a module constant, would be an
    operator control that works in one place and not the other -- the exact
    shape of the defect being fixed.
    """
    conn = _seed_for_sweep(tmp_path)
    conn.close()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"curator": {"current_ttl_days": 0}}), encoding="utf-8")
    monkeypatch.setenv("OCBRAIN_CONFIG", str(config_path))

    assert main(["--db", str(tmp_path / "core.sqlite"), "wiki-volatility"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["gains_a_ttl"] == 0
    assert plan["shortened"] == 0
    assert plan["already_expired"] == 0
    assert plan["unchanged"] == plan["serving"]

    before = _attribute_snapshot(connect(tmp_path / "core.sqlite"))
    assert (
        main(["--db", str(tmp_path / "core.sqlite"), "wiki-volatility", "--apply", "--yes"]) == 0
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["rewritten"] == 0
    assert _attribute_snapshot(connect(tmp_path / "core.sqlite")) == before
