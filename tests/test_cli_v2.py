"""Lane 5 — v0.2 CLI wiring (spec §8, test plan row)."""

from __future__ import annotations

import json

import pytest

from ocbrain.cli import build_parser, main
from ocbrain.db import connect, init_db, upsert_knowledge
from ocbrain.safeguards import quarantine_knowledge


def _parse(argv):
    return build_parser().parse_args(argv)


def test_new_subcommands_parse_and_dispatch():
    assert _parse(["autopilot"]).func.__name__ == "cmd_autopilot"
    ap = _parse(["autopilot", "--stage", "review", "--stage", "promote", "--dry-run"])
    assert ap.stages == ["review", "promote"] and ap.dry_run is True

    assert _parse(["quarantine", "list"]).func.__name__ == "cmd_quarantine_list"
    rel = _parse(["quarantine", "release", "know_1", "--actor", "human:jon", "--reason", "fp"])
    assert rel.func.__name__ == "cmd_quarantine_release"

    lab = _parse(["label", "know_1", "--outcome", "bad", "--note", "n"])
    assert lab.func.__name__ == "cmd_label" and lab.outcome == "bad"

    assert _parse(["dataset-mine", "--dataset", "sft"]).func.__name__ == "cmd_dataset_mine"
    exp = _parse(["dataset-export", "--min-scope", "workspace", "--min-label", "neutral"])
    assert exp.min_scope == "workspace" and exp.min_label == "neutral"
    assert _parse(["dataset-stats"]).func.__name__ == "cmd_dataset_stats"


def test_propose_command_is_gone():
    # The human-gate proposal command was deleted in v0.2 (spec §5.1-4).
    with pytest.raises(SystemExit):
        _parse(["propose", "know_1"])


def test_allow_writes_is_accepted_noop_flag():
    # Kept for back-compat so existing plists/scripts don't break (spec §8).
    args = _parse(["mcp", "--allow-writes"])
    assert args.func.__name__ == "cmd_mcp" and args.allow_writes is True


def test_dataset_stats_dispatch(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    conn = connect(db)
    init_db(conn)
    conn.commit()

    rc = main(["--db", str(db), "dataset-stats"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "dataset-stats"
    assert set(payload["datasets"]) == {"sft", "dpo", "persona"}


def test_quarantine_list_and_release_dispatch(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    conn = connect(db)
    init_db(conn)
    kid = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", slug="q", title="t",
        subject="s", predicate="p", value_text="v", status="current",
        confidence=0.9, origin="autopilot",
    )
    quarantine_knowledge(conn, kid, reason="secret_leak", actor="ocbrain-autopilot")
    conn.commit()

    assert main(["--db", str(db), "quarantine", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 1 and listed["quarantined"][0]["id"] == kid

    rc = main(
        ["--db", str(db), "quarantine", "release", kid, "--actor", "human:jon", "--reason", "fp"]
    )
    assert rc == 0
    released = json.loads(capsys.readouterr().out)
    assert released["released"] is True
    assert released["knowledge"]["quarantine_reason"] is None


def test_label_dispatch_records_signal(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    conn = connect(db)
    init_db(conn)
    kid = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", slug="l", title="t",
        subject="s2", predicate="p2", value_text="v", status="current",
        confidence=0.9, origin="autopilot",
    )
    conn.commit()

    assert main(["--db", str(db), "label", kid, "--outcome", "bad", "--note", "wrong"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "bad"

    check = connect(db)
    row = check.execute(
        "SELECT polarity, weight FROM signal_events WHERE knowledge_id = ?", (kid,)
    ).fetchone()
    assert row["polarity"] == "bad" and row["weight"] == 0.9
