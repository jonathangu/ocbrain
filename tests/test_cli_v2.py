"""Core and extracted-companion CLI wiring."""

from __future__ import annotations

import json

import pytest

from ocbrain.cli import build_parser as build_core_parser
from ocbrain.db import connect, init_db, upsert_knowledge
from ocbrain_ops.cli import build_parser as build_ops_parser
from ocbrain_ops.cli import main as ops_main
from ocbrain_ops.safeguards import quarantine_knowledge
from ocbrain_training.cli import build_parser as build_training_parser
from ocbrain_training.cli import main as training_main


def _parse_core(argv):
    return build_core_parser().parse_args(argv)


def _parse_ops(argv):
    return build_ops_parser().parse_args(argv)


def _parse_training(argv):
    return build_training_parser().parse_args(argv)


def test_new_subcommands_parse_and_dispatch():
    assert _parse_ops(["autopilot"]).func.__name__ == "cmd_autopilot"
    ap = _parse_ops(
        ["autopilot", "--stage", "review", "--stage", "promote", "--dry-run"]
    )
    assert ap.stages == ["review", "promote"] and ap.dry_run is True

    assert _parse_ops(["quarantine", "list"]).func.__name__ == "cmd_quarantine_list"
    rel = _parse_ops(
        ["quarantine", "release", "know_1", "--actor", "human:jon", "--reason", "fp"]
    )
    assert rel.func.__name__ == "cmd_quarantine_release"

    lab = _parse_ops(["label", "know_1", "--outcome", "bad", "--note", "n"])
    assert lab.func.__name__ == "cmd_label" and lab.outcome == "bad"

    assert (
        _parse_training(["dataset-mine", "--dataset", "sft"]).func.__name__
        == "cmd_mine"
    )
    exp = _parse_training(
        ["dataset-export", "--min-scope", "workspace", "--min-label", "neutral"]
    )
    assert exp.min_scope == "workspace" and exp.min_label == "neutral"
    assert _parse_training(["dataset-stats"]).func.__name__ == "cmd_stats"


def test_propose_command_is_gone():
    # The human-gate proposal command was deleted in v0.2 (spec §5.1-4).
    with pytest.raises(SystemExit):
        _parse_core(["propose", "know_1"])


def test_allow_writes_is_accepted_noop_flag():
    # Kept for back-compat so existing plists/scripts don't break (spec §8).
    args = _parse_core(["mcp", "--allow-writes"])
    assert args.func.__name__ == "cmd_mcp" and args.allow_writes is True


def test_dataset_stats_dispatch(tmp_path, capsys):
    db = tmp_path / "training.sqlite"

    rc = training_main(["--training-db", str(db), "dataset-stats"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "dataset-stats"
    assert set(payload["datasets"]) == {"sft", "dpo", "persona"}


def test_quarantine_list_and_release_dispatch(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    ops_db = tmp_path / "ops.sqlite"
    conn = connect(db)
    init_db(conn)
    kid = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", slug="q", title="t",
        subject="s", predicate="p", value_text="v", status="current",
        confidence=0.9, origin="autopilot",
    )
    quarantine_knowledge(conn, kid, reason="secret_leak", actor="ocbrain-autopilot")
    conn.commit()

    assert (
        ops_main(["--ops-db", str(ops_db), "--legacy-db", str(db), "quarantine", "list"])
        == 0
    )
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 1 and listed["quarantined"][0]["id"] == kid

    rc = ops_main(
        [
            "--ops-db",
            str(ops_db),
            "--legacy-db",
            str(db),
            "quarantine",
            "release",
            kid,
            "--actor",
            "human:jon",
            "--reason",
            "fp",
        ]
    )
    assert rc == 0
    released = json.loads(capsys.readouterr().out)
    assert released["released"] is True
    assert released["knowledge"]["quarantine_reason"] is None


def test_label_dispatch_records_signal(tmp_path, capsys):
    db = tmp_path / "cli.sqlite"
    ops_db = tmp_path / "ops.sqlite"
    conn = connect(db)
    init_db(conn)
    kid = upsert_knowledge(
        conn, knowledge_type="value", gate="auto", slug="l", title="t",
        subject="s2", predicate="p2", value_text="v", status="current",
        confidence=0.9, origin="autopilot",
    )
    conn.commit()

    assert (
        ops_main(
            [
                "--ops-db",
                str(ops_db),
                "--legacy-db",
                str(db),
                "label",
                kid,
                "--outcome",
                "bad",
                "--note",
                "wrong",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["outcome"] == "bad"

    check = connect(db)
    row = check.execute(
        "SELECT polarity, weight FROM signal_events WHERE knowledge_id = ?", (kid,)
    ).fetchone()
    assert row["polarity"] == "bad" and row["weight"] == 0.9
