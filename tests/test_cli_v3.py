"""Lane v3-integration — v0.3 CLI wiring (autopilot profiles).

The full profile execution path is covered hermetically in ``test_autopilot.py``;
here we only assert the CLI surface: ``--profile`` parses, is mutually exclusive
with ``--stage``, and is threaded through ``cmd_autopilot`` to ``run_autopilot``.
"""

from __future__ import annotations

import pytest

from ocbrain.db import connect, init_db
from ocbrain_ops import cli
from ocbrain_ops.cli import build_parser, main


def _parse(argv):
    return build_parser().parse_args(argv)


def test_autopilot_profile_flag_parses():
    ap = _parse(["autopilot", "--profile", "light"])
    assert ap.func.__name__ == "cmd_autopilot"
    assert ap.profile == "light"
    assert ap.stages is None

    heavy = _parse(["autopilot", "--profile", "heavy", "--dry-run"])
    assert heavy.profile == "heavy" and heavy.dry_run is True


def test_autopilot_stage_flag_still_parses_without_profile():
    ap = _parse(["autopilot", "--stage", "review", "--stage", "promote"])
    assert ap.stages == ["review", "promote"]
    assert ap.profile is None


def test_profile_and_stage_are_mutually_exclusive_at_parse():
    with pytest.raises(SystemExit):
        _parse(["autopilot", "--profile", "light", "--stage", "review"])


def test_cmd_autopilot_threads_profile_through(tmp_path, monkeypatch, capsys):
    captured: dict = {}

    def fake_run_autopilot(conn, cfg, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "stages": {}, "run_id": "run_x"}

    monkeypatch.setattr(cli, "run_autopilot", fake_run_autopilot)

    db = tmp_path / "cli.sqlite"
    ops_db = tmp_path / "ops.sqlite"
    conn = connect(db)
    init_db(conn)
    conn.commit()

    rc = main(
        ["--ops-db", str(ops_db), "--legacy-db", str(db), "autopilot", "--profile", "light"]
    )
    assert rc == 0
    assert captured["profile"] == "light"
    assert captured["stages"] is None
    capsys.readouterr()  # drain


def test_cmd_autopilot_default_has_no_profile(tmp_path, monkeypatch, capsys):
    captured: dict = {}

    def fake_run_autopilot(conn, cfg, **kwargs):
        captured.update(kwargs)
        return {"status": "ok", "stages": {}, "run_id": "run_x"}

    monkeypatch.setattr(cli, "run_autopilot", fake_run_autopilot)

    db = tmp_path / "cli.sqlite"
    ops_db = tmp_path / "ops.sqlite"
    conn = connect(db)
    init_db(conn)
    conn.commit()

    rc = main(["--ops-db", str(ops_db), "--legacy-db", str(db), "autopilot"])
    assert rc == 0
    assert captured["profile"] is None
    capsys.readouterr()
