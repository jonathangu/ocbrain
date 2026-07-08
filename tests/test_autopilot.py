"""Lane 5 — autopilot stage machine (spec §4, test plan row)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

from ocbrain import autopilot
from ocbrain.config import load_config
from ocbrain.db import connect, init_db
from ocbrain.fsutil import file_lock


def _cfg(tmp_path: Path):
    cfg = load_config()
    roots = tmp_path / "roots"
    roots.mkdir(exist_ok=True)
    return dataclasses.replace(
        cfg,
        autopilot=dataclasses.replace(
            cfg.autopilot,
            lock_path=str(tmp_path / "autopilot.lock"),
            snapshot_dir=str(tmp_path / "snaps"),
            snapshot_keep=3,
            stage_budget_seconds=30,
        ),
        review=dataclasses.replace(cfg.review, session_roots=[str(roots)]),
        judge=dataclasses.replace(cfg.judge, enabled=False),
        dataset=dataclasses.replace(
            cfg.dataset,
            export_dir=str(tmp_path / "datasets"),
            # Point persona git mining at a dir with no .git so it never touches
            # the real workspace repos during tests.
            persona_git_repos=[str(tmp_path / "norepo")],
            learning_db=str(tmp_path / "no-learning.db"),
            commitments_path=str(tmp_path / "no-commitments.json"),
            cron_state_path=str(tmp_path / "no-cron.json"),
        ),
    )


def _db(tmp_path: Path):
    path = tmp_path / "ap.sqlite"
    conn = connect(path)
    init_db(conn)
    return conn, path


def test_full_run_stage_order_and_ledger(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    result = autopilot.run_autopilot(conn, cfg, db_path=path)

    assert result["status"] == "ok"
    # Every dispatched stage ran, in the declared order.
    assert list(result["stages"].keys()) == list(autopilot.STAGE_NAMES)
    # Ledger row written with matching status.
    row = conn.execute(
        "SELECT status FROM autopilot_runs WHERE id = ?", (result["run_id"],)
    ).fetchone()
    assert row["status"] == "ok"


def test_flock_single_instance(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    with file_lock(cfg.autopilot.lock_path) as acquired:
        assert acquired
        # A second instance must find the lock held and exit immediately.
        result = autopilot.run_autopilot(conn, cfg, db_path=path)
    assert result["status"] == "locked"
    assert result["stages"] == {}
    assert conn.execute("SELECT COUNT(*) FROM autopilot_runs").fetchone()[0] == 0


def test_snapshot_daily_skip(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    now = datetime(2026, 7, 8, tzinfo=UTC)
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, now=now)

    first = autopilot.stage_snapshot(ctx)
    assert first["changed"] == 1 and Path(first["path"]).exists()
    second = autopilot.stage_snapshot(ctx)
    assert second["changed"] == 0 and second["skipped"] == "exists"


def test_snapshot_rotation_keep_three(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    snap_dir = Path(cfg.autopilot.snapshot_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    for day in ("20260101", "20260102", "20260103", "20260104"):
        (snap_dir / f"ocbrain-{day}.sqlite").write_text("old")

    now = datetime(2026, 2, 1, tzinfo=UTC)
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, now=now)
    autopilot.stage_snapshot(ctx)

    remaining = sorted(p.name for p in snap_dir.glob("ocbrain-*.sqlite"))
    assert len(remaining) == 3
    assert "ocbrain-20260101.sqlite" not in remaining  # oldest rotated out
    assert "ocbrain-20260201.sqlite" in remaining  # today's kept


def test_stage_failure_is_partial_and_continues(tmp_path, monkeypatch):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)

    def boom(ctx):
        raise RuntimeError("autolabel exploded")

    monkeypatch.setitem(autopilot.STAGES, "autolabel", boom)
    result = autopilot.run_autopilot(conn, cfg, db_path=path)

    assert result["status"] == "partial"
    assert "error" in result["stages"]["autolabel"]
    # Later independent stages still ran despite the mid-pipeline failure.
    assert "promote" in result["stages"] and "dataset_export" in result["stages"]


def test_migrate_failure_aborts(tmp_path, monkeypatch):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)

    def boom(ctx):
        raise RuntimeError("migrate exploded")

    monkeypatch.setitem(autopilot.STAGES, "migrate", boom)
    result = autopilot.run_autopilot(conn, cfg, db_path=path)

    assert result["status"] == "error"
    # Aborts before any downstream stage.
    assert "promote" not in result["stages"]
    assert result["error"] and "migrate" in result["error"]


def test_stage_time_budget_zero_processes_nothing(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    # A settled transcript exists, but a zero budget must stop before yielding it.
    roots = Path(cfg.review.session_roots[0])
    (roots / "s.jsonl").write_text('{"role":"user","content":"hi"}\n')

    ctx = autopilot.AutopilotContext(
        conn=conn, cfg=cfg, db_path=path, stage_budget_seconds=0.0
    )
    result = autopilot.stage_review(ctx)
    assert result["changed"] == 0


def test_dry_run_skips_snapshot_and_export_and_ledger(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    result = autopilot.run_autopilot(conn, cfg, db_path=path, dry_run=True)

    assert result["dry_run"] is True
    assert result["stages"]["snapshot"]["skipped"] == "dry_run"
    assert result["stages"]["dataset_export"]["skipped"] == "dry_run"
    # No ledger row committed on a dry run.
    assert conn.execute("SELECT COUNT(*) FROM autopilot_runs").fetchone()[0] == 0
