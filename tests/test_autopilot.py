"""Lane 5 — autopilot stage machine (spec §4, test plan row)."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from ocbrain import autopilot
from ocbrain.config import load_config
from ocbrain.db import connect, init_db
from ocbrain.fsutil import file_lock, snapshot_sqlite


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


def test_side_paths_anchor_to_db_dir_no_cwd_pollution(tmp_path, monkeypatch):
    """Lock/snapshot/export side-files derive from the *target DB* path, not the
    config-anchored CWD tree — a run against a copy DB writes NOTHING outside the
    copy's own directory (regression: copy runs polluted the live repo tree)."""
    db_dir = tmp_path / "dbhome"
    db_dir.mkdir()
    path = db_dir / "ap.sqlite"
    conn = connect(path)
    init_db(conn)

    base = _cfg(tmp_path)
    # Restore the RELATIVE defaults so derivation (not an absolute override) runs.
    cfg = dataclasses.replace(
        base,
        autopilot=dataclasses.replace(
            base.autopilot,
            lock_path="data/autopilot.lock",
            snapshot_dir="data/snapshots/",
        ),
        dataset=dataclasses.replace(base.dataset, export_dir="data/datasets"),
    )

    # A pristine CWD that must stay untouched (no config-anchored 'data/' tree).
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    now = datetime(2026, 7, 8, tzinfo=UTC)
    result = autopilot.run_autopilot(conn, cfg, db_path=path, now=now)
    assert result["status"] == "ok"

    # Everything landed beside the DB…
    assert (db_dir / "autopilot.lock").exists()
    snap = db_dir / "snapshots" / "ocbrain-20260708.sqlite"
    assert snap.exists()
    assert (db_dir / "datasets" / "manifest.json").exists()
    # …the snapshot is a valid standalone DB…
    verify = sqlite3.connect(snap)
    assert verify.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    verify.close()
    # …and the CWD is pristine (nothing leaked to the config-anchored tree).
    assert not (cwd / "data").exists()


def test_snapshot_backup_is_valid_standalone_db(tmp_path):
    """Snapshot uses the online backup API: a valid, self-contained copy even
    with a live connection open on the source (no torn copy, source WAL intact)."""
    src = tmp_path / "src.sqlite"
    conn = connect(src)
    init_db(conn)
    conn.executemany(
        "INSERT INTO retrieval_uses (id, served_at) VALUES (?, ?)",
        [(f"r{i}", "2026-07-08T00:00:00Z") for i in range(500)],
    )
    conn.commit()

    dest = tmp_path / "snap.sqlite"
    # conn stays open (mimics autopilot's live connection during stage 1).
    snapshot_sqlite(src, dest)

    verify = sqlite3.connect(dest)
    assert verify.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    n = verify.execute("SELECT COUNT(*) FROM retrieval_uses").fetchone()[0]
    verify.close()
    conn.close()
    assert n == 500  # transactionally consistent committed image


def test_poisoned_connection_still_records_ledger(tmp_path):
    """A stage that leaves the shared connection raising ``file is not a
    database`` must not both crash the run and lose the ledger — the run is
    still recorded via a fresh connection (spec §4.2 durable record)."""
    cfg = _cfg(tmp_path)
    path = tmp_path / "ap.sqlite"

    class _PoisonLedger(sqlite3.Connection):
        def execute(self, sql, *args):  # type: ignore[override]
            stripped = sql.strip().upper()
            if "AUTOPILOT_RUNS" in stripped and stripped.startswith("INSERT"):
                raise sqlite3.DatabaseError("file is not a database")
            return super().execute(sql, *args)

    conn = sqlite3.connect(path, factory=_PoisonLedger)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    # Must not raise even though the ledger INSERT poisons on the shared handle.
    result = autopilot.run_autopilot(conn, cfg, db_path=path)
    assert result["status"] in {"ok", "partial"}

    # The run is durably recorded via the fresh-connection fallback.
    verify = sqlite3.connect(path)
    verify.row_factory = sqlite3.Row
    row = verify.execute(
        "SELECT status FROM autopilot_runs WHERE id = ?", (result["run_id"],)
    ).fetchone()
    verify.close()
    assert row is not None
