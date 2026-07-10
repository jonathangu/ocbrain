"""Lane 5 — autopilot stage machine (spec §4, test plan row)."""

from __future__ import annotations

import dataclasses
import json
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
        # Keep excerpt_render hermetic: the operator's real config points targets
        # at live workspace MEMORY.md files; a test must never write to those.
        excerpt_render=dataclasses.replace(cfg.excerpt_render, targets=[]),
        # Keep the embed stage hermetic: disabled means embed_pending self-skips
        # with no network egress even if OPENAI_API_KEY is present in the env.
        embed=dataclasses.replace(cfg.embed, enabled=False),
        dataset=dataclasses.replace(
            cfg.dataset,
            export_dir=str(tmp_path / "datasets"),
            # Point persona git mining at a dir with no .git so it never touches
            # the real workspace repos during tests.
            persona_git_repos=[str(tmp_path / "norepo")],
            learning_db=str(tmp_path / "no-learning.db"),
            commitments_path=str(tmp_path / "no-commitments.json"),
            cron_state_path=str(tmp_path / "no-cron.json"),
            # Keep the memory-glob harvest hermetic (the operator's real config may
            # point at workspace doctrine files outside tmp_path).
            memory_globs=[],
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


def test_run_checkpoints_visible_progress_and_clears_deadman(tmp_path, monkeypatch):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    observed: dict[str, object] = {}

    def migrate_stage(ctx):
        return {"action": "migrate", "changed": 0}

    def inspect_after_migrate(ctx):
        fresh = connect(path)
        fresh.row_factory = sqlite3.Row
        row = fresh.execute(
            "SELECT status, finished_at, stages_json FROM autopilot_runs "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        live = fresh.execute(
            "SELECT last_heartbeat_at, deadman_due_at FROM loop_liveness "
            "WHERE loop_id='autopilot' AND run_id='manual'"
        ).fetchone()
        observed["status"] = row["status"]
        observed["finished_at"] = row["finished_at"]
        observed["stages"] = json.loads(row["stages_json"])
        observed["heartbeat"] = live["last_heartbeat_at"]
        observed["due"] = live["deadman_due_at"]
        fresh.close()
        return {"action": "maintain", "changed": 0}

    monkeypatch.setitem(autopilot.STAGES, "migrate", migrate_stage)
    monkeypatch.setitem(autopilot.STAGES, "maintain", inspect_after_migrate)
    result = autopilot.run_autopilot(conn, cfg, db_path=path, stages=["migrate", "maintain"])

    assert observed["status"] == "running"
    assert observed["finished_at"] is None
    assert "migrate" in observed["stages"]
    assert observed["heartbeat"] is not None and observed["due"] is not None
    final = conn.execute(
        "SELECT status, finished_at FROM autopilot_runs WHERE id = ?", (result["run_id"],)
    ).fetchone()
    assert final["status"] == "ok" and final["finished_at"] is not None
    assert (
        conn.execute(
            "SELECT deadman_due_at FROM loop_liveness WHERE loop_id='autopilot' AND run_id='manual'"
        ).fetchone()[0]
        is None
    )


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

    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, stage_budget_seconds=0.0)
    result = autopilot.stage_review(ctx)
    assert result["changed"] == 0


def test_budget_for_resolves_per_stage_override(tmp_path):
    # R2: stage_budgets overrides one stage; every other budget-aware stage
    # falls back to the shared stage_budget_seconds.
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg,
        autopilot=dataclasses.replace(cfg.autopilot, stage_budgets={"dataset_mine": 900}),
    )
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, stage_budget_seconds=30.0)
    assert ctx.budget_for("dataset_mine") == 900.0
    assert ctx.budget_for("tripwires") == 30.0
    assert ctx.budget_for("autolabel") == 30.0
    # A None shared budget disables budgets even with no override.
    ctx_none = dataclasses.replace(ctx, stage_budget_seconds=None)
    assert ctx_none.budget_for("dataset_mine") is None


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


def test_stage_names_include_embed_after_autolabel():
    names = list(autopilot.STAGE_NAMES)
    assert "embed" in names
    assert names.index("embed") == names.index("autolabel") + 1
    assert "embed" in autopilot.STAGES


def test_profile_light_runs_fast_subset_with_embed(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    result = autopilot.run_autopilot(conn, cfg, db_path=path, profile="light")

    assert result["status"] == "ok"
    ran = list(result["stages"].keys())
    # light = migrate → review → autolabel → embed → tripwires → promote →
    # excerpt_render → maintain, ordered by STAGE_NAMES; embed injected after
    # autolabel, excerpt_render after promote.
    assert ran == [
        "migrate",
        "review",
        "autolabel",
        "embed",
        "tripwires",
        "promote",
        "excerpt_render",
        "maintain",
    ]
    # The fast profile skips the expensive stages.
    assert "snapshot" not in ran
    assert "dataset_mine" not in ran and "dataset_export" not in ran


def test_profile_heavy_runs_full_cycle_with_embed(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    result = autopilot.run_autopilot(conn, cfg, db_path=path, profile="heavy")

    assert result["status"] == "ok"
    ran = list(result["stages"].keys())
    # heavy is the full sequence — every declared stage, embed included.
    assert ran == list(autopilot.STAGE_NAMES)
    assert "embed" in ran


def test_profile_injects_embed_only_once_when_config_already_lists_it(tmp_path):
    base = _cfg(tmp_path)
    profiles = dict(base.autopilot.profiles)
    profiles["custom"] = ["migrate", "autolabel", "embed", "maintain"]
    cfg = dataclasses.replace(
        base, autopilot=dataclasses.replace(base.autopilot, profiles=profiles)
    )
    stages = autopilot._resolve_profile_stages(cfg, "custom")
    assert stages.count("embed") == 1


def test_unknown_profile_raises(tmp_path):
    cfg = _cfg(tmp_path)
    try:
        autopilot._resolve_profile_stages(cfg, "nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError for unknown profile")


def test_profile_and_stages_are_mutually_exclusive(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    try:
        autopilot.run_autopilot(conn, cfg, db_path=path, profile="light", stages=["maintain"])
    except ValueError as exc:
        assert "profile" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError when both profile and stages given")


def test_light_and_heavy_share_one_lock(tmp_path):
    # profile_locks == "shared": a held autopilot lock blocks BOTH profiles, so an
    # overlapping light/heavy fire skips cleanly instead of double-running.
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.autopilot.profile_locks == "shared"
    with file_lock(cfg.autopilot.lock_path) as acquired:
        assert acquired
        light = autopilot.run_autopilot(conn, cfg, db_path=path, profile="light")
        heavy = autopilot.run_autopilot(conn, cfg, db_path=path, profile="heavy")
    assert light["status"] == "locked" and heavy["status"] == "locked"


def test_maintain_wires_catalog_archival(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)
    result = autopilot.stage_maintain(ctx)
    # Archival is wired in and reports its own sub-result when enabled.
    assert "archive" in result
    assert result["archive"]["action"] == "archive-catalog"
    assert result["liveness"]["action"] == "liveness-check"
    assert result["changed"] >= 0


def test_maintain_skips_archival_when_disabled(tmp_path):
    conn, path = _db(tmp_path)
    base = _cfg(tmp_path)
    cfg = dataclasses.replace(base, archive=dataclasses.replace(base.archive, enabled=False))
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)
    result = autopilot.stage_maintain(ctx)
    assert "archive" not in result
    assert result["liveness"]["action"] == "liveness-check"


def test_embed_stage_skipped_on_dry_run(tmp_path):
    conn, path = _db(tmp_path)
    base = _cfg(tmp_path)
    # Even with embedding enabled, a dry run must make no egress call.
    cfg = dataclasses.replace(base, embed=dataclasses.replace(base.embed, enabled=True))
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, dry_run=True)
    result = autopilot.stage_embed(ctx)
    assert result == {"action": "embed", "changed": 0, "skipped": "dry_run"}


def test_embed_stage_self_skips_when_disabled(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)  # embed disabled in the fixture
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)
    result = autopilot.stage_embed(ctx)
    assert result["action"] == "embed" and result["changed"] == 0
    assert result.get("status") == "skipped"


def test_harvest_memory_globs_imports_doctrine(tmp_path):
    """dataset.memory_globs pulls curated doctrine files in as memory_file evidence.

    These live OUTSIDE the transcript session roots (e.g. a per-workspace MEMORY.md
    carrying founder doctrine that the transcript harvest never reaches).
    """
    conn, path = _db(tmp_path)
    memdir = tmp_path / "doctrine"
    memdir.mkdir()
    (memdir / "MEMORY.md").write_text(
        "# Doctrine\n\nUse 'ready' for ripeness and 'available' for stock.\n",
        encoding="utf-8",
    )
    base = _cfg(tmp_path)
    cfg = dataclasses.replace(
        base,
        dataset=dataclasses.replace(base.dataset, memory_globs=[str(memdir / "*.md")]),
    )
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)
    assert autopilot._harvest_memory_globs(ctx, None) == 1
    assert conn.in_transaction is False
    row = conn.execute(
        "SELECT source_uri FROM evidence WHERE source_type = 'memory_file'"
    ).fetchone()
    assert row is not None and row["source_uri"].endswith("MEMORY.md")
    # Re-running does not create a duplicate evidence row (content_hash dedup).
    autopilot._harvest_memory_globs(ctx, None)
    count = conn.execute(
        "SELECT COUNT(*) FROM evidence WHERE source_type = 'memory_file'"
    ).fetchone()[0]
    assert count == 1


def _inject_row(conn, subject: str, value_text: str) -> str:
    from ocbrain.db import upsert_knowledge

    return upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        origin="loop",
        subject=subject,
        predicate="note",
        value_text=value_text,
        status="current",
        inject=True,
        confidence=0.9,
    )


def test_stage_names_include_excerpt_render_after_promote():
    names = list(autopilot.STAGE_NAMES)
    assert "excerpt_render" in names
    assert names.index("excerpt_render") == names.index("promote") + 1
    assert "excerpt_render" in autopilot.STAGES


def test_excerpt_render_stage_skips_with_no_targets(tmp_path):
    conn, path = _db(tmp_path)
    cfg = _cfg(tmp_path)  # excerpt_render.targets defaults empty
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)
    result = autopilot.stage_excerpt_render(ctx)
    assert result == {"action": "excerpt_render", "changed": 0, "skipped": "no_targets"}


def test_excerpt_render_stage_skipped_on_dry_run(tmp_path):
    conn, path = _db(tmp_path)
    base = _cfg(tmp_path)
    target = tmp_path / "AGENTS.md"
    cfg = dataclasses.replace(
        base,
        excerpt_render=dataclasses.replace(base.excerpt_render, targets=[str(target)]),
    )
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path, dry_run=True)
    result = autopilot.stage_excerpt_render(ctx)
    assert result == {"action": "excerpt_render", "changed": 0, "skipped": "dry_run"}
    assert not target.exists()  # dry run wrote nothing


def test_excerpt_render_stage_writes_targets_and_is_idempotent(tmp_path):
    conn, path = _db(tmp_path)
    _inject_row(conn, "cache-policy", "cache TTL is thirty seconds")
    conn.commit()
    base = _cfg(tmp_path)
    target = tmp_path / "workspace" / "MEMORY.md"
    target.parent.mkdir(parents=True)
    target.write_text("# Memory\n\nExisting doctrine line.\n", encoding="utf-8")
    cfg = dataclasses.replace(
        base,
        excerpt_render=dataclasses.replace(base.excerpt_render, targets=[str(target)]),
    )
    ctx = autopilot.AutopilotContext(conn=conn, cfg=cfg, db_path=path)

    first = autopilot.stage_excerpt_render(ctx)
    assert first["changed"] == 1
    text = target.read_text(encoding="utf-8")
    assert "BEGIN OCBRAIN MANAGED BLOCK" in text
    assert "Existing doctrine line." in text  # content outside markers preserved
    assert "cache-policy" in text

    # Second render with unchanged knowledge writes nothing (mtime preserved).
    mtime_before = target.stat().st_mtime_ns
    second = autopilot.stage_excerpt_render(ctx)
    assert second["changed"] == 0
    assert second["targets"][0]["skipped"] == "unchanged"
    assert target.stat().st_mtime_ns == mtime_before
