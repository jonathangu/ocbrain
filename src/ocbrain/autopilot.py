"""Autopilot pipeline — the v0.2 autonomy loop (spec §4, lane 5).

``run_autopilot`` is the 14-stage state machine launchd runs every 30 minutes
(spec §9). It is single-instance (``fcntl.flock`` via :func:`fsutil.file_lock`),
takes a daily rotated SQLite snapshot before touching anything, and drives every
downstream stage the other lanes built — harvest, injection-scan, post-turn
review, compile, autolabel, tripwires, promote, maintain, dataset mine/export.

Failure semantics (spec §4.2): each independent stage runs in its own
try/except and records its :class:`~ocbrain.maintenance.MaintenanceResult`-shaped
dict (or an error) in the ``autopilot_runs`` ledger, downgrading the run to
``partial`` but continuing. The two foundational stages — snapshot (1) and
migrate (2) — abort the whole run with status ``error`` on failure, because
every later stage assumes a snapshotted, migrated DB.

Idempotency lives in the stages themselves (watermarks, stable ids, UNIQUE
constraints); autopilot only sequences them, commits after each success so a
kill mid-run loses no committed progress, and is safe to run back-to-back.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.autolabel import autolabel
from ocbrain.config import OcbrainConfig, load_config
from ocbrain.db import now_iso
from ocbrain.events import canonical_json, rebuild_projection
from ocbrain.fsutil import file_fingerprint, file_lock, snapshot_sqlite
from ocbrain.ids import stable_id
from ocbrain.maintenance import heal_conflicts, prune_knowledge
from ocbrain.promote import demote_and_decay, promote_to_memory
from ocbrain.review import review_sessions
from ocbrain.safeguards import (
    auto_decide_compilations,
    run_tripwires,
    scan_evidence_for_injection,
)

# Ordered stage names (stage 0 lock and stage 13 finalize are handled by the
# runner itself, not the dispatch table).
STAGE_NAMES: tuple[str, ...] = (
    "snapshot",
    "migrate",
    "harvest",
    "injection_scan",
    "review",
    "compile",
    "autolabel",
    "tripwires",
    "promote",
    "maintain",
    "dataset_mine",
    "dataset_export",
)
# Stages whose failure aborts the whole run (spec §4.2).
ABORT_STAGES: frozenset[str] = frozenset({"snapshot", "migrate"})


@dataclass
class AutopilotContext:
    conn: sqlite3.Connection
    cfg: OcbrainConfig
    db_path: Path | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    dry_run: bool = False
    roots: list[str] | None = None
    repos: list[str] | None = None
    stage_budget_seconds: float | None = None

    @property
    def session_roots(self) -> list[str]:
        return self.roots if self.roots is not None else list(self.cfg.review.session_roots)


# --------------------------------------------------------------------------- #
# Stage 1 — snapshot
# --------------------------------------------------------------------------- #
def _rotate_snapshots(snapshot_dir: Path, keep: int) -> list[str]:
    snaps = sorted(snapshot_dir.glob("ocbrain-*.sqlite"))
    removed: list[str] = []
    while len(snaps) > max(keep, 0):
        victim = snaps.pop(0)
        try:
            victim.unlink()
            for suffix in ("-wal", "-shm"):
                sidecar = victim.with_name(victim.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            removed.append(str(victim))
        except OSError:
            break
    return removed


def stage_snapshot(ctx: AutopilotContext) -> dict[str, Any]:
    if ctx.db_path is None or str(ctx.db_path) == ":memory:":
        return {"action": "snapshot", "changed": 0, "skipped": "no_db_path"}
    snapshot_dir = Path(ctx.cfg.autopilot.snapshot_dir).expanduser()
    dest = snapshot_dir / f"ocbrain-{ctx.now.strftime('%Y%m%d')}.sqlite"
    if dest.exists():
        return {"action": "snapshot", "changed": 0, "skipped": "exists", "path": str(dest)}
    if ctx.dry_run:
        return {"action": "snapshot", "changed": 0, "skipped": "dry_run", "path": str(dest)}
    snapshot_sqlite(ctx.db_path, dest)
    removed = _rotate_snapshots(snapshot_dir, ctx.cfg.autopilot.snapshot_keep)
    return {"action": "snapshot", "changed": 1, "path": str(dest), "rotated": removed}


# --------------------------------------------------------------------------- #
# Stage 2 — migrate
# --------------------------------------------------------------------------- #
def stage_migrate(ctx: AutopilotContext) -> dict[str, Any]:
    from ocbrain.db import init_db

    init_db(ctx.conn)
    return {"action": "migrate", "changed": 0}


# --------------------------------------------------------------------------- #
# Stage 3 — harvest (existing history import path, fingerprint-gated)
# --------------------------------------------------------------------------- #
def stage_harvest(ctx: AutopilotContext) -> dict[str, Any]:
    # Lazy import: cli.py imports autopilot at module load, so importing cli at
    # autopilot import-time would be circular. By the time harvest runs, cli is
    # already fully loaded.
    from ocbrain.cli import (
        history_files,
        import_history_file,
        imported_history_sources,
    )

    deadline = _deadline(ctx)
    files = history_files([Path(r).expanduser() for r in ctx.session_roots])
    existing = imported_history_sources(ctx.conn)
    imported = 0
    skipped = 0
    for path in files:
        if _expired(deadline):
            break
        from ocbrain.fsutil import history_runtime

        key = (str(path), f"{history_runtime(path)}_history_file")
        if key in existing:
            continue
        try:
            result = import_history_file(
                ctx.conn,
                path,
                project=None,
                privacy_scope="workspace",
                max_bytes=200_000,
            )
        except (OSError, UnicodeError, ValueError):
            skipped += 1
            continue
        if result is None:
            skipped += 1
            continue
        existing.add((result["path"], f"{result['runtime']}_history_file"))
        imported += 1
    return {"action": "harvest", "changed": imported, "imported": imported, "skipped": skipped}


# --------------------------------------------------------------------------- #
# Stage 4 — injection scan
# --------------------------------------------------------------------------- #
def stage_injection_scan(ctx: AutopilotContext) -> dict[str, Any]:
    return scan_evidence_for_injection(ctx.conn).as_dict()


# --------------------------------------------------------------------------- #
# Stage 5 — post-turn review of settled sessions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ReviewSession:
    """Adapter exposing the review contract's attribute names over a parsed
    :class:`~ocbrain.dataset.transcripts.Session`.

    The mining lane's ``Session`` DTO uses ``session_id`` / ``source_uri`` / a
    tuple of ``Turn``; review (lane 3) reads ``session_key`` / ``path`` /
    ``mtime_ns`` / ``fingerprint``. This wrapper bridges the two so review can
    watermark per-file (via ``path`` + ``fingerprint``) and link candidates to
    the harvested evidence rows keyed on the transcript path.
    """

    turns: Any
    session_key: str
    path: str
    agent: str | None
    occurred_at: str | None
    mtime_ns: int
    fingerprint: str


def _iter_settled_sessions(
    ctx: AutopilotContext, deadline: float | None
) -> Iterator[_ReviewSession]:
    from ocbrain.dataset.transcripts import iter_transcript_files, parse_transcript

    settle_seconds = ctx.cfg.review.settle_minutes * 60
    now_ns = time.time_ns()
    author_ids = ctx.cfg.dataset.persona_author_ids
    direct_agents = ctx.cfg.dataset.persona_direct_agents
    for path in iter_transcript_files(ctx.session_roots):
        if _expired(deadline):
            return
        try:
            stat = path.stat()
        except OSError:
            continue
        if (now_ns - stat.st_mtime_ns) / 1e9 < settle_seconds:
            continue  # not settled yet
        try:
            session = parse_transcript(
                path,
                author_ids=author_ids,
                direct_agents=direct_agents,
                tool_result_truncate=ctx.cfg.dataset.tool_result_truncate,
            )
        except (OSError, UnicodeError, ValueError):
            continue
        if session is None or not session.turns:
            continue
        yield _ReviewSession(
            turns=session.turns,
            session_key=session.session_id,
            path=session.source_uri,
            agent=session.agent,
            occurred_at=session.occurred_at,
            mtime_ns=stat.st_mtime_ns,
            fingerprint=file_fingerprint(path),
        )


def stage_review(ctx: AutopilotContext) -> dict[str, Any]:
    deadline = _deadline(ctx)
    sessions = _iter_settled_sessions(ctx, deadline)
    return review_sessions(ctx.conn, sessions, ctx.cfg, now_ns=time.time_ns())


# --------------------------------------------------------------------------- #
# Stage 6 — compile (auto-decide undecided proposals; single rebuild)
# --------------------------------------------------------------------------- #
def stage_compile(ctx: AutopilotContext) -> dict[str, Any]:
    return auto_decide_compilations(ctx.conn).as_dict()


# --------------------------------------------------------------------------- #
# Stage 7 — autolabel
# --------------------------------------------------------------------------- #
def stage_autolabel(ctx: AutopilotContext) -> dict[str, Any]:
    return autolabel(
        ctx.conn,
        ctx.cfg,
        now=ctx.now,
        time_budget_seconds=ctx.stage_budget_seconds,
    )


# --------------------------------------------------------------------------- #
# Stage 8 — tripwires
# --------------------------------------------------------------------------- #
def stage_tripwires(ctx: AutopilotContext) -> dict[str, Any]:
    return run_tripwires(ctx.conn, ctx.cfg, now=ctx.now).as_dict()


# --------------------------------------------------------------------------- #
# Stage 9 — promote / demote
# --------------------------------------------------------------------------- #
def stage_promote(ctx: AutopilotContext) -> dict[str, Any]:
    promoted = promote_to_memory(ctx.conn, ctx.cfg, now=ctx.now)
    demoted = demote_and_decay(ctx.conn, ctx.cfg, now=ctx.now)
    changed = int(promoted.get("changed", 0)) + int(demoted.get("changed", 0))
    return {"action": "promote", "changed": changed, "promote": promoted, "demote": demoted}


# --------------------------------------------------------------------------- #
# Stage 10 — maintain (prune + heal)
# --------------------------------------------------------------------------- #
def stage_maintain(ctx: AutopilotContext) -> dict[str, Any]:
    pruned = prune_knowledge(ctx.conn, now=ctx.now).as_dict()
    healed = heal_conflicts(ctx.conn, now=ctx.now).as_dict()
    changed = int(pruned["changed"]) + int(healed["changed"])
    return {"action": "maintain", "changed": changed, "prune": pruned, "heal": healed}


# --------------------------------------------------------------------------- #
# Stage 11 — dataset mine
# --------------------------------------------------------------------------- #
def stage_dataset_mine(ctx: AutopilotContext) -> dict[str, Any]:
    from ocbrain.dataset import mine_all

    return mine_all(
        ctx.conn,
        cfg=ctx.cfg,
        roots=ctx.session_roots,
        repos=ctx.repos,
        time_budget_seconds=ctx.stage_budget_seconds,
    )


# --------------------------------------------------------------------------- #
# Stage 12 — dataset export
# --------------------------------------------------------------------------- #
def stage_dataset_export(ctx: AutopilotContext) -> dict[str, Any]:
    if ctx.dry_run:
        return {"action": "dataset-export", "changed": 0, "skipped": "dry_run"}
    from ocbrain.dataset.export import export_all

    return export_all(ctx.conn, cfg=ctx.cfg, now=ctx.now)


STAGES: dict[str, Callable[[AutopilotContext], dict[str, Any]]] = {
    "snapshot": stage_snapshot,
    "migrate": stage_migrate,
    "harvest": stage_harvest,
    "injection_scan": stage_injection_scan,
    "review": stage_review,
    "compile": stage_compile,
    "autolabel": stage_autolabel,
    "tripwires": stage_tripwires,
    "promote": stage_promote,
    "maintain": stage_maintain,
    "dataset_mine": stage_dataset_mine,
    "dataset_export": stage_dataset_export,
}


# --------------------------------------------------------------------------- #
# Time-budget helpers
# --------------------------------------------------------------------------- #
def _deadline(ctx: AutopilotContext) -> float | None:
    if ctx.stage_budget_seconds is None:
        return None
    return time.monotonic() + ctx.stage_budget_seconds


def _expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_autopilot(
    conn: sqlite3.Connection,
    cfg: OcbrainConfig | None = None,
    *,
    db_path: Path | str | None = None,
    now: datetime | None = None,
    stages: list[str] | None = None,
    dry_run: bool = False,
    roots: list[str] | None = None,
    repos: list[str] | None = None,
) -> dict[str, Any]:
    """Run the autopilot pipeline under a single-instance lock (spec §4).

    ``stages`` restricts the run to a subset (``migrate`` is always run first so
    the schema is present). Returns a summary dict; when the lock is held by
    another instance, returns ``{"status": "locked"}`` immediately (spec §4.1).
    """
    cfg = cfg or load_config()
    lock_path = Path(cfg.autopilot.lock_path).expanduser()
    with file_lock(lock_path) as acquired:
        if not acquired:
            return {"status": "locked", "stages": {}}
        return _run_locked(
            conn,
            cfg,
            db_path=Path(db_path) if db_path is not None else None,
            now=now or datetime.now(UTC),
            stages=stages,
            dry_run=dry_run,
            roots=roots,
            repos=repos,
        )


def _run_locked(
    conn: sqlite3.Connection,
    cfg: OcbrainConfig,
    *,
    db_path: Path | None,
    now: datetime,
    stages: list[str] | None,
    dry_run: bool,
    roots: list[str] | None,
    repos: list[str] | None,
) -> dict[str, Any]:
    ctx = AutopilotContext(
        conn=conn,
        cfg=cfg,
        db_path=db_path,
        now=now,
        dry_run=dry_run,
        roots=roots,
        repos=repos,
        stage_budget_seconds=float(cfg.autopilot.stage_budget_seconds),
    )

    selected = _resolve_stages(stages)
    started_at = now.isoformat(timespec="microseconds")
    run_id = stable_id("run", started_at)
    stage_results: dict[str, Any] = {}
    status = "ok"
    run_error: str | None = None

    for name in selected:
        stage_fn = STAGES[name]
        began = time.monotonic()
        try:
            result = stage_fn(ctx)
            result["elapsed_seconds"] = round(time.monotonic() - began, 4)
            stage_results[name] = result
            if not dry_run:
                conn.commit()
        except Exception as exc:  # noqa: BLE001 - per-stage isolation (spec §4.2)
            if not dry_run:
                conn.rollback()
            stage_results[name] = {
                "action": name,
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - began, 4),
            }
            if name in ABORT_STAGES:
                status = "error"
                run_error = f"{name}: {exc}"
                break
            status = "partial"

    # compile stage already runs one rebuild internally; nothing extra here.
    finished_at = now_iso()
    if not dry_run:
        _write_run_ledger(
            conn, run_id, started_at, finished_at, status, stage_results, run_error
        )
        conn.commit()

    return {
        "status": status,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": stage_results,
        "error": run_error,
        "dry_run": dry_run,
    }


def _resolve_stages(stages: list[str] | None) -> list[str]:
    if not stages:
        return list(STAGE_NAMES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")
    # Always ensure schema is present before any single stage runs.
    ordered = ["migrate"] if "migrate" not in stages else []
    ordered += [s for s in STAGE_NAMES if s in stages]
    return ordered


def _write_run_ledger(
    conn: sqlite3.Connection,
    run_id: str,
    started_at: str,
    finished_at: str,
    status: str,
    stage_results: dict[str, Any],
    error: str | None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO autopilot_runs
          (id, started_at, finished_at, status, stages_json, error)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            started_at,
            finished_at,
            status,
            canonical_json(stage_results),
            error,
        ),
    )


# ``rebuild_projection`` is imported for parity with the compile stage's
# contract; it is exercised inside ``auto_decide_compilations``. Re-exported so
# ops scripts can force a rebuild without importing events directly.
__all__ = [
    "AutopilotContext",
    "STAGES",
    "STAGE_NAMES",
    "rebuild_projection",
    "run_autopilot",
]
