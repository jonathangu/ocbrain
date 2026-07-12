"""Autopilot pipeline — the autonomy loop (spec §4, lane 5; v0.3 profiles).

``run_autopilot`` is the stage state machine launchd runs on a timer. It is
single-instance (``fcntl.flock`` via :func:`fsutil.file_lock`), takes a daily
rotated SQLite snapshot before touching anything, and drives every downstream
stage the other lanes built — harvest, injection-scan, post-turn review,
compile, autolabel, embed, tripwires, promote, maintain (prune + heal +
catalog archival), dataset mine/export.

**Profiles (v0.3).** The old single 30-minute full cycle (34–45 min) overran its
own timer, so the driver now runs one of two named stage sequences from
``cfg.autopilot.profiles``: a fast ``light`` cycle (every 15 min) and a full
``heavy`` cycle (hourly). Both contend for the *same* autopilot lock
(``cfg.autopilot.profile_locks == "shared"``) so an overlapping fire finds the
lock held and skips cleanly (``{"status": "locked"}``). The ``embed`` stage runs
after ``autolabel`` in every profile — injected here, mirroring the migrate-first
guarantee in :func:`_resolve_stages`, so ``cfg.autopilot.profiles`` stays the
literal operator-facing sequence.

Failure semantics (spec §4.2): each independent stage runs in its own
try/except and records its :class:`~ocbrain.maintenance.MaintenanceResult`-shaped
dict (or an error) in the ``autopilot_runs`` ledger, downgrading the run to
``partial`` but continuing. The two foundational stages — snapshot (1) and
migrate (2) — abort the whole run with status ``error`` on failure, because
every later stage assumes a snapshotted, migrated DB.

Idempotency lives in the stages themselves (watermarks, stable ids, UNIQUE
constraints); autopilot only sequences them, commits after each success so a
kill mid-run loses no committed stage work, and is safe to run back-to-back. A
``running`` ledger row and profile deadman are committed before the first stage,
then checkpointed after every stage so an independent stallcheck can distinguish
a slow cycle from a disappeared one.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.autolabel import autolabel
from ocbrain.config import OcbrainConfig, load_config
from ocbrain.db import DB_BUSY_TIMEOUT_MS, now_iso
from ocbrain.events import canonical_json, rebuild_projection
from ocbrain.fsutil import (
    checkpoint_sqlite_wal,
    file_fingerprint,
    file_lock,
    snapshot_sqlite,
)
from ocbrain.ids import stable_id
from ocbrain.maintenance import (
    archive_unreferenced_catalog,
    check_loop_liveness,
    heal_conflicts,
    prune_knowledge,
)
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
    "embed",
    "tripwires",
    "promote",
    "excerpt_render",
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

    def budget_for(self, stage: str) -> float | None:
        """Per-stage wall-clock budget in seconds (spec §4.2).

        A stage listed in ``cfg.autopilot.stage_budgets`` uses its own value
        (e.g. ``dataset_mine`` at 900s); every other budget-aware stage falls
        back to the shared ``stage_budget_seconds``. ``None`` disables the
        budget entirely (unbounded run).
        """
        if self.stage_budget_seconds is None:
            return None
        override = self.cfg.autopilot.stage_budgets.get(stage)
        return float(override) if override is not None else self.stage_budget_seconds


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
    snapshot_dir = _anchor_side_path(ctx.cfg.autopilot.snapshot_dir, ctx.db_path)
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
        current_history_fingerprints,
        history_files,
        import_history_file,
        imported_history_sources,
    )

    deadline = _deadline(ctx, "harvest")
    roots = [Path(r).expanduser() for r in ctx.session_roots]
    files = history_files(roots)
    # Exact-file roots are operator-curated high-value ledgers. Process them
    # before large transcript trees so a busy five-minute harvest budget can
    # never starve a canonical snapshot at the end of the global path order.
    exact_roots = {str(path) for path in roots if path.is_file()}
    files.sort(key=lambda path: (str(path) not in exact_roots, path))
    existing = imported_history_sources(ctx.conn)
    current_fingerprints = current_history_fingerprints(ctx.conn)
    imported = 0
    skipped = 0
    lock_retries = 0
    for path in files:
        if _expired(deadline):
            break
        from ocbrain.fsutil import history_runtime

        key = (str(path), f"{history_runtime(path)}_history_file")
        fingerprint = file_fingerprint(path)
        if current_fingerprints.get(key) == fingerprint:
            continue
        try:
            result, retries = _retry_sqlite_locked(
                ctx,
                deadline,
                lambda current_path=path: import_history_file(
                    ctx.conn,
                    current_path,
                    project=None,
                    privacy_scope="workspace",
                    max_bytes=200_000,
                ),
            )
            lock_retries += retries
        except (OSError, UnicodeError, ValueError):
            skipped += 1
            continue
        if result is None:
            skipped += 1
            continue
        # Reading and normalizing the next history file can consume the whole
        # stage budget. Close this file's import transaction first.
        ctx.conn.commit()
        existing.add((result["path"], f"{result['runtime']}_history_file"))
        current_fingerprints[key] = fingerprint
        imported += 1
    mem = _harvest_memory_globs(ctx, deadline)
    imported += mem
    return {
        "action": "harvest",
        "changed": imported,
        "imported": imported,
        "memory_imported": mem,
        "skipped": skipped,
        "lock_retries": lock_retries,
    }


def _retry_sqlite_locked(
    ctx: AutopilotContext,
    deadline: float | None,
    operation: Callable[[], Any],
) -> tuple[Any, int]:
    """Retry one idempotent harvest write while its stage budget remains.

    Import writes use stable ids/upserts, and every failed SQLite statement is
    rolled back before retry. Non-lock failures propagate unchanged.
    """
    retries = 0
    maximum = max(0, int(ctx.cfg.autopilot.sqlite_lock_retries))
    while True:
        try:
            return operation(), retries
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or retries >= maximum:
                raise
            with contextlib.suppress(sqlite3.Error):
                ctx.conn.rollback()
            delay = float(ctx.cfg.autopilot.sqlite_lock_backoff_seconds) * (2**retries)
            if deadline is not None and time.monotonic() + delay >= deadline:
                raise
            retries += 1
            time.sleep(delay)


def _harvest_memory_globs(ctx: AutopilotContext, deadline: float | None) -> int:
    """Import curated memory/doctrine files (``dataset.memory_globs``) as evidence.

    These are high-value files OUTSIDE the transcript session roots (e.g. a
    per-workspace ``MEMORY.md`` carrying founder doctrine). ``upsert_evidence``
    dedups on ``(source_uri, content_hash)`` so re-imports are idempotent.
    """
    import glob
    import os

    from ocbrain.cli import import_memory_file

    globs = list(ctx.cfg.dataset.memory_globs)
    if not globs:
        return 0
    seen: set[str] = set()
    imported = 0
    for pattern in globs:
        expanded = os.path.expanduser(str(pattern))
        for match in sorted(glob.glob(expanded, recursive=True)):
            if _expired(deadline):
                return imported
            path = Path(match)
            key = str(path)
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            try:
                result = import_memory_file(
                    ctx.conn,
                    path,
                    project=None,
                    privacy_scope="workspace",
                    max_bytes=200_000,
                )
            except (OSError, UnicodeError, ValueError):
                continue
            if result is not None:
                # Never hold the writer while the next doctrine file is read.
                ctx.conn.commit()
                imported += 1
    return imported


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
    from ocbrain.config import founder_ids as _founder_ids
    from ocbrain.dataset.transcripts import iter_transcript_files, parse_transcript

    settle_seconds = ctx.cfg.review.settle_minutes * 60
    now_ns = time.time_ns()
    author_ids = ctx.cfg.dataset.persona_author_ids
    direct_agents = ctx.cfg.dataset.persona_direct_agents
    founder_ids = _founder_ids(ctx.cfg)
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
                founder_ids=founder_ids,
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
    deadline = _deadline(ctx, "review")
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
        time_budget_seconds=ctx.budget_for("autolabel"),
    )


# --------------------------------------------------------------------------- #
# Stage 7b — embed (semantic vectors for attribution; v0.3)
# --------------------------------------------------------------------------- #
def stage_embed(ctx: AutopilotContext) -> dict[str, Any]:
    """Embed pending knowledge rows for vector attribution (v0.3).

    Runs after ``autolabel`` in every profile so freshly-labelled rows are
    embeddable in the same cycle. ``embed_pending`` is self-limiting: it
    early-returns a ``skipped`` summary when embedding is disabled, no API key is
    present, or the daily USD cap is spent — so no network egress happens on
    those paths. A dry run skips it entirely (the stage makes real egress calls).
    """
    if ctx.dry_run:
        return {"action": "embed", "changed": 0, "skipped": "dry_run"}
    from ocbrain.embed import embed_pending

    return embed_pending(ctx.conn, ctx.cfg, now=ctx.now)


# --------------------------------------------------------------------------- #
# Stage 8 — tripwires
# --------------------------------------------------------------------------- #
def stage_tripwires(ctx: AutopilotContext) -> dict[str, Any]:
    return run_tripwires(
        ctx.conn,
        ctx.cfg,
        now=ctx.now,
        time_budget_seconds=ctx.budget_for("tripwires"),
    ).as_dict()


# --------------------------------------------------------------------------- #
# Stage 9 — promote / demote
# --------------------------------------------------------------------------- #
def stage_promote(ctx: AutopilotContext) -> dict[str, Any]:
    promoted = promote_to_memory(ctx.conn, ctx.cfg, now=ctx.now)
    demoted = demote_and_decay(ctx.conn, ctx.cfg, now=ctx.now)
    changed = int(promoted.get("changed", 0)) + int(demoted.get("changed", 0))
    return {"action": "promote", "changed": changed, "promote": promoted, "demote": demoted}


# --------------------------------------------------------------------------- #
# Stage 9b — excerpt_render (render the injectable memory view into runtime files)
# --------------------------------------------------------------------------- #
def stage_excerpt_render(ctx: AutopilotContext) -> dict[str, Any]:
    """Render the promoted, injectable memory view into runtime files (v0.3).

    Runs after ``promote`` so the just-settled injectable set is what lands in
    each target file's managed block. For every path in
    ``cfg.excerpt_render.targets`` it writes/updates ONLY the
    ``BEGIN/END OCBRAIN MANAGED BLOCK`` region — content outside the markers is
    preserved (these are agent-owned memory files). The char budget is
    ``promote.max_chars``. Rendering is idempotent: an unchanged block is not
    rewritten (mtime preserved) and logs no ``served`` retrieval, so a quiet
    cycle touches nothing. Quarantined / unscanned / non-injected rows never
    reach the block (``build_excerpt`` filters them). A dry run skips it entirely
    (the stage writes files and logs served retrievals). Each target is isolated:
    a bad path records an error without failing the others.
    """
    if ctx.dry_run:
        return {"action": "excerpt_render", "changed": 0, "skipped": "dry_run"}
    targets = list(ctx.cfg.excerpt_render.targets)
    if not targets:
        return {"action": "excerpt_render", "changed": 0, "skipped": "no_targets"}
    from ocbrain.excerpt import render_excerpt_file

    results: list[dict[str, Any]] = []
    changed = 0
    for raw in targets:
        path = Path(raw).expanduser()
        try:
            res = render_excerpt_file(
                ctx.conn,
                path,
                runtime="autopilot",
                scope=ctx.cfg.excerpt_render.scope,
                limit=ctx.cfg.excerpt_render.limit,
                max_chars=ctx.cfg.promote.max_chars,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            res = {"path": str(path), "changed": 0, "error": str(exc)}
        results.append(res)
        changed += int(res.get("changed", 0))
    return {"action": "excerpt_render", "changed": changed, "targets": results}


# --------------------------------------------------------------------------- #
# Stage 10 — maintain (prune + heal + independent liveness observer)
# --------------------------------------------------------------------------- #
def stage_maintain(ctx: AutopilotContext) -> dict[str, Any]:
    pruned = prune_knowledge(ctx.conn, now=ctx.now).as_dict()
    healed = heal_conflicts(ctx.conn, now=ctx.now).as_dict()
    # Stallcheck cannot reliably watch its own death. The independently
    # scheduled light/heavy autopilot consumes its heartbeat/deadman ledger so a
    # missing watchdog becomes durable tripwire evidence.
    liveness = check_loop_liveness(ctx.conn, now=ctx.now).as_dict()
    changed = int(pruned["changed"]) + int(healed["changed"]) + int(liveness["changed"])
    result: dict[str, Any] = {
        "action": "maintain",
        "changed": changed,
        "prune": pruned,
        "heal": healed,
        "liveness": liveness,
    }
    # v0.3: sweep never-referenced stale catalog docs out of the working set so
    # the judge + rebuild stop paying for the 101k-file backlog. Reversible
    # (status flip only) and idempotent — already-archived rows are never
    # re-selected. Off when cfg.archive.enabled is false.
    if ctx.cfg.archive.enabled:
        archived = archive_unreferenced_catalog(
            ctx.conn,
            older_than_days=ctx.cfg.archive.catalog_never_referenced_days,
            batch_cap=ctx.cfg.archive.batch_cap,
            now=ctx.now,
        ).as_dict()
        result["archive"] = archived
        result["changed"] = changed + int(archived["changed"])
    return result


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
        time_budget_seconds=ctx.budget_for("dataset_mine"),
    )


# --------------------------------------------------------------------------- #
# Stage 12 — dataset export
# --------------------------------------------------------------------------- #
def stage_dataset_export(ctx: AutopilotContext) -> dict[str, Any]:
    if ctx.dry_run:
        return {"action": "dataset-export", "changed": 0, "skipped": "dry_run"}
    from ocbrain.dataset.export import export_all

    out_dir = _anchor_side_path(ctx.cfg.dataset.export_dir, ctx.db_path)
    return export_all(ctx.conn, cfg=ctx.cfg, now=ctx.now, export_dir=out_dir)


STAGES: dict[str, Callable[[AutopilotContext], dict[str, Any]]] = {
    "snapshot": stage_snapshot,
    "migrate": stage_migrate,
    "harvest": stage_harvest,
    "injection_scan": stage_injection_scan,
    "review": stage_review,
    "compile": stage_compile,
    "autolabel": stage_autolabel,
    "embed": stage_embed,
    "tripwires": stage_tripwires,
    "promote": stage_promote,
    "excerpt_render": stage_excerpt_render,
    "maintain": stage_maintain,
    "dataset_mine": stage_dataset_mine,
    "dataset_export": stage_dataset_export,
}


# --------------------------------------------------------------------------- #
# Side-path resolution
# --------------------------------------------------------------------------- #
def _anchor_side_path(cfg_value: str, db_path: Path | str | None) -> Path:
    """Resolve a lock / snapshot / export side path for the run.

    An *absolute* config value is an explicit override and is honored as-is. A
    *relative* default (e.g. ``data/snapshots/``) is anchored beside the target
    DB — its leaf name under the DB file's parent directory — so an autopilot
    run against a copy DB writes its lock, snapshot, and dataset side-files next
    to that copy and never pollutes the config-anchored (CWD) tree. The raw
    relative value is used only when there is no real DB path (``:memory:`` /
    ``None``), preserving the legacy CWD-relative behavior for in-memory runs.
    """
    p = Path(cfg_value).expanduser()
    if p.is_absolute():
        return p
    if db_path is None or str(db_path) == ":memory:":
        return p
    return Path(db_path).expanduser().resolve().parent / p.name


# --------------------------------------------------------------------------- #
# Time-budget helpers
# --------------------------------------------------------------------------- #
def _deadline(ctx: AutopilotContext, stage: str | None = None) -> float | None:
    budget = ctx.budget_for(stage) if stage is not None else ctx.stage_budget_seconds
    if budget is None:
        return None
    return time.monotonic() + budget


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
    profile: str | None = None,
    dry_run: bool = False,
    roots: list[str] | None = None,
    repos: list[str] | None = None,
) -> dict[str, Any]:
    """Run the autopilot pipeline under a single-instance lock (spec §4).

    ``stages`` restricts the run to a subset (``migrate`` is always run first so
    the schema is present). ``profile`` selects a named stage sequence from
    ``cfg.autopilot.profiles`` (v0.3) — ``light`` (fast, 15-min timer) or
    ``heavy`` (full, hourly); the two are mutually exclusive with ``stages``.
    Every profile gets the ``embed`` stage injected after ``autolabel``.

    All profiles share one lock (``cfg.autopilot.profile_locks == "shared"``), so
    when the lock is held by another instance this returns ``{"status":
    "locked"}`` immediately (spec §4.1) and an overlapping light/heavy fire skips
    cleanly. Returns a summary dict.
    """
    cfg = cfg or load_config()
    if profile is not None:
        if stages:
            raise ValueError("pass either 'profile' or 'stages', not both")
        stages = _resolve_profile_stages(cfg, profile)
    lock_path = _anchor_side_path(cfg.autopilot.lock_path, db_path)
    with file_lock(lock_path) as acquired:
        if not acquired:
            return {"status": "locked", "stages": {}}
        profile_key = profile or ("manual" if stages else "full")
        return _run_locked(
            conn,
            cfg,
            db_path=Path(db_path) if db_path is not None else None,
            now=now or datetime.now(UTC),
            stages=stages,
            dry_run=dry_run,
            roots=roots,
            repos=repos,
            profile_key=profile_key,
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
    profile_key: str,
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
    deadman_seconds = _profile_deadman_seconds(cfg, profile_key, len(selected))

    if not dry_run:
        # Commit observability before work begins. A hard kill can no longer
        # erase the run from history, and the separate stallcheck process has a
        # durable deadline it can inspect even while this file lock stays held.
        _write_run_progress_resilient(
            conn,
            db_path,
            run_id=run_id,
            started_at=started_at,
            finished_at=None,
            status="running",
            stage_results=stage_results,
            error=None,
            profile_key=profile_key,
            deadman_seconds=deadman_seconds,
            active=True,
        )

    for name in selected:
        stage_fn = STAGES[name]
        began = time.monotonic()
        try:
            result = stage_fn(ctx)
            result["elapsed_seconds"] = round(time.monotonic() - began, 4)
            if not dry_run:
                conn.commit()
                if name == "dataset_mine" and cfg.autopilot.checkpoint_after_dataset_mine:
                    result["wal_checkpoint"] = checkpoint_sqlite_wal(
                        conn,
                        db_path,
                        minimum_bytes=cfg.autopilot.checkpoint_wal_min_bytes,
                    )
            stage_results[name] = result
        except Exception as exc:  # noqa: BLE001 - per-stage isolation (spec §4.2)
            if not dry_run:
                # A failing stage may have left the connection unusable; a
                # rollback that itself raises must not escape the isolation.
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
            stage_results[name] = {
                "action": name,
                "error": str(exc),
                "elapsed_seconds": round(time.monotonic() - began, 4),
            }
            if name in ABORT_STAGES:
                status = "error"
                run_error = f"{name}: {exc}"
            else:
                status = "partial"

        if not dry_run:
            _write_run_progress_resilient(
                conn,
                db_path,
                run_id=run_id,
                started_at=started_at,
                finished_at=None,
                status="running",
                stage_results=stage_results,
                error=run_error,
                profile_key=profile_key,
                deadman_seconds=deadman_seconds,
                active=True,
            )
        if status == "error":
            break

    # compile stage already runs one rebuild internally; nothing extra here.
    finished_at = now_iso()
    if not dry_run:
        _write_run_progress_resilient(
            conn,
            db_path,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            stage_results=stage_results,
            error=run_error,
            profile_key=profile_key,
            deadman_seconds=deadman_seconds,
            active=False,
        )

    return {
        "status": status,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "stages": stage_results,
        "error": run_error,
        "dry_run": dry_run,
    }


def _resolve_profile_stages(cfg: OcbrainConfig, profile: str) -> list[str]:
    """Resolve a named profile to its stage list, guaranteeing ``embed`` (v0.3).

    ``cfg.autopilot.profiles`` carries the literal operator-facing sequences
    (``light`` / ``heavy``). The ``embed`` stage runs after ``autolabel`` in every
    profile, but is injected *here* rather than baked into the config literal so
    the config stays the clean source of truth — the exact same discipline
    :func:`_resolve_stages` uses to guarantee ``migrate`` runs first. Injection is
    idempotent (a profile that already lists ``embed`` is left untouched) and only
    applies to profiles that actually label (``autolabel`` present).

    Final stage ordering is imposed by :func:`_resolve_stages` from
    ``STAGE_NAMES``, so ``embed`` lands in its canonical slot (after ``autolabel``)
    regardless of where it is inserted in the set here.
    """
    profiles = cfg.autopilot.profiles
    if profile not in profiles:
        known = ", ".join(sorted(profiles)) or "(none configured)"
        raise ValueError(f"unknown profile {profile!r}; known profiles: {known}")
    stages = list(profiles[profile])
    if "embed" not in stages and "autolabel" in stages:
        stages.insert(stages.index("autolabel") + 1, "embed")
    return stages


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


def _profile_deadman_seconds(cfg: OcbrainConfig, profile_key: str, stage_count: int) -> int:
    configured = cfg.autopilot.profile_deadman_seconds.get(profile_key)
    if configured is not None:
        return max(60, int(configured))
    # Custom profiles still get a conservative deadline without requiring a
    # second config edit. The grace period covers an overlapping scheduler tick.
    derived = int(cfg.autopilot.stage_budget_seconds) * max(1, stage_count) + 1800
    return max(3600, derived)


def _upsert_autopilot_liveness(
    conn: sqlite3.Connection,
    *,
    profile_key: str,
    checkpoint_at: str,
    deadman_seconds: int,
    active: bool,
) -> None:
    checkpoint = datetime.fromisoformat(checkpoint_at.replace("Z", "+00:00"))
    due_at = (
        datetime.fromtimestamp(checkpoint.timestamp() + deadman_seconds, tz=UTC).isoformat()
        if active
        else None
    )
    conn.execute(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        ) VALUES ('autopilot', ?, ?, ?, ?, ?)
        ON CONFLICT(loop_id, run_id) DO UPDATE SET
          last_heartbeat_at = excluded.last_heartbeat_at,
          last_ledger_write_at = excluded.last_ledger_write_at,
          expected_interval_seconds = excluded.expected_interval_seconds,
          deadman_due_at = excluded.deadman_due_at
        """,
        (profile_key, checkpoint_at, checkpoint_at, deadman_seconds, due_at),
    )


def _write_run_progress_resilient(
    conn: sqlite3.Connection,
    db_path: Path | None,
    *,
    run_id: str,
    started_at: str,
    finished_at: str | None,
    status: str,
    stage_results: dict[str, Any],
    error: str | None,
    profile_key: str,
    deadman_seconds: int,
    active: bool,
) -> None:
    """Checkpoint the run ledger and its deadman in one short transaction.

    Progress is written before work and after every stage. Because a
    stage can leave that connection's SQLite handle in a bad state (e.g. a
    ``file is not a database`` from an interrupted/torn snapshot or a poisoned
    WAL), fall back to a fresh short-lived connection. The ledger row and
    liveness row commit together, so neither observer can get ahead of the
    other. Only a genuinely unwritable on-disk DB is allowed to propagate.
    """
    checkpoint_at = finished_at or now_iso()

    def write(target: sqlite3.Connection) -> None:
        _write_run_ledger(target, run_id, started_at, finished_at, status, stage_results, error)
        _upsert_autopilot_liveness(
            target,
            profile_key=profile_key,
            checkpoint_at=checkpoint_at,
            deadman_seconds=deadman_seconds,
            active=active,
        )

    try:
        write(conn)
        conn.commit()
        return
    except sqlite3.Error:
        with contextlib.suppress(sqlite3.Error):
            conn.rollback()
        if db_path is None or str(db_path) == ":memory:":
            raise
    fresh = sqlite3.connect(Path(db_path))
    fresh.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS}")
    try:
        write(fresh)
        fresh.commit()
    finally:
        fresh.close()


def _write_run_ledger(
    conn: sqlite3.Connection,
    run_id: str,
    started_at: str,
    finished_at: str | None,
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
