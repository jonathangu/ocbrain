"""Explicit command line surface for the optional operations companion."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from ocbrain.config import load_config
from ocbrain.db import connect, counts, get_knowledge, now_iso
from ocbrain.events import rebuild_projection
from ocbrain.scope import ScopeContext

from ocbrain_ops.autolabel import Signal, record_signal
from ocbrain_ops.autopilot import run_autopilot
from ocbrain_ops.dream import dream
from ocbrain_ops.feedback import feedback_coverage
from ocbrain_ops.loops import LoopIngestOptions, dry_run_loop_ingest, write_loop_ingest
from ocbrain_ops.maintenance import check_loop_liveness, heal_conflicts, prune_knowledge
from ocbrain_ops.publicsafety import scan as public_safety_scan
from ocbrain_ops.safeguards import release_quarantine
from ocbrain_ops.store import DEFAULT_OPS_DB
from ocbrain_ops.teacher import hosted_teacher_request

from . import __version__


def _context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project")
    parser.add_argument("--repo")
    parser.add_argument("--client")
    parser.add_argument("--task")
    parser.add_argument("--session")
    parser.add_argument("--runtime")


def _context(args: argparse.Namespace) -> ScopeContext:
    return ScopeContext(
        project=getattr(args, "project", None),
        repo=getattr(args, "repo", None),
        client=getattr(args, "client", None),
        task=getattr(args, "task", None),
        session=getattr(args, "session", None),
        runtime=getattr(args, "runtime", None),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocbrain-ops",
        description="Optional manual OCBrain operations; no scheduler is installed",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--ops-db", type=Path, default=DEFAULT_OPS_DB)
    parser.add_argument(
        "--legacy-db",
        type=Path,
        help="explicit archived v0.x monolith for compatibility-only mutating commands",
    )
    parser.add_argument("--core-db", type=Path, help="optional v1 core opened read-only")
    parser.add_argument("--pretty", action="store_true")
    commands = parser.add_subparsers(dest="command")

    autopilot = commands.add_parser("autopilot")
    selected = autopilot.add_mutually_exclusive_group()
    selected.add_argument("--stage", action="append", dest="stages")
    selected.add_argument("--profile")
    autopilot.add_argument("--dry-run", action="store_true")
    autopilot.set_defaults(func=cmd_autopilot)

    quarantine = commands.add_parser("quarantine")
    q_commands = quarantine.add_subparsers(dest="quarantine_command")
    q_list = q_commands.add_parser("list")
    q_list.add_argument("--limit", type=int, default=100)
    q_list.set_defaults(func=cmd_quarantine_list)
    q_release = q_commands.add_parser("release")
    q_release.add_argument("knowledge_id")
    q_release.add_argument("--actor", required=True)
    q_release.add_argument("--reason", required=True)
    q_release.set_defaults(func=cmd_quarantine_release)
    quarantine.set_defaults(func=cmd_quarantine_list, limit=100)

    label = commands.add_parser("label")
    label.add_argument("knowledge_id")
    label.add_argument("--outcome", choices=["good", "bad"], required=True)
    label.add_argument("--note", default="")
    label.set_defaults(func=cmd_label)

    loop = commands.add_parser("loop-ingest")
    loop.add_argument("--loop-id", required=True)
    loop.add_argument("--run-id", required=True)
    loop.add_argument("--artifacts", type=Path, required=True)
    loop.add_argument("--ledger", type=Path)
    loop.add_argument("--backlog", type=Path)
    mode = loop.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    loop.add_argument("--json", action="store_true")
    loop.set_defaults(func=cmd_loop_ingest)

    prune = commands.add_parser("prune")
    prune.add_argument("--ttl-days", type=int, default=30)
    prune.add_argument("--unhelpful-ttl-days", type=int, default=14)
    prune.add_argument("--archive-stale-days", type=int)
    prune.set_defaults(func=cmd_prune)

    heal = commands.add_parser("heal")
    heal.add_argument("--numeric-threshold", type=float, default=0.0)
    heal.set_defaults(func=cmd_heal)

    liveness = commands.add_parser("liveness-check")
    liveness.add_argument("--runner-ledger", type=Path)
    liveness.set_defaults(func=cmd_liveness)

    dream_parser = commands.add_parser("event-dream")
    _context_args(dream_parser)
    dream_parser.add_argument("--since-ts")
    dream_parser.add_argument("--target", default="local_model")
    dream_parser.add_argument("--record-egress", action="store_true")
    dream_parser.add_argument("--limit", type=int, default=20)
    dream_parser.set_defaults(func=cmd_dream)

    teacher = commands.add_parser("event-teacher-request")
    _context_args(teacher)
    teacher.add_argument("--query")
    teacher.add_argument("--objective", default="compile_scoped_beliefs")
    teacher.add_argument("--model", default="hosted_teacher")
    teacher.add_argument("--limit", type=int, default=20)
    teacher.add_argument("--no-record", action="store_true")
    teacher.set_defaults(func=cmd_teacher)

    feedback = commands.add_parser("retrieval-feedback-stats")
    feedback.set_defaults(func=cmd_feedback_stats)

    safety = commands.add_parser("public-safety-check")
    safety.add_argument("--diff-range")
    safety.add_argument("--root", type=Path)
    safety.add_argument("--json", action="store_true")
    safety.set_defaults(func=cmd_public_safety)

    hooks = commands.add_parser("install-hooks")
    hooks.add_argument("--root", type=Path)
    hooks.set_defaults(func=cmd_install_hooks)
    return parser


def _output(args: argparse.Namespace, value: Any) -> None:
    print(json.dumps(value, indent=2 if args.pretty else None, sort_keys=True))


def _legacy(args: argparse.Namespace) -> sqlite3.Connection:
    if args.legacy_db is None:
        raise ValueError(
            "this compatibility command requires explicit --legacy-db; "
            "it will never default to or mutate the v1 core"
        )
    path = args.legacy_db.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"legacy database not found: {path}")
    return connect(path)


def _read_source(args: argparse.Namespace) -> sqlite3.Connection:
    path = args.core_db or args.legacy_db
    if path is None:
        raise ValueError("pass --core-db or --legacy-db for this read-only command")
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"database not found: {resolved}")
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    try:
        return int(args.func(args))
    except ValueError as exc:
        _output(args, {"action": args.command, "status": "blocked", "error": str(exc)})
        return 2


def legacy_dispatch(
    argv: list[str] | None = None,
    *,
    db: Path | str | None = None,
    pretty: bool = False,
) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    del db  # The core database is never inherited as a companion write target.
    global_values, command_values = _extract_global_options(
        values,
        value_options={"--ops-db", "--legacy-db", "--core-db"},
    )
    if pretty and "--pretty" not in global_values:
        global_values.append("--pretty")
    return main([*global_values, *command_values])


def _extract_global_options(
    values: list[str],
    *,
    value_options: set[str],
) -> tuple[list[str], list[str]]:
    """Move companion-global flags ahead of an exact lazy-dispatch command."""
    global_values: list[str] = []
    command_values: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in value_options and index + 1 < len(values):
            global_values.extend((value, values[index + 1]))
            index += 2
            continue
        if value == "--pretty":
            global_values.append(value)
            index += 1
            continue
        command_values.append(value)
        index += 1
    return global_values, command_values


def loop_ingest_main() -> int:
    return legacy_dispatch(["loop-ingest", *sys.argv[1:]])


def cmd_autopilot(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    result = run_autopilot(
        conn,
        load_config(),
        db_path=args.legacy_db,
        stages=args.stages,
        profile=args.profile,
        dry_run=args.dry_run,
    )
    _output(args, result)
    return 0


def cmd_quarantine_list(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    rows = conn.execute(
        "SELECT id, slug, title, quarantine_reason, quality_label, updated_at "
        "FROM knowledge WHERE quarantine_reason IS NOT NULL "
        "ORDER BY updated_at DESC, id LIMIT ?",
        (args.limit,),
    ).fetchall()
    _output(args, {"quarantined": [dict(row) for row in rows], "count": len(rows)})
    return 0


def cmd_quarantine_release(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    released = release_quarantine(conn, args.knowledge_id, actor=args.actor, reason=args.reason)
    conn.commit()
    row = get_knowledge(conn, args.knowledge_id)
    _output(
        args,
        {
            "knowledge_id": args.knowledge_id,
            "released": released,
            "knowledge": dict(row) if row else None,
        },
    )
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    if get_knowledge(conn, args.knowledge_id) is None:
        raise ValueError(f"knowledge not found: {args.knowledge_id}")
    signal_id = record_signal(
        conn,
        Signal(
            kind="manual_label",
            polarity=args.outcome,
            weight=0.9,
            source="session",
            source_ref=f"manual:{args.knowledge_id}",
            knowledge_id=args.knowledge_id,
            details={"note": args.note} if args.note else {"manual": True},
            occurred_at=now_iso(),
        ),
    )
    conn.commit()
    _output(
        args,
        {
            "knowledge_id": args.knowledge_id,
            "signal_id": signal_id,
            "outcome": args.outcome,
        },
    )
    return 0


def cmd_loop_ingest(args: argparse.Namespace) -> int:
    options = LoopIngestOptions(
        loop_id=args.loop_id,
        run_id=args.run_id,
        artifacts_root=args.artifacts,
        ledger=args.ledger,
        backlog=args.backlog,
        dry_run=not args.apply,
    )
    if args.apply:
        conn = _legacy(args)
        result = write_loop_ingest(conn, options)
        conn.commit()
    else:
        result = dry_run_loop_ingest(options)
    _output(args, result)
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    result = prune_knowledge(
        conn,
        ttl_days=args.ttl_days,
        unhelpful_ttl_days=args.unhelpful_ttl_days,
        archive_stale_days=args.archive_stale_days,
    )
    conn.commit()
    _output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_heal(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    result = heal_conflicts(conn, numeric_threshold=args.numeric_threshold)
    conn.commit()
    _output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_liveness(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    result = check_loop_liveness(conn, runner_ledger=args.runner_ledger)
    conn.commit()
    _output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    conn = _legacy(args)
    result = dream(
        conn,
        context=_context(args),
        since_ts=args.since_ts,
        target=args.target,
        record_egress=args.record_egress,
        limit=args.limit,
    )
    conn.commit()
    _output(args, result)
    return 0


def cmd_teacher(args: argparse.Namespace) -> int:
    if not load_config().teacher.enabled:
        _output(
            args,
            {
                "action": "event-teacher-request",
                "call_performed": False,
                "status": "blocked",
                "reason": "hosted_teacher_disabled_by_default",
            },
        )
        return 2
    conn = _legacy(args)
    result = hosted_teacher_request(
        conn,
        context=_context(args),
        query=args.query,
        objective=args.objective,
        model=args.model,
        limit=args.limit,
        record=not args.no_record,
    )
    rebuild_projection(conn)
    conn.commit()
    _output(args, result)
    return 0


def cmd_feedback_stats(args: argparse.Namespace) -> int:
    _output(args, feedback_coverage(_read_source(args)))
    return 0


def _repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    try:
        value = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path.cwd()
    return Path(value) if value else Path.cwd()


def cmd_public_safety(args: argparse.Namespace) -> int:
    result = public_safety_scan(_repo_root(args.root), diff_range=args.diff_range)
    _output(args, result.to_dict())
    return 0 if result.ok else 1


def cmd_install_hooks(args: argparse.Namespace) -> int:
    root = _repo_root(args.root)
    source = root / "ops/hooks/pre-push"
    target = root / ".git/hooks/pre-push"
    if not source.is_file() or not target.parent.is_dir():
        raise ValueError("tracked pre-push hook or .git/hooks directory is missing")
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    _output(args, {"action": "install-hooks", "path": str(target), "target": str(source)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
