"""Explicit command line surface for the optional training companion."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from ocbrain.config import load_config
from ocbrain.fsutil import checkpoint_sqlite_wal

from ocbrain_training.dataset.calibration import import_calibrations
from ocbrain_training.dataset.classify import classify_examples
from ocbrain_training.dataset.curate import import_persona_curation
from ocbrain_training.dataset.export import export_all
from ocbrain_training.dataset.grade import grade_examples
from ocbrain_training.dataset.mine_persona import mine_persona
from ocbrain_training.dataset.mine_sft import mine_sft
from ocbrain_training.dataset.pilot import (
    prepare_blind_pairs,
    prepare_multiblind,
    prepare_pilot,
    record_training_result,
    score_blind_ratings,
    score_multiblind,
)
from ocbrain_training.dataset.selection import (
    finalize_training_pack,
    select_training_pack,
    selected_pack_stats,
)
from ocbrain_training.dataset.stats import dataset_stats
from ocbrain_training.retrieval_eval import expand_runtime_matrix, run_benchmark
from ocbrain_training.store import DEFAULT_TRAINING_DB, connect_training

from . import __version__


def _add_dataset_choice(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=["sft", "dpo", "persona"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocbrain-training",
        description="Optional, explicitly invoked OCBrain training tools",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--training-db", type=Path, default=DEFAULT_TRAINING_DB)
    parser.add_argument(
        "--core-db",
        type=Path,
        default=Path("~/.ocbrain/core.sqlite").expanduser(),
        help="read-only v1 core used only by retrieval benchmarks",
    )
    parser.add_argument("--pretty", action="store_true")
    commands = parser.add_subparsers(dest="command")

    mine = commands.add_parser("dataset-mine", help="Mine SFT/DPO/persona examples")
    _add_dataset_choice(mine)
    mine.add_argument("--limit", type=int)
    mine.add_argument("--time-budget", type=float)
    mine.add_argument("--verified-only", action="store_true")
    mine.set_defaults(func=cmd_mine)

    curate = commands.add_parser("dataset-persona-curate")
    curate.add_argument("--input", type=Path, required=True)
    curate.set_defaults(func=cmd_persona_curate)

    calibration = commands.add_parser("dataset-calibration-import")
    calibration.add_argument("--input", type=Path, required=True)
    calibration.set_defaults(func=cmd_calibration_import)

    grade = commands.add_parser("dataset-grade", help="Grade with a loopback-only local model")
    _add_dataset_choice(grade)
    grade.add_argument("--limit", type=int)
    grade.add_argument("--endpoint")
    grade.add_argument("--model")
    grade.add_argument("--source-uri-prefix")
    grade.add_argument("--force", action="store_true")
    grade.add_argument(
        "--train-class",
        action="append",
        dest="train_classes",
        choices=[
            "train_voice",
            "train_judgment",
            "train_skill",
            "retrieval_only",
            "exclude",
        ],
    )
    grade.add_argument("--selected-only", action="store_true")
    grade.set_defaults(func=cmd_grade)

    classify = commands.add_parser("dataset-classify")
    classify.add_argument("--limit", type=int)
    classify.add_argument("--force", action="store_true")
    classify.set_defaults(func=cmd_classify)

    select = commands.add_parser("dataset-pack-select")
    _add_pack_args(select, defaults=(2000, 300, 500))
    select.add_argument("--seed", default="ocbrain-v04-selected-pack-v1")
    select.set_defaults(func=cmd_pack_select)

    finalize = commands.add_parser("dataset-pack-finalize")
    _add_pack_args(finalize, defaults=(1000, 200, 300))
    finalize.add_argument("--min-grade", type=float, default=0.8)
    finalize.set_defaults(func=cmd_pack_finalize)

    pack_stats = commands.add_parser("dataset-pack-stats")
    pack_stats.add_argument("--min-grade", type=float, default=0.8)
    pack_stats.set_defaults(func=cmd_pack_stats)

    export = commands.add_parser("dataset-export")
    _add_dataset_choice(export)
    export.add_argument("--min-scope")
    export.add_argument("--min-label")
    export.add_argument("--min-grade", type=float)
    export.add_argument("--output-dir", type=Path)
    export.add_argument("--verified-only", action="store_true")
    export.set_defaults(func=cmd_export)

    stats = commands.add_parser("dataset-stats")
    stats.set_defaults(func=cmd_stats)

    pilot = commands.add_parser("dataset-pilot-prepare")
    pilot.add_argument("--output-dir", type=Path)
    pilot.add_argument("--min-grade", type=float)
    pilot.add_argument("--eval-prompts", type=int, default=100)
    pilot.add_argument("--seed", default="ocbrain-voice-pilot-v3")
    pilot.add_argument("--base-model")
    pilot.add_argument("--base-model-source")
    pilot.add_argument("--base-model-revision")
    pilot.add_argument("--eval-from", type=Path)
    pilot.add_argument("--legacy-sentinel-from", type=Path)
    pilot.add_argument("--diagnostic-small-pack", action="store_true")
    pilot.add_argument("--training-iterations", type=int, default=25)
    pilot.set_defaults(func=cmd_pilot_prepare)

    blind = commands.add_parser("dataset-pilot-blind")
    blind.add_argument("--pilot-dir", type=Path, required=True)
    blind.add_argument("--candidate-responses", type=Path, required=True)
    blind.add_argument("--seed", default="ocbrain-blind-v1")
    blind.set_defaults(func=cmd_pilot_blind)

    score = commands.add_parser("dataset-pilot-score")
    score.add_argument("--pilot-dir", type=Path, required=True)
    score.add_argument("--ratings", type=Path, required=True)
    score.set_defaults(func=cmd_pilot_score)

    multiblind = commands.add_parser("dataset-pilot-multiblind")
    multiblind.add_argument("--pilot-dir", type=Path, required=True)
    multiblind.add_argument("--response", action="append", required=True)
    multiblind.add_argument("--seed", default="ocbrain-multiblind-v1")
    multiblind.set_defaults(func=cmd_pilot_multiblind)

    multiscore = commands.add_parser("dataset-pilot-multiscore")
    multiscore.add_argument("--pilot-dir", type=Path, required=True)
    multiscore.add_argument("--ratings", type=Path, required=True)
    multiscore.set_defaults(func=cmd_pilot_multiscore)

    record = commands.add_parser("dataset-pilot-record-training")
    record.add_argument("--pilot-dir", type=Path, required=True)
    record.add_argument("--iterations", type=int, required=True)
    record.add_argument("--train-loss", type=float, required=True)
    record.add_argument("--validation-loss", type=float, required=True)
    record.add_argument("--exit-code", type=int, required=True)
    record.set_defaults(func=cmd_pilot_record)

    benchmark = commands.add_parser("retrieval-benchmark")
    benchmark.add_argument("--input", type=Path, required=True)
    benchmark.add_argument("--allow-small", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    expand = commands.add_parser("retrieval-benchmark-expand")
    expand.add_argument("--input", type=Path, required=True)
    expand.add_argument("--output", type=Path, required=True)
    expand.set_defaults(func=cmd_benchmark_expand)
    return parser


def _add_pack_args(parser: argparse.ArgumentParser, *, defaults: tuple[int, int, int]) -> None:
    parser.add_argument("--sft", type=int, default=defaults[0])
    parser.add_argument("--dpo", type=int, default=defaults[1])
    parser.add_argument("--persona", type=int, default=defaults[2])


def _output(args: argparse.Namespace, value: Any) -> None:
    print(json.dumps(value, indent=2 if args.pretty else None, sort_keys=True))


def _open(args: argparse.Namespace):
    return connect_training(args.training_db)


def _open_core_read_only(args: argparse.Namespace) -> sqlite3.Connection:
    path = args.core_db.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"core database not found: {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        build_parser().print_help()
        return 2
    return int(args.func(args))


def legacy_dispatch(
    argv: list[str] | None = None,
    *,
    db: Path | str | None = None,
    pretty: bool = False,
) -> int:
    """Adapter used by the core's lazy exact-command entry-point dispatcher."""
    values = list(sys.argv[1:] if argv is None else argv)
    # ``db`` is the core dispatcher's database and is intentionally ignored:
    # the training companion must never silently write the brain database.
    del db
    global_values, command_values = _extract_global_options(
        values,
        value_options={"--training-db", "--core-db"},
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


def cmd_mine(args: argparse.Namespace) -> int:
    if args.dataset not in {"sft", "persona"}:
        _output(
            args,
            {
                "action": "dataset-mine",
                "changed": 0,
                "status": "blocked",
                "reason": "event_source_adapter_not_migrated",
                "detail": (
                    "v1 training mining currently supports explicit --dataset sft or persona; "
                    "DPO/all-source mining needs the read-only core source adapter"
                ),
            },
        )
        return 2
    conn = _open(args)
    cfg = load_config()
    roots = list(cfg.review.session_roots)
    kwargs = {
        "cfg": cfg,
        "roots": roots,
        "limit": args.limit,
        "time_budget_seconds": args.time_budget,
    }
    if args.dataset == "sft":
        result = mine_sft(conn, **kwargs)
    else:
        result = mine_persona(conn, verified_only=args.verified_only, **kwargs)
    conn.commit()
    if cfg.autopilot.checkpoint_after_dataset_mine:
        result["wal_checkpoint"] = checkpoint_sqlite_wal(
            conn,
            args.training_db,
            minimum_bytes=cfg.autopilot.checkpoint_wal_min_bytes,
        )
    _output(args, result)
    return 0


def cmd_persona_curate(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = import_persona_curation(conn, args.input, cfg=load_config())
    conn.commit()
    _output(args, result)
    return 0


def cmd_calibration_import(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = import_calibrations(conn, args.input)
    conn.commit()
    _output(args, result)
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = grade_examples(
        conn,
        cfg=load_config(),
        datasets=[args.dataset] if args.dataset else None,
        limit=args.limit,
        endpoint=args.endpoint,
        model=args.model,
        force=args.force,
        source_uri_prefix=args.source_uri_prefix,
        train_classes=args.train_classes,
        selected_only=args.selected_only,
    )
    conn.commit()
    _output(args, result)
    return 1 if result.get("status") in {"error", "blocked", "locked"} else 0


def cmd_classify(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = classify_examples(conn, force=args.force, limit=args.limit)
    conn.commit()
    _output(args, result)
    return 0


def _targets(args: argparse.Namespace) -> dict[str, int]:
    return {"sft": args.sft, "dpo": args.dpo, "persona": args.persona}


def cmd_pack_select(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = select_training_pack(conn, targets=_targets(args), seed=args.seed)
    conn.commit()
    _output(args, result)
    return 0


def cmd_pack_finalize(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = finalize_training_pack(conn, targets=_targets(args), min_grade=args.min_grade)
    conn.commit()
    _output(args, result)
    return 0


def cmd_pack_stats(args: argparse.Namespace) -> int:
    conn = _open(args)
    _output(args, selected_pack_stats(conn, min_grade=args.min_grade))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    conn = _open(args)
    result = export_all(
        conn,
        cfg=load_config(),
        datasets=[args.dataset] if args.dataset else None,
        min_scope=args.min_scope,
        min_label=args.min_label,
        min_grade=args.min_grade,
        verified_only=args.verified_only,
        export_dir=args.output_dir,
    )
    conn.commit()
    _output(args, result)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    _output(args, dataset_stats(_open(args)))
    return 0


def _training_enabled(args: argparse.Namespace, action: str) -> bool:
    if load_config().dataset.training_enabled:
        return True
    _output(
        args,
        {
            "action": action,
            "changed": 0,
            "reason": "dataset_training_disabled_by_default",
            "status": "blocked",
        },
    )
    return False


def cmd_pilot_prepare(args: argparse.Namespace) -> int:
    if not _training_enabled(args, "dataset-pilot-prepare"):
        return 2
    conn = _open(args)
    try:
        result = prepare_pilot(
            conn,
            cfg=load_config(),
            output_dir=args.output_dir,
            min_grade=args.min_grade,
            eval_prompts=args.eval_prompts,
            seed=args.seed,
            base_model=args.base_model,
            base_model_source=args.base_model_source,
            base_model_revision=args.base_model_revision,
            eval_from=args.eval_from,
            training_iterations=args.training_iterations,
            quality_gates=not args.diagnostic_small_pack,
            sentinel_from=args.legacy_sentinel_from,
        )
    except RuntimeError as exc:
        _output(args, {"action": "dataset-pilot-prepare", "status": "blocked", "error": str(exc)})
        return 1
    _output(args, result)
    return 0


def cmd_pilot_blind(args: argparse.Namespace) -> int:
    _output(args, prepare_blind_pairs(args.pilot_dir, args.candidate_responses, seed=args.seed))
    return 0


def cmd_pilot_score(args: argparse.Namespace) -> int:
    _output(args, score_blind_ratings(args.pilot_dir, args.ratings))
    return 0


def cmd_pilot_multiblind(args: argparse.Namespace) -> int:
    response_sets: dict[str, Path] = {}
    for raw in args.response:
        name, separator, value = str(raw).partition("=")
        if not separator or name not in {"base", "tuned", "frontier"} or not value:
            raise ValueError("--response must be base=PATH, tuned=PATH, or frontier=PATH")
        response_sets[name] = Path(value).expanduser()
    _output(args, prepare_multiblind(args.pilot_dir, response_sets, seed=args.seed))
    return 0


def cmd_pilot_multiscore(args: argparse.Namespace) -> int:
    _output(args, score_multiblind(args.pilot_dir, args.ratings))
    return 0


def cmd_pilot_record(args: argparse.Namespace) -> int:
    if not _training_enabled(args, "dataset-pilot-record-training"):
        return 2
    _output(
        args,
        record_training_result(
            args.pilot_dir,
            iterations=args.iterations,
            train_loss=args.train_loss,
            validation_loss=args.validation_loss,
            exit_code=args.exit_code,
        ),
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    result = run_benchmark(
        _open_core_read_only(args),
        args.input,
        require_cases=1 if args.allow_small else 100,
    )
    _output(args, result)
    return 0


def cmd_benchmark_expand(args: argparse.Namespace) -> int:
    _output(args, expand_runtime_matrix(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
