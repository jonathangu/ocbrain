from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocbrain.autolabel import Signal, record_signal
from ocbrain.autopilot import run_autopilot
from ocbrain.config import load_config
from ocbrain.dataset import mine_all
from ocbrain.dataset.export import export_all
from ocbrain.dataset.stats import dataset_stats
from ocbrain.db import (
    DEFAULT_DB_PATH,
    PUBLIC_SCOPES,
    connect,
    counts,
    get_knowledge,
    init_db,
    knowledge_digest,
    link_knowledge_evidence,
    list_knowledge,
    mark_knowledge_stale,
    now_iso,
    search,
    upsert_evidence,
    upsert_knowledge,
    upsert_search_index,
)
from ocbrain.dream import dream
from ocbrain.egress import egress_preview
from ocbrain.events import (
    decide_compilation,
    event_core_digest,
    evidence_id_for,
    list_compilation_proposals,
    propose_compilation,
    rebuild_projection,
    record_correction,
    record_evidence,
    record_tombstone,
)
from ocbrain.fsutil import file_fingerprint, history_runtime
from ocbrain.ids import content_hash, stable_id
from ocbrain.loops import LoopIngestOptions, dry_run_loop_ingest, write_loop_ingest
from ocbrain.maintenance import check_loop_liveness, heal_conflicts, prune_knowledge
from ocbrain.mcp import serve
from ocbrain.retrieve import retrieve
from ocbrain.safeguards import release_quarantine
from ocbrain.scope import ScopeContext, ScopeTag, global_scope, resolve_write_scope
from ocbrain.teacher import hosted_teacher_request
from ocbrain.text import compact_whitespace, redact_secrets, title_from_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocbrain", description="OCBrain final-spec brain")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize the SQLite ledger")
    init_parser.set_defaults(func=cmd_init)

    evidence_parser = subparsers.add_parser("evidence", help="Append immutable evidence")
    evidence_parser.add_argument("--claim")
    evidence_parser.add_argument("--input", type=Path)
    evidence_parser.add_argument("--source-type", default="closeout")
    evidence_parser.add_argument("--source-runtime")
    evidence_parser.add_argument("--source-uri")
    evidence_parser.add_argument("--artifact-uri")
    evidence_parser.add_argument("--artifact-hash")
    evidence_parser.add_argument("--verifier-status", default="unknown")
    evidence_parser.add_argument("--project")
    evidence_parser.add_argument("--privacy-scope", default="workspace")
    evidence_parser.set_defaults(func=cmd_evidence)

    knowledge_parser = subparsers.add_parser("knowledge", help="List knowledge rows")
    knowledge_parser.add_argument("--status")
    knowledge_parser.add_argument("--type")
    knowledge_parser.add_argument("--include-private", action="store_true")
    knowledge_parser.add_argument("--limit", type=int, default=20)
    knowledge_parser.set_defaults(func=cmd_knowledge)

    promote_parser = subparsers.add_parser("value", help="Upsert one value knowledge row")
    promote_parser.add_argument("--subject", required=True)
    promote_parser.add_argument("--predicate", required=True)
    typed_value = promote_parser.add_mutually_exclusive_group(required=True)
    typed_value.add_argument("--text")
    typed_value.add_argument("--number", type=float)
    typed_value.add_argument("--bool", choices=["true", "false"])
    promote_parser.add_argument("--unit")
    promote_parser.add_argument("--target-value", type=float)
    promote_parser.add_argument("--status", default="candidate")
    promote_parser.add_argument("--inject", action="store_true")
    promote_parser.add_argument("--confidence", type=float)
    promote_parser.add_argument("--project")
    promote_parser.add_argument("--privacy-scope", default="workspace")
    promote_parser.set_defaults(func=cmd_value)

    search_parser = subparsers.add_parser("search", help="Search evidence and knowledge")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--include-private", action="store_true")
    search_parser.add_argument("--project")
    search_parser.add_argument("--type")
    search_parser.add_argument("--status")
    search_parser.add_argument("--loop-id")
    search_parser.add_argument("--family")
    search_parser.set_defaults(func=cmd_search)

    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview the exact scoped retrieval payload from the event-sourced core",
    )
    preview_parser.add_argument("query")
    add_context_args(preview_parser)
    preview_parser.add_argument("--limit", type=int, default=12)
    preview_parser.add_argument("--cross-scope", action="store_true")
    preview_parser.add_argument("--at-ts")
    preview_parser.set_defaults(func=cmd_preview)

    ingest_parser = subparsers.add_parser(
        "event-ingest",
        help="Append scoped evidence to the event-sourced core",
    )
    ingest_parser.add_argument("--body", required=True)
    ingest_parser.add_argument("--kind", default="observation")
    ingest_parser.add_argument("--writer", default="ocbrain")
    ingest_parser.add_argument("--artifact-ref")
    add_context_args(ingest_parser)
    ingest_parser.add_argument("--global-doctrine", action="store_true")
    ingest_parser.set_defaults(func=cmd_event_ingest)

    compile_parser = subparsers.add_parser(
        "event-compile",
        help="Append and optionally approve a compiled belief event",
    )
    compile_parser.add_argument("--belief-id", required=True)
    compile_parser.add_argument("--body", required=True)
    compile_parser.add_argument("--evidence-id", action="append", default=[])
    compile_parser.add_argument("--confidence", type=float)
    compile_parser.add_argument(
        "--reward-band",
        choices=["discard", "weak", "moderate", "strong"],
    )
    compile_parser.add_argument("--approve", action="store_true")
    add_context_args(compile_parser)
    compile_parser.add_argument("--global-doctrine", action="store_true")
    compile_parser.set_defaults(func=cmd_event_compile)

    correct_parser = subparsers.add_parser(
        "event-correct",
        help="Append a durable correction and synchronously rebuild the projection",
    )
    correct_parser.add_argument(
        "--target-layer",
        choices=["evidence", "knowledge", "belief"],
        required=True,
    )
    correct_parser.add_argument("--target-id", required=True)
    correct_parser.add_argument(
        "--op",
        choices=["mark_wrong", "edit", "pin", "demote", "reframe", "retract"],
        required=True,
    )
    correct_parser.add_argument("--body")
    correct_parser.add_argument("--author", default="human:jonathan")
    correct_parser.add_argument("--hard", action="store_true")
    correct_parser.set_defaults(func=cmd_event_correct)

    forget_parser = subparsers.add_parser(
        "event-forget",
        help="Append a tombstone and synchronously rebuild the projection",
    )
    forget_parser.add_argument("--target", required=True)
    forget_parser.add_argument("--mode", choices=["soft", "shred"], default="soft")
    forget_parser.add_argument("--reason")
    forget_parser.add_argument("--approved-by", default="human:jonathan")
    forget_parser.set_defaults(func=cmd_event_forget)

    dream_parser = subparsers.add_parser(
        "event-dream",
        help="Batch scoped evidence into pending compilation proposals",
    )
    add_context_args(dream_parser)
    dream_parser.add_argument("--since-ts")
    dream_parser.add_argument("--target", default="local_model")
    dream_parser.add_argument("--record-egress", action="store_true")
    dream_parser.add_argument("--limit", type=int, default=20)
    dream_parser.set_defaults(func=cmd_event_dream)

    proposals_parser = subparsers.add_parser(
        "event-proposals",
        help="List pending or decided event-core compilation proposals",
    )
    add_context_args(proposals_parser)
    proposals_parser.add_argument("--include-decided", action="store_true")
    proposals_parser.add_argument("--limit", type=int, default=50)
    proposals_parser.set_defaults(func=cmd_event_proposals)

    decide_parser = subparsers.add_parser(
        "event-decide",
        help="Append a gate decision for one compilation proposal",
    )
    decide_parser.add_argument("--proposal-event-id", required=True)
    decide_parser.add_argument(
        "--decision",
        choices=["approve", "reject", "edit", "shadow"],
        required=True,
    )
    decide_parser.add_argument("--actor", default="human:jonathan")
    decide_parser.add_argument("--edited-body")
    decide_parser.add_argument("--reason")
    decide_parser.set_defaults(func=cmd_event_decide)

    event_digest_parser = subparsers.add_parser(
        "event-digest",
        help="Return scoped event-core digest, pending proposals, and current beliefs",
    )
    add_context_args(event_digest_parser)
    event_digest_parser.add_argument("--since-ts")
    event_digest_parser.add_argument("--limit", type=int, default=20)
    event_digest_parser.set_defaults(func=cmd_event_digest)

    egress_parser = subparsers.add_parser(
        "egress-preview",
        help="Preview scope-filtered evidence before local or hosted teacher egress",
    )
    egress_parser.add_argument("--target", default="hosted_teacher")
    egress_parser.add_argument("--query")
    egress_parser.add_argument("--record", action="store_true")
    add_context_args(egress_parser)
    egress_parser.set_defaults(func=cmd_egress_preview)

    teacher_parser = subparsers.add_parser(
        "event-teacher-request",
        help="Prepare a hosted-teacher request package without dispatch",
    )
    add_context_args(teacher_parser)
    teacher_parser.add_argument("--query")
    teacher_parser.add_argument("--objective", default="compile_scoped_beliefs")
    teacher_parser.add_argument("--model", default="hosted_teacher")
    teacher_parser.add_argument("--limit", type=int, default=20)
    teacher_parser.add_argument("--no-record", action="store_true")
    teacher_parser.set_defaults(func=cmd_event_teacher_request)

    backfill_parser = subparsers.add_parser(
        "event-backfill",
        help="Backfill current legacy knowledge into the scoped event-sourced core",
    )
    backfill_parser.add_argument("--limit", type=int, default=100)
    backfill_parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Maximum planned/imported items to include in command output",
    )
    backfill_parser.add_argument(
        "--all",
        action="store_true",
        help="Backfill all remaining matching current legacy rows in one transaction",
    )
    backfill_parser.add_argument("--project")
    backfill_parser.add_argument("--type")
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify the next legacy rows without appending event-core writes",
    )
    backfill_parser.set_defaults(func=cmd_event_backfill)

    import_memory_parser = subparsers.add_parser(
        "import-memory",
        help="Import markdown memory files as source-backed doc knowledge",
    )
    import_memory_parser.add_argument("paths", nargs="+", type=Path)
    import_memory_parser.add_argument("--project", default="workspace")
    import_memory_parser.add_argument("--privacy-scope", default="workspace")
    import_memory_parser.add_argument("--limit", type=int)
    import_memory_parser.add_argument(
        "--max-bytes",
        type=int,
        default=50_000,
        help="Maximum UTF-8 bytes to index per file",
    )
    import_memory_parser.set_defaults(func=cmd_import_memory)

    import_history_parser = subparsers.add_parser(
        "import-history",
        help="Import runtime transcript/history files as source-backed doc knowledge",
    )
    import_history_parser.add_argument("paths", nargs="*", type=Path)
    import_history_parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Newline-delimited file containing history file paths",
    )
    import_history_parser.add_argument("--project", default="workspace")
    import_history_parser.add_argument("--privacy-scope", default="workspace")
    import_history_parser.add_argument("--limit", type=int)
    import_history_parser.add_argument(
        "--max-bytes",
        type=int,
        default=20_000,
        help="Maximum UTF-8 bytes to index per history file",
    )
    import_history_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Commit after this many imported files",
    )
    import_history_parser.set_defaults(func=cmd_import_history)

    digest_parser = subparsers.add_parser("digest", help="Show current knowledge digest")
    digest_parser.add_argument("--project")
    digest_parser.add_argument("--limit", type=int, default=12)
    digest_parser.add_argument("--include-private", action="store_true")
    digest_parser.set_defaults(func=cmd_digest)

    loop_ingest_parser = subparsers.add_parser(
        "loop-ingest", help="Dry-run or apply loop result envelopes as evidence/knowledge"
    )
    loop_ingest_parser.add_argument("--loop-id", required=True)
    loop_ingest_parser.add_argument("--run-id", required=True)
    loop_ingest_parser.add_argument("--artifacts", required=True, type=Path)
    loop_ingest_parser.add_argument("--ledger", type=Path)
    loop_ingest_parser.add_argument("--backlog", type=Path)
    loop_mode = loop_ingest_parser.add_mutually_exclusive_group()
    loop_mode.add_argument("--dry-run", action="store_true")
    loop_mode.add_argument("--apply", action="store_true")
    loop_ingest_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    loop_ingest_parser.set_defaults(func=cmd_loop_ingest)

    stale_parser = subparsers.add_parser("mark-stale", help="Mark knowledge stale")
    stale_parser.add_argument("knowledge_id")
    stale_parser.add_argument("--reason", default="user_request")
    stale_parser.set_defaults(func=cmd_mark_stale)

    prune_parser = subparsers.add_parser(
        "prune", help="Mark unreferenced expired knowledge stale or archived"
    )
    prune_parser.add_argument("--ttl-days", type=int, default=30)
    prune_parser.add_argument("--unhelpful-ttl-days", type=int, default=14)
    prune_parser.add_argument("--archive-stale-days", type=int)
    prune_parser.set_defaults(func=cmd_prune)

    heal_parser = subparsers.add_parser(
        "heal", help="Supersede conflicting current value knowledge"
    )
    heal_parser.add_argument("--numeric-threshold", type=float, default=0.0)
    heal_parser.set_defaults(func=cmd_heal)

    liveness_parser = subparsers.add_parser(
        "liveness-check", help="Open loop liveness tripwires from runner deadman rows"
    )
    liveness_parser.add_argument("--runner-ledger", type=Path)
    liveness_parser.set_defaults(func=cmd_liveness_check)

    mcp_parser = subparsers.add_parser("mcp", help="Run stdio MCP server")
    mcp_parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Deprecated no-op flag retained for back-compat (spec §5.1-7)",
    )
    mcp_parser.set_defaults(func=cmd_mcp)

    # --- v0.2 autonomy + dataset factory (spec §8) --------------------------
    autopilot_parser = subparsers.add_parser(
        "autopilot", help="Run the autonomy pipeline (harvest→…→dataset-export)"
    )
    autopilot_select = autopilot_parser.add_mutually_exclusive_group()
    autopilot_select.add_argument(
        "--stage",
        action="append",
        dest="stages",
        help="Run only this stage (repeatable); default runs all stages",
    )
    autopilot_select.add_argument(
        "--profile",
        dest="profile",
        help=(
            "Run a named stage profile from cfg.autopilot.profiles "
            "(e.g. 'light' every 15 min, 'heavy' hourly); embed runs after autolabel"
        ),
    )
    autopilot_parser.add_argument("--dry-run", action="store_true")
    autopilot_parser.set_defaults(func=cmd_autopilot)

    quarantine_parser = subparsers.add_parser(
        "quarantine", help="List or release quarantined knowledge"
    )
    quarantine_sub = quarantine_parser.add_subparsers(dest="quarantine_command")
    q_list = quarantine_sub.add_parser("list", help="List quarantined knowledge rows")
    q_list.add_argument("--limit", type=int, default=100)
    q_list.set_defaults(func=cmd_quarantine_list)
    q_release = quarantine_sub.add_parser("release", help="Release a quarantined row")
    q_release.add_argument("knowledge_id")
    q_release.add_argument("--actor", required=True)
    q_release.add_argument("--reason", required=True)
    q_release.set_defaults(func=cmd_quarantine_release)
    quarantine_parser.set_defaults(func=cmd_quarantine_list)

    label_parser = subparsers.add_parser(
        "label", help="Record a manual good/bad quality signal on a knowledge row"
    )
    label_parser.add_argument("knowledge_id")
    label_parser.add_argument("--outcome", choices=["good", "bad"], required=True)
    label_parser.add_argument("--note", default="")
    label_parser.set_defaults(func=cmd_label)

    dataset_mine_parser = subparsers.add_parser(
        "dataset-mine", help="Mine SFT/DPO/persona examples from transcripts"
    )
    dataset_mine_parser.add_argument("--dataset", choices=["sft", "dpo", "persona"])
    dataset_mine_parser.add_argument("--limit", type=int)
    dataset_mine_parser.add_argument("--time-budget", type=float, dest="time_budget")
    dataset_mine_parser.add_argument("--verified-only", action="store_true")
    dataset_mine_parser.set_defaults(func=cmd_dataset_mine)

    dataset_curate_parser = subparsers.add_parser(
        "dataset-persona-curate",
        help="Import explicit private persona prompt/response JSONL",
    )
    dataset_curate_parser.add_argument("--input", type=Path, required=True)
    dataset_curate_parser.set_defaults(func=cmd_dataset_persona_curate)

    dataset_calibration_parser = subparsers.add_parser(
        "dataset-calibration-import",
        help="Import private human preferences with reasons and ideal corrections",
    )
    dataset_calibration_parser.add_argument("--input", type=Path, required=True)
    dataset_calibration_parser.set_defaults(func=cmd_dataset_calibration_import)

    dataset_grade_parser = subparsers.add_parser(
        "dataset-grade", help="Grade examples with a loopback-only local LLM"
    )
    dataset_grade_parser.add_argument("--dataset", choices=["sft", "dpo", "persona"])
    dataset_grade_parser.add_argument("--limit", type=int)
    dataset_grade_parser.add_argument("--endpoint")
    dataset_grade_parser.add_argument("--model")
    dataset_grade_parser.add_argument(
        "--source-uri-prefix",
        help="Grade only examples whose local provenance URI begins with this value",
    )
    dataset_grade_parser.add_argument("--force", action="store_true")
    dataset_grade_parser.add_argument(
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
        help="Grade only examples in one or more weights/retrieval classes",
    )
    dataset_grade_parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Grade only the deterministic v0.4 selected training pack",
    )
    dataset_grade_parser.set_defaults(func=cmd_dataset_grade)

    dataset_classify_parser = subparsers.add_parser(
        "dataset-classify",
        help="Classify examples as weights training, retrieval-only, or excluded",
    )
    dataset_classify_parser.add_argument("--limit", type=int)
    dataset_classify_parser.add_argument("--force", action="store_true")
    dataset_classify_parser.set_defaults(func=cmd_dataset_classify)

    dataset_pack_select_parser = subparsers.add_parser(
        "dataset-pack-select",
        help="Select the deterministic local v0.4 training pack",
    )
    dataset_pack_select_parser.add_argument("--sft", type=int, default=2000)
    dataset_pack_select_parser.add_argument("--dpo", type=int, default=300)
    dataset_pack_select_parser.add_argument("--persona", type=int, default=500)
    dataset_pack_select_parser.add_argument("--seed", default="ocbrain-v04-selected-pack-v1")
    dataset_pack_select_parser.set_defaults(func=cmd_dataset_pack_select)

    dataset_pack_stats_parser = subparsers.add_parser(
        "dataset-pack-stats",
        help="Report selected-pack local grade coverage and passing counts",
    )
    dataset_pack_stats_parser.add_argument("--min-grade", type=float, default=0.8)
    dataset_pack_stats_parser.set_defaults(func=cmd_dataset_pack_stats)

    feedback_stats_parser = subparsers.add_parser(
        "retrieval-feedback-stats",
        help="Report explicit/inferred retrieval feedback coverage",
    )
    feedback_stats_parser.set_defaults(func=cmd_retrieval_feedback_stats)

    retrieval_benchmark_parser = subparsers.add_parser(
        "retrieval-benchmark",
        help="Run a frozen, scope-aware retrieval benchmark without returning corpus text",
    )
    retrieval_benchmark_parser.add_argument("--input", type=Path, required=True)
    retrieval_benchmark_parser.add_argument(
        "--allow-small", action="store_true", help="Allow fewer than 100 cases for diagnostics"
    )
    retrieval_benchmark_parser.set_defaults(func=cmd_retrieval_benchmark)

    retrieval_benchmark_expand_parser = subparsers.add_parser(
        "retrieval-benchmark-expand",
        help="Expand a private 25-case base across four supported runtimes",
    )
    retrieval_benchmark_expand_parser.add_argument("--input", type=Path, required=True)
    retrieval_benchmark_expand_parser.add_argument("--output", type=Path, required=True)
    retrieval_benchmark_expand_parser.set_defaults(func=cmd_retrieval_benchmark_expand)

    dataset_export_parser = subparsers.add_parser(
        "dataset-export", help="Export deterministic JSONL datasets + manifest"
    )
    dataset_export_parser.add_argument("--dataset", choices=["sft", "dpo", "persona"])
    dataset_export_parser.add_argument("--min-scope", dest="min_scope")
    dataset_export_parser.add_argument("--min-label", dest="min_label")
    dataset_export_parser.add_argument("--min-grade", type=float, dest="min_grade")
    dataset_export_parser.add_argument("--output-dir", type=Path, dest="output_dir")
    dataset_export_parser.add_argument("--verified-only", action="store_true")
    dataset_export_parser.set_defaults(func=cmd_dataset_export)

    dataset_stats_parser = subparsers.add_parser(
        "dataset-stats", help="Report dataset growth by label/scope/source/week"
    )
    dataset_stats_parser.set_defaults(func=cmd_dataset_stats)

    pilot_prepare_parser = subparsers.add_parser(
        "dataset-pilot-prepare", help="Build the eval-first local fine-tune pilot pack"
    )
    pilot_prepare_parser.add_argument("--output-dir", type=Path)
    pilot_prepare_parser.add_argument("--min-grade", type=float)
    pilot_prepare_parser.add_argument("--eval-prompts", type=int, default=100)
    pilot_prepare_parser.add_argument("--seed", default="ocbrain-voice-pilot-v3")
    pilot_prepare_parser.add_argument("--base-model")
    pilot_prepare_parser.add_argument("--base-model-source")
    pilot_prepare_parser.add_argument("--base-model-revision")
    pilot_prepare_parser.add_argument(
        "--eval-from",
        type=Path,
        help="Reuse a prior pilot's prompts/references/rubric byte-for-byte",
    )
    pilot_prepare_parser.add_argument(
        "--legacy-sentinel-from",
        type=Path,
        help="Preserve a prior frozen eval separately and exclude all its sources from train",
    )
    pilot_prepare_parser.add_argument(
        "--diagnostic-small-pack",
        action="store_true",
        help="Disable v0.4 corpus-size/train-class gates for diagnostics only",
    )
    pilot_prepare_parser.add_argument("--training-iterations", type=int, default=25)
    pilot_prepare_parser.set_defaults(func=cmd_dataset_pilot_prepare)

    pilot_blind_parser = subparsers.add_parser(
        "dataset-pilot-blind", help="Randomize reference/model answers for blind scoring"
    )
    pilot_blind_parser.add_argument("--pilot-dir", type=Path, required=True)
    pilot_blind_parser.add_argument("--candidate-responses", type=Path, required=True)
    pilot_blind_parser.add_argument("--seed", default="ocbrain-blind-v1")
    pilot_blind_parser.set_defaults(func=cmd_dataset_pilot_blind)

    pilot_score_parser = subparsers.add_parser(
        "dataset-pilot-score", help="Score completed blind voice/taste ratings"
    )
    pilot_score_parser.add_argument("--pilot-dir", type=Path, required=True)
    pilot_score_parser.add_argument("--ratings", type=Path, required=True)
    pilot_score_parser.set_defaults(func=cmd_dataset_pilot_score)

    pilot_multiblind_parser = subparsers.add_parser(
        "dataset-pilot-multiblind",
        help="Build a blinded Jonathan/base/tuned/frontier evaluation pack",
    )
    pilot_multiblind_parser.add_argument("--pilot-dir", type=Path, required=True)
    pilot_multiblind_parser.add_argument(
        "--response",
        action="append",
        required=True,
        help="One of base=/path, tuned=/path, frontier=/path; repeat three times",
    )
    pilot_multiblind_parser.add_argument("--seed", default="ocbrain-multiblind-v1")
    pilot_multiblind_parser.set_defaults(func=cmd_dataset_pilot_multiblind)

    pilot_multiscore_parser = subparsers.add_parser(
        "dataset-pilot-multiscore", help="Score completed four-way blind rankings"
    )
    pilot_multiscore_parser.add_argument("--pilot-dir", type=Path, required=True)
    pilot_multiscore_parser.add_argument("--ratings", type=Path, required=True)
    pilot_multiscore_parser.set_defaults(func=cmd_dataset_pilot_multiscore)

    pilot_record_parser = subparsers.add_parser(
        "dataset-pilot-record-training", help="Record verified local adapter evidence"
    )
    pilot_record_parser.add_argument("--pilot-dir", type=Path, required=True)
    pilot_record_parser.add_argument("--iterations", type=int, required=True)
    pilot_record_parser.add_argument("--train-loss", type=float, required=True)
    pilot_record_parser.add_argument("--validation-loss", type=float, required=True)
    pilot_record_parser.add_argument("--exit-code", type=int, required=True)
    pilot_record_parser.set_defaults(func=cmd_dataset_pilot_record_training)

    # --- public-safety enforcement (tracked-tree scanner + hooks) ----------
    public_safety_parser = subparsers.add_parser(
        "public-safety-check",
        help="Scan the tracked tree for private data before it reaches the public repo",
    )
    public_safety_parser.add_argument(
        "--diff-range",
        help="git range (e.g. origin/main..HEAD) to scan added lines for new secrets",
    )
    public_safety_parser.add_argument(
        "--root", type=Path, help="repo root (default: git toplevel of the cwd)"
    )
    public_safety_parser.add_argument("--json", action="store_true", help="Emit JSON output")
    public_safety_parser.set_defaults(func=cmd_public_safety_check)

    install_hooks_parser = subparsers.add_parser(
        "install-hooks", help="Symlink tracked git hooks (ops/hooks) into .git/hooks"
    )
    install_hooks_parser.add_argument(
        "--root", type=Path, help="repo root (default: git toplevel of the cwd)"
    )
    install_hooks_parser.set_defaults(func=cmd_install_hooks)

    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None and Path(sys.argv[0]).name == "brain-loop-ingest":
        argv = ["loop-ingest", *sys.argv[1:]]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input and args.command is None:
        args.command = "evidence"
        args.func = cmd_evidence
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


def output(args: argparse.Namespace, payload) -> None:
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project")
    parser.add_argument("--repo")
    parser.add_argument("--client")
    parser.add_argument("--task")
    parser.add_argument("--session")
    parser.add_argument("--runtime")


def context_from_args(args: argparse.Namespace) -> ScopeContext:
    return ScopeContext(
        project=getattr(args, "project", None),
        repo=getattr(args, "repo", None),
        client=getattr(args, "client", None),
        task=getattr(args, "task", None),
        session=getattr(args, "session", None),
        runtime=getattr(args, "runtime", None),
    )


def open_db(args: argparse.Namespace):
    conn = connect(args.db)
    init_db(conn)
    return conn


def cmd_init(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(args, {"db": str(args.db), "counts": counts(conn)})
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    conn = open_db(args)
    claim, raw = evidence_claim(args)
    source_uri = args.source_uri or (str(args.input) if args.input else None)
    evidence_id = upsert_evidence(
        conn,
        source_type=args.source_type,
        source_runtime=args.source_runtime,
        source_uri=source_uri,
        content_hash=content_hash(raw),
        claim=claim,
        artifact_uri=args.artifact_uri,
        artifact_hash=args.artifact_hash,
        verifier_status=args.verifier_status,
        project=args.project,
        privacy_scope=args.privacy_scope,
    )
    conn.commit()
    output(args, {"evidence_id": evidence_id, "counts": counts(conn)})
    return 0


def evidence_claim(args: argparse.Namespace) -> tuple[str, str]:
    if args.claim:
        return compact_whitespace(args.claim), args.claim
    if args.input:
        text = args.input.read_text(encoding="utf-8", errors="replace")
        return compact_whitespace(text[:1200]), text
    raise ValueError("pass --claim or --input")


def cmd_knowledge(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scopes = None if args.include_private else PUBLIC_SCOPES
    rows = [
        dict(row)
        for row in list_knowledge(
            conn,
            status=args.status,
            knowledge_type=args.type,
            scopes=scopes,
            limit=args.limit,
        )
    ]
    output(args, {"knowledge": rows})
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    conn = open_db(args)
    value_bool = None
    if args.bool is not None:
        value_bool = args.bool == "true"
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="value",
        gate="auto",
        subject=args.subject,
        predicate=args.predicate,
        value_numeric=args.number,
        value_text=args.text,
        value_bool=value_bool,
        unit=args.unit,
        target_value=args.target_value,
        status=args.status,
        inject=args.inject,
        confidence=args.confidence,
        content_hash=content_hash(
            json.dumps(
                {
                    "subject": args.subject,
                    "predicate": args.predicate,
                    "number": args.number,
                    "text": args.text,
                    "bool": value_bool,
                },
                sort_keys=True,
            )
        ),
        project=args.project,
        privacy_scope=args.privacy_scope,
    )
    conn.commit()
    output(args, {"knowledge_id": knowledge_id, "counts": counts(conn)})
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scopes = None if args.include_private else PUBLIC_SCOPES
    filters = {
        key: value
        for key, value in {
            "project": args.project,
            "type": args.type,
            "status": args.status,
            "loop_id": args.loop_id,
            "family": args.family,
        }.items()
        if value
    }
    rows = [
        dict(row) for row in search(conn, args.query, args.limit, scopes=scopes, filters=filters)
    ]
    output(args, {"query": args.query, "filters": filters, "results": rows})
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(
        args,
        retrieve(
            conn,
            args.query,
            context=context_from_args(args),
            limit=args.limit,
            cross_scope=args.cross_scope,
            at_ts=args.at_ts,
        ),
    )
    return 0


def cmd_event_ingest(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scope = global_scope() if args.global_doctrine else resolve_write_scope(context_from_args(args))
    event_id = record_evidence(
        conn,
        body=args.body,
        kind=args.kind,
        context=context_from_args(args),
        scope=scope,
        writer=args.writer,
        session_id=args.session,
        artifact_ref=args.artifact_ref,
    )
    conn.commit()
    output(args, {"event_id": event_id, "scope": scope.to_dict(), "counts": counts(conn)})
    return 0


def cmd_event_compile(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scope = global_scope() if args.global_doctrine else resolve_write_scope(context_from_args(args))
    proposal_id = propose_compilation(
        conn,
        belief_id=args.belief_id,
        body=args.body,
        evidence_ids=args.evidence_id,
        scope=scope,
        confidence=args.confidence,
        reward_band=args.reward_band,
        session_id=args.session,
    )
    decision_id = None
    if args.approve:
        decision_id = decide_compilation(conn, proposal_event_id=proposal_id, decision="approve")
    else:
        rebuild_projection(conn)
    conn.commit()
    output(
        args,
        {
            "proposal_event_id": proposal_id,
            "decision_event_id": decision_id,
            "scope": scope.to_dict(),
            "counts": counts(conn),
        },
    )
    return 0


def cmd_event_correct(args: argparse.Namespace) -> int:
    conn = open_db(args)
    event_id = record_correction(
        conn,
        target_layer=args.target_layer,
        target_id=args.target_id,
        op=args.op,
        body=args.body,
        author=args.author,
        hard=args.hard,
    )
    conn.commit()
    output(args, {"event_id": event_id, "kind": "correction_recorded", "counts": counts(conn)})
    return 0


def cmd_event_forget(args: argparse.Namespace) -> int:
    conn = open_db(args)
    event_id = record_tombstone(
        conn,
        target=args.target,
        mode=args.mode,
        reason=args.reason,
        approved_by=args.approved_by,
    )
    conn.commit()
    output(args, {"event_id": event_id, "kind": "tombstone_recorded", "counts": counts(conn)})
    return 0


def cmd_event_dream(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = dream(
        conn,
        context=context_from_args(args),
        since_ts=args.since_ts,
        target=args.target,
        record_egress=args.record_egress,
        limit=args.limit,
    )
    conn.commit()
    output(args, result)
    return 0


def cmd_event_proposals(args: argparse.Namespace) -> int:
    conn = open_db(args)
    context = context_from_args(args)
    proposals = list_compilation_proposals(
        conn,
        context=context,
        include_decided=args.include_decided,
        limit=args.limit,
    )
    output(args, {"proposals": proposals})
    return 0


def cmd_event_decide(args: argparse.Namespace) -> int:
    conn = open_db(args)
    event_id = decide_compilation(
        conn,
        proposal_event_id=args.proposal_event_id,
        decision=args.decision,
        actor=args.actor,
        edited_body=args.edited_body,
        reason=args.reason,
    )
    conn.commit()
    output(args, {"event_id": event_id, "decision": args.decision, "counts": counts(conn)})
    return 0


def cmd_event_digest(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(
        args,
        event_core_digest(
            conn,
            context=context_from_args(args),
            since_ts=args.since_ts,
            limit=args.limit,
        ),
    )
    return 0


def cmd_egress_preview(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = egress_preview(
        conn,
        context=context_from_args(args),
        target=args.target,
        query=args.query,
        record=args.record,
    )
    if args.record:
        conn.commit()
    output(args, result)
    return 0


def cmd_event_teacher_request(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = hosted_teacher_request(
        conn,
        context=context_from_args(args),
        query=args.query,
        objective=args.objective,
        model=args.model,
        limit=args.limit,
        record=not args.no_record,
    )
    if not args.no_record:
        conn.commit()
    output(args, result)
    return 0


def cmd_event_backfill(args: argparse.Namespace) -> int:
    conn = open_db(args)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    rebuild_projection(conn)
    limit = None if args.all else args.limit
    rows = legacy_rows_for_backfill(
        conn,
        limit=limit,
        project=args.project,
        knowledge_type=args.type,
    )
    planned = [legacy_backfill_plan_item(row) for row in rows]
    if args.dry_run:
        output(
            args,
            {
                "dry_run": True,
                "would_import": len(planned),
                "scope_counts": scope_counts(planned),
                "items": planned[: args.sample_limit],
                "items_sampled": len(planned) > args.sample_limit,
                "counts": counts(conn),
            },
        )
        return 0
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for row, plan_item in zip(rows, planned, strict=True):
        belief_id = f"legacy:{row['id']}"
        if current_belief_exists(conn, belief_id):
            skipped.append({"knowledge_id": row["id"], "reason": "already_projected"})
            continue
        scope = scope_from_legacy_row(row)
        body = legacy_row_body(row)
        artifact_ref = row["body_uri"] or row["id"]
        kind = f"legacy_{row['type']}"
        source_evidence_id = evidence_id_for(
            body=body,
            kind=kind,
            artifact_ref=artifact_ref,
            scope=scope,
        )
        evidence_event_id = record_evidence(
            conn,
            body=body,
            kind=kind,
            scope=scope,
            writer="ocbrain-backfill",
            artifact_ref=artifact_ref,
        )
        proposal_id = propose_compilation(
            conn,
            belief_id=belief_id,
            body=body,
            evidence_ids=[source_evidence_id],
            scope=scope,
            confidence=row["confidence"],
            writer="ocbrain-backfill",
            check_hard_block=False,
        )
        decision_id = decide_compilation(
            conn,
            proposal_event_id=proposal_id,
            decision="approve",
            actor="ocbrain-backfill",
            rebuild=False,
            check_existing=False,
        )
        imported.append(
            {
                "knowledge_id": row["id"],
                "belief_id": belief_id,
                "scope_id": plan_item["scope_id"],
                "scope_type": plan_item["scope_type"],
                "classification_reason": plan_item["classification_reason"],
                "evidence_id": source_evidence_id,
                "evidence_event_id": evidence_event_id,
                "decision_event_id": decision_id,
            }
        )
    rebuild_projection(conn)
    conn.commit()
    output(
        args,
        {
            "imported": len(imported),
            "scope_counts": scope_counts(imported),
            "skipped": skipped,
            "items": imported[: args.sample_limit],
            "items_sampled": len(imported) > args.sample_limit,
            "counts": counts(conn),
        },
    )
    return 0


def legacy_rows_for_backfill(
    conn,
    *,
    limit: int | None,
    project: str | None = None,
    knowledge_type: str | None = None,
):
    clauses = ["status = 'current'"]
    params: list[str | int] = []
    if project:
        clauses.append("project = ?")
        params.append(project)
    if knowledge_type:
        clauses.append("type = ?")
        params.append(knowledge_type)
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(limit)
    return list(
        conn.execute(
            f"""
            SELECT *
            FROM knowledge
            WHERE {" AND ".join(clauses)}
              AND NOT EXISTS (
                SELECT 1
                FROM current_beliefs
                WHERE current_beliefs.belief_id = 'legacy:' || knowledge.id
              )
            ORDER BY updated_at DESC, id ASC
            {limit_clause}
            """,
            params,
        )
    )


def legacy_backfill_plan_item(row) -> dict[str, str]:
    classification = classify_legacy_row(row)
    scope = classification["scope"]
    return {
        "knowledge_id": row["id"],
        "belief_id": f"legacy:{row['id']}",
        "knowledge_type": row["type"],
        "project": row["project"] or "",
        "privacy_scope": row["privacy_scope"],
        "scope_type": scope.scope_type,
        "scope_id": scope.scope_id,
        "visibility": scope.visibility,
        "egress_policy": scope.egress_policy,
        "provenance": scope.provenance,
        "classification_reason": ";".join(classification["reasons"]),
    }


def scope_counts(items: list[dict[str, str]]) -> dict[str, int]:
    counts_by_scope: dict[str, int] = {}
    for item in items:
        scope_id = item.get("scope_id") or "unknown"
        counts_by_scope[scope_id] = counts_by_scope.get(scope_id, 0) + 1
    return dict(sorted(counts_by_scope.items()))


def current_belief_exists(conn, belief_id: str) -> bool:
    return bool(
        conn.execute(
            "SELECT 1 FROM current_beliefs WHERE belief_id = ? LIMIT 1", (belief_id,)
        ).fetchone()
    )


def scope_from_legacy_row(row) -> ScopeTag:
    return classify_legacy_row(row)["scope"]


def classify_legacy_row(row) -> dict[str, object]:
    text = " ".join(
        str(value or "")
        for value in (
            row["project"],
            row["title"],
            row["subject"],
            row["predicate"],
            row["body_uri"],
            row["doc_kind"],
        )
    ).lower()
    privacy_scope = row["privacy_scope"]
    if "bihua" in text or "cormorant" in text:
        return {
            "scope": ScopeTag(
                "client",
                "client:bihua",
                visibility="confidential",
                egress_policy="local_only",
                provenance="inferred",
            ),
            "reasons": ["matched cormorant/bihua client terms"],
        }
    if "pelican" in text:
        return {
            "scope": ScopeTag(
                "personal_finance",
                "personal_finance:pelican",
                visibility="confidential",
                egress_policy="local_only",
                provenance="inferred",
            ),
            "reasons": ["matched pelican personal-finance terms"],
        }
    if "bountiful" in text or "backyard-ripe" in text:
        return {
            "scope": ScopeTag("project", "project:bountiful", provenance="inferred"),
            "reasons": ["matched bountiful/backyard-ripe project terms"],
        }
    if "ocbrain" in text or row["project"] == "ocbrain":
        return {
            "scope": ScopeTag("project", "project:ocbrain", provenance="inferred"),
            "reasons": ["matched ocbrain project terms"],
        }
    if privacy_scope == "public":
        return {
            "scope": global_scope(),
            "reasons": ["legacy row is public"],
        }
    return {
        "scope": resolve_write_scope(ScopeContext()),
        "reasons": ["no narrow scope signal; quarantined as legacy unscoped"],
    }


def legacy_row_body(row) -> str:
    if row["type"] == "value":
        value = row["value_text"]
        if row["value_bool"] is not None:
            value = str(bool(row["value_bool"]))
        elif row["value_numeric"] is not None:
            value = str(row["value_numeric"])
        return f"{row['subject']} {row['predicate']} {value}".strip()
    return " ".join(str(value or "") for value in (row["title"], row["body_uri"])).strip()


def cmd_import_memory(args: argparse.Namespace) -> int:
    conn = open_db(args)
    files = memory_files(args.paths)
    if args.limit is not None:
        files = files[: args.limit]
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for path in files:
        try:
            result = import_memory_file(
                conn,
                path,
                project=args.project,
                privacy_scope=args.privacy_scope,
                max_bytes=args.max_bytes,
            )
        except OSError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        if result is None:
            skipped.append({"path": str(path), "reason": "empty"})
        else:
            imported.append(result)
    conn.commit()
    output(
        args,
        {
            "imported": len(imported),
            "skipped": skipped,
            "files": imported,
            "counts": counts(conn),
        },
    )
    return 0


def cmd_import_history(args: argparse.Namespace) -> int:
    conn = open_db(args)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    files = history_files(args.paths, manifests=args.manifest)
    if not files:
        raise ValueError("pass at least one history path or --manifest")
    if args.limit is not None:
        files = files[: args.limit]
    existing_sources = imported_history_sources(conn)
    imported = 0
    existing = 0
    by_runtime: dict[str, int] = {}
    samples: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    batch_size = max(args.batch_size, 1)
    for path in files:
        source_key = (str(path), f"{history_runtime(path)}_history_file")
        if source_key in existing_sources:
            existing += 1
            continue
        try:
            result = import_history_file(
                conn,
                path,
                project=args.project,
                privacy_scope=args.privacy_scope,
                max_bytes=args.max_bytes,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        if result is None:
            skipped.append({"path": str(path), "reason": "empty"})
            continue
        imported += 1
        by_runtime[result["runtime"]] = by_runtime.get(result["runtime"], 0) + 1
        if len(samples) < 20:
            samples.append(result)
        existing_sources.add((result["path"], f"{result['runtime']}_history_file"))
        if imported % batch_size == 0:
            conn.commit()
    conn.commit()
    output(
        args,
        {
            "imported": imported,
            "existing": existing,
            "by_runtime": by_runtime,
            "sample_files": samples,
            "skipped_count": len(skipped),
            "skipped": skipped[:50],
            "counts": counts(conn),
        },
    )
    return 0


def memory_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.suffix.lower() == ".md":
            files.append(path)
    return sorted(dict.fromkeys(path.resolve() for path in files))


HISTORY_SUFFIXES = (
    ".jsonl",
    ".trajectory.jsonl",
    ".jsonl.codex-app-server.json",
    ".json",
    ".md",
)
HISTORY_GLOBS = ("*.jsonl", "*.trajectory.jsonl", "*.jsonl.codex-app-server.json", "*.json", "*.md")


def history_files(paths: list[Path], *, manifests: list[Path] | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for manifest in manifests or []:
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            candidate = Path(line.strip())
            key = str(candidate)
            if key not in seen and candidate.is_file() and is_history_file(candidate):
                files.append(candidate)
                seen.add(key)
    for path in paths:
        if path.is_dir():
            for pattern in HISTORY_GLOBS:
                for candidate in path.rglob(pattern):
                    key = str(candidate)
                    if key not in seen and candidate.is_file():
                        files.append(candidate)
                        seen.add(key)
        elif path.is_file() and is_history_file(path):
            key = str(path)
            if key not in seen:
                files.append(path)
                seen.add(key)
    return sorted(files)


def is_history_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES)


def import_memory_file(
    conn,
    path: Path,
    *,
    project: str | None,
    privacy_scope: str,
    max_bytes: int,
) -> dict[str, str] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None
    truncated = raw.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
    text = redact_secrets(truncated)
    title = title_from_text(text, path.stem)
    source_uri = str(path)
    digest = content_hash(raw)
    evidence_id = upsert_evidence(
        conn,
        source_type="memory_file",
        source_runtime="openclaw",
        source_uri=source_uri,
        content_hash=digest,
        claim=f"Memory file {path.name}: {compact_whitespace(text[:900])}",
        artifact_uri=source_uri,
        artifact_hash=digest,
        verifier_status="not_required",
        project=project,
        privacy_scope=privacy_scope,
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug=memory_slug(path),
        title=title,
        body_uri=source_uri,
        doc_kind="memory",
        status="current",
        confidence=0.7,
        content_hash=digest,
        project=project,
        privacy_scope=privacy_scope,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="derived_from")
    if text:
        upsert_search_index(
            conn,
            knowledge_id,
            "knowledge:doc",
            title,
            text,
            source_uri,
        )
    return {"path": source_uri, "evidence_id": evidence_id, "knowledge_id": knowledge_id}


def import_history_file(
    conn,
    path: Path,
    *,
    project: str | None,
    privacy_scope: str,
    max_bytes: int,
) -> dict[str, str] | None:
    size = path.stat().st_size
    if size == 0:
        return None
    source_uri = str(path)
    runtime = history_runtime(path)
    digest = file_fingerprint(path)
    text = redact_secrets(history_text_window(path, max_bytes=max_bytes))
    title = history_title(path, runtime)
    evidence_id = upsert_evidence(
        conn,
        source_type=f"{runtime}_history_file",
        source_runtime=runtime,
        source_uri=source_uri,
        content_hash=digest,
        claim=(
            f"{runtime} history file {path.name} ({size} bytes, fingerprinted): "
            f"{compact_whitespace(text[:900])}"
        ),
        artifact_uri=source_uri,
        artifact_hash=digest,
        verifier_status="not_required",
        project=project,
        privacy_scope=privacy_scope,
    )
    knowledge_id = upsert_knowledge(
        conn,
        knowledge_type="doc",
        gate="auto",
        slug=history_slug(path, runtime),
        title=title,
        body_uri=source_uri,
        doc_kind=f"{runtime}_history",
        status="current",
        confidence=0.55,
        content_hash=digest,
        project=project,
        privacy_scope=privacy_scope,
    )
    link_knowledge_evidence(conn, knowledge_id, evidence_id, relation="derived_from")
    if text:
        upsert_search_index(
            conn,
            knowledge_id,
            "knowledge:doc",
            title,
            text,
            source_uri,
        )
    return {
        "path": source_uri,
        "runtime": runtime,
        "evidence_id": evidence_id,
        "knowledge_id": knowledge_id,
    }


def imported_history_sources(conn) -> set[tuple[str, str]]:
    return {
        (row["source_uri"], row["source_type"])
        for row in conn.execute(
            """
            SELECT source_uri, source_type
            FROM evidence
            WHERE source_uri IS NOT NULL
              AND source_type IN (
                'openclaw_history_file',
                'codex_history_file',
                'claude_history_file',
                'unknown_history_file'
              )
            """
        )
    }


def history_text_window(path: Path, *, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= max_bytes:
            data = handle.read()
        else:
            head_len = max_bytes // 2
            tail_len = max_bytes - head_len
            head = handle.read(head_len)
            handle.seek(max(size - tail_len, 0))
            tail = handle.read(tail_len)
            marker = f"\n\n[... {size - max_bytes} bytes omitted from middle ...]\n\n".encode()
            data = head + marker + tail
    return data.decode("utf-8", errors="replace")


def history_title(path: Path, runtime: str) -> str:
    stem = path.name
    for suffix in HISTORY_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{runtime} history: {stem}"[:160]


def memory_slug(path: Path) -> str:
    parts = [part for part in path.with_suffix("").parts if part not in {"/", ""}]
    tail = parts[-3:] if len(parts) > 3 else parts
    slug = "-".join(tail).lower()
    return "".join(char if char.isalnum() else "-" for char in slug).strip("-")


def history_slug(path: Path, runtime: str) -> str:
    parts = [part for part in path.with_suffix("").parts if part not in {"/", ""}]
    tail = parts[-4:] if len(parts) > 4 else parts
    slug = f"{runtime}-history-{'-'.join(tail)}-{stable_id('path', str(path))[5:13]}"
    return "".join(char if char.isalnum() else "-" for char in slug.lower()).strip("-")


def cmd_digest(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scopes = None if args.include_private else PUBLIC_SCOPES
    output(args, knowledge_digest(conn, project=args.project, scopes=scopes, limit=args.limit))
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
        conn = open_db(args)
        result = write_loop_ingest(conn, options)
    else:
        result = dry_run_loop_ingest(options)
    if args.json:
        output(args, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_mark_stale(args: argparse.Namespace) -> int:
    conn = open_db(args)
    if not mark_knowledge_stale(conn, args.knowledge_id, reason=args.reason):
        raise ValueError(f"knowledge not found: {args.knowledge_id}")
    conn.commit()
    row = get_knowledge(conn, args.knowledge_id)
    output(args, {"knowledge": dict(row)})
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = prune_knowledge(
        conn,
        ttl_days=args.ttl_days,
        unhelpful_ttl_days=args.unhelpful_ttl_days,
        archive_stale_days=args.archive_stale_days,
    )
    conn.commit()
    output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_heal(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = heal_conflicts(conn, numeric_threshold=args.numeric_threshold)
    conn.commit()
    output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_liveness_check(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = check_loop_liveness(conn, runner_ledger=args.runner_ledger)
    conn.commit()
    output(args, result.as_dict() | {"counts": counts(conn)})
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    return serve(args.db, allow_writes=args.allow_writes)


# --------------------------------------------------------------------------- #
# v0.2 autonomy + dataset factory commands (spec §8)
# --------------------------------------------------------------------------- #
def cmd_autopilot(args: argparse.Namespace) -> int:
    conn = open_db(args)
    cfg = load_config()
    result = run_autopilot(
        conn,
        cfg,
        db_path=args.db,
        stages=args.stages,
        profile=getattr(args, "profile", None),
        dry_run=args.dry_run,
    )
    output(args, result)
    return 0


def cmd_quarantine_list(args: argparse.Namespace) -> int:
    conn = open_db(args)
    limit = getattr(args, "limit", 100)
    rows = conn.execute(
        """
        SELECT id, slug, title, quarantine_reason, quality_label, updated_at
        FROM knowledge
        WHERE quarantine_reason IS NOT NULL
        ORDER BY updated_at DESC, id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    output(args, {"quarantined": [dict(row) for row in rows], "count": len(rows)})
    return 0


def cmd_quarantine_release(args: argparse.Namespace) -> int:
    conn = open_db(args)
    released = release_quarantine(conn, args.knowledge_id, actor=args.actor, reason=args.reason)
    conn.commit()
    row = get_knowledge(conn, args.knowledge_id)
    output(
        args,
        {
            "knowledge_id": args.knowledge_id,
            "released": released,
            "knowledge": dict(row) if row else None,
        },
    )
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    conn = open_db(args)
    row = get_knowledge(conn, args.knowledge_id)
    if row is None:
        raise ValueError(f"knowledge not found: {args.knowledge_id}")
    polarity = "good" if args.outcome == "good" else "bad"
    signal = Signal(
        kind="manual_label",
        polarity=polarity,
        weight=0.9,
        source="session",
        source_ref=f"manual:{args.knowledge_id}",
        knowledge_id=args.knowledge_id,
        details={"note": args.note} if args.note else {"manual": True},
        occurred_at=now_iso(),
    )
    signal_id = record_signal(conn, signal)
    conn.commit()
    output(
        args,
        {"knowledge_id": args.knowledge_id, "signal_id": signal_id, "outcome": args.outcome},
    )
    return 0


def cmd_dataset_mine(args: argparse.Namespace) -> int:
    conn = open_db(args)
    cfg = load_config()
    roots = list(cfg.review.session_roots)
    budget = getattr(args, "time_budget", None)
    limit = getattr(args, "limit", None)
    verified_only = getattr(args, "verified_only", False)
    dataset = getattr(args, "dataset", None)
    if dataset == "sft":
        from ocbrain.dataset.mine_sft import mine_sft

        result = mine_sft(conn, cfg=cfg, roots=roots, limit=limit, time_budget_seconds=budget)
    elif dataset == "dpo":
        from ocbrain.dataset.mine_dpo import mine_dpo

        result = mine_dpo(conn, cfg=cfg, roots=roots, limit=limit, time_budget_seconds=budget)
    elif dataset == "persona":
        from ocbrain.dataset.mine_persona import mine_persona

        result = mine_persona(
            conn,
            cfg=cfg,
            roots=roots,
            verified_only=verified_only,
            limit=limit,
            time_budget_seconds=budget,
        )
    else:
        result = mine_all(
            conn,
            cfg=cfg,
            roots=roots,
            verified_only=verified_only,
            time_budget_seconds=budget,
        )
    conn.commit()
    if cfg.autopilot.checkpoint_after_dataset_mine:
        from ocbrain.fsutil import checkpoint_sqlite_wal

        db_path = getattr(args, "db", None)
        result["wal_checkpoint"] = checkpoint_sqlite_wal(
            conn,
            db_path,
            minimum_bytes=cfg.autopilot.checkpoint_wal_min_bytes,
        )
    output(args, result)
    return 0


def cmd_dataset_persona_curate(args: argparse.Namespace) -> int:
    from ocbrain.dataset.curate import import_persona_curation

    conn = open_db(args)
    result = import_persona_curation(conn, args.input, cfg=load_config())
    conn.commit()
    output(args, result)
    return 0


def cmd_dataset_calibration_import(args: argparse.Namespace) -> int:
    from ocbrain.dataset.calibration import import_calibrations

    conn = open_db(args)
    result = import_calibrations(conn, args.input)
    conn.commit()
    output(args, result)
    return 0


def cmd_dataset_export(args: argparse.Namespace) -> int:
    conn = open_db(args)
    cfg = load_config()
    dataset = getattr(args, "dataset", None)
    result = export_all(
        conn,
        cfg=cfg,
        datasets=[dataset] if dataset else None,
        min_scope=getattr(args, "min_scope", None),
        min_label=getattr(args, "min_label", None),
        min_grade=getattr(args, "min_grade", None),
        verified_only=getattr(args, "verified_only", False),
        export_dir=getattr(args, "output_dir", None),
    )
    conn.commit()
    output(args, result)
    return 0


def cmd_dataset_grade(args: argparse.Namespace) -> int:
    from ocbrain.dataset.grade import grade_examples

    conn = open_db(args)
    cfg = load_config()
    dataset = getattr(args, "dataset", None)
    result = grade_examples(
        conn,
        cfg=cfg,
        datasets=[dataset] if dataset else None,
        limit=getattr(args, "limit", None),
        endpoint=getattr(args, "endpoint", None),
        model=getattr(args, "model", None),
        force=getattr(args, "force", False),
        source_uri_prefix=getattr(args, "source_uri_prefix", None),
        train_classes=getattr(args, "train_classes", None),
        selected_only=bool(getattr(args, "selected_only", False)),
    )
    conn.commit()
    output(args, result)
    return 1 if result.get("status") in {"error", "blocked", "locked"} else 0


def cmd_dataset_classify(args: argparse.Namespace) -> int:
    from ocbrain.dataset.classify import classify_examples

    conn = open_db(args)
    result = classify_examples(
        conn,
        force=bool(getattr(args, "force", False)),
        limit=getattr(args, "limit", None),
    )
    conn.commit()
    output(args, result)
    return 0


def cmd_dataset_pack_select(args: argparse.Namespace) -> int:
    from ocbrain.dataset.selection import select_training_pack

    conn = open_db(args)
    result = select_training_pack(
        conn,
        targets={"sft": args.sft, "dpo": args.dpo, "persona": args.persona},
        seed=args.seed,
    )
    conn.commit()
    output(args, result)
    return 0


def cmd_dataset_pack_stats(args: argparse.Namespace) -> int:
    from ocbrain.dataset.selection import selected_pack_stats

    conn = open_db(args)
    output(args, selected_pack_stats(conn, min_grade=args.min_grade))
    return 0


def cmd_retrieval_feedback_stats(args: argparse.Namespace) -> int:
    from ocbrain.feedback import feedback_coverage

    conn = open_db(args)
    output(args, feedback_coverage(conn))
    return 0


def cmd_retrieval_benchmark(args: argparse.Namespace) -> int:
    from ocbrain.retrieval_eval import run_benchmark

    conn = open_db(args)
    result = run_benchmark(
        conn,
        args.input,
        require_cases=1 if bool(getattr(args, "allow_small", False)) else 100,
    )
    output(args, result)
    return 0


def cmd_retrieval_benchmark_expand(args: argparse.Namespace) -> int:
    from ocbrain.retrieval_eval import expand_runtime_matrix

    result = expand_runtime_matrix(args.input, args.output)
    output(args, result)
    return 0


def cmd_dataset_stats(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(args, dataset_stats(conn))
    return 0


def cmd_dataset_pilot_prepare(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import prepare_pilot

    conn = open_db(args)
    try:
        result = prepare_pilot(
            conn,
            cfg=load_config(),
            output_dir=getattr(args, "output_dir", None),
            min_grade=getattr(args, "min_grade", None),
            eval_prompts=getattr(args, "eval_prompts", 100),
            seed=getattr(args, "seed", "ocbrain-voice-pilot-v3"),
            base_model=getattr(args, "base_model", None),
            base_model_source=getattr(args, "base_model_source", None),
            base_model_revision=getattr(args, "base_model_revision", None),
            eval_from=getattr(args, "eval_from", None),
            training_iterations=getattr(args, "training_iterations", 25),
            quality_gates=not bool(getattr(args, "diagnostic_small_pack", False)),
            sentinel_from=getattr(args, "legacy_sentinel_from", None),
        )
    except RuntimeError as exc:
        output(
            args,
            {
                "action": "dataset-pilot-prepare",
                "changed": 0,
                "status": "blocked",
                "error": str(exc),
            },
        )
        return 1
    output(args, result)
    return 0


def cmd_dataset_pilot_blind(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import prepare_blind_pairs

    result = prepare_blind_pairs(
        args.pilot_dir,
        args.candidate_responses,
        seed=args.seed,
    )
    output(args, result)
    return 0


def cmd_dataset_pilot_score(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import score_blind_ratings

    result = score_blind_ratings(args.pilot_dir, args.ratings)
    output(args, result)
    return 0


def cmd_dataset_pilot_multiblind(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import prepare_multiblind

    response_sets: dict[str, Path] = {}
    for raw in args.response:
        name, separator, value = str(raw).partition("=")
        if not separator or name not in {"base", "tuned", "frontier"} or not value:
            raise ValueError("--response must be base=PATH, tuned=PATH, or frontier=PATH")
        response_sets[name] = Path(value).expanduser()
    result = prepare_multiblind(args.pilot_dir, response_sets, seed=args.seed)
    output(args, result)
    return 0


def cmd_dataset_pilot_multiscore(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import score_multiblind

    result = score_multiblind(args.pilot_dir, args.ratings)
    output(args, result)
    return 0


def cmd_dataset_pilot_record_training(args: argparse.Namespace) -> int:
    from ocbrain.dataset.pilot import record_training_result

    result = record_training_result(
        args.pilot_dir,
        iterations=args.iterations,
        train_loss=args.train_loss,
        validation_loss=args.validation_loss,
        exit_code=args.exit_code,
    )
    output(args, result)
    return 0


# --------------------------------------------------------------------------- #
# public-safety enforcement (keep private data out of the public repo)
# --------------------------------------------------------------------------- #
def _resolve_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    import subprocess

    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path.cwd()
    return Path(top) if top else Path.cwd()


def cmd_public_safety_check(args: argparse.Namespace) -> int:
    from ocbrain.publicsafety import scan

    root = _resolve_repo_root(getattr(args, "root", None))
    result = scan(root, diff_range=getattr(args, "diff_range", None))
    if getattr(args, "json", False):
        output(args, result.to_dict())
    else:
        print(result.report(), file=sys.stderr)
    return 0 if result.ok else 1


def cmd_install_hooks(args: argparse.Namespace) -> int:
    import os

    root = _resolve_repo_root(getattr(args, "root", None))
    hooks_src = root / "ops" / "hooks"
    hooks_dst = root / ".git" / "hooks"
    if not hooks_src.is_dir():
        raise ValueError(f"no tracked hooks directory at {hooks_src}")
    if not hooks_dst.is_dir():
        raise ValueError(f"no .git/hooks directory at {hooks_dst} (not a git working copy?)")
    installed: list[dict[str, str]] = []
    for hook in sorted(hooks_src.iterdir()):
        if hook.name.startswith(".") or not hook.is_file():
            continue
        target = hooks_dst / hook.name
        rel = os.path.relpath(hook, hooks_dst)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(rel)
        os.chmod(hook, 0o755)
        installed.append({"hook": hook.name, "link": str(target), "points_to": rel})
    output(args, {"installed": installed, "count": len(installed)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
