from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain import __version__
from ocbrain.briefing import (
    DEFAULT_BRIEFING_BUDGET_CHARS,
    build_briefing,
    build_ledger,
)
from ocbrain.bundle import export_bundle, import_bundle
from ocbrain.compact import (
    COMPACT_VERSION,
    DEFAULT_COSINE_FLOOR,
    DEFAULT_MERGE_LIMIT,
    adjudicate,
    apply_compaction,
    plan_compaction,
    undo_command,
    undo_merge,
)
from ocbrain.config import ConfigError, describe_config, load_config
from ocbrain.core_ops import (
    backup_database,
    database_status,
    doctor,
    restore_database,
    sync_core,
)
from ocbrain.core_v1 import (
    append_core_event,
    get_core_v1_belief,
    get_core_v1_evidence,
    init_core_v1,
    is_core_v1,
    migrate_core_v1_columns,
    reclassify_no_coverage_receipts,
    record_core_v1_evidence,
    resolve_object_id,
)
from ocbrain.curation import apply_curated_manifest
from ocbrain.curator import (
    DEFAULT_MEASURED_TTL_DAYS,
    DEFAULT_VOLATILE_TTL_DAYS,
    PROVIDER_DEFAULTS,
)
from ocbrain.db import (
    DEFAULT_DB_PATH,
    PUBLIC_SCOPES,
    connect,
    counts,
    init_db,
    knowledge_digest,
    link_knowledge_evidence,
    list_knowledge,
    search,
    upsert_evidence,
    upsert_knowledge,
    upsert_search_index,
)
from ocbrain.deslop import rewindowed_evidence_id
from ocbrain.egress import egress_preview
from ocbrain.events import (
    SKILL_TELEMETRY_KINDS,
    canonical_json,
    decide_compilation,
    event_core_digest,
    evidence_id_for,
    list_compilation_proposals,
    propose_compilation,
    rebuild_projection,
    record_correction,
    record_evidence,
    record_tombstone,
    validate_skill_telemetry,
)
from ocbrain.fsutil import file_fingerprint, history_runtime, snapshot_sqlite
from ocbrain.history_window import (
    _PRIVATE_KEY_BEGIN_RE,
    HistoryWindow,
    build_history_window,
    history_text_window,
)
from ocbrain.hybrid import build_vector_index, vector_status
from ocbrain.hygiene import CLASSES as HYGIENE_CLASSES
from ocbrain.hygiene import (
    DEFAULT_BATCH_CAP,
    DEFAULT_RESTATEMENT_THRESHOLD,
    apply_retirements,
    plan_retirements,
    restore,
    supersede,
    verify_serving_invariants,
)
from ocbrain.ids import content_hash, stable_id
from ocbrain.mcp import serve
from ocbrain.mcp_v1 import (
    build_context_v1,
    correct_v1,
    decide_proposal_v1,
    digest_v1,
    forget_v1,
    ingest_v1,
    proposals_v1,
    record_context_v1,
    search_v1,
)
from ocbrain.retrieve import retrieve
from ocbrain.scope import (
    DEFAULT_GLOBAL_SCOPE_ID,
    SCOPE_TYPES,
    ScopeContext,
    ScopeTag,
    fold_scope_component,
    global_scope,
    hosted_egress_refusal_reason,
    resolve_scope_alias,
    resolve_write_scope,
)
from ocbrain.text import (
    compact_whitespace,
    find_probable_secret_leaks,
    redact_secrets,
    title_from_text,
)

PRIVACY_SCOPES = ("private", "workspace", "project", "public")


def build_parser() -> argparse.ArgumentParser:
    """Build the core-only v1 CLI parser.

    Training and operations commands are exact-name lazy entry points handled
    before parsing; they never become imports or apparent core subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="ocbrain",
        description="Local shared-context bridge for Codex, Claude Code, and OpenClaw",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--pretty", action="store_true")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("init", help="Initialize a fresh event-authoritative v1 core").set_defaults(
        func=cmd_init
    )
    commands.add_parser("status", help="Inspect core health without changing it").set_defaults(
        func=cmd_status
    )
    sync = commands.add_parser("sync", help="Boundedly reconcile local core projections")
    sync.add_argument("--max-events", type=int, default=1_000)
    sync.add_argument("--time-budget", type=float, default=10.0)
    sync.add_argument(
        "--full",
        action="store_true",
        help=(
            "discard every projected row and refold the whole ledger from scratch. "
            "Ignores --max-events"
        ),
    )
    sync.set_defaults(func=cmd_sync)

    vector_build = commands.add_parser(
        "vector-build",
        help="Explicitly rebuild the disposable loopback-only dense index",
    )
    vector_build.add_argument("--output", type=Path)
    vector_build.add_argument("--model")
    vector_build.add_argument("--endpoint")
    vector_build.add_argument("--batch-size", type=int, default=8)
    vector_build.set_defaults(func=cmd_vector_build)
    vector_status_parser = commands.add_parser(
        "vector-status", help="Inspect the local derived dense index"
    )
    vector_status_parser.add_argument("--sidecar", type=Path)
    vector_status_parser.set_defaults(func=cmd_vector_status)
    curated_apply = commands.add_parser(
        "curated-apply",
        help="Apply a source-hash-verified curated-memory manifest",
    )
    curated_apply.add_argument("manifest", type=Path)
    curated_apply.add_argument("--actor", default="human-curated:operator")
    curated_apply.add_argument(
        "--allow-hosted-egress",
        action="store_true",
        help="Acknowledge that hosted_ok fact bodies may be delivered to a hosted model",
    )
    curated_apply.set_defaults(func=cmd_curated_apply)

    feedback_repair = commands.add_parser(
        "feedback-repair",
        help="Reclassify relevance verdicts filed on zero-item retrievals as no_coverage",
    )
    feedback_repair.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the selected receipts (default: report only)",
    )
    feedback_repair.set_defaults(func=cmd_feedback_repair)

    hygiene_parser = commands.add_parser(
        "hygiene",
        help="Retire expired, never-retrieved, or badly-judged beliefs",
    )
    hygiene_parser.add_argument(
        "--class",
        dest="classes",
        action="append",
        choices=list(HYGIENE_CLASSES),
        help="restrict to one class; repeatable (default: all)",
    )
    hygiene_parser.add_argument(
        "--apply",
        action="store_true",
        help="soft-retract the selected beliefs (default: report only)",
    )
    hygiene_parser.add_argument("--batch-cap", type=int, default=DEFAULT_BATCH_CAP)
    hygiene_parser.add_argument(
        "--restatement-threshold",
        type=float,
        default=DEFAULT_RESTATEMENT_THRESHOLD,
        help=(
            "token overlap above which two served beliefs count as one fact "
            "restated; lower retires more aggressively"
        ),
    )
    hygiene_parser.add_argument(
        "--restore",
        metavar="BELIEF_ID",
        help="put a soft-retracted belief back into service, then exit",
    )
    hygiene_parser.add_argument(
        "--supersede",
        nargs=2,
        metavar=("BELIEF_ID", "SUCCESSOR_ID"),
        help="mark one belief superseded by another, then exit",
    )
    hygiene_parser.set_defaults(func=cmd_hygiene)

    compact_parser = commands.add_parser(
        "compact",
        help="Collapse historical near-duplicate beliefs into one, reversibly",
    )
    compact_parser.add_argument(
        "--cosine",
        type=float,
        default=DEFAULT_COSINE_FLOOR,
        help="cosine at or above which two same-scope beliefs enter one cluster",
    )
    compact_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_MERGE_LIMIT,
        help="maximum beliefs one run may retire; the rest are reported as deferred",
    )
    compact_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="print the plan and write nothing (the default)",
    )
    compact_parser.add_argument(
        "--apply",
        action="store_true",
        help="retire the planned losers; additionally requires --yes",
    )
    compact_parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm that a human read the plan; --apply does nothing without it",
    )
    compact_parser.add_argument(
        "--undo",
        metavar="BELIEF_ID",
        help="put one merged-away belief back into service, then exit",
    )
    compact_parser.add_argument(
        "--provider", choices=sorted(PROVIDER_DEFAULTS), default="anthropic"
    )
    compact_parser.add_argument("--model", default="", help="override the provider default")
    compact_parser.add_argument(
        "--allow-hosted-egress",
        action="store_true",
        help=(
            "allow eligible belief bodies to be sent to the configured hosted "
            "provider for adjudication; otherwise the ambiguous tail stays undecided"
        ),
    )
    compact_parser.add_argument(
        "--json", action="store_true", help="emit the plan as JSON instead of a report"
    )
    compact_parser.set_defaults(func=cmd_compact)

    volatility_parser = commands.add_parser(
        "wiki-volatility",
        help="Re-date serving beliefs by volatility class; prints a plan by default",
    )
    volatility_parser.add_argument(
        "--volatile-days",
        type=int,
        default=DEFAULT_VOLATILE_TTL_DAYS,
        help="TTL for a belief naming a host, version, credential or live state",
    )
    volatility_parser.add_argument(
        "--measured-days",
        type=int,
        default=DEFAULT_MEASURED_TTL_DAYS,
        help="TTL for a measurement or result",
    )
    volatility_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="print the plan and write nothing (the default)",
    )
    volatility_parser.add_argument(
        "--apply",
        action="store_true",
        help="write the new expiries; additionally requires --yes",
    )
    volatility_parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm that a human read the plan, including already_expired",
    )
    volatility_parser.add_argument(
        "--actor", default="operator", help="who authorised the re-dating"
    )
    volatility_parser.set_defaults(func=cmd_wiki_volatility)

    config_parser = commands.add_parser(
        "config",
        help="Show the effective configuration and where each value came from",
    )
    config_parser.add_argument(
        "--section",
        help="restrict output to one section (e.g. curator, retrieval)",
    )
    config_parser.add_argument(
        "--changed-only",
        action="store_true",
        help="show only values that differ from the shipped default",
    )
    config_parser.set_defaults(func=cmd_config)

    doctor_parser = commands.add_parser("doctor", help="Check the core and stdio MCP")
    doctor_parser.add_argument("--timeout", type=float, default=8.0)
    doctor_parser.add_argument("--launcher", type=Path)
    doctor_parser.add_argument(
        "--ops",
        action="store_true",
        help=(
            "Also verify the machine against its ops manifest: launchd jobs "
            "loaded with the intended env, hooks byte-identical to their repo "
            "sources, control files present. The DB can be healthy while the "
            "wiring around it is not; this is the check that notices."
        ),
    )
    doctor_parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "With --ops: snapshot the current wiring as the intended wiring "
            "before checking. Run once on deployment day, and again whenever "
            "a wiring change is deliberate."
        ),
    )
    doctor_parser.add_argument(
        "--ops-manifest",
        type=Path,
        help="Manifest location (default ~/.ocbrain/ops-manifest.json)",
    )
    doctor_parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help=(
            "With --ops --write-manifest: replace an existing manifest after "
            "reviewing the intended wiring change. Without this flag, bootstrap "
            "refuses to erase an existing drift baseline."
        ),
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    runtime = commands.add_parser("runtime-check", help="Probe all three client integrations")
    runtime.add_argument("--timeout", type=float, default=12.0)
    runtime.add_argument("--launcher", type=Path)
    runtime.set_defaults(func=cmd_runtime_check)

    backup = commands.add_parser("backup", help="Create a verified online SQLite backup")
    backup.add_argument("--output", type=Path, required=True)
    backup.add_argument("--manifest", type=Path)
    backup.set_defaults(func=cmd_backup)
    restore = commands.add_parser("restore", help="Restore a backup to a fresh path")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument("--output-db", type=Path, required=True)
    restore.add_argument("--manifest", type=Path)
    restore.set_defaults(func=cmd_restore)
    migrate = commands.add_parser("core-migrate-v1", help="Build archive-first v1 outputs")
    migrate.add_argument("--core-db", type=Path, required=True)
    migrate.add_argument("--archive-db", type=Path, required=True)
    migrate.add_argument("--manifest", type=Path, required=True)
    migrate.add_argument("--training-db", type=Path)
    migrate.add_argument("--ops-db", type=Path)
    migrate.add_argument("--plan", action="store_true")
    migrate.set_defaults(func=cmd_core_migrate_v1)

    export_parser = commands.add_parser(
        "export-bundle",
        help="Export selected strict-v1 evidence to a fresh local bundle file",
    )
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--evidence-id", action="append", required=True)
    export_parser.add_argument("--approve-egress", action="store_true")
    add_context_args(export_parser)
    export_parser.set_defaults(func=cmd_export_bundle)

    import_parser = commands.add_parser(
        "import-bundle",
        help="Validate a local evidence bundle; append only with --apply",
    )
    import_parser.add_argument("path", type=Path)
    import_parser.add_argument("--project", required=True)
    import_parser.add_argument("--apply", action="store_true")
    import_parser.set_defaults(func=cmd_import_bundle)

    evidence = commands.add_parser("evidence", help="Append source-backed evidence")
    evidence.add_argument("--claim")
    evidence.add_argument("--input", type=Path)
    evidence.add_argument("--source-type", default="closeout")
    evidence.add_argument("--source-runtime")
    evidence.add_argument("--source-uri")
    evidence.add_argument("--artifact-uri")
    evidence.add_argument("--artifact-hash")
    evidence.add_argument("--verifier-status", default="unknown")
    evidence.add_argument("--project")
    evidence.add_argument("--privacy-scope", default="workspace")
    evidence.set_defaults(func=cmd_evidence)

    knowledge = commands.add_parser("knowledge", help="List compatibility knowledge rows")
    knowledge.add_argument("--status")
    knowledge.add_argument("--type")
    knowledge.add_argument("--include-private", action="store_true")
    knowledge.add_argument("--limit", type=int, default=20)
    knowledge.set_defaults(func=cmd_knowledge)
    value = commands.add_parser("value", help="Upsert one typed compatibility value")
    value.add_argument("--subject", required=True)
    value.add_argument("--predicate", required=True)
    typed = value.add_mutually_exclusive_group(required=True)
    typed.add_argument("--text")
    typed.add_argument("--number", type=float)
    typed.add_argument("--bool", choices=["true", "false"])
    value.add_argument("--unit")
    value.add_argument("--target-value", type=float)
    value.add_argument("--status", default="candidate")
    value.add_argument("--inject", action="store_true")
    value.add_argument("--confidence", type=float)
    value.add_argument("--project")
    value.add_argument("--privacy-scope", default="workspace")
    value.set_defaults(func=cmd_value)
    search_parser = commands.add_parser("search", help="Search scoped brain objects")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--include-private", action="store_true")
    search_parser.add_argument("--project")
    search_parser.add_argument("--type")
    search_parser.add_argument("--status")
    search_parser.add_argument("--loop-id")
    search_parser.add_argument("--family")
    search_parser.set_defaults(func=cmd_search)

    preview = commands.add_parser("preview", help="Preview a stable shared-context retrieval")
    preview.add_argument("query")
    add_context_args(preview)
    preview.add_argument("--limit", type=int, default=12)
    preview.add_argument("--cross-scope", action="store_true")
    preview.add_argument("--at-ts")
    preview.set_defaults(func=cmd_preview)

    ingest = commands.add_parser("event-ingest", help="Append scoped event evidence")
    ingest.add_argument("--body", required=True)
    ingest.add_argument("--kind", default="observation")
    ingest.add_argument("--writer", default="ocbrain")
    ingest.add_argument("--artifact-ref")
    add_context_args(ingest)
    ingest.add_argument("--global-doctrine", action="store_true")
    ingest.set_defaults(func=cmd_event_ingest)
    compile_parser = commands.add_parser("event-compile", help="Propose a compiled belief")
    compile_parser.add_argument("--belief-id", required=True)
    compile_parser.add_argument("--body", required=True)
    compile_parser.add_argument("--evidence-id", action="append", default=[])
    compile_parser.add_argument("--confidence", type=float)
    compile_parser.add_argument("--approve", action="store_true")
    add_context_args(compile_parser)
    compile_parser.add_argument("--global-doctrine", action="store_true")
    compile_parser.set_defaults(func=cmd_event_compile)
    correct = commands.add_parser("event-correct", help="Append a durable correction")
    correct.add_argument(
        "--target-layer",
        choices=["evidence", "knowledge", "belief"],
        required=True,
    )
    correct.add_argument("--target-id", required=True)
    correct.add_argument(
        "--op",
        choices=["mark_wrong", "edit", "pin", "demote", "reframe", "retract", "restore"],
        required=True,
    )
    correct.add_argument("--body")
    correct.add_argument("--author", default="human:jonathan")
    correct.add_argument("--hard", action="store_true")
    correct.set_defaults(func=cmd_event_correct)
    scope_promote = commands.add_parser(
        "scope-promote",
        help="Move approved durable beliefs to a wider scope (never a wider egress)",
    )
    scope_promote.add_argument("--belief-id", action="append", default=[])
    scope_promote.add_argument(
        "--select-durable-preferences",
        action="store_true",
        help=(
            "also promote every current, served workspace wiki_fact whose lifecycle is "
            "durable and whose category is preference/decision/workflow/system"
        ),
    )
    scope_promote.add_argument("--to-scope-type", choices=sorted(SCOPE_TYPES))
    scope_promote.add_argument("--to-scope-id")
    scope_promote.add_argument(
        "--approved-by",
        required=True,
        help="the human accountable for widening these beliefs; recorded in the event",
    )
    scope_promote.add_argument("--reason")
    scope_promote.add_argument("--dry-run", action="store_true")
    scope_promote.set_defaults(func=cmd_scope_promote)

    egress_promote = commands.add_parser(
        "egress-promote",
        help=(
            "Lift current beliefs' egress to hosted delivery (never confidential/secret); "
            "a human-attributable decision recorded as egress_promoted events"
        ),
    )
    egress_promote.add_argument("belief_id", nargs="*", default=[])
    egress_promote.add_argument(
        "--to",
        choices=("hosted_ok", "approval_required"),
        default="hosted_ok",
        help="target egress policy (default hosted_ok)",
    )
    egress_promote.add_argument(
        "--scope-id",
        help="promote every current belief in this scope_id (e.g. project:coframe-personalization)",
    )
    egress_promote.add_argument(
        "--provenance",
        help="with --scope-id, restrict the selection to this scope_provenance",
    )
    egress_promote.add_argument(
        "--approved-by",
        required=True,
        help="the human accountable for lifting egress; recorded in the event",
    )
    egress_promote.add_argument("--reason", required=True)
    egress_promote.add_argument("--dry-run", action="store_true")
    egress_promote.set_defaults(func=cmd_egress_promote)

    forget = commands.add_parser("event-forget", help="Append a tombstone")
    forget.add_argument("--target", required=True)
    forget.add_argument("--mode", choices=["soft", "shred"], default="soft")
    forget.add_argument("--reason")
    forget.add_argument("--approved-by", default="human:jonathan")
    forget.set_defaults(func=cmd_event_forget)
    proposals = commands.add_parser("event-proposals", help="List compilation proposals")
    add_context_args(proposals)
    proposals.add_argument("--include-decided", action="store_true")
    proposals.add_argument("--limit", type=int, default=50)
    proposals.set_defaults(func=cmd_event_proposals)
    decide = commands.add_parser("event-decide", help="Gate one compilation proposal")
    decide.add_argument("--proposal-event-id", required=True)
    decide.add_argument(
        "--decision",
        choices=["approve", "reject", "edit", "shadow"],
        required=True,
    )
    decide.add_argument("--actor", default="human:jonathan")
    decide.add_argument("--edited-body")
    decide.add_argument("--reason")
    decide.set_defaults(func=cmd_event_decide)
    hosted_queue = commands.add_parser(
        "hosted-queue",
        help="List approval_required evidence eligible for hosted approval (read-only)",
    )
    hosted_queue.add_argument("--project", help="limit to evidence scoped project:PROJECT")
    hosted_queue.add_argument("--writer", help="limit to evidence recorded by this writer")
    hosted_queue.add_argument("--since", help="limit to rows recorded at or after this ISO ts")
    hosted_queue.add_argument("--limit", type=int, default=200)
    hosted_queue.set_defaults(func=cmd_hosted_queue)
    hosted_approve = commands.add_parser(
        "hosted-approve",
        help="Promote approval_required evidence to hosted_ok beliefs (human-gated, CLI-only)",
    )
    hosted_approve.add_argument(
        "evidence_id",
        nargs="*",
        help="evidence id(s) to approve; or use --all-from-queue",
    )
    hosted_approve.add_argument(
        "--approved-by",
        required=True,
        help="the human deciding this, spelled human:NAME (recorded as the decision actor)",
    )
    hosted_approve.add_argument("--reason", help="why this promotion is approved")
    hosted_approve.add_argument(
        "--all-from-queue",
        action="store_true",
        help="select every evidence row the hosted-queue would list under the given filters",
    )
    hosted_approve.add_argument("--project", help="target project scope for the compiled beliefs")
    hosted_approve.add_argument("--writer", help="queue filter: writer")
    hosted_approve.add_argument("--since", help="queue filter: recorded at or after this ISO ts")
    hosted_approve.add_argument("--limit", type=int, default=200)
    hosted_approve.add_argument(
        "--dry-run", action="store_true", help="report what would be approved; write nothing"
    )
    hosted_approve.set_defaults(func=cmd_hosted_approve)
    event_digest = commands.add_parser("event-digest", help="Show scoped current event state")
    add_context_args(event_digest)
    event_digest.add_argument("--since-ts")
    event_digest.add_argument("--limit", type=int, default=20)
    event_digest.set_defaults(func=cmd_event_digest)
    egress = commands.add_parser("egress-preview", help="Preview scope-filtered egress")
    egress.add_argument("--target", default="hosted_teacher")
    egress.add_argument("--query")
    egress.add_argument("--record", action="store_true")
    add_context_args(egress)
    egress.set_defaults(func=cmd_egress_preview)

    backfill = commands.add_parser("event-backfill", help="Explicitly backfill legacy rows")
    backfill.add_argument("--limit", type=int, default=100)
    backfill.add_argument("--sample-limit", type=int, default=100)
    backfill.add_argument("--all", action="store_true")
    backfill.add_argument("--project")
    backfill.add_argument("--type")
    backfill.add_argument("--dry-run", action="store_true")
    backfill.set_defaults(func=cmd_event_backfill)

    import_memory = commands.add_parser(
        "import-memory",
        help="Import markdown sources into the local core",
    )
    import_memory.add_argument("paths", nargs="+", type=Path)
    import_memory.add_argument("--project", default="workspace")
    import_memory.add_argument("--privacy-scope", choices=PRIVACY_SCOPES, default="private")
    import_memory.add_argument("--limit", type=int)
    import_memory.add_argument("--max-bytes", type=int, default=50_000)
    import_memory.add_argument("--dry-run", action="store_true")
    import_memory.add_argument(
        "--evidence-only",
        action="store_true",
        help="append memory-file evidence without promoting the whole file into a serving belief",
    )
    import_memory.set_defaults(func=cmd_import_memory)

    import_history = commands.add_parser(
        "import-history",
        help="Import runtime transcript sources into the local core",
    )
    import_history.add_argument("paths", nargs="*", type=Path)
    import_history.add_argument("--manifest", action="append", type=Path, default=[])
    import_history.add_argument("--project", default="workspace")
    import_history.add_argument("--privacy-scope", choices=PRIVACY_SCOPES, default="private")
    import_history.add_argument("--limit", type=int)
    import_history.add_argument("--max-bytes", type=int, default=20_000)
    import_history.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Deprecated: history imports commit per file to bound SQLite writer-lock windows",
    )
    import_history.add_argument(
        "--evidence-only",
        action="store_true",
        help="append transcript evidence without promoting whole transcripts into serving beliefs",
    )
    import_history.add_argument("--dry-run", action="store_true")
    import_history.set_defaults(func=cmd_import_history)

    digest = commands.add_parser("digest", help="Show current scoped knowledge")
    digest.add_argument("--project")
    digest.add_argument("--limit", type=int, default=12)
    digest.add_argument("--include-private", action="store_true")
    digest.set_defaults(func=cmd_digest)
    briefing = commands.add_parser(
        "briefing",
        help="Deterministic, bounded session-start reorientation for one scope",
    )
    add_context_args(briefing)
    briefing.add_argument("--budget-chars", type=int, default=DEFAULT_BRIEFING_BUDGET_CHARS)
    briefing.add_argument(
        "--repo-root",
        help=(
            "Local directory to resolve goal spec pointers against. This is not "
            "--repo: --repo is part of the retrieval scope and narrows what the "
            "briefing reports, while this only tells the pointer check where to "
            "look on disk."
        ),
    )
    briefing.add_argument(
        "--text",
        action="store_true",
        help="Print the rendered briefing only. This is what a SessionStart hook wants.",
    )
    briefing.set_defaults(func=cmd_briefing)
    ledger = commands.add_parser(
        "ledger", help="Which tasks are verified done, which failed, which are in flight"
    )
    add_context_args(ledger)
    ledger.add_argument("--task-ref", help="One task's full attempt chain, ignoring scope")
    ledger.add_argument("--limit", type=int, default=25)
    ledger.set_defaults(func=cmd_ledger)
    mcp_parser = commands.add_parser("mcp", help="Run the core stdio MCP server")
    mcp_parser.add_argument("--profile", choices=["runtime", "admin"], default="runtime")
    mcp_parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="deprecated alias for --profile admin",
    )
    mcp_parser.add_argument(
        "--delivery-target",
        choices=["local_model", "hosted_model"],
        help=(
            "delivery filter for served memory; default local_model. "
            "Overrides OCBRAIN_DELIVERY_TARGET."
        ),
    )
    mcp_parser.add_argument("--active-db-file", type=Path, help=argparse.SUPPRESS)
    mcp_parser.set_defaults(func=cmd_mcp)
    # Public-safety enforcement lives in core, not an optional companion: it is
    # the gate that keeps private paths and secrets out of this public repo, and
    # CI runs it on every push. A guard nobody has installed is not a guard.
    public_safety_parser = commands.add_parser(
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

    # Standing health scorecard. Read-only by construction, so it is safe to
    # point at the live core an MCP server is serving from, and cheap enough to
    # run hourly. Exits non-zero on any alarm so cron can gate on it.
    selftest_parser = commands.add_parser(
        "selftest", help="Score core health against documented thresholds (read-only)"
    )
    selftest_parser.add_argument(
        "--since-days", type=int, default=30, help="window for windowed metrics (default: 30)"
    )
    selftest_parser.add_argument(
        "--baseline", type=Path, help="a previously saved scorecard to diff against"
    )
    selftest_parser.add_argument("--out", type=Path, help="write this scorecard to a JSON file")
    selftest_parser.add_argument(
        "--transcript-root",
        type=Path,
        help="where session transcripts live (default: ~/.claude/projects)",
    )
    # SUPPRESS, not store_true: --pretty is a global flag, and a subparser
    # default would overwrite the global value with False whenever the flag was
    # given before the subcommand. Suppressing leaves the parent's value alone,
    # so both `ocbrain --pretty selftest` and `ocbrain selftest --pretty` work.
    selftest_parser.add_argument(
        "--pretty",
        action="store_true",
        default=argparse.SUPPRESS,
        help="render the human-readable table instead of JSON",
    )
    selftest_parser.set_defaults(func=cmd_selftest)

    install_hooks_parser = commands.add_parser(
        "install-hooks", help="Symlink tracked git hooks (ops/hooks) into .git/hooks"
    )
    install_hooks_parser.add_argument(
        "--root", type=Path, help="repo root (default: git toplevel of the cwd)"
    )
    install_hooks_parser.set_defaults(func=cmd_install_hooks)

    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
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


def scope_for_privacy(project: str | None, privacy_scope: str) -> ScopeTag:
    if privacy_scope == "public":
        return global_scope()
    base = resolve_write_scope(ScopeContext(project=project))
    if privacy_scope == "private":
        return ScopeTag(
            base.scope_type,
            base.scope_id,
            visibility="confidential",
            egress_policy="prohibited",
            provenance="explicit",
        )
    return ScopeTag(
        base.scope_type,
        base.scope_id,
        visibility="internal",
        egress_policy="local_only",
        provenance="explicit",
    )


def open_db(args: argparse.Namespace):
    conn = connect(args.db)
    if is_core_v1(conn):
        if migrate_core_v1_columns(conn):
            conn.commit()
        return conn
    has_tables = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        is not None
    )
    if has_tables:
        init_db(conn)
    else:
        init_core_v1(conn)
    return conn


def open_existing_core_v1(path: Path):
    """Open an existing strict-v1 core without creating or migrating a path."""
    resolved = path.expanduser()
    if not resolved.is_file():
        raise ValueError(f"strict-v1 core database does not exist: {resolved}")
    conn = connect(resolved)
    if not is_core_v1(conn):
        conn.close()
        raise ValueError("bundle commands require an initialized strict-v1 core")
    return conn


def cmd_export_bundle(args: argparse.Namespace) -> int:
    conn = open_existing_core_v1(args.db)
    try:
        result = export_bundle(
            conn,
            args.output,
            evidence_ids=args.evidence_id,
            context=context_from_args(args),
            approve_egress=args.approve_egress,
        )
    finally:
        conn.close()
    output(args, result)
    return 0


def cmd_import_bundle(args: argparse.Namespace) -> int:
    if not args.apply:
        output(
            args,
            import_bundle(None, args.path, project=args.project, apply=False),
        )
        return 0
    conn = open_existing_core_v1(args.db)
    try:
        result = import_bundle(conn, args.path, project=args.project, apply=True)
    finally:
        conn.close()
    output(args, result)
    return 0


def v1_counts(conn) -> dict[str, int]:
    result = {
        name: int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
        for name in ("brain_events", "evidence_objects", "current_beliefs", "retrieval_uses")
    }
    # Stable output aliases ease automation migration without recreating the
    # retired relational tables inside the v1 database.
    result["evidence"] = result["evidence_objects"]
    result["knowledge"] = result["current_beliefs"]
    return result


def compatibility_refusal(args: argparse.Namespace, command: str, detail: str) -> int:
    output(
        args,
        {
            "action": command,
            "status": "blocked",
            "reason": "legacy_compatibility_command_on_v1_core",
            "detail": detail,
        },
    )
    return 2


def cmd_init(args: argparse.Namespace) -> int:
    existed = args.db.expanduser().exists()
    conn = connect(args.db)
    has_tables = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        is not None
    )
    if not existed or not has_tables:
        init_core_v1(conn)
    elif is_core_v1(conn):
        pass
    else:
        # Existing v0.x ledgers keep their compatibility schema. Migration to
        # v1 remains an explicit archive-first command and never happens here.
        init_db(conn)
    if is_core_v1(conn):
        conn.commit()
        payload = {"db": str(args.db), "core": "v1", "database": database_status(args.db)}
    else:
        payload = {"db": str(args.db), "core": "legacy", "counts": counts(conn)}
    output(args, payload)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    result = database_status(args.db)
    output(
        args,
        {
            "action": "status",
            "database": result,
            "operating_model": {
                "core": "explicit one-shot commands plus stdio MCP",
                "scheduler_installed_by_core": False,
            },
        },
    )
    return 0 if result.get("healthy") else 1


def cmd_sync(args: argparse.Namespace) -> int:
    result = sync_core(
        args.db,
        max_events=args.max_events,
        time_budget_seconds=args.time_budget,
        full=args.full,
    )
    output(args, result)
    return 0 if result["status"] == "ok" else 3


def cmd_vector_build(args: argparse.Namespace) -> int:
    if not args.db.expanduser().is_file():
        raise ValueError(f"strict-v1 core database does not exist: {args.db.expanduser()}")
    result = build_vector_index(
        args.db,
        output_path=args.output,
        model=args.model,
        endpoint=args.endpoint,
        batch_size=args.batch_size,
    )
    output(args, {"action": "vector-build", **result})
    return 0


def cmd_vector_status(args: argparse.Namespace) -> int:
    result = vector_status(args.db, sidecar_path=args.sidecar)
    output(args, {"action": "vector-status", **result})
    return 0 if result.get("healthy") else 1


def cmd_curated_apply(args: argparse.Namespace) -> int:
    conn = open_existing_core_v1(args.db)
    try:
        result = apply_curated_manifest(
            conn,
            args.manifest,
            actor=args.actor,
            allow_hosted_egress=args.allow_hosted_egress,
        )
    finally:
        conn.close()
    output(args, {"action": "curated-apply", **result})
    return 0


def cmd_feedback_repair(args: argparse.Namespace) -> int:
    """Report, or on ``--apply`` rewrite, relevance verdicts on empty retrievals.

    Deliberately opt-in with a reporting default: the rows are live history, and
    an automatic migration would rewrite an operator's corpus on the next open
    of a version they did not choose to run.
    """
    conn = open_existing_core_v1(args.db)
    try:
        report = reclassify_no_coverage_receipts(conn, apply=bool(args.apply))
        if args.apply:
            conn.commit()
    finally:
        conn.close()
    output(args, {"action": "feedback-repair", **report})
    return 0


def cmd_hygiene(args: argparse.Namespace) -> int:
    conn = open_existing_core_v1(args.db)
    try:
        if args.supersede:
            belief_id, successor_id = args.supersede
            result = supersede(conn, belief_id=belief_id, successor_id=successor_id)
            output(args, {"action": "hygiene", "superseded": result})
            return 0
        if args.restore:
            output(
                args,
                {"action": "hygiene", "restored": restore(conn, belief_id=args.restore)},
            )
            return 0
        plan = plan_retirements(
            conn,
            classes=tuple(args.classes) if args.classes else HYGIENE_CLASSES,
            batch_cap=args.batch_cap,
            restatement_threshold=args.restatement_threshold,
        )
        if args.apply:
            plan = apply_retirements(conn, plan)
        # Report the sample, not every id: a capped run can still name hundreds.
        verbose_keys = {"targets", "applied_belief_ids"}
        payload = {key: value for key, value in plan.items() if key not in verbose_keys}
        payload["target_sample"] = plan.get("targets", [])[:12]
        payload["invariants"] = verify_serving_invariants(conn)
        # Distinct from the plan's `applied` count, which only exists after a run.
        payload["apply_requested"] = bool(args.apply)
    finally:
        conn.close()
    output(args, {"action": "hygiene", **payload})
    return 0


def cmd_wiki_volatility(args: argparse.Namespace) -> int:
    """Plan, or on an explicit double confirmation apply, volatility-class TTLs.

    Dry run by default and dry run unless *both* ``--apply`` and ``--yes`` are
    given, because the number that matters here -- how many serving beliefs the
    new scheme has already expired -- is only knowable from the plan.
    """
    from ocbrain.curator import (
        apply_volatility_ttl,
        curator_runtime_settings,
        plan_volatility_ttl,
    )

    # The same off switch the compile path honours. `curator.current_ttl_days=0`
    # means "this brain does not expire beliefs"; a sweep that re-dated them
    # anyway would be the operator control that works in one place only.
    current_ttl_days = curator_runtime_settings()["current_ttl_days"]
    conn = open_existing_core_v1(args.db)
    try:
        if args.apply and args.yes:
            result = apply_volatility_ttl(
                conn,
                actor=args.actor,
                current_ttl_days=current_ttl_days,
                volatile_ttl_days=args.volatile_days,
                measured_ttl_days=args.measured_days,
            )
            result["applied"] = True
        else:
            result = plan_volatility_ttl(
                conn,
                current_ttl_days=current_ttl_days,
                volatile_ttl_days=args.volatile_days,
                measured_ttl_days=args.measured_days,
            )
            result["applied"] = False
            if args.apply and not args.yes:
                result["refused"] = "--apply needs --yes; nothing was written"
    finally:
        conn.close()
    output(args, {"action": "wiki-volatility", **result})
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    # The one command an operator runs to debug their config must report a
    # malformed file, not crash on it.
    try:
        report = describe_config()
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc
    if args.section:
        if args.section not in report["sections"]:
            available = ", ".join(sorted(report["sections"]))
            raise SystemExit(f"unknown section {args.section!r}; available: {available}")
        report["sections"] = {args.section: report["sections"][args.section]}
    if args.changed_only:
        report["sections"] = {
            name: {k: v for k, v in entries.items() if v["source"] != "default"}
            for name, entries in report["sections"].items()
        }
        report["sections"] = {n: e for n, e in report["sections"].items() if e}
    output(args, {"action": "config", **report})
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ops = bool(getattr(args, "ops", False))
    write_manifest = bool(getattr(args, "write_manifest", False))
    replace_manifest = bool(getattr(args, "replace_manifest", False))
    ops_manifest = getattr(args, "ops_manifest", None)
    if not ops and (write_manifest or replace_manifest or ops_manifest is not None):
        output(
            args,
            {
                "action": "doctor",
                "status": "failed",
                "healthy": False,
                "error": "--write-manifest, --replace-manifest, and --ops-manifest require --ops",
            },
        )
        return 2
    if replace_manifest and not write_manifest:
        output(
            args,
            {
                "action": "doctor",
                "status": "failed",
                "healthy": False,
                "error": "--replace-manifest requires --write-manifest",
            },
        )
        return 2
    result = doctor(
        args.db,
        timeout_seconds=args.timeout,
        launcher=args.launcher,
        check_clients=False,
    )
    if ops:
        from ocbrain.opscheck import ops_check, write_ops_manifest

        if write_manifest:
            try:
                write_ops_manifest(ops_manifest, replace=replace_manifest)
            except OSError as exc:
                result["ops"] = {
                    "action": "ops-manifest-write",
                    "status": "failed",
                    "healthy": False,
                    "error": str(exc),
                }
                result["healthy"] = False
                result["status"] = "failed"
                output(args, result)
                return 2
        result["ops"] = ops_check(ops_manifest)
        result["healthy"] = bool(result["healthy"]) and bool(result["ops"]["healthy"])
        result["status"] = "ok" if result["healthy"] else "failed"
    output(args, result)
    return 0 if result["healthy"] else 1


def cmd_runtime_check(args: argparse.Namespace) -> int:
    result = doctor(
        args.db,
        timeout_seconds=args.timeout,
        launcher=args.launcher,
        check_clients=True,
    )
    output(args, result)
    return 0 if result["healthy"] else 1


def cmd_backup(args: argparse.Namespace) -> int:
    result = backup_database(args.db, args.output, manifest=args.manifest)
    output(args, {"action": "backup", "status": "verified"} | result)
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    result = restore_database(args.backup, args.output_db, manifest=args.manifest)
    output(args, {"action": "restore", "status": "verified"} | result)
    return 0


def cmd_core_migrate_v1(args: argparse.Namespace) -> int:
    from ocbrain.v1_migration import migrate_core_v1, migration_plan

    if args.plan:
        result = migration_plan(
            args.db,
            args.core_db,
            args.archive_db,
            args.manifest,
            training=args.training_db,
            ops=args.ops_db,
        )
        output(args, result)
        return 0 if result["ready"] else 2
    result = migrate_core_v1(
        args.db,
        args.core_db,
        args.archive_db,
        args.manifest,
        training=args.training_db,
        ops=args.ops_db,
    )
    output(args, result)
    return 0


def cmd_evidence(args: argparse.Namespace) -> int:
    conn = open_db(args)
    claim, raw = evidence_claim(args)
    source_uri = args.source_uri or (str(args.input) if args.input else None)
    if is_core_v1(conn):
        scope = scope_for_privacy(args.project, args.privacy_scope)
        body = redact_secrets(raw if args.input else claim)
        evidence_id, event_id = record_core_v1_evidence(
            conn,
            body=body,
            kind=args.source_type,
            scope=scope,
            writer=args.source_runtime or "ocbrain-cli",
            artifact_ref=args.artifact_uri or source_uri,
        )
        conn.commit()
        output(
            args,
            {
                "event_id": event_id,
                "evidence_id": evidence_id,
                "scope": scope.to_dict(),
                "counts": v1_counts(conn),
            },
        )
        return 0
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
    if is_core_v1(conn):
        clauses: list[str] = []
        params: list[object] = []
        if args.status:
            clauses.append("status=?")
            params.append(args.status)
        if args.type:
            clauses.append("belief_type=?")
            params.append(args.type)
        if not args.include_private:
            clauses.append("visibility NOT IN ('confidential','secret')")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM current_beliefs {where} "
            "ORDER BY pinned DESC, last_compiled_at DESC, belief_id LIMIT ?",
            (*params, args.limit),
        )
        output(
            args,
            {
                "schema_version": "ocbrain.knowledge.v1",
                "knowledge": [dict(row) for row in rows],
            },
        )
        return 0
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
    if is_core_v1(conn):
        value = args.text
        if args.number is not None:
            value = str(args.number)
        elif value_bool is not None:
            value = str(value_bool).lower()
        rendered = " ".join(
            part for part in (args.subject, args.predicate, value, args.unit) if part is not None
        )
        scope = scope_for_privacy(args.project, args.privacy_scope)
        evidence_id, evidence_event_id = record_core_v1_evidence(
            conn,
            body=rendered,
            kind="typed_value",
            scope=scope,
            writer="ocbrain-cli",
        )
        belief_id = stable_id("belief", "value", args.subject, args.predicate, scope.scope_id)
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "schema_version": "ocbrain.compilation.v1",
                "subject": {"kind": "belief", "id": belief_id},
                "belief_id": belief_id,
                "body": rendered,
                "evidence_ids": [evidence_id],
                "scope": scope.to_dict(),
                "confidence": args.confidence,
            },
            writer="ocbrain-cli",
        )
        decision = None
        if args.status == "current" or args.inject:
            decision = decide_proposal_v1(
                conn,
                proposal_event_id=proposal_id,
                decision="approve",
                actor="ocbrain-cli",
                edited_body=None,
                reason="explicit current/inject value command",
            )
        conn.commit()
        output(
            args,
            {
                "belief_id": belief_id,
                "evidence_id": evidence_id,
                "evidence_event_id": evidence_event_id,
                "proposal_event_id": proposal_id,
                "decision_event_id": decision["event_id"] if decision else None,
                "status": "current" if decision else "candidate",
                "scope": scope.to_dict(),
                "counts": v1_counts(conn),
            },
        )
        return 0
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
    if is_core_v1(conn):
        result = search_v1(
            conn,
            args.query,
            context=ScopeContext(project=args.project, runtime="cli"),
            limit=args.limit,
            cross_scope=args.include_private,
        )
        conn.commit()
        output(args, result)
        return 0
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
    if is_core_v1(conn):
        if args.at_ts:
            return compatibility_refusal(
                args,
                "preview",
                "historical folding is not exposed by the v1 shared-context packet",
            )
        context = context_from_args(args)
        packet, handles = build_context_v1(
            conn,
            args.query,
            context=context,
            limit=args.limit,
            cross_scope=args.cross_scope,
        )
        retrieval_id = record_context_v1(conn, packet, handles, context=context)
        packet["retrieval_use_id"] = retrieval_id
        packet["retrieval_use_status"] = "recorded"
        conn.commit()
        output(args, packet)
        return 0
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
    body = args.body
    kind = args.kind
    if kind in SKILL_TELEMETRY_KINDS:
        envelope = validate_skill_telemetry(body)
        if envelope["kind"] != kind:
            raise ValueError("skill telemetry body kind must match event-ingest kind")
        body = canonical_json(envelope)
    scope = global_scope() if args.global_doctrine else resolve_write_scope(context_from_args(args))
    if is_core_v1(conn):
        if args.global_doctrine:
            evidence_id, event_id = record_core_v1_evidence(
                conn,
                body=body,
                kind=kind,
                scope=scope,
                writer=args.writer,
                session_id=args.session,
                artifact_ref=args.artifact_ref,
            )
            result = {"event_id": event_id, "evidence_id": evidence_id, "kind": args.kind}
        else:
            result = ingest_v1(
                conn,
                body=body,
                kind=kind,
                context=context_from_args(args),
                writer=args.writer,
                session_id=args.session,
                artifact_ref=args.artifact_ref,
            )
        conn.commit()
        output(args, result | {"scope": scope.to_dict(), "counts": v1_counts(conn)})
        return 0
    event_id = record_evidence(
        conn,
        body=body,
        kind=kind,
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
    if is_core_v1(conn):
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "schema_version": "ocbrain.compilation.v1",
                "subject": {"kind": "belief", "id": args.belief_id},
                "belief_id": args.belief_id,
                "body": args.body,
                "evidence_ids": args.evidence_id,
                "scope": scope.to_dict(),
                "confidence": args.confidence,
            },
            writer="ocbrain-cli",
            session_id=args.session,
        )
        decision = None
        if args.approve:
            decision = decide_proposal_v1(
                conn,
                proposal_event_id=proposal_id,
                decision="approve",
                actor="ocbrain-cli",
                edited_body=None,
                reason="explicit --approve",
            )
        conn.commit()
        output(
            args,
            {
                "proposal_event_id": proposal_id,
                "decision_event_id": decision["event_id"] if decision else None,
                "scope": scope.to_dict(),
                "counts": v1_counts(conn),
            },
        )
        return 0
    proposal_id = propose_compilation(
        conn,
        belief_id=args.belief_id,
        body=args.body,
        evidence_ids=args.evidence_id,
        scope=scope,
        confidence=args.confidence,
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
    if is_core_v1(conn):
        result = correct_v1(
            conn,
            layer=args.target_layer,
            target=args.target_id,
            op=args.op,
            body=args.body,
            actor=args.author,
            hard=args.hard,
        )
        conn.commit()
        output(args, result | {"counts": v1_counts(conn)})
        return 0
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
    if is_core_v1(conn):
        result = forget_v1(
            conn,
            target=args.target,
            mode=args.mode,
            reason=args.reason,
            actor=args.approved_by,
        )
        conn.commit()
        output(args, result | {"counts": v1_counts(conn)})
        return 0
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


# A workspace-scoped durable wiki fact in one of these categories is doctrine:
# it describes how the operator works, not what one project is doing. Left in
# `project:workspace` it is invisible to every other project, which is how a
# brain ends up with 97 beliefs that most retrievals cannot reach.
DURABLE_PREFERENCE_CATEGORIES = ("preference", "decision", "workflow", "system")
DURABLE_PREFERENCE_SQL = """
SELECT belief_id FROM current_beliefs
WHERE scope_id='project:workspace' AND status='current' AND serve=1
  AND belief_type='wiki_fact'
  AND json_extract(attributes_json, '$.lifecycle')='durable'
  AND json_extract(attributes_json, '$.category') IN (?, ?, ?, ?)
ORDER BY belief_id
"""


def cmd_scope_promote(args: argparse.Namespace) -> int:
    """Emit the `scope_promoted` event the projector has always known how to apply.

    The event, its projection, and its rebuild path all shipped; only a way to
    write one was missing, which is why zero beliefs in a real brain are global.

    A promotion widens *reach* and never *egress*. Each belief keeps its own
    visibility and egress_policy, so a `local_only` belief promoted to
    `global:doctrine` becomes recallable from every project on this machine and
    is still refused for hosted delivery — `_delivery_sql('hosted_model')`
    requires `egress_policy='hosted_ok'`, which the promotion does not grant.
    """
    conn = open_db(args)
    if not is_core_v1(conn):
        return compatibility_refusal(
            args, "scope-promote", "scope promotion requires an event-authoritative v1 core"
        )
    scope_type = args.to_scope_type
    scope_id = args.to_scope_id
    belief_ids = list(dict.fromkeys(args.belief_id))
    if args.select_durable_preferences:
        selected = conn.execute(DURABLE_PREFERENCE_SQL, DURABLE_PREFERENCE_CATEGORIES).fetchall()
        belief_ids.extend(str(row["belief_id"]) for row in selected)
        belief_ids = list(dict.fromkeys(belief_ids))
        scope_type = scope_type or "global"
        scope_id = scope_id or DEFAULT_GLOBAL_SCOPE_ID
    if not belief_ids:
        output(args, {"action": "scope-promote", "status": "no_beliefs_selected", "promoted": []})
        return 0
    if not scope_type or not scope_id:
        output(
            args,
            {
                "action": "scope-promote",
                "status": "blocked",
                "reason": "target_scope_required",
                "detail": "pass --to-scope-type and --to-scope-id, or --select-durable-preferences",
            },
        )
        return 2

    promoted: list[dict[str, Any]] = []
    unchanged: list[str] = []
    missing: list[str] = []
    for belief_id in belief_ids:
        belief = get_core_v1_belief(conn, belief_id)
        if belief is None or belief.get("status") != "current":
            missing.append(belief_id)
            continue
        current = belief["scope"]
        # Carry visibility and egress through verbatim. A promotion that also
        # relaxed them would be an egress decision wearing a scope decision's
        # clothes, and nothing downstream would show the difference.
        target = ScopeTag(
            scope_type,
            scope_id,
            visibility=str(current["visibility"]),
            egress_policy=str(current["egress_policy"]),
            provenance="scope_promoted",
        )
        if str(current["scope_type"]) == scope_type and str(current["scope_id"]) == scope_id:
            unchanged.append(belief["canonical_id"])
            continue
        entry = {
            "belief_id": belief["canonical_id"],
            "from_scope": {"scope_type": current["scope_type"], "scope_id": current["scope_id"]},
            "to_scope": target.to_dict(),
        }
        if not args.dry_run:
            entry["event_id"] = append_core_event(
                conn,
                "scope_promoted",
                {
                    "schema_version": "ocbrain.scope-promotion.v1",
                    "subject": {"kind": "belief", "id": belief["canonical_id"]},
                    "belief_id": belief["canonical_id"],
                    "scope": target.to_dict(),
                    "approved_by": args.approved_by,
                    "reason": args.reason,
                },
                writer="ocbrain-cli",
                project=True,
            )
        promoted.append(entry)
    if not args.dry_run:
        conn.commit()
    output(
        args,
        {
            "action": "scope-promote",
            "status": "planned" if args.dry_run else "applied",
            "dry_run": bool(args.dry_run),
            "approved_by": args.approved_by,
            "promoted": promoted,
            "unchanged": unchanged,
            "missing": missing,
            "counts": v1_counts(conn),
        },
    )
    return 0


# Hosted approval is the human-gated bridge between "this evidence may not
# reach a hosted model without a decision" and "this belief may". The verbs are
# CLI-only on purpose: nothing in the MCP runtime or admin catalogue can stamp
# an egress policy, so an unattended write can never reach hosted-model
# delivery — it can only ask (a hosted_egress_proposal) or stay narrowed.
HOSTED_QUEUE_SCHEMA_VERSION = "ocbrain.hosted-queue.v1"
HOSTED_APPROVAL_PROPOSAL_SCHEMA = "ocbrain.compilation.v1"
HOSTED_APPROVAL_CONFIDENCE = 0.8
HOSTED_QUEUE_HEAD_CHARS = 120
# The approval is worth nothing if an agent can type it. `human:NAME` is the
# same spelling scope-promote's `--approved-by` and event-forget use.
_HOSTED_APPROVED_BY_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9._-]*$")

_HOSTED_QUEUE_EVIDENCE_SQL = """
SELECT e.evidence_id, e.kind, e.body, e.body_head, e.scope_type, e.scope_id,
       e.visibility, e.egress_policy, e.recorded_at, e.source_runtime AS writer
FROM evidence_objects AS e
WHERE e.egress_policy = 'approval_required'
  AND e.visibility NOT IN ('confidential', 'secret')
  AND NOT EXISTS (
    SELECT 1 FROM belief_evidence AS be
    JOIN current_beliefs AS cb ON cb.belief_id = be.belief_id
    WHERE be.evidence_id = e.evidence_id AND cb.status = 'current'
  )
{filters}
ORDER BY e.recorded_at DESC
LIMIT ?
"""

# A widening request is pending for exactly as long as the evidence it names is
# still uncompiled: once a human-approved path compiles it, the request has
# been answered and drops out of the queue on its own.
_HOSTED_QUEUE_PROPOSAL_SQL = """
SELECT p.id, p.ts, p.writer, p.body_json
FROM brain_events AS p
WHERE p.kind = 'hosted_egress_proposal'
  AND NOT EXISTS (
    SELECT 1 FROM belief_evidence AS be
    JOIN current_beliefs AS cb ON cb.belief_id = be.belief_id
    WHERE be.evidence_id = json_extract(p.body_json, '$.evidence_id')
      AND cb.status = 'current'
  )
{filters}
ORDER BY p.rowid DESC
LIMIT ?
"""


def _hosted_queue_filters(args: argparse.Namespace) -> dict[str, str | None]:
    return {
        "project": getattr(args, "project", None),
        "writer": getattr(args, "writer", None),
        "since": getattr(args, "since", None),
    }


def _hosted_queue_rows(
    conn,
    *,
    project: str | None = None,
    writer: str | None = None,
    since: str | None = None,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Eligible evidence rows plus pending widening proposals. Reads only."""
    clauses: list[str] = []
    params: list[Any] = []
    if project:
        folded = fold_scope_component(project) or str(project)
        clauses.append("AND e.scope_type='project' AND e.scope_id=?")
        params.append(f"project:{folded}")
    if writer:
        clauses.append("AND e.source_runtime=?")
        params.append(str(writer))
    if since:
        clauses.append("AND e.recorded_at>=?")
        params.append(str(since))
    evidence_rows = [
        dict(row)
        for row in conn.execute(
            _HOSTED_QUEUE_EVIDENCE_SQL.format(filters=" ".join(clauses)),
            (*params, max(1, limit)),
        )
    ]
    proposal_clauses: list[str] = []
    proposal_params: list[Any] = []
    if project:
        folded = fold_scope_component(project) or str(project)
        proposal_clauses.append(
            "AND json_extract(p.body_json, '$.requested_scope.scope_id')=?"
        )
        proposal_params.append(f"project:{folded}")
    if writer:
        proposal_clauses.append("AND p.writer=?")
        proposal_params.append(str(writer))
    if since:
        proposal_clauses.append("AND p.ts>=?")
        proposal_params.append(str(since))
    proposal_rows = []
    for row in conn.execute(
        _HOSTED_QUEUE_PROPOSAL_SQL.format(filters=" ".join(proposal_clauses)),
        (*proposal_params, max(1, limit)),
    ):
        body = json.loads(row["body_json"])
        proposal_rows.append(
            {
                "proposal_event_id": str(row["id"]),
                "ts": str(row["ts"]),
                "evidence_id": str(body.get("evidence_id")),
                "writer": str(row["writer"]),
                "requested_scope": body.get("requested_scope"),
                "inferred_scope": body.get("inferred_scope"),
                "body_head": body.get("body_head"),
            }
        )
    return evidence_rows, proposal_rows


def _hosted_head(text: str | None, limit: int = HOSTED_QUEUE_HEAD_CHARS) -> str:
    value = compact_whitespace(str(text or ""))
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _hosted_entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": str(row["evidence_id"]),
        "kind": str(row["kind"]),
        "writer": str(row["writer"] or ""),
        "scope_id": str(row["scope_id"]),
        "recorded_at": str(row["recorded_at"]),
        "body_head": _hosted_head(row["body_head"] or row["body"]),
    }


def _evidence_has_current_belief(conn, evidence_id: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM belief_evidence AS be "
            "JOIN current_beliefs AS cb ON cb.belief_id = be.belief_id "
            "WHERE be.evidence_id=? AND cb.status='current' LIMIT 1",
            (evidence_id,),
        ).fetchone()
        is not None
    )


def _requested_project_from_proposal(conn, evidence_id: str) -> tuple[str, str] | None:
    """(project, proposal_event_id) the evidence's own widening request asked for, if any.

    Approving the queue answers that request; landing at the row's task scope would
    keep the egress and drop the reach it named. Visibility never comes from here, and
    the gauntlet has already run, so a client-scoped row still needs ``--project``.
    """
    row = conn.execute(
        "SELECT id, body_json FROM brain_events WHERE kind='hosted_egress_proposal' "
        "AND json_extract(body_json,'$.evidence_id')=? ORDER BY rowid DESC LIMIT 1",
        (evidence_id,),
    ).fetchone()
    if row is None:
        return None
    requested = (json.loads(row["body_json"]) or {}).get("requested_scope") or {}
    if requested.get("scope_type") != "project":
        return None
    scope_id = str(requested.get("scope_id") or "")
    if not scope_id.startswith("project:") or len(scope_id) <= len("project:"):
        return None
    return scope_id[len("project:") :], str(row["id"])


def _hosted_target_scope(project: str | None, row: dict[str, Any]) -> ScopeTag:
    """The project scope the approved belief lands in, hosted-safe by decree.

    Only a human CLI decision reaches this point, so this is the single place
    an `approval_required` row is allowed to become `hosted_ok`. When no
    project is derivable the belief keeps the evidence's own scope: reach
    never widens past what the row already had.
    """
    if project:
        folded = fold_scope_component(project) or str(project)
        return ScopeTag(
            "project",
            f"project:{folded}",
            visibility="internal",
            egress_policy="hosted_ok",
            provenance="human_approved_hosted",
        )
    scope_type = str(row["scope_type"])
    return ScopeTag(
        scope_type,
        str(row["scope_id"]),
        visibility="internal",
        egress_policy="hosted_ok",
        provenance="human_approved_hosted",
    )


def _hosted_approve_one(
    conn,
    *,
    evidence_id: str,
    approved_by: str,
    reason: str | None,
    project: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    canonical = resolve_object_id(conn, evidence_id)

    def refused(rsn: str, detail: str | None = None) -> dict[str, Any]:
        entry: dict[str, Any] = {"evidence_id": evidence_id, "status": "refused", "reason": rsn}
        if detail:
            entry["detail"] = detail
        return entry

    row = conn.execute(
        "SELECT evidence_id, kind, body, body_head, scope_type, scope_id, visibility, "
        "egress_policy, recorded_at FROM evidence_objects WHERE evidence_id=?",
        (canonical,),
    ).fetchone()
    if row is None:
        return refused("not_found", f"no evidence row: {canonical}")
    if str(row["visibility"]) in {"confidential", "secret"}:
        return refused(
            f"visibility_{row['visibility']}",
            "hosted approval refuses confidential and secret evidence",
        )
    if str(row["egress_policy"]) != "approval_required":
        return refused(
            "egress_not_approval_required",
            f"current egress policy is {row['egress_policy']}",
        )
    if _evidence_has_current_belief(conn, canonical):
        return refused("already_compiled", "a current belief already covers this evidence")
    if project and str(row["scope_type"]) == "project":
        folded = fold_scope_component(project) or str(project)
        target_scope_id = f"project:{folded}"
        if resolve_scope_alias(row["scope_id"]) != resolve_scope_alias(target_scope_id):
            return refused(
                "project_scope_mismatch",
                "project-scoped evidence may only be approved into its existing project",
            )
    if str(row["scope_type"]) == "client" and not project:
        return refused(
            "client_scope_requires_project",
            "pass --project to name the project this client evidence may serve",
        )
    body_text = compact_whitespace(str(row["body"] or row["body_head"] or ""))
    leaks = find_probable_secret_leaks(body_text)
    if leaks:
        return refused(
            "secret_leak_body",
            f"body trips the public-safety scanner: {', '.join(leaks)}",
        )
    scope_source = "cli_project" if project else "evidence_row"
    requested = _requested_project_from_proposal(conn, canonical)
    answers_proposal = requested[1] if requested is not None else None
    if not project and requested is not None:
        project = requested[0]
        scope_source = "requested_by_proposal"
    target = _hosted_target_scope(project, dict(row))
    belief_id = stable_id("belief", body_text, target.scope_id)
    evidence_ids = [
        str(item["evidence_id"])
        for item in conn.execute(
            "SELECT be.evidence_id FROM belief_evidence AS be "
            "JOIN current_beliefs AS cb ON cb.belief_id=be.belief_id "
            "WHERE be.belief_id=? AND cb.status='current' "
            "ORDER BY be.created_at, be.evidence_id",
            (belief_id,),
        )
    ]
    if canonical not in evidence_ids:
        evidence_ids.append(canonical)
    if dry_run:
        return {
            "evidence_id": canonical,
            "status": "planned",
            "belief_id": belief_id,
            "scope": target.to_dict(),
            "scope_source": scope_source,
        }
    attributes: dict[str, Any] = {"approved_by": approved_by}
    if reason:
        attributes["approval_reason"] = reason
    if answers_proposal:
        attributes["answers_proposal"] = answers_proposal
    proposal_event_id = append_core_event(
        conn,
        "compilation_proposed",
        {
            "schema_version": HOSTED_APPROVAL_PROPOSAL_SCHEMA,
            "subject": {"kind": "belief", "id": belief_id},
            "belief_id": belief_id,
            "body": body_text,
            # Two independently approved evidence rows may converge on the
            # same body+scope-derived belief id. A compilation decision replaces
            # that belief's support set, so carry the existing current supports
            # forward before adding this approval rather than dropping the
            # earlier evidence link.
            "evidence_ids": evidence_ids,
            "scope": target.to_dict(),
            "confidence": HOSTED_APPROVAL_CONFIDENCE,
            "attributes": attributes,
        },
        writer="ocbrain-cli",
    )
    decision = decide_proposal_v1(
        conn,
        proposal_event_id=proposal_event_id,
        decision="approve",
        actor=approved_by,
        edited_body=None,
        reason=reason or "hosted approval queue",
    )
    return {
        "evidence_id": canonical,
        "status": "approved",
        "belief_id": belief_id,
        "proposal_event_id": proposal_event_id,
        "decision_event_id": decision["event_id"],
        "scope": target.to_dict(),
        "scope_source": scope_source,
    }


def cmd_hosted_queue(args: argparse.Namespace) -> int:
    """List approval_required evidence and pending widening requests (read-only).

    Opens the core without migrating or committing anything: this verb must be
    safe to point at the live operator database between other people's writes.
    """
    conn = connect(args.db)
    try:
        if not is_core_v1(conn):
            return compatibility_refusal(
                args, "hosted-queue", "hosted approval requires an event-authoritative v1 core"
            )
        evidence_rows, proposal_rows = _hosted_queue_rows(
            conn,
            project=args.project,
            writer=args.writer,
            since=args.since,
            limit=args.limit,
        )
        output(
            args,
            {
                "action": "hosted-queue",
                "schema_version": HOSTED_QUEUE_SCHEMA_VERSION,
                "count": len(evidence_rows),
                "proposal_count": len(proposal_rows),
                "queue": [_hosted_entry(row) for row in evidence_rows],
                "proposals": [
                    {**row, "body_head": _hosted_head(row.get("body_head"))}
                    for row in proposal_rows
                ],
                "filters": _hosted_queue_filters(args),
            },
        )
        return 0
    finally:
        conn.close()


def cmd_hosted_approve(args: argparse.Namespace) -> int:
    """Promote queued evidence to hosted_ok beliefs through the approved-compile path.

    Every compiled belief is minted by the same proposal + decide events
    `event-compile --approve` writes, with the decision's actor recorded as the
    `--approved-by human:NAME` value. Confidential and secret rows are refused
    outright; `--dry-run` runs the full eligibility gauntlet and writes nothing.
    """
    conn = connect(args.db)
    try:
        if not is_core_v1(conn):
            return compatibility_refusal(
                args, "hosted-approve", "hosted approval requires an event-authoritative v1 core"
            )
        approved_by = str(args.approved_by or "").strip()
        if not _HOSTED_APPROVED_BY_RE.match(approved_by):
            output(
                args,
                {
                    "action": "hosted-approve",
                    "status": "blocked",
                    "reason": "invalid_approved_by",
                    "detail": "--approved-by must spell the deciding human as human:NAME",
                },
            )
            return 2
        selected = [str(item) for item in args.evidence_id]
        if args.all_from_queue:
            evidence_rows, proposal_rows = _hosted_queue_rows(
                conn,
                project=args.project,
                writer=args.writer,
                since=args.since,
                limit=args.limit,
            )
            selected.extend(str(row["evidence_id"]) for row in evidence_rows)
            # Project-filtered widening requests live in ``proposal_rows``:
            # their underlying evidence is deliberately still task/session
            # scoped, so it cannot match the project-scoped evidence query.
            # Treat both queue sections as selectable; the stable dedupe below
            # handles evidence that appears in both sections without approving
            # it twice.
            selected.extend(str(row["evidence_id"]) for row in proposal_rows)
        selected = list(dict.fromkeys(selected))
        if not selected:
            if args.all_from_queue:
                output(
                    args,
                    {
                        "action": "hosted-approve",
                        "status": "nothing_queued",
                        "dry_run": bool(args.dry_run),
                        "approved_by": approved_by,
                        "approved": [],
                        "refused": [],
                        "counts": v1_counts(conn),
                    },
                )
                return 0
            output(
                args,
                {
                    "action": "hosted-approve",
                    "status": "blocked",
                    "reason": "no_evidence_selected",
                    "detail": "pass evidence id(s) or --all-from-queue",
                },
            )
            return 2
        approved: list[dict[str, Any]] = []
        refused: list[dict[str, Any]] = []
        for evidence_id in selected:
            outcome = _hosted_approve_one(
                conn,
                evidence_id=evidence_id,
                approved_by=approved_by,
                reason=args.reason,
                project=args.project,
                dry_run=bool(args.dry_run),
            )
            if outcome["status"] == "refused":
                refused.append(outcome)
            else:
                approved.append(outcome)
        if not args.dry_run:
            conn.commit()
        if approved:
            status = "planned" if args.dry_run else "applied"
            exit_code = 0
        else:
            status = "blocked"
            exit_code = 2
        output(
            args,
            {
                "action": "hosted-approve",
                "status": status,
                "dry_run": bool(args.dry_run),
                "approved_by": approved_by,
                "approved": approved,
                "refused": refused,
                "counts": v1_counts(conn),
            },
        )
        return exit_code
    finally:
        conn.close()


def cmd_egress_promote(args: argparse.Namespace) -> int:
    """Emit ``egress_promoted`` events: a human-attributable lift to hosted egress.

    ``scope-promote`` widens *reach* and never egress; this is the separate,
    human-only decision it deliberately refuses to make. The event changes
    egress ONLY — scope, visibility, body, confidence, and evidence ride
    through verbatim — and it refuses beliefs whose visibility is
    ``confidential`` or ``secret``, which can never be ``hosted_ok`` (the same
    rule ``curated-apply`` enforces; see
    :func:`ocbrain.scope.hosted_egress_refusal_reason`). Refusals are reported,
    never silently skipped.
    """
    conn = open_db(args)
    if not is_core_v1(conn):
        return compatibility_refusal(
            args, "egress-promote", "egress promotion requires an event-authoritative v1 core"
        )
    approved_by = str(args.approved_by or "").strip()
    if not _HOSTED_APPROVED_BY_RE.match(approved_by):
        output(
            args,
            {
                "action": "egress-promote",
                "status": "blocked",
                "reason": "invalid_approved_by",
                "detail": "--approved-by must spell the deciding human as human:NAME",
            },
        )
        return 2
    belief_ids = list(dict.fromkeys(args.belief_id))
    if args.scope_id:
        clauses = ["scope_id=?", "status='current'"]
        params: list[Any] = [args.scope_id]
        if args.provenance:
            clauses.append("scope_provenance=?")
            params.append(args.provenance)
        selected = conn.execute(
            "SELECT belief_id FROM current_beliefs "
            f"WHERE {' AND '.join(clauses)} ORDER BY belief_id",
            params,
        ).fetchall()
        belief_ids.extend(str(row["belief_id"]) for row in selected)
        belief_ids = list(dict.fromkeys(belief_ids))
    if not belief_ids:
        output(
            args,
            {
                "action": "egress-promote",
                "status": "no_beliefs_selected",
                "promoted": [],
                "unchanged": [],
                "refused": [],
            },
        )
        return 0

    promoted: list[dict[str, Any]] = []
    unchanged: list[str] = []
    refused: list[dict[str, str]] = []
    missing: list[str] = []
    for belief_id in belief_ids:
        belief = get_core_v1_belief(conn, belief_id)
        if belief is None or belief.get("status") != "current":
            missing.append(belief_id)
            continue
        scope = belief["scope"]
        from_egress = str(scope["egress_policy"])
        refusal = hosted_egress_refusal_reason(str(scope["visibility"]), args.to)
        if refusal is not None:
            refused.append({"belief_id": belief["canonical_id"], "reason": refusal})
            continue
        if from_egress == args.to:
            unchanged.append(belief["canonical_id"])
            continue
        entry = {
            "belief_id": belief["canonical_id"],
            "from_egress": from_egress,
            "to_egress": args.to,
        }
        if not args.dry_run:
            entry["event_id"] = append_core_event(
                conn,
                "egress_promoted",
                {
                    "schema_version": "ocbrain.egress-promotion.v1",
                    "subject": {"kind": "belief", "id": belief["canonical_id"]},
                    "belief_id": belief["canonical_id"],
                    # Audit history, not authority: the projector applies the
                    # target policy to the belief's current scope and refuses
                    # to touch anything else.
                    "scope": {
                        **scope,
                        "egress_policy": args.to,
                        "provenance": "egress_promoted",
                    },
                    "from_egress_policy": from_egress,
                    "to_egress_policy": args.to,
                    "approved_by": approved_by,
                    "reason": args.reason,
                },
                writer="ocbrain-cli",
                project=True,
            )
        promoted.append(entry)
    if not args.dry_run:
        conn.commit()
    output(
        args,
        {
            "action": "egress-promote",
            "status": "planned" if args.dry_run else "applied",
            "dry_run": bool(args.dry_run),
            "approved_by": approved_by,
            "promoted": promoted,
            "unchanged": unchanged,
            "refused": refused,
            "missing": missing,
            "counts": v1_counts(conn),
        },
    )
    return 0


def cmd_event_proposals(args: argparse.Namespace) -> int:
    conn = open_db(args)
    if is_core_v1(conn):
        output(
            args,
            proposals_v1(
                conn,
                limit=args.limit,
                include_decided=args.include_decided,
            ),
        )
        return 0
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
    if is_core_v1(conn):
        result = decide_proposal_v1(
            conn,
            proposal_event_id=args.proposal_event_id,
            decision=args.decision,
            actor=args.actor,
            edited_body=args.edited_body,
            reason=args.reason,
        )
        conn.commit()
        output(args, result | {"counts": v1_counts(conn)})
        return 0
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
    if is_core_v1(conn):
        if args.since_ts:
            return compatibility_refusal(
                args,
                "event-digest",
                "the v1 CLI digest currently exposes current projected state only",
            )
        result = digest_v1(conn, context=context_from_args(args), limit=args.limit)
        result["proposals"] = proposals_v1(
            conn,
            limit=args.limit,
            include_decided=False,
        )["proposals"]
        output(args, result)
        return 0
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


def cmd_event_backfill(args: argparse.Namespace) -> int:
    conn = open_db(args)
    if is_core_v1(conn):
        return compatibility_refusal(
            args,
            "event-backfill",
            "a v1 core has no in-place legacy relational rows; use core-migrate-v1 "
            "from the v0.x source",
        )
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
    imported: list[dict[str, object]] = []
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
    selection_skipped: list[dict[str, str]] = []
    files = memory_files(args.paths, skipped=selection_skipped)
    if args.limit is not None:
        files = files[: args.limit]
    if args.dry_run:
        return emit_import_dry_run(
            args,
            files,
            history=False,
            skipped=selection_skipped,
        )

    conn = open_db(args)
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = list(selection_skipped)
    for path in files:
        try:
            if is_core_v1(conn):
                result = import_memory_file_v1(
                    conn,
                    path,
                    project=args.project,
                    privacy_scope=args.privacy_scope,
                    max_bytes=args.max_bytes,
                    activate=not args.evidence_only,
                )
            else:
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
    payload = {
        "imported": sum(1 for item in imported if item.get("changed", True)),
        "existing": sum(1 for item in imported if item.get("changed") is False),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "files": imported,
        "counts": v1_counts(conn) if is_core_v1(conn) else counts(conn),
    }
    output(args, payload)
    return 0


def cmd_import_history(args: argparse.Namespace) -> int:
    selection_skipped: list[dict[str, str]] = []
    files = history_files(
        args.paths,
        manifests=args.manifest,
        skipped=selection_skipped,
    )
    if not files and not selection_skipped:
        raise ValueError("pass at least one history path or --manifest")
    if args.limit is not None:
        files = files[: args.limit]
    if args.dry_run:
        return emit_import_dry_run(
            args,
            files,
            history=True,
            skipped=selection_skipped,
        )

    conn = open_db(args)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    core_v1 = is_core_v1(conn)
    existing_sources = set() if core_v1 else imported_history_sources(conn)
    if core_v1:
        # Persisted stat-fingerprint gate: unchanged files skip the full
        # redact-and-read pass. import_source_v1 stays authoritative for any
        # file whose fingerprint changed; this only fast-paths files that
        # provably have not changed since their last completed import.
        current_fingerprints = load_v1_history_fingerprints(conn)
    else:
        current_fingerprints = current_history_fingerprints(conn)
    imported = 0
    existing = 0
    by_runtime: dict[str, int] = {}
    samples: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = list(selection_skipped)
    for path in files:
        source_key = (str(path), f"{history_runtime(path)}_history_file")
        fingerprint = file_fingerprint(path)
        if current_fingerprints.get(source_key) == fingerprint:
            existing += 1
            continue
        try:
            if is_core_v1(conn):
                result = import_history_file_v1(
                    conn,
                    path,
                    project=args.project,
                    privacy_scope=args.privacy_scope,
                    max_bytes=args.max_bytes,
                    activate=not args.evidence_only,
                )
            else:
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
        if result.get("changed", True):
            imported += 1
        else:
            existing += 1
        by_runtime[result["runtime"]] = by_runtime.get(result["runtime"], 0) + 1
        if len(samples) < 20:
            samples.append(result)
        existing_sources.add((result["path"], f"{result['runtime']}_history_file"))
        current_fingerprints[source_key] = fingerprint
        # Commit after every file. History files can take minutes to redact,
        # and an implicit write transaction held across that work monopolises
        # SQLite's single writer slot: every other local writer (MCP
        # ingest/closeout/retrieval logging from Codex/Claude/Cursor/Hermes)
        # then fails with "database is locked". Per-file commits bound the
        # writer window to one file's actual DB writes; the slow redaction
        # of the next file happens outside any transaction.
        if core_v1:
            store_v1_history_fingerprints(conn, current_fingerprints)
        conn.commit()
    if core_v1:
        store_v1_history_fingerprints(conn, current_fingerprints)
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
            "counts": v1_counts(conn) if is_core_v1(conn) else counts(conn),
        },
    )
    return 0


def memory_files(paths: list[Path], *, skipped: list[dict[str, str]] | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def consider(candidate: Path, *, sweep_root: Path | None = None) -> None:
        resolved = candidate.resolve()
        key = str(resolved)
        if sweep_root is not None:
            root = sweep_root.resolve()
            if not path_is_within(resolved, root):
                if skipped is not None:
                    skipped.append({"path": key, "reason": "outside_sweep_root"})
                return
            if has_hidden_descendant(candidate, root) or has_hidden_descendant(resolved, root):
                if skipped is not None:
                    skipped.append({"path": key, "reason": "hidden_path"})
                return
        if key in seen:
            return
        seen.add(key)
        files.append(resolved)

    for path in paths:
        if path.is_dir():
            root = path.resolve()
            for candidate in root.rglob("*.md"):
                if candidate.is_file():
                    consider(candidate, sweep_root=root)
        elif path.suffix.lower() == ".md":
            consider(path)
    return sorted(files)


HISTORY_SUFFIXES = (
    ".jsonl",
    ".trajectory.jsonl",
    ".jsonl.codex-app-server.json",
    ".json",
    ".md",
)
HISTORY_GLOBS = ("*.jsonl", "*.trajectory.jsonl", "*.jsonl.codex-app-server.json", "*.json", "*.md")

SENSITIVE_HISTORY_FILENAMES = frozenset(
    {
        "auth.json",
        "credentials.json",
        ".credentials.json",
        "config.json",
        "settings.json",
        "secrets.json",
        "mcp.json",
        "keychain.json",
    }
)


def is_sensitive_history_file(path: Path) -> bool:
    """Return true for credential-shaped files that must never be harvested."""
    names = {path.name.lower(), path.resolve().name.lower()}
    return any(
        name in SENSITIVE_HISTORY_FILENAMES
        or name.startswith(".env")
        or name.endswith((".pem", ".key"))
        for name in names
    )


def has_hidden_descendant(candidate: Path, sweep_root: Path) -> bool:
    relative = candidate.relative_to(sweep_root)
    return any(part.startswith(".") for part in relative.parts)


def path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def history_files(
    paths: list[Path],
    *,
    manifests: list[Path] | None = None,
    skipped: list[dict[str, str]] | None = None,
) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def consider(candidate: Path, *, sweep_root: Path | None = None) -> None:
        resolved = candidate.resolve()
        key = str(resolved)
        if sweep_root is not None:
            root = sweep_root.resolve()
            if not path_is_within(resolved, root):
                if skipped is not None:
                    skipped.append({"path": key, "reason": "outside_sweep_root"})
                return
            if has_hidden_descendant(candidate, root) or has_hidden_descendant(resolved, root):
                if skipped is not None:
                    skipped.append({"path": key, "reason": "hidden_path"})
                return
        if is_sensitive_history_file(candidate):
            if skipped is not None:
                skipped.append({"path": key, "reason": "sensitive_filename"})
            return
        if key in seen:
            return
        seen.add(key)
        files.append(resolved)

    for manifest in manifests or []:
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            candidate = Path(line.strip())
            if line.strip() and candidate.is_file() and has_history_suffix(candidate):
                consider(candidate)
    for path in paths:
        if path.is_dir():
            root = path.resolve()
            for pattern in HISTORY_GLOBS:
                for candidate in root.rglob(pattern):
                    if candidate.is_file():
                        consider(candidate, sweep_root=root)
        elif path.is_file() and has_history_suffix(path):
            consider(path)
    return sorted(files)


def has_history_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in HISTORY_SUFFIXES)


def is_history_file(path: Path) -> bool:
    return has_history_suffix(path) and not is_sensitive_history_file(path)


def emit_import_dry_run(
    args: argparse.Namespace,
    files: list[Path],
    *,
    history: bool,
    skipped: list[dict[str, str]] | None = None,
) -> int:
    """Inspect source files without opening, creating, or mutating SQLite."""
    planned: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = list(skipped or [])
    secret_leaks: list[dict[str, object]] = []
    for path in files:
        try:
            nonempty, leaks, residue = inspect_import_source(path)
        except (OSError, UnicodeError, MemoryError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
            continue
        if not nonempty:
            rejected.append({"path": str(path), "reason": "empty"})
            continue
        if leaks or residue:
            secret_leaks.append(
                {
                    "path": str(path),
                    "leaks": leaks,
                    "redaction_residue": residue,
                }
            )
        item = {"path": str(path)}
        if history:
            item["runtime"] = history_runtime(path)
        planned.append(item)
    output(
        args,
        {
            "dry_run": True,
            "database_touched": False,
            "privacy_scope": args.privacy_scope,
            "would_import": len(planned),
            "skipped_count": len(rejected),
            "skipped": rejected[:50],
            "secret_leak_count": len(secret_leaks),
            "secret_leaks": secret_leaks[:50],
            "files": planned[:50],
        },
    )
    return 0


def inspect_import_source(path: Path) -> tuple[bool, list[str], list[str]]:
    """Stream a source to classify secrets without retaining its full contents."""
    nonempty = False
    leaks: set[str] = set()
    residue: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for raw_line in handle:
            nonempty = nonempty or bool(raw_line.strip())
            leaks.update(find_probable_secret_leaks(raw_line))
            if _PRIVATE_KEY_BEGIN_RE.search(raw_line):
                leaks.add("private_key")
            residue.update(find_probable_secret_leaks(redact_secrets(raw_line)))
    return nonempty, sorted(leaks), sorted(residue)


def import_memory_file_v1(
    conn,
    path: Path,
    *,
    project: str | None,
    privacy_scope: str,
    max_bytes: int,
    activate: bool = True,
) -> dict[str, object] | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None
    redacted = redact_secrets(raw)
    truncated = redacted.encode("utf-8", errors="replace")[:max_bytes].decode(
        "utf-8", errors="replace"
    )
    text = truncated
    runtime = history_runtime(path)
    return import_source_v1(
        conn,
        path=path,
        text=text,
        title=title_from_text(text, path.stem),
        source_type="memory_file",
        runtime=runtime,
        project=project,
        privacy_scope=privacy_scope,
        confidence=0.7,
        activate=activate,
    )


def import_history_file_v1(
    conn,
    path: Path,
    *,
    project: str | None,
    privacy_scope: str,
    max_bytes: int,
    activate: bool = True,
) -> dict[str, object] | None:
    if path.stat().st_size == 0:
        return None
    runtime = history_runtime(path)
    # Transcript windows are the one kind of evidence stored by reference: they
    # were 99.13% of all evidence body bytes and 82.4% of the ledger, and the
    # file they came from is still on disk. Everything else stays inline.
    window = build_history_window(path, max_bytes=max_bytes, source_uri=str(path.resolve()))
    return import_source_v1(
        conn,
        path=path,
        text=window.text,
        title=history_title(path, runtime),
        source_type=f"{runtime}_history_file",
        runtime=runtime,
        project=project,
        privacy_scope=privacy_scope,
        confidence=0.55,
        activate=activate,
        window=window,
    )


def import_source_v1(
    conn,
    *,
    path: Path,
    text: str,
    title: str,
    source_type: str,
    runtime: str,
    project: str | None,
    privacy_scope: str,
    confidence: float,
    activate: bool = True,
    window: HistoryWindow | None = None,
) -> dict[str, object]:
    """Record one local file as evidence, and compile it into one belief.

    ``window`` marks the source as a transcript stored by reference. The
    evidence id is still derived from the full window text, so identity is
    exactly what it was when the text was stored inline; only where the text
    lives changes. The *belief* body then carries the recorded head rather than
    the whole window -- not a size optimization but a correctness one: a
    re-windowed transcript must compare unchanged, and it cannot do that
    against a body the store no longer has.
    """
    source_uri = str(path.resolve())
    scope = scope_for_privacy(project, privacy_scope)
    evidence_id = stable_id("evd", text, source_type, source_uri, scope.scope_id)
    evidence_event_id = None
    summary_text = window.head if window is not None else text
    # A transcript is imported as a head plus a sliding tail, so every append
    # mints a new content-addressed id for the same session. When the head is
    # byte-identical, adopt the stored row wholesale -- id *and* summary text --
    # so the belief compares unchanged too and the import is a true no-op.
    # Reusing only the id would still re-propose the belief on every harvest,
    # which appends the transcript to the ledger a second time. See
    # ocbrain.deslop.
    known = get_core_v1_evidence(conn, evidence_id) is not None
    if not known and (
        rewindowed := rewindowed_evidence_id(
            conn, source_uri=source_uri, kind=source_type, text=text
        )
    ):
        stored = get_core_v1_evidence(conn, rewindowed) or {}
        evidence_id = rewindowed
        summary_text = (
            str(stored.get("body_head") or stored.get("body") or summary_text)
            if window is not None
            else str(stored.get("body") or summary_text)
        )
        known = True
    if not known:
        evidence_id, evidence_event_id = record_core_v1_evidence(
            conn,
            body="" if window is not None else text,
            kind=source_type,
            scope=scope,
            writer=f"ocbrain-import:{runtime}",
            artifact_ref=source_uri,
            body_ref=window.body_ref if window is not None else None,
            body_head=window.head if window is not None else None,
            identity_body=text if window is not None else None,
        )

    belief_id = stable_id("belief", "source", source_type, source_uri)
    belief_body = f"{title}\n\n{summary_text}".strip()
    existing = get_core_v1_belief(conn, belief_id)
    unchanged = bool(
        existing
        and existing.get("body") == belief_body
        and evidence_id in existing.get("evidence_ids", [])
        and existing.get("scope") == scope.to_dict()
        and existing.get("status") == "current"
        and existing.get("serve")
    )
    proposal_id = None
    decision_id = None
    if activate and not unchanged:
        proposal_id = append_core_event(
            conn,
            "compilation_proposed",
            {
                "schema_version": "ocbrain.compilation.v1",
                "subject": {"kind": "belief", "id": belief_id},
                "belief_id": belief_id,
                "body": belief_body,
                "evidence_ids": [evidence_id],
                "scope": scope.to_dict(),
                "confidence": confidence,
            },
            writer=f"ocbrain-import:{runtime}",
        )
        decision = decide_proposal_v1(
            conn,
            proposal_event_id=proposal_id,
            decision="approve",
            actor="ocbrain-import",
            edited_body=None,
            reason="explicit local source import",
        )
        decision_id = decision["event_id"]
    return {
        "path": source_uri,
        "runtime": runtime,
        "source_type": source_type,
        "evidence_id": evidence_id,
        "knowledge_id": belief_id,
        "belief_id": belief_id,
        "evidence_event_id": evidence_event_id,
        "proposal_event_id": proposal_id,
        "decision_event_id": decision_id,
        "changed": bool(evidence_event_id or (activate and not unchanged)),
    }


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
    redacted = redact_secrets(raw)
    text = redacted.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
    title = title_from_text(text, path.stem)
    source_uri = str(path)
    runtime = history_runtime(path)
    digest = content_hash(raw)
    evidence_id = upsert_evidence(
        conn,
        source_type="memory_file",
        source_runtime=runtime,
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
    text = history_text_window(path, max_bytes=max_bytes)
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


_V1_HISTORY_FINGERPRINTS_KEY = "history_file_fingerprints_v1"


def _v1_fingerprint_key(source_key: tuple[str, str]) -> str:
    return json.dumps([source_key[0], source_key[1]])


def load_v1_history_fingerprints(conn) -> dict[tuple[str, str], str]:
    """Load the persisted stat-fingerprint gate for v1 history imports.

    Stored as one JSON blob in ``schema_meta`` so no schema migration is
    needed. Keys are ``[path, source_type]`` JSON pairs; values are
    ``file_fingerprint`` digests from the last completed import.
    """
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (_V1_HISTORY_FINGERPRINTS_KEY,)
    ).fetchone()
    if not row:
        return {}
    try:
        raw = json.loads(row[0])
    except (TypeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    loaded: dict[tuple[str, str], str] = {}
    for key, fingerprint in raw.items():
        try:
            path, source_type = json.loads(key)
        except (TypeError, ValueError):
            continue
        if isinstance(path, str) and isinstance(source_type, str) and isinstance(fingerprint, str):
            loaded[(path, source_type)] = fingerprint
    return loaded


def store_v1_history_fingerprints(conn, fingerprints: dict[tuple[str, str], str]) -> None:
    blob = json.dumps(
        {_v1_fingerprint_key(key): value for key, value in fingerprints.items()},
        sort_keys=True,
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
        (_V1_HISTORY_FINGERPRINTS_KEY, blob),
    )


def current_history_fingerprints(conn) -> dict[tuple[str, str], str]:
    """Return the fingerprint backing each currently searchable history doc.

    Evidence is immutable, so a changing append-only source can have several
    historical evidence rows.  The current knowledge row carries the fingerprint
    whose search index is active and is therefore the correct idempotency gate.
    """
    # The strict v1 import path is already event-idempotent and intentionally
    # has no legacy ``knowledge`` table. Returning no pre-read fingerprints
    # lets ``import_source_v1`` make the authoritative changed/unchanged
    # decision without querying a retired projection.
    if is_core_v1(conn):
        return {}
    return {
        (row["body_uri"], f"{row['doc_kind'].removesuffix('_history')}_history_file"): row[
            "content_hash"
        ]
        for row in conn.execute(
            """
            SELECT body_uri, doc_kind, content_hash
            FROM knowledge
            WHERE type = 'doc'
              AND body_uri IS NOT NULL
              AND doc_kind IN ('openclaw_history', 'codex_history',
                               'claude_history', 'unknown_history')
              AND content_hash IS NOT NULL
            """
        )
    }


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
    if is_core_v1(conn):
        output(
            args,
            digest_v1(
                conn,
                context=ScopeContext(project=args.project, runtime="cli"),
                limit=args.limit,
            ),
        )
        return 0
    scopes = None if args.include_private else PUBLIC_SCOPES
    output(args, knowledge_digest(conn, project=args.project, scopes=scopes, limit=args.limit))
    return 0


def cmd_briefing(args: argparse.Namespace) -> int:
    """Print the session-start briefing.

    ``--text`` exists for the SessionStart hook: Claude Code injects a hook's
    stdout into the conversation verbatim since 2.1.0, so what the hook prints
    is what lands in the window. JSON there would spend the budget on braces.
    """
    conn = open_db(args)
    repo_root = getattr(args, "repo_root", None)
    payload = build_briefing(
        conn,
        context=context_from_args(args),
        budget_chars=args.budget_chars,
        repo_root=Path(repo_root) if repo_root else None,
    )
    if args.text:
        print(payload["text"])
        return 0
    output(args, payload)
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(
        args,
        build_ledger(
            conn,
            context=context_from_args(args),
            task_ref=args.task_ref,
            limit=args.limit,
        ),
    )
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    import os

    from ocbrain.scope import normalize_delivery_target

    # Local coding agents get full-fidelity local delivery by default. Hosted
    # (egress-filtered) delivery stays available via --delivery-target or the
    # OCBRAIN_DELIVERY_TARGET env, for feeding a hosted teacher model.
    selected = getattr(args, "delivery_target", None) or os.environ.get(
        "OCBRAIN_DELIVERY_TARGET"
    )
    return serve(
        args.db,
        allow_writes=args.allow_writes,
        profile=args.profile,
        active_db_file=getattr(args, "active_db_file", None),
        delivery_target=normalize_delivery_target(selected or None),
    )
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


def cmd_selftest(args: argparse.Namespace) -> int:
    from ocbrain.selftest import (
        SelftestError,
        diff_scorecards,
        exit_code,
        open_readonly,
        render_pretty,
        run_selftest,
    )

    try:
        conn = open_readonly(args.db)
    except SelftestError as error:
        print(str(error), file=sys.stderr)
        return 2
    try:
        scorecard = run_selftest(
            conn,
            since_days=args.since_days,
            transcript_root=getattr(args, "transcript_root", None),
        )
    finally:
        conn.close()

    baseline_path = getattr(args, "baseline", None)
    if baseline_path is not None:
        try:
            baseline = json.loads(Path(baseline_path).expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"cannot read baseline {baseline_path}: {error}", file=sys.stderr)
            return 2
        scorecard["baseline_diff"] = diff_scorecards(baseline, scorecard)

    out_path = getattr(args, "out", None)
    if out_path is not None:
        target = Path(out_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")

    if getattr(args, "pretty", False):
        print(render_pretty(scorecard), end="")
    else:
        print(json.dumps(scorecard, indent=2, sort_keys=True))
    return exit_code(scorecard)


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


def _compact_side_by_side(cluster: dict[str, Any], *, index: int, heading: str) -> list[str]:
    """One cluster rendered so a human can actually adjudicate the machine.

    Every body in full, beside its id, scope, confidence and evidence count, and
    the pipeline's own verdict spelled out underneath. This block is the entire
    point of the dry run: an operator authorises a merge by reading it, so it
    must carry enough to disagree with, not a count.
    """
    survivor = cluster.get("survivor")
    lines = [
        "",
        f"{heading} {index}  scope={cluster['scope_id']}  members={cluster['size']}"
        f"  min_cosine={cluster['min_cosine']}  min_jaccard={cluster['min_jaccard']}",
        "-" * 78,
    ]
    for member in cluster["members"]:
        if survivor is None:
            marker = "  ."
        elif member["belief_id"] == survivor:
            marker = "KEEP"
        elif member["belief_id"] in (cluster.get("losers") or []):
            marker = "MERGE"
        else:
            marker = "  ."
        lines.append(
            f"  [{marker:>5}] {member['belief_id']}  key={member['key'] or '(none)'}"
            f"  conf={member['confidence']}  evidence={member['evidence_count']}"
        )
        for chunk in textwrap.wrap(member["body"], width=72) or [""]:
            lines.append(f"          {chunk}")
    if cluster.get("stage"):
        lines.append(f"  stage: {cluster['stage']}")
    if cluster.get("reason"):
        for chunk in textwrap.wrap(f"verdict: {cluster['reason']}", width=76):
            lines.append(f"  {chunk}")
    if cluster.get("deferred_reason"):
        lines.append(f"  deferred: {cluster['deferred_reason']}")
    lines.append(
        f"  evidence: shared={cluster['shares_evidence']} "
        f"identical={cluster['identical_evidence']} (reported only; decides nothing)"
    )
    if cluster.get("egress_audit_id"):
        lines.append(f"  egress audit: {cluster['egress_audit_id']}")
    return lines


def render_compaction(plan: dict[str, Any], *, applied: dict[str, Any] | None) -> str:
    """The dry-run report a human reads before authorising anything."""
    if not plan.get("measured"):
        return (
            f"compaction not measured: {plan.get('detail')}\n"
            f"serving beliefs: {plan.get('serving')}\n"
            "build the sidecar with `ocbrain vector-build`, then re-run."
        )
    stages = plan["stages"]
    lines = [
        "=" * 78,
        f"OCBrain compaction plan  ({COMPACT_VERSION})",
        "=" * 78,
        f"serving beliefs        : {plan['serving']}",
        f"cosine floor           : {plan['cosine_floor']}   (same scope only)",
        f"restatement threshold  : {plan['restatement_threshold']}",
        f"merge cap              : {plan['limit']} losers per run",
        "",
        "cascade",
        f"  clusters found                : {stages['clusters_found']}",
        f"  excluded by guard             : {stages['excluded_by_guard']}",
        f"  resolved mechanically         : "
        f"{stages['resolved_identical_bodies'] + stages['resolved_restatement']}"
        f"  (identical bodies {stages['resolved_identical_bodies']},"
        f" restatement {stages['resolved_restatement']})",
        f"  escalated to a model          : {stages['escalated']}",
        f"    withheld, not egress-eligible: {stages['withheld_egress']}",
        f"    left undecided, no provider  : {stages['not_adjudicated']}",
        f"    adjudicated -> merge         : {stages['adjudicated_merge']}",
        f"    adjudicated -> coexist       : {stages['adjudicated_coexist']}",
        f"  hosted calls made             : {plan['hosted_calls']}",
        "",
        f"proposed merges        : {len(plan['merges'])}"
        f"  retiring {plan['would_retire']} beliefs",
        f"serving after          : {plan['serving_after']}",
        f"deferred over the cap  : {len(plan['deferred'])}",
    ]
    for index, cluster in enumerate(plan["merges"], 1):
        lines += _compact_side_by_side(cluster, index=index, heading="MERGE")
    for index, cluster in enumerate(plan["deferred"], 1):
        lines += _compact_side_by_side(cluster, index=index, heading="DEFERRED")
    for index, cluster in enumerate(plan.get("coexisting") or [], 1):
        lines += _compact_side_by_side(cluster, index=index, heading="COEXIST")
    for index, cluster in enumerate(plan["excluded"], 1):
        lines += _compact_side_by_side(cluster, index=index, heading="EXCLUDED")
    if applied is None:
        if plan.get("hosted_calls"):
            dry_run_status = (
                "DRY RUN. No beliefs were changed. "
                f"Hosted adjudication recorded {plan['hosted_calls']} egress audit(s)."
            )
        else:
            dry_run_status = "DRY RUN. Nothing was written."
        lines += [
            "",
            "=" * 78,
            dry_run_status,
            "Authorise with: ocbrain compact --apply --yes",
            "=" * 78,
        ]
    else:
        lines += [
            "",
            "=" * 78,
            f"APPLIED. {applied['retired']} beliefs retired behind their survivors.",
            "Every one is a soft supersession: the body is still readable with",
            "  brain.get <belief_id> mode=as_stored",
            "and mode=resolve follows the pointer to the survivor.",
            "",
            "Undo any single merge with:",
        ]
        for entry in applied["applied"]:
            lines.append(f"  {undo_command(entry['belief_id'])}")
        for entry in applied["failed"]:
            lines.append(f"  FAILED {entry['belief_id']}: {entry['error']}")
        lines.append("=" * 78)
    return "\n".join(lines)


def _compaction_adjudicator(conn, args: argparse.Namespace, egress_policies: tuple[str, ...]):
    """The hosted adjudicator, or ``None`` without explicit authority and a key.

    A provider credential proves reachability, not operator intent. Compaction
    therefore requires ``--allow-hosted-egress`` before it can send even an
    otherwise eligible belief body. Missing authority or credentials degrade
    the run rather than failing it: the mechanical plan remains useful and the
    ambiguous tail is reported as undecided.
    """
    if not args.allow_hosted_egress:
        return None
    defaults = PROVIDER_DEFAULTS[args.provider]
    model = args.model or defaults["model"]
    api_key = os.environ.get(defaults["api_key_env"])
    if not api_key:
        return None

    def adjudicator(members: list[str], beliefs: dict[str, Any]) -> dict[str, Any]:
        return adjudicate(
            conn,
            members,
            beliefs,
            provider=args.provider,
            model=model,
            api_key=api_key,
            base_url=defaults["base_url"],
            egress_policies=egress_policies,
        )

    return adjudicator


def _ensure_compaction_snapshot(db: Path, *, started_at: float) -> Path:
    """A snapshot no older than this run, taken now if none exists.

    The precondition is "recoverable", not "backed up at some point". A snapshot
    from before the run is only a restore point if nothing has been written
    since, so the age test is against the run's own start rather than a fixed
    window.
    """
    resolved = db.expanduser().resolve()
    snapshot_dir = resolved.parent / "backups"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in snapshot_dir.glob("*.sqlite")
        if path.stat().st_mtime >= started_at
    ]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return snapshot_sqlite(resolved, snapshot_dir / f"pre-compact-{stamp}.sqlite")


def cmd_compact(args: argparse.Namespace) -> int:
    started_at = time.time()
    conn = open_existing_core_v1(args.db)
    try:
        if args.undo:
            result = undo_merge(conn, belief_id=args.undo)
            output(args, {"action": "compact", "undone": result})
            return 0
        curator_cfg = load_config().curator
        egress_policies = tuple(curator_cfg.egress_policies)
        visibilities = tuple(curator_cfg.visibilities)
        plan = plan_compaction(
            conn,
            cosine_floor=args.cosine,
            limit=args.limit,
            adjudicator=_compaction_adjudicator(conn, args, egress_policies),
            egress_policies=egress_policies,
            visibilities=visibilities,
        )
        applied = None
        if args.apply:
            # Two independent gates. `--apply` says what to do; `--yes` says a
            # human read the plan above and meant it. A single flag would let a
            # shell history entry or a copied command line retire beliefs that
            # nobody reviewed, which is the one thing this command must not do.
            if not args.yes:
                print(render_compaction(plan, applied=None))
                print("\nrefusing to apply: --apply also requires --yes")
                return 0
            snapshot = _ensure_compaction_snapshot(args.db, started_at=started_at)
            print(f"snapshot covering this run: {snapshot}")
            applied = apply_compaction(conn, plan)
    finally:
        conn.close()
    if args.json:
        output(args, {"action": "compact", "plan": plan, "applied": applied})
        return 0
    print(render_compaction(plan, applied=applied))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
