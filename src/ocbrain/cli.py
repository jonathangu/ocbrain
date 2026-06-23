from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocbrain.classifier import classify_event, classify_text, is_low_value_evidence_line
from ocbrain.db import (
    DEFAULT_DB_PATH,
    EventInput,
    add_evidence,
    backfill_candidate_claim_keys,
    connect,
    counts,
    init_db,
    insert_candidate,
    iter_untriaged_events,
    list_candidate_decisions,
    list_candidates,
    mark_event_triaged,
    now_iso,
    review_group_candidates,
    review_groups,
    search,
    transition_candidate,
    transition_candidate_group,
    upsert_event,
)
from ocbrain.eval import SampleSpec, evaluate, write_reports
from ocbrain.excerpt import write_excerpt
from ocbrain.ids import stable_id
from ocbrain.ingest import (
    IngestOptions,
    default_history_roots,
    event_from_file,
    iter_candidate_files,
)
from ocbrain.mcp import serve
from ocbrain.proposals import write_proposal
from ocbrain.schema import Evidence
from ocbrain.temporal import temporal_supersession_groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocbrain", description="OpenClawBrain Lite governor")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize the SQLite ledger")
    init_parser.set_defaults(func=cmd_init)

    closeout_parser = subparsers.add_parser("closeout", help="Classify one artifact")
    closeout_parser.add_argument("--input", required=True, type=Path)
    closeout_parser.add_argument(
        "--store", action="store_true", help="Store candidates in the ledger"
    )
    closeout_parser.set_defaults(func=cmd_closeout)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest safe local history")
    ingest_parser.add_argument("roots", nargs="*", type=Path)
    ingest_parser.add_argument("--history-profile", action="store_true")
    ingest_parser.add_argument("--workspace", type=Path, default=Path.cwd().parent)
    ingest_parser.add_argument("--limit", type=int)
    ingest_parser.add_argument("--max-file-bytes", type=int, default=512_000)
    ingest_parser.add_argument("--include-yaml", action="store_true")
    ingest_parser.set_defaults(func=cmd_ingest)

    triage_parser = subparsers.add_parser("triage", help="Classify untriaged events")
    triage_parser.add_argument("--limit", type=int)
    triage_parser.set_defaults(func=cmd_triage)

    rebuild_parser = subparsers.add_parser(
        "rebuild-candidates",
        help="Reclassify events and stale old generic draft candidates",
    )
    rebuild_parser.add_argument("--limit", type=int)
    rebuild_parser.add_argument("--apply", action="store_true")
    rebuild_parser.set_defaults(func=cmd_rebuild_candidates)

    backfill_preview_parser = subparsers.add_parser(
        "backfill-preview",
        help="Apply a candidate rebuild to a copied DB and report before/after quality",
    )
    backfill_preview_parser.add_argument("--output-dir", required=True, type=Path)
    backfill_preview_parser.add_argument("--limit", type=int)
    backfill_preview_parser.add_argument("--sample-size", type=int, default=160)
    backfill_preview_parser.add_argument("--per-target", type=int, default=40)
    backfill_preview_parser.add_argument("--seed", type=int, default=20260621)
    backfill_preview_parser.add_argument("--min-score-delta", type=float, default=0.0)
    backfill_preview_parser.add_argument("--allow-score-drop", action="store_true")
    backfill_preview_parser.add_argument("--fail-on-leak", action="store_true")
    backfill_preview_parser.set_defaults(func=cmd_backfill_preview)

    invalidate_temporal_parser = subparsers.add_parser(
        "invalidate-temporal",
        help="Mark older duplicate temporal facts stale and record invalidations",
    )
    invalidate_temporal_parser.add_argument("--limit", type=int)
    invalidate_temporal_parser.add_argument("--apply", action="store_true")
    invalidate_temporal_parser.set_defaults(func=cmd_invalidate_temporal)

    search_parser = subparsers.add_parser("search", help="Search ingested history")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--include-private", action="store_true")
    search_parser.set_defaults(func=cmd_search)

    digest_parser = subparsers.add_parser("digest", help="Show ledger counts")
    digest_parser.set_defaults(func=cmd_digest)

    candidates_parser = subparsers.add_parser("candidates", help="List candidates")
    candidates_parser.add_argument("--target")
    candidates_parser.add_argument("--status")
    candidates_parser.add_argument("--scope")
    candidates_parser.add_argument("--limit", type=int, default=20)
    candidates_parser.set_defaults(func=cmd_candidates)

    backfill_parser = subparsers.add_parser(
        "backfill-claim-keys",
        help="Derive claim keys for existing candidates from redacted evidence",
    )
    backfill_parser.add_argument("--limit", type=int)
    backfill_parser.set_defaults(func=cmd_backfill_claim_keys)

    propose_parser = subparsers.add_parser(
        "propose", help="Write proposal markdown for a candidate"
    )
    propose_parser.add_argument("candidate_id")
    propose_parser.add_argument("--output-dir", type=Path, default=Path("proposals"))
    propose_parser.add_argument("--allow-draft", action="store_true")
    propose_parser.add_argument("--actor", default="ocbrain")
    propose_parser.set_defaults(func=cmd_propose)

    review_parser = subparsers.add_parser("review", help="Review candidate queue")
    review_subparsers = review_parser.add_subparsers(dest="review_command")

    review_list = review_subparsers.add_parser("list", help="List grouped candidates")
    review_list.add_argument("--status", default="draft")
    review_list.add_argument("--target")
    review_list.add_argument("--limit", type=int, default=20)
    review_list.add_argument("--include-low-value", action="store_true")
    review_list.set_defaults(func=cmd_review_list)

    review_inspect = review_subparsers.add_parser("inspect", help="Inspect one candidate")
    review_inspect.add_argument("candidate_id")
    review_inspect.set_defaults(func=cmd_review_inspect)

    review_inspect_group = review_subparsers.add_parser(
        "inspect-group", help="Inspect candidates in one grouped claim"
    )
    review_inspect_group.add_argument("--target", required=True)
    review_inspect_group.add_argument("--claim-key", required=True)
    review_inspect_group.add_argument("--status", default="draft")
    review_inspect_group.add_argument("--limit", type=int, default=20)
    review_inspect_group.set_defaults(func=cmd_review_inspect_group)

    for action in ("approve", "reject", "defer"):
        action_parser = review_subparsers.add_parser(action, help=f"{action.title()} candidate")
        action_parser.add_argument("candidate_id")
        action_parser.add_argument("--actor", default="jon")
        action_parser.add_argument("--reason", required=True)
        action_parser.set_defaults(func=cmd_review_transition, review_action=action)

        group_parser = review_subparsers.add_parser(
            f"{action}-group", help=f"{action.title()} all candidates in one grouped claim"
        )
        group_parser.add_argument("--target", required=True)
        group_parser.add_argument("--claim-key", required=True)
        group_parser.add_argument("--status", default="draft")
        group_parser.add_argument("--actor", default="jon")
        group_parser.add_argument("--reason", required=True)
        group_parser.add_argument("--limit", type=int)
        group_parser.set_defaults(func=cmd_review_transition_group, review_action=action)

    excerpt_parser = subparsers.add_parser("excerpt", help="Write a managed native context block")
    excerpt_parser.add_argument("--output", required=True, type=Path)
    excerpt_parser.add_argument(
        "--runtime",
        choices=["codex", "claude", "openclaw", "generic"],
        default="generic",
    )
    excerpt_parser.add_argument("--scope")
    excerpt_parser.add_argument("--status")
    excerpt_parser.add_argument(
        "--include-draft",
        action="store_true",
        help="Allow draft/deferred candidates in the generated context block",
    )
    excerpt_parser.add_argument("--limit", type=int, default=12)
    excerpt_parser.set_defaults(func=cmd_excerpt)

    eval_parser = subparsers.add_parser("eval", help="Score candidate quality and safety")
    eval_parser.add_argument("--sample-size", type=int, default=100)
    eval_parser.add_argument("--per-target", type=int)
    eval_parser.add_argument("--seed", type=int, default=20260621)
    eval_parser.add_argument("--targets", help="Comma-separated target filter")
    eval_parser.add_argument("--output-json", type=Path)
    eval_parser.add_argument("--output-md", type=Path)
    eval_parser.add_argument("--sample-output-limit", type=int, default=200)
    eval_parser.add_argument("--fail-under", type=float)
    eval_parser.add_argument("--fail-on-leak", action="store_true")
    eval_parser.set_defaults(func=cmd_eval)

    mcp_parser = subparsers.add_parser("mcp", help="Run stdio MCP server")
    mcp_parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Expose write-capable MCP tools; off by default",
    )
    mcp_parser.set_defaults(func=cmd_mcp)

    # Compatibility for the initial ocbrain-closeout script usage.
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input and args.command is None:
        args.command = "closeout"
        args.store = False
        args.func = cmd_closeout
    if not args.command:
        parser.print_help()
        return 2
    return args.func(args)


def output(args: argparse.Namespace, payload) -> None:
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))


def open_db(args: argparse.Namespace):
    conn = connect(args.db)
    init_db(conn)
    return conn


def cmd_init(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(args, {"db": str(args.db), "status": "ready", "counts": counts(conn)})
    return 0


def cmd_closeout(args: argparse.Namespace) -> int:
    options = IngestOptions()
    event = event_from_file(args.input, options)
    candidates = classify_event(event) if event else []
    payload = {"input": str(args.input), "candidates": [item.to_dict() for item in candidates]}
    if args.store:
        conn = open_db(args)
        stored_ids: list[str] = []
        if event and upsert_event(conn, event):
            for candidate in candidates:
                candidate_id = insert_candidate(conn, candidate, event.id)
                if candidate_id:
                    stored_ids.append(candidate_id)
            mark_event_triaged(conn, event.id)
            conn.commit()
        payload["stored_candidate_ids"] = stored_ids
    output(args, payload)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = open_db(args)
    options = IngestOptions(max_file_bytes=args.max_file_bytes, include_yaml=args.include_yaml)
    roots = list(args.roots)
    if args.history_profile:
        roots.extend(default_history_roots(args.workspace))
    if not roots:
        roots = [Path.cwd()]

    seen = inserted = skipped = 0
    for path in iter_candidate_files(roots, options):
        if args.limit is not None and seen >= args.limit:
            break
        seen += 1
        event = event_from_file(path, options)
        if event is None:
            skipped += 1
            continue
        if upsert_event(conn, event):
            excerpt = event.summary[:1000]
            add_evidence(
                conn,
                event.id,
                Evidence(uri=event.source_uri, excerpt=excerpt),
                event.source_type,
            )
            inserted += 1
        else:
            skipped += 1
    conn.commit()
    output(args, {"seen": seen, "inserted": inserted, "skipped": skipped, "counts": counts(conn)})
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    conn = open_db(args)
    events = list(iter_untriaged_events(conn, args.limit))
    inserted = 0
    for event in events:
        evidence = [Evidence(uri=event["source_uri"], excerpt=event["summary"][:1000])]
        candidates = classify_text(
            event["body"],
            evidence=evidence,
            title_hint=event["title"],
            source_type=event["source_type"],
        )
        for candidate in candidates:
            if insert_candidate(conn, candidate, event["id"]):
                inserted += 1
        mark_event_triaged(conn, event["id"])
    conn.commit()
    output(
        args,
        {"events_triaged": len(events), "candidates_inserted": inserted, "counts": counts(conn)},
    )
    return 0


def cmd_rebuild_candidates(args: argparse.Namespace) -> int:
    conn = open_db(args)
    result = rebuild_candidates(conn, limit=args.limit, apply=args.apply)
    output(
        args,
        {
            **result,
            "counts": counts(conn),
        },
    )
    return 0


def rebuild_candidates(conn, *, limit: int | None = None, apply: bool = False) -> dict:
    events = list(
        conn.execute("SELECT * FROM events ORDER BY ingested_at ASC LIMIT ?", (limit or -1,))
    )
    draft_candidates_to_stale = candidates_inserted = 0
    for event in events:
        event_input = event_input_from_row(event)
        candidates = classify_event(event_input)
        stale_ids = [
            row["id"]
            for row in conn.execute(
                """
                SELECT id, title, body
                FROM candidates
                WHERE event_id = ? AND status = 'draft'
                """,
                (event["id"],),
            )
            if is_stale_rebuild_candidate(row)
        ]
        if apply:
            for candidate_id in stale_ids:
                conn.execute(
                    "UPDATE candidates SET status = 'stale', updated_at = ? WHERE id = ?",
                    (now_iso(), candidate_id),
                )
            for candidate in candidates:
                if insert_candidate(conn, candidate, event["id"]):
                    candidates_inserted += 1
        draft_candidates_to_stale += len(stale_ids)
    if apply:
        conn.commit()
    return {
        "apply": apply,
        "events_seen": len(events),
        "generic_candidates_to_stale": draft_candidates_to_stale,
        "draft_candidates_to_stale": draft_candidates_to_stale,
        "candidates_inserted": candidates_inserted,
    }


def cmd_backfill_preview(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_db = args.output_dir / "backfill-preview.sqlite"
    if preview_db.exists():
        preview_db.unlink()

    source_conn = open_db(args)
    preview_conn = connect(preview_db)
    source_conn.backup(preview_conn)
    init_db(preview_conn)

    spec = SampleSpec(
        sample_size=args.sample_size,
        seed=args.seed,
        per_target=args.per_target,
    )
    before_distribution = candidate_distribution(source_conn)
    before_report = evaluate(
        source_conn,
        spec,
        db_label=str(args.db),
        sample_output_limit=120,
    )
    rebuild_result = rebuild_candidates(preview_conn, limit=args.limit, apply=True)
    after_distribution = candidate_distribution(preview_conn)
    after_report = evaluate(
        preview_conn,
        spec,
        db_label=str(preview_db),
        sample_output_limit=120,
    )

    before_score = before_report["summary"]["overall_score"]
    after_score = after_report["summary"]["overall_score"]
    score_delta = round(after_score - before_score, 3)
    leak_count = after_report["leakage"]["probable_secret_count"]
    score_ok = args.allow_score_drop or score_delta >= args.min_score_delta
    leak_ok = not args.fail_on_leak or leak_count == 0
    passed = score_ok and leak_ok
    payload = {
        "gate": {
            "passed": passed,
            "score_ok": score_ok,
            "leak_ok": leak_ok,
            "min_score_delta": args.min_score_delta,
            "allow_score_drop": args.allow_score_drop,
        },
        "source_db": str(args.db),
        "preview_db": str(preview_db),
        "score_delta": score_delta,
        "before": {
            "summary": before_report["summary"],
            "distribution": before_distribution,
        },
        "after": {
            "summary": after_report["summary"],
            "distribution": after_distribution,
        },
        "rebuild": {
            **rebuild_result,
            "counts": counts(preview_conn),
        },
        "distribution_diff": distribution_diff(before_distribution, after_distribution),
    }
    summary_path = args.output_dir / "backfill-preview.json"
    before_report_path = args.output_dir / "backfill-preview-before.json"
    after_report_path = args.output_dir / "backfill-preview-after.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    before_report_path.write_text(
        json.dumps(before_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    after_report_path.write_text(
        json.dumps(after_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output(args, {**payload, "output_json": str(summary_path)})
    return 0 if passed else 1


def candidate_distribution(conn) -> dict[str, object]:
    active_by_source_target_status = [
        dict(row)
        for row in conn.execute(
            """
            SELECT
              COALESCE(events.source_type, 'none') AS source_type,
              candidates.target,
              candidates.status,
              COUNT(*) AS count
            FROM candidates
            LEFT JOIN events ON events.id = candidates.event_id
            WHERE candidates.status != 'stale'
            GROUP BY source_type, candidates.target, candidates.status
            ORDER BY count DESC, source_type, candidates.target, candidates.status
            """
        )
    ]
    by_status = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM candidates GROUP BY status ORDER BY status"
        )
    }
    active_by_target = {
        row["target"]: row["count"]
        for row in conn.execute(
            """
            SELECT target, COUNT(*) AS count
            FROM candidates
            WHERE status != 'stale'
            GROUP BY target
            ORDER BY count DESC
            """
        )
    }
    return {
        "by_status": by_status,
        "active_by_target": active_by_target,
        "active_by_source_target_status": active_by_source_target_status,
    }


def distribution_diff(
    before: dict[str, object],
    after: dict[str, object],
) -> list[dict[str, object]]:
    def keyed_rows(distribution: dict[str, object]) -> dict[tuple[str, str, str], int]:
        rows = distribution["active_by_source_target_status"]
        return {
            (row["source_type"], row["target"], row["status"]): row["count"]
            for row in rows
        }

    before_rows = keyed_rows(before)
    after_rows = keyed_rows(after)
    diffs = []
    for key in sorted(set(before_rows) | set(after_rows)):
        before_count = before_rows.get(key, 0)
        after_count = after_rows.get(key, 0)
        delta = after_count - before_count
        if delta:
            source_type, target, status = key
            diffs.append(
                {
                    "source_type": source_type,
                    "target": target,
                    "status": status,
                    "before": before_count,
                    "after": after_count,
                    "delta": delta,
                }
            )
    return sorted(diffs, key=lambda item: abs(item["delta"]), reverse=True)


def cmd_invalidate_temporal(args: argparse.Namespace) -> int:
    conn = open_db(args)
    groups = temporal_supersession_groups(conn, limit=args.limit)
    invalidations = []
    for group in groups:
        reason = f"superseded temporal fact: {group['subject_key']}"
        for old_candidate_id in group["stale_candidate_ids"]:
            invalidation_id = stable_invalidation_id(
                old_candidate_id,
                group["keep_candidate_id"],
                reason,
            )
            invalidations.append(
                {
                    "id": invalidation_id,
                    "old_candidate_id": old_candidate_id,
                    "new_candidate_id": group["keep_candidate_id"],
                    "reason": reason,
                }
            )
            if args.apply:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO invalidations (
                      id, old_candidate_id, new_candidate_id, reason, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        invalidation_id,
                        old_candidate_id,
                        group["keep_candidate_id"],
                        reason,
                        now_iso(),
                    ),
                )
                conn.execute(
                    "UPDATE candidates SET status = 'stale', updated_at = ? WHERE id = ?",
                    (now_iso(), old_candidate_id),
                )
    if args.apply:
        conn.commit()
    output(
        args,
        {
            "apply": args.apply,
            "groups": groups,
            "invalidations": invalidations,
            "groups_seen": len(groups),
            "candidates_to_stale": len(invalidations),
            "counts": counts(conn),
        },
    )
    return 0


def stable_invalidation_id(old_candidate_id: str, new_candidate_id: str, reason: str) -> str:
    return stable_id("inv", old_candidate_id, new_candidate_id, reason)


def event_input_from_row(row) -> EventInput:
    return EventInput(
        id=row["id"],
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        content_hash=row["content_hash"],
        title=row["title"],
        summary=row["summary"],
        body=row["body"],
        scope=row["scope"],
        metadata=json.loads(row["metadata_json"] or "{}"),
        created_at=row["created_at"],
    )


def is_generic_candidate_body(body: str) -> bool:
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in (
            "artifact appears to",
            "route to",
            "create a patch suggestion only",
            "extract only concise",
            "no strong memory/wiki/skill/policy",
            "from source: ---",
            "from source: - **session key**",
            "from source: - status:",
            "from source: pagetype:",
            "from source: - openclaw home:",
            "brain loaded: runtime hook registered",
            "from source: status ok",
            "from source: openclawbrain",
            "from source: {",
            "from source: date:",
            "from source: - bundle verdict:",
            "from source: bundle verdict:",
            "from source: format:",
            "from source: - severity:",
            "from source: severity:",
            'from source: "command":',
            "from source: owner:",
            "from source: - owner:",
        )
    )


def is_stale_rebuild_candidate(row) -> bool:
    return is_generic_candidate_body(row["body"]) or has_low_value_candidate_title(row["title"])


def has_low_value_candidate_title(title: str) -> bool:
    if ": " not in title:
        return False
    _, suffix = title.split(": ", 1)
    suffix = suffix.strip()
    if not suffix:
        return True
    return is_low_value_evidence_line(suffix)


def cmd_search(args: argparse.Namespace) -> int:
    conn = open_db(args)
    scopes = None if args.include_private else ("workspace", "project", "public")
    rows = [dict(row) for row in search(conn, args.query, args.limit, scopes=scopes)]
    output(args, {"query": args.query, "results": rows})
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    conn = open_db(args)
    output(args, counts(conn))
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    conn = open_db(args)
    rows = [
        dict(row)
        for row in list_candidates(conn, args.target, args.status, args.scope, args.limit)
    ]
    output(args, {"candidates": rows})
    return 0


def cmd_backfill_claim_keys(args: argparse.Namespace) -> int:
    conn = open_db(args)
    updated = backfill_candidate_claim_keys(conn, args.limit)
    conn.commit()
    output(args, {"updated": updated, "counts": counts(conn)})
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    conn = open_db(args)
    path = write_proposal(
        conn,
        args.candidate_id,
        args.output_dir,
        allow_unapproved=args.allow_draft,
        actor=args.actor,
    )
    output(args, {"candidate_id": args.candidate_id, "proposal": str(path)})
    return 0


def cmd_review_list(args: argparse.Namespace) -> int:
    conn = open_db(args)
    rows = [
        dict(row)
        for row in review_groups(
            conn,
            status=args.status,
            target=args.target,
            limit=args.limit,
            include_low_value=args.include_low_value,
        )
    ]
    output(args, {"groups": rows})
    return 0


def cmd_review_inspect(args: argparse.Namespace) -> int:
    conn = open_db(args)
    candidate = conn.execute(
        "SELECT * FROM candidates WHERE id = ?",
        (args.candidate_id,),
    ).fetchone()
    if candidate is None:
        raise ValueError(f"candidate not found: {args.candidate_id}")
    decisions = [dict(row) for row in list_candidate_decisions(conn, args.candidate_id)]
    output(args, {"candidate": dict(candidate), "decisions": decisions})
    return 0


def cmd_review_inspect_group(args: argparse.Namespace) -> int:
    conn = open_db(args)
    rows = [
        dict(row)
        for row in review_group_candidates(
            conn,
            target=args.target,
            claim_key=args.claim_key,
            status=args.status,
            limit=args.limit,
        )
    ]
    output(
        args,
        {
            "target": args.target,
            "claim_key": args.claim_key,
            "status": args.status,
            "count": len(rows),
            "candidates": rows,
        },
    )
    return 0


def cmd_review_transition(args: argparse.Namespace) -> int:
    conn = open_db(args)
    next_status = {
        "approve": "approved",
        "reject": "rejected",
        "defer": "deferred",
    }[args.review_action]
    decision_id = transition_candidate(
        conn,
        args.candidate_id,
        action=args.review_action,
        next_status=next_status,
        actor=args.actor,
        reason=args.reason,
    )
    conn.commit()
    output(
        args,
        {
            "candidate_id": args.candidate_id,
            "decision_id": decision_id,
            "status": next_status,
        },
    )
    return 0


def cmd_review_transition_group(args: argparse.Namespace) -> int:
    conn = open_db(args)
    next_status = {
        "approve": "approved",
        "reject": "rejected",
        "defer": "deferred",
    }[args.review_action]
    decision_ids = transition_candidate_group(
        conn,
        target=args.target,
        claim_key=args.claim_key,
        status=args.status,
        action=f"{args.review_action}_group",
        next_status=next_status,
        actor=args.actor,
        reason=args.reason,
        limit=args.limit,
    )
    conn.commit()
    output(
        args,
        {
            "target": args.target,
            "claim_key": args.claim_key,
            "previous_status": args.status,
            "status": next_status,
            "changed": len(decision_ids),
            "decision_ids": decision_ids,
        },
    )
    return 0


def cmd_excerpt(args: argparse.Namespace) -> int:
    conn = open_db(args)
    path = write_excerpt(
        conn,
        args.output,
        args.runtime,
        args.scope,
        args.limit,
        args.status,
        include_draft=args.include_draft,
    )
    output(args, {"runtime": args.runtime, "output": str(path)})
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    conn = open_db(args)
    targets = tuple(item.strip() for item in (args.targets or "").split(",") if item.strip())
    spec = SampleSpec(
        sample_size=args.sample_size,
        seed=args.seed,
        targets=targets,
        per_target=args.per_target,
    )
    report = evaluate(
        conn,
        spec,
        db_label=str(args.db),
        sample_output_limit=args.sample_output_limit,
    )
    write_reports(report, output_json=args.output_json, output_md=args.output_md)
    if args.output_json or args.output_md:
        payload = {"report": report["summary"]}
        if args.output_md:
            payload["output_md"] = str(args.output_md)
    else:
        payload = report
    if args.output_json:
        payload = {**payload, "output_json": str(args.output_json)}
    output(args, payload)
    if args.fail_on_leak and report["leakage"]["probable_secret_count"]:
        return 1
    if args.fail_under is not None and report["summary"]["overall_score"] < args.fail_under:
        return 1
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    return serve(args.db, allow_writes=args.allow_writes)


if __name__ == "__main__":
    raise SystemExit(main())
