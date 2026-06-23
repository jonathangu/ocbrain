from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocbrain.db import (
    DEFAULT_DB_PATH,
    PUBLIC_SCOPES,
    connect,
    counts,
    get_knowledge,
    init_db,
    knowledge_digest,
    list_knowledge,
    mark_knowledge_stale,
    search,
    upsert_evidence,
    upsert_knowledge,
)
from ocbrain.ids import content_hash
from ocbrain.loops import LoopIngestOptions, dry_run_loop_ingest, write_loop_ingest
from ocbrain.mcp import serve
from ocbrain.proposals import write_proposal
from ocbrain.text import compact_whitespace


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

    propose_parser = subparsers.add_parser(
        "propose", help="Write proposal markdown for human-gated knowledge"
    )
    propose_parser.add_argument("knowledge_id")
    propose_parser.add_argument("--output-dir", type=Path, default=Path("proposals"))
    propose_parser.add_argument("--allow-unapproved", action="store_true")
    propose_parser.add_argument("--actor", default="ocbrain")
    propose_parser.set_defaults(func=cmd_propose)

    stale_parser = subparsers.add_parser("mark-stale", help="Mark knowledge stale")
    stale_parser.add_argument("knowledge_id")
    stale_parser.add_argument("--reason", default="user_request")
    stale_parser.set_defaults(func=cmd_mark_stale)

    mcp_parser = subparsers.add_parser("mcp", help="Run stdio MCP server")
    mcp_parser.add_argument(
        "--allow-writes",
        action="store_true",
        help="Expose write-capable MCP tools; off by default",
    )
    mcp_parser.set_defaults(func=cmd_mcp)

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
        dict(row)
        for row in search(conn, args.query, args.limit, scopes=scopes, filters=filters)
    ]
    output(args, {"query": args.query, "filters": filters, "results": rows})
    return 0


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


def cmd_propose(args: argparse.Namespace) -> int:
    conn = open_db(args)
    path = write_proposal(
        conn,
        args.knowledge_id,
        args.output_dir,
        allow_unapproved=args.allow_unapproved,
        actor=args.actor,
    )
    output(args, {"knowledge_id": args.knowledge_id, "proposal": str(path)})
    return 0


def cmd_mark_stale(args: argparse.Namespace) -> int:
    conn = open_db(args)
    if not mark_knowledge_stale(conn, args.knowledge_id, reason=args.reason):
        raise ValueError(f"knowledge not found: {args.knowledge_id}")
    conn.commit()
    row = get_knowledge(conn, args.knowledge_id)
    output(args, {"knowledge": dict(row)})
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    return serve(args.db, allow_writes=args.allow_writes)


if __name__ == "__main__":
    raise SystemExit(main())
