#!/usr/bin/env python3
"""Compile high-signal OCBrain evidence into a sparse, human-readable wiki.

An explicit operator-invoked hosted operation. Dry-run by default: without
``--apply`` it prints what it would send and makes no network call. Only
already-redacted, bounded evidence bodies that pass project, visibility, and
egress gates are eligible, and raw transcripts never are (they are excluded by
kind).

``--project`` is repeatable and ``--projects-from-config`` curates every scope
in ``curator.projects``. The loop lives here rather than in the caller because
each project carries its own digest in one shared ``state.json``: a shell loop
over ``--project`` would overwrite the previous project's digest and make every
scheduled cycle re-bill a hosted call for every project. Each project is gated
on its own digest, so a scope whose evidence has not changed costs nothing, and
a scope with too little eligible evidence is skipped and reported rather than
billed.

Which egress policies qualify is the operator's declaration, set once in
``curator.egress_policies`` or overridden with ``--egress-policy``. It ships as
``hosted_ok`` only. ``prohibited`` egress and ``secret`` visibility are refused
in code and cannot be enabled. Every applied run writes an ``egress_audits`` row
naming exactly what was sent, before it is sent.

Providers: ``anthropic`` (default), ``openai``, ``moonshot``. See
``ocbrain.curator`` for the provider backends and the local claim validation
every provider's output must pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocbrain.config import load_config
from ocbrain.core_v1 import is_core_v1
from ocbrain.curator import (
    PROVIDER_DEFAULTS,
    WIKI_STATE_SCHEMA,
    apply_claims,
    input_digest,
    load_env_value,
    now_iso,
    project_digests,
    record_curation_egress,
    request_claims,
    resolve_selection_policy,
    select_evidence,
    validate_claims,
)
from ocbrain.db import connect
from ocbrain.wiki import current_wiki_beliefs, materialize_wiki

WIKI_DIR_NAME = "wiki"


def emit(payload: dict) -> None:
    """One JSON object per line, so a promote log stays greppable."""
    print(json.dumps(payload, sort_keys=True))


def close_project(
    by_status: dict[str, list[str]],
    summary: dict,
    status: str,
    extra: dict | None = None,
) -> None:
    """Report one project's outcome and file it under that outcome.

    Every project a run considered produces exactly one line, whatever happened
    to it. A scope that was skipped has to be visible in the promote log rather
    than inferred from its absence.
    """
    by_status.setdefault(status, []).append(str(summary["project"]))
    emit(summary | {"status": status} | (extra or {}))


def resolve_projects(args, configured: list[str]) -> list[str]:
    """Which project scopes this run curates, in order, without duplicates.

    An explicit ``--project`` always wins, so an operator can curate one scope by
    hand on a machine whose config lists forty.
    """
    if args.projects:
        chosen = list(args.projects)
    elif args.projects_from_config:
        chosen = list(configured)
    else:
        chosen = ["workspace"]
    return list(dict.fromkeys(name.strip() for name in chosen if str(name).strip()))


def read_state(state_path: Path) -> dict:
    """Read ``state.json``, treating an unreadable or malformed file as absent.

    A corrupt cursor must not stop the curator; the cost of re-reading it as
    empty is one extra hosted call per project, and the cost of raising is a
    wiki that stays frozen until someone notices.
    """
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=sorted(PROVIDER_DEFAULTS),
        help="hosted model provider (default: anthropic)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".common",
        help="dotenv file consulted when the API key env var is unset",
    )
    parser.add_argument("--api-key-env", help="defaults to the provider's usual variable")
    parser.add_argument("--base-url", help="defaults to the provider's endpoint")
    parser.add_argument("--model", help="defaults to the provider's mid-tier model")
    parser.add_argument(
        "--project",
        action="append",
        dest="projects",
        metavar="NAME",
        help=(
            "project scope to curate; repeatable. Overrides "
            "--projects-from-config. Defaults to `workspace`"
        ),
    )
    parser.add_argument(
        "--projects-from-config",
        action="store_true",
        help="curate every scope listed in curator.projects",
    )
    parser.add_argument(
        "--min-evidence-per-project",
        type=int,
        help=(
            "skip (and report) a project with fewer eligible objects than this, "
            "instead of spending a hosted call on it. Defaults to "
            "curator.min_evidence_per_project"
        ),
    )
    parser.add_argument("--max-evidence", type=int, default=260)
    parser.add_argument("--max-beliefs", type=int, default=24)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8_000,
        help=(
            "output token budget. On models with adaptive thinking this budget "
            "covers thinking as well as the visible answer, so leave headroom"
        ),
    )
    parser.add_argument(
        "--current-ttl-days",
        type=int,
        default=90,
        help=(
            "expiry stamped on 'current' lifecycle claims so they can age out; "
            "0 disables expiry. Durable claims never expire"
        ),
    )
    parser.add_argument("--wiki-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--egress-policy",
        action="append",
        choices=["hosted_ok", "approval_required"],
        help=(
            "evidence egress policy the curator may read; repeatable. Overrides "
            "curator.egress_policies from config. `local_only` and `prohibited` "
            "egress and `secret` visibility are never eligible and cannot be enabled"
        ),
    )
    parser.add_argument(
        "--allow-hosted-egress",
        action="store_true",
        help=(
            "explicitly authorize bounded approval-required evidence and wiki facts "
            "for this hosted compilation; local-only, prohibited, confidential, and "
            "secret objects stay excluded"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even when the input digest is unchanged since the last run",
    )
    args = parser.parse_args()

    defaults = PROVIDER_DEFAULTS[args.provider]
    model = args.model or defaults["model"]
    api_key_env = args.api_key_env or defaults["api_key_env"]
    base_url = args.base_url or defaults["base_url"]

    db_path = args.db.expanduser()
    wiki_dir = (args.wiki_dir or db_path.parent / WIKI_DIR_NAME).expanduser()
    state_path = wiki_dir / "state.json"
    conn = connect(db_path)
    try:
        if not is_core_v1(conn):
            raise ValueError("database is not an OCBrain v1 core")
        # The operator's standing declaration of what their curator may read,
        # from config (OCBRAIN_CURATOR_EGRESS_POLICIES / the config file), with a
        # CLI override. `local_only`, `prohibited`, and `secret` are refused in
        # code either way.
        curator_cfg = load_config().curator
        egress_policies = args.egress_policy or curator_cfg.egress_policies
        resolved_egress, resolved_visibility = resolve_selection_policy(
            egress_policies=egress_policies,
            visibilities=curator_cfg.visibilities,
            allow_hosted_egress=bool(args.allow_hosted_egress),
        )
        projects = resolve_projects(args, list(curator_cfg.projects))
        min_evidence = (
            curator_cfg.min_evidence_per_project
            if args.min_evidence_per_project is None
            else args.min_evidence_per_project
        )
        max_beliefs = max(1, min(args.max_beliefs, 40))
        prior_digests = project_digests(read_state(state_path))
        # Only a project that actually completed advances its cursor. A failed or
        # skipped project keeps the digest it had, so the next cycle retries it
        # instead of recording a run that never happened.
        next_digests = dict(prior_digests)

        # Resolved on first use, not up front: a cycle where every project is
        # digest-unchanged must still exit 0 on a machine with no key configured.
        api_key: str | None = None
        totals = {
            "eligible_evidence": 0,
            "hosted_calls": 0,
            "claims_accepted": 0,
            "claims_rejected": 0,
            "beliefs_applied": 0,
            "beliefs_unchanged": 0,
            "beliefs_blocked": 0,
            "beliefs_superseded": 0,
            "beliefs_coexist_marked": 0,
            "beliefs_deferred": 0,
            "beliefs_pending_deduped": 0,
        }
        by_status: dict[str, list[str]] = {}
        for project in projects:
            evidence = select_evidence(
                conn,
                limit=max(1, args.max_evidence),
                project=project,
                egress_policies=resolved_egress,
                visibilities=resolved_visibility,
            )
            existing = current_wiki_beliefs(
                conn,
                project=project,
                hosted_egress=True,
                allow_approval_required=bool(args.allow_hosted_egress),
            )
            digest = input_digest(evidence, existing)
            summary = {
                "action": "wiki-curate",
                "project": project,
                "apply": bool(args.apply),
                "provider": args.provider,
                "model": model,
                "eligible_evidence": len(evidence),
                "eligible_kinds": sorted({str(row["kind"]) for row in evidence}),
                "input_characters": sum(len(str(row["body"])) for row in evidence),
                "input_digest": digest,
                "prior_digest_matches": prior_digests.get(project) == digest,
                "raw_transcripts_eligible": False,
                "confidential_or_prohibited_eligible": False,
                "hosted_egress_acknowledged": bool(args.allow_hosted_egress),
                "egress_policies": list(resolved_egress),
                "visibilities": list(resolved_visibility),
            }
            totals["eligible_evidence"] += len(evidence)

            if not args.apply:
                close_project(by_status, summary, "preview")
                continue
            if not evidence:
                close_project(by_status, summary, "no_eligible_evidence")
                continue
            # A scope too thin to be worth a hosted call is reported as skipped,
            # never dropped silently: "the curator ignores this project" has to
            # be visible in the promote log, not inferred from its absence.
            if len(evidence) < min_evidence:
                close_project(
                    by_status,
                    summary,
                    "skipped_thin_project",
                    {"min_evidence_per_project": min_evidence},
                )
                continue
            # An unchanged digest means the model would see the same input for
            # this project and produce the same wiki, so a quiet scope is free.
            if summary["prior_digest_matches"] and not args.force:
                close_project(by_status, summary, "unchanged_no_api_call")
                continue

            # A missing credential is a whole-run precondition, not one project's
            # problem, so it is raised rather than filed as a per-project failure.
            if api_key is None:
                api_key = load_env_value(args.env_file.expanduser(), api_key_env)
                if not api_key:
                    raise ValueError(f"{api_key_env} is not configured")

            try:
                # Record what is about to leave the machine before it leaves.
                audit_id = record_curation_egress(
                    conn,
                    evidence=evidence,
                    provider=args.provider,
                    model=model,
                    project=project,
                    egress_policies=resolved_egress,
                )
                response = request_claims(
                    provider=args.provider,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    evidence=evidence,
                    existing=existing,
                    max_beliefs=max_beliefs,
                    max_tokens=max(1_000, args.max_tokens),
                )
                claims, rejected = validate_claims(
                    response,
                    evidence=evidence,
                    max_beliefs=max_beliefs,
                    existing=existing,
                )
                if not claims:
                    raise RuntimeError(
                        f"no quote-validated beliefs survived; rejected={rejected[:8]}"
                    )
                applied = apply_claims(
                    conn,
                    claims,
                    model=model,
                    project=project,
                    provider=args.provider,
                    current_ttl_days=max(0, args.current_ttl_days),
                )
            except Exception as exc:  # noqa: BLE001 - one bad scope must not stop the rest
                # Do not advance this project's digest. Letting the exception out
                # would also strand every earlier project's completed work
                # unmaterialized, and re-bill all of them on the next cycle.
                totals["hosted_calls"] += 1
                close_project(
                    by_status, summary, "failed", {"error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            totals["hosted_calls"] += 1
            totals["claims_accepted"] += len(claims)
            totals["claims_rejected"] += len(rejected)
            totals["beliefs_applied"] += len(applied["applied"])
            totals["beliefs_unchanged"] += len(applied["unchanged"])
            totals["beliefs_blocked"] += len(applied["blocked"])
            totals["beliefs_superseded"] += len(applied["superseded"])
            totals["beliefs_coexist_marked"] += len(applied["coexist_marked"])
            totals["beliefs_deferred"] += len(applied["deferred"])
            totals["beliefs_pending_deduped"] += len(applied["pending_deduped"])
            next_digests[project] = digest
            close_project(
                by_status,
                summary,
                "completed",
                {
                    "egress_audit_id": audit_id,
                    "accepted": len(claims),
                    "rejected": len(rejected),
                    "applied": len(applied["applied"]),
                    "unchanged": len(applied["unchanged"]),
                    "blocked": len(applied["blocked"]),
                    "superseded": len(applied["superseded"]),
                    "coexist_marked": len(applied["coexist_marked"]),
                    "deferred": len(applied["deferred"]),
                    "pending_deduped": len(applied["pending_deduped"]),
                    "rejection_sample": rejected[:8],
                },
            )

        failed = by_status.get("failed", [])
        rollup = {
            "action": "wiki-curate-rollup",
            "apply": bool(args.apply),
            "provider": args.provider,
            "model": model,
            "projects": projects,
            "projects_by_status": dict(sorted(by_status.items())),
            **totals,
        }
        # Nothing was compiled, so there is nothing new to write out and no
        # cursor to advance. Rebuilding the tree anyway would churn the wiki
        # directory on every quiet cycle for no change.
        if totals["hosted_calls"] == len(failed):
            emit(rollup)
            return 1 if failed else 0

        now = now_iso()
        wiki_count = materialize_wiki(
            conn,
            wiki_dir,
            run={
                "schema_version": WIKI_STATE_SCHEMA,
                "at": now,
                "action": "wiki-curate",
                "provider": args.provider,
                "model": model,
                "projects": {
                    name: {"input_digest": value, "at": now}
                    for name, value in sorted(next_digests.items())
                },
                "evidence_count": totals["eligible_evidence"],
                "accepted_count": totals["claims_accepted"],
                "rejected_count": totals["claims_rejected"],
                "applied_count": totals["beliefs_applied"],
                "unchanged_count": totals["beliefs_unchanged"],
                "blocked_count": totals["beliefs_blocked"],
                "superseded_count": totals["beliefs_superseded"],
                "coexist_marked_count": totals["beliefs_coexist_marked"],
                "deferred_count": totals["beliefs_deferred"],
                "pending_deduped_count": totals["beliefs_pending_deduped"],
            },
        )
        emit(
            rollup
            | {
                "wiki_current_beliefs": wiki_count,
                "wiki_index": str(wiki_dir / "index.md"),
            }
        )
        return 1 if failed else 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
