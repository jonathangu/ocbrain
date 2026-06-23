# ocbrain Long-Running Build Loop

Date opened: 2026-06-21 15:50 PDT

## Why This Exists

The first `ocbrain` pass produced a working local prototype too quickly to count as the
assignment being complete. Treat commit `b554799` as seed code and baseline evidence,
not as the finished OpenClawBrain.

This loop turns the prototype into a real consolidation governor through repeated
measure-build-check cycles. Each cycle must leave a small artifact, runnable evidence,
and a next-step pointer that survives context compaction.

## Operating Rules

- Work in loops, not one giant closeout.
- Keep `TASKS.md` as the principal truth for whether the build is active.
- Keep this file as the repo-local loop contract.
- Put worker outputs under `/Users/guclaw/.openclaw/workspace/task-artifacts/ocbrain-build-loop/`.
- Put compact machine-readable status under `/Users/guclaw/.openclaw/workspace/task-status/ocbrain-build-loop.json`.
- Prefer dry-run and proposal output until review quality is proven.
- Do not create cron jobs, mutate live memory/wiki/skills/policy, or publish remotely without Jonathan's explicit approval.
- Each loop must end with evidence: tests, corpus stats, sample audits, or integration proof.

## Loop Cadence

1. Check in with Jonathan at the start of each major loop or when blocked.
2. Run a bounded audit or build step.
3. Write an artifact with findings, evidence, and next action.
4. Update the status file.
5. Update `TASKS.md` only at phase boundaries.
6. Commit meaningful repo changes in small commits.

## Phase Gates

### Loop 0: Reopen And Audit

Goal: demote the fast build to a prototype, inspect what exists, and identify the
highest-risk gaps.

Exit evidence:

- active `TASKS.md` entry
- status file initialized
- at least three independent audit artifacts
- repo tests still pass

### Loop 1: Quality Harness

Goal: create an eval harness that samples ingested candidates and grades target,
evidence, confidence, redaction, duplicate, and actionability quality.

Exit evidence:

- repeatable CLI command for evaluation
- JSON/Markdown report output
- tests for evaluator logic
- baseline report over the local historical DB

### Loop 2: Consolidation Review UX

Goal: make candidate review practical for a human/operator before any live writes.

Exit evidence:

- review queue CLI
- approve/reject/defer commands
- proposal grouping/deduplication
- audit trail for decisions

### Loop 3: Runtime Integration Proof

Goal: prove Codex/OpenClaw/Claude can consume compact native excerpts and MCP search
without bloating context or bypassing native instruction surfaces.

Exit evidence:

- generated AGENTS/CLAUDE/OpenClaw excerpt samples
- MCP smoke with representative queries
- documented install/config path
- no live mutation unless explicitly approved

### Loop 4: Historical Backfill Iterations

Goal: run multiple bounded passes over history, improve filters/classifiers, and track
how quality changes over time.

Exit evidence:

- before/after candidate distributions
- false positive/false negative examples
- redaction leakage audit
- narrowed ingestion profiles by source type

### Loop 5: Loop-Aware Brain Ingest

Goal: make ocbrain understand autonomous loop result envelopes without becoming
the loop runner, using the final evidence/knowledge core instead of parallel
loop tables.

Exit evidence:

- `ocbrain.loop_result.v1` envelope validation
- dry-run `brain-loop-ingest` command
- deterministic run summary, metric, experiment-family, candidate, and tripwire output
- tests proving dry-run ingest writes nothing
- explicit `--apply` mode writes loop-tagged evidence/knowledge rows and is idempotent
- `family_scores` rollup is derivable from loop-tagged rows

### Loop 6: Scheduler Readiness

Goal: prepare a scheduled consolidation loop without enabling it yet.

Exit evidence:

- dry-run schedule command
- idempotence proof
- failure recovery notes
- approval checklist for enabling cron/automation

## Current Next Action

Continue Loop 5 loop-aware brain ingest:

- add loop digest/wiki draft writers before any human-gated promotion
- keep scheduled dry-run readiness queued until loop evidence ingestion is stable
