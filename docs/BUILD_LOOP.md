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

### Loop 1: Final Core Quality Harness

Goal: keep tests focused on the final evidence/knowledge core: typed values,
identity-spine dedupe, source-backed rendering, privacy gates, and human-gated
capability proposals.

Exit evidence:

- repeatable test/ruff/compileall commands
- tests for evidence/knowledge/link behavior
- tests for human-gated proposal-first behavior
- tests proving legacy table removal

### Loop 2: Knowledge Proposal UX

Goal: make human-gated knowledge practical to inspect before any live writes.

Exit evidence:

- proposal markdown for human-gated knowledge rows
- stale/supersession operations over `knowledge`
- evidence links included in proposals
- no live skill/policy/wiki/memory apply

### Loop 3: Runtime Integration Proof

Goal: prove Codex/OpenClaw/Claude can consume compact native excerpts and MCP search
without bloating context or bypassing native instruction surfaces.

Exit evidence:

- generated AGENTS/CLAUDE/OpenClaw excerpt samples
- MCP smoke with representative queries
- documented install/config path
- no live mutation unless explicitly approved

### Loop 4: Runtime Install And Public Surface

Goal: publish and install the lightweight brain, then update public surfaces to
point at `ocbrain`.

Exit evidence:

- public GitHub repo
- local MCP install for Codex/Claude/OpenClaw
- MCP smoke proof
- public site updates

### Loop 5: Loop-Aware Brain Ingest

Goal: make ocbrain understand autonomous loop result envelopes without becoming
the loop runner, using the final evidence/knowledge core instead of parallel
loop tables.

Exit evidence:

- `ocbrain.loop_result.v1` envelope validation
- dry-run `brain-loop-ingest` command
- deterministic run summary, metric, experiment-family, knowledge, and tripwire output
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

Continue Loop 5 final-spec work:

- add prune/heal jobs over `knowledge`
- add liveness watcher wiring over runner ledger/deadman timestamps
- keep scheduled dry-run readiness queued until loop evidence ingestion is stable
