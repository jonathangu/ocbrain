# ocbrain End-to-End Build Plan

Date: 2026-06-22

## North Star

Build ocbrain into an OpenClaw-owned consolidation governor that turns finished work
and historical runtime traces into the smallest useful durable artifact on the right
surface, then serves compact context back through both MCP and native runtime files.

The design deliberately uses both access layers:

- MCP is the shared query and retrieval layer.
- Native excerpts are the high-adherence layer for Codex, Claude, and OpenClaw.

## Safety Boundaries

- Local-first and source-backed.
- Scope every event, candidate, artifact, retrieval, and output.
- Treat runtime output as context, not instructions.
- Keep skills and policy proposal-first.
- Do not mutate live memory, wiki, skills, policy, cron, packages, or remote surfaces
  without explicit approval.
- Prefer preview copies and eval gates before broad historical DB mutation.

## Target Architecture

### Store

- `events`: completed work, sessions, docs, artifacts, status files.
- `evidence`: source pointers and bounded excerpts.
- `candidates`: proposed memory/wiki/skill/policy/ignore records.
- `artifact_links`: links from reviewed candidates to emitted artifacts.
- `retrieval_uses`: served records and later usefulness/outcome.
- `invalidations`: contradiction and supersession edges.
- FTS search index for fast local retrieval.

### Writers

- Memory writer: stage source-backed operational facts.
- Wiki writer: stage or apply managed wiki blocks only after quality gates.
- Skill writer: Skill Workshop pending proposals only.
- Policy writer: patch suggestion artifact only.
- Native excerpt writer: budgeted, scoped, removable managed blocks.

### Runtime Surfaces

- Codex: `AGENTS.md`, Codex skills, and MCP search.
- Claude: `CLAUDE.md`, `.claude/rules/`, Claude skills, and MCP search.
- OpenClaw: `MEMORY.md`, daily memory, memory wiki, Skill Workshop, policy proposals, and MCP.
- Future runtimes: MCP first, plus a compact native excerpt if the runtime has one.

## Build Phases

### Phase A: Ledger And Historical Intake

Status: mostly implemented.

- Safe historical file walker over workspace memory, artifacts, task artifacts/status,
  docs, selected sessions, and wiki source pages.
- Redaction before storage.
- Scope inference.
- SQLite ledger with events/evidence/candidates/retrieval/invalidations.
- Evaluation harness and leak guard.

Current Loop 4 work is hardening this phase for large retrospective ingestion.

### Phase B: Candidate Quality And Review

Status: partially implemented.

- Deterministic classifier.
- Review queue list/inspect/approve/reject/defer.
- Grouped duplicate review.
- Proposal gates so draft candidates do not write proposals by default.
- Stale structural rebuild detection.

Next:

- Add source-type/target diff reporting and preview gates before every broad apply.
- Improve evidence/title alignment after boilerplate skipping.
- Add temporal invalidation heuristics for latest/current/installed/version facts.

### Phase C: MCP And Native Read Proof

Status: implemented as controlled proof.

- Read-only MCP search/digest/get by default.
- Reviewed-by-default gates for `brain.get`.
- Controlled fixture proves non-empty Codex/Claude/OpenClaw excerpts.
- Draft/private read paths require explicit opt-in.

Next:

- Add retrieval-use logging for MCP/native serves.
- Expand MCP resources after search/get semantics are stable.

### Phase D: Proposal Writers

Status: partially implemented.

- Markdown proposal writer exists.
- Review approval gate exists.

Next:

- Memory proposal writer with temporal/supersession metadata.
- Wiki draft/apply writer with managed-block preservation.
- Skill Workshop proposal integration.
- Policy patch suggestion artifacts.

### Phase E: Runtime Install Path

Status: documented proof, not enabled live.

- Keep read-only MCP as the default runtime integration.
- Keep write-capable MCP hidden unless explicitly enabled.
- Generate native excerpt samples before writing into real runtime roots.
- Budget Codex/Claude/OpenClaw excerpts well below runtime caps.

### Phase F: Scheduled Dry Run

Status: queued; do not enable yet.

- Prepare an idempotent scheduled dry-run command.
- Produce daily digest artifacts to Jon.
- Run proposal-only for at least a week.
- Add failure recovery notes and explicit approval checklist before cron.

## Compaction Resume

On context compaction or `/new`, resume with:

```bash
cd /Users/guclaw/.openclaw/workspace
sed -n '1,220p' TASKS.md
sed -n '1,220p' task-status/ocbrain-build-loop.json
cd /Users/guclaw/.openclaw/workspace/ocbrain
git status --short
PYTHONPATH=src uv run --no-project --with pytest --with ruff --python /opt/homebrew/bin/python3.13 python -m pytest
```

Then continue the `next_action` in `task-status/ocbrain-build-loop.json`.
