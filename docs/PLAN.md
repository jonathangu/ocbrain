# ocbrain End-to-End Build Plan

Date: 2026-06-21

## Goal

Build `ocbrain` into a local OpenClawBrain Lite consolidation governor that can:

1. Ingest local historical OpenClaw/Codex/Claude/workspace artifacts safely.
2. Store source-backed events/evidence/candidates in SQLite.
3. Classify durable lessons into `memory`, `wiki`, `skill`, `policy`, or `ignore`.
4. Search and digest indexed history.
5. Emit proposal markdown rather than mutating live skills/policy.
6. Expose a small MCP-ready interface for shared runtime access.

## Safety Defaults

- Local only.
- No secret-bearing config ingestion by default.
- Exclude `.env`, key files, OpenClaw config JSON, credentials, databases, media, build dirs, and package caches.
- Store excerpts and hashes, not giant raw archives.
- Skills and policy remain proposal-first.
- No cron jobs in this build.
- No package/runtime upgrades in this build.

## Phases

### Phase 1: Ledger

Add SQLite schema:

- `events`
- `evidence`
- `candidates`
- `artifact_links`
- `retrieval_uses`
- `invalidations`
- FTS index for event/evidence search

Status: implemented.

### Phase 2: Ingest

Add a safe historical file walker:

- Markdown/text artifacts
- daily memory files
- task artifacts/status
- selected session `.jsonl`
- wiki source pages
- docs

Use path heuristics to assign source type and scope.

Status: implemented with safety exclusions.

### Phase 3: Triage

Classify untriaged events and store candidates.

Status: implemented with deterministic conservative heuristics.

### Phase 4: Search And Digest

Add:

- `ocbrain search`
- `ocbrain digest`
- `ocbrain candidates list/inspect`

Status: implemented as `search`, `digest`, and `candidates`.

### Phase 5: Proposals

Write candidate proposal files under `proposals/`:

- memory proposal
- wiki proposal
- skill proposal
- policy patch suggestion

Status: implemented as markdown proposal writer.

### Phase 6: MCP-Ready Serving

Add a stdio JSON-RPC MCP server skeleton with:

- `brain.search`
- `brain.get`
- `brain.digest`
- `brain.propose`
- resource listing/read for current digest and candidate records

Status: implemented for stdio JSON-RPC with digest resource and search/digest/get/propose tools.

### Phase 7: Historical Smoke Run

Run a bounded ingest over safe workspace history and verify:

- events inserted
- candidates generated
- search works
- digest works
- proposals can be emitted

Status: completed against `data/ocbrain.sqlite`.

Historical smoke evidence from 2026-06-21:

- Full safe historical ingest: 5,330 files seen, 5,202 events inserted, 128 skipped.
- Full deterministic triage: 5,202 events triaged, 8,453 candidates inserted.
- Candidate distribution: 2,698 wiki, 1,655 policy, 1,476 memory, 388 skill, 2,236 ignore.
- Search smoke: `OpenClawBrain` returned archived task artifacts and memory files.
- Proposal smoke: wrote wiki proposal markdown from a candidate.
- Excerpt smoke: wrote a managed Codex `AGENTS.md` block.
- MCP smoke: `initialize` and `tools/list` returned expected server/tools.

## Compaction Resume Protocol

If context compacts, resume by running:

```bash
cd /Users/guclaw/.openclaw/workspace/ocbrain
git status --short
sed -n '1,220p' docs/PLAN.md
PYTHONPATH=src python3 -m pytest
```

Then continue the first unfinished phase above.
