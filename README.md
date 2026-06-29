# ocbrain

Lightweight shared brain for Codex, Claude Code, OpenClaw, and future runtimes.

`ocbrain` follows the final OpenClawBrain spec: immutable evidence goes in,
compiled current knowledge comes out. It is a librarian/compiler, not an
autopilot. It never runs loops, enqueues work, applies policy, installs skills,
or pushes irreversible change.

## Core Model

- `evidence`: immutable, append-only, hash-pinned claims about what happened.
- `knowledge`: current belief compiled from evidence, with `type`, lifecycle,
  privacy scope, and gate.
- `knowledge_evidence`: support/contradiction/derivation links.
- `memory`: a SQLite view over current injectable knowledge.

Knowledge types:

- `value`: facts and metrics.
- `doc`: readable wiki/procedure pages.
- `capability`: executable or loadable skills/procedures.

The bright line is readable versus executable or prescriptive. Capabilities,
high-risk knowledge, and prescriptive constraints are human-gated and
proposal-first.

## Status

The legacy `events`/`candidates`/review-queue model has been removed from the
active schema and CLI. Startup drops those old tables if they exist.

Current surfaces:

- SQLite ledger: `evidence`, `knowledge`, `knowledge_evidence`, `retrieval_uses`,
  `loop_liveness`, `family_scores`, and `memory`.
- CLI: `evidence`, `value`, `knowledge`, `import-memory`, `import-history`,
  `search`, `digest`, `loop-ingest`, `propose`, `mark-stale`, `prune`, `heal`,
  `liveness-check`, `mcp`.
- MCP: `brain.search`, `brain.get`, `brain.digest`, `brain.feedback`,
  write-gated `brain.propose`, write-gated `brain.mark_stale`.
- Resources: `brain://digest/current`, `brain://wiki/{slug}`,
  `brain://loop/families`.

For a full product and engineering walkthrough, read
[`docs/ULTIMATE_GUIDE.md`](docs/ULTIMATE_GUIDE.md). For a pickup guide for
another agent, read [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md).

## Quick Start

```bash
uv run --with-editable . ocbrain init
uv run --with-editable . ocbrain evidence --claim "Codex emitted evidence."
uv run --with-editable . ocbrain value \
  --subject runtime:codex \
  --predicate shared_brain \
  --bool true \
  --status current \
  --inject
uv run --with-editable . ocbrain import-memory MEMORY.md memory/
uv run --with-editable . ocbrain --pretty digest
```

`import-memory` converts markdown memory files into final-spec immutable
evidence plus current `doc` knowledge, then indexes redacted document text so
`search`, `digest`, and MCP tools can return source-backed context.

To harvest local runtime transcript stores:

```bash
uv run --with-editable . ocbrain import-history \
  ~/.openclaw/agents ~/.openclaw/commitments ~/.openclaw/media/inbound \
  ~/.codex/sessions ~/.codex/archived_sessions \
  ~/.claude/projects ~/.claude/sessions ~/.claude/tasks
```

`import-history` catalogs every matched history file as evidence plus current
`doc` knowledge. It records a source path, file-size/mtime fingerprint, and a
bounded redacted head/tail text window so large transcript trees stay usable.
Repeated imports skip already-harvested source paths before reading excerpts.

## MCP

```bash
uv run --with-editable . ocbrain --db data/ocbrain.sqlite mcp
```

Installed launcher:

```bash
scripts/ocbrain-mcp
```

Routine MCP is read-first. Write-capable tools are hidden unless the server is
launched with `--allow-writes`:

```bash
uv run --with-editable . ocbrain --db data/ocbrain.sqlite mcp --allow-writes
```

`brain.search` supports final-spec filters:

```json
{
  "query": "typecheck narrowing failures",
  "filters": {
    "loop_id": "repo-quality-loop",
    "family": "typecheck_narrowing",
    "project": "ocbrain"
  }
}
```

`brain.search` and `brain.get` return `retrieval_use_id` values. Call
`brain.feedback` with `helpful`, `used`, `irrelevant`, `ignored`, or `harmful`
to record usefulness. With `--allow-writes`, `brain.feedback` can also approve
or reject human-gated candidate knowledge:

```json
{ "id": "know_...", "decision": "approve", "actor": "jon" }
```

## Loop Ingest

`ocbrain` observes loop result envelopes as evidence; it does not run the loop.

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --dry-run \
  --json
```

Apply mode writes loop-tagged evidence/knowledge rows and refreshes
`family_scores`:

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --apply \
  --json
```

Kept loop results require verifier evidence whose target hash matches the
changed artifact hash. Mismatches fail ingest and write tripwire evidence.
Failed loop results must include `failure_class` as `approach`, `precondition`,
`infra`, `safety`, or `unknown`. Only `approach` failures count toward an
`exhausted` family; `precondition` and `infra` failures mark the family
`blocked` and stage repair context instead of suppressing the family.
Forced-exploration envelopes can set `forced_exploration=true`; ingest records
whether those attempts found improvement.

## Maintenance

Maintenance commands are designed for OpenClaw cron/heartbeat lanes, but no cron
is installed by this repo.

```bash
uv run --with-editable . ocbrain prune \
  --ttl-days 30 \
  --unhelpful-ttl-days 14 \
  --archive-stale-days 90
uv run --with-editable . ocbrain heal --numeric-threshold 0.01
uv run --with-editable . ocbrain liveness-check --runner-ledger loops/runner.sqlite
```

`prune` marks unreferenced expired knowledge `stale`, decays served-but-never
useful knowledge on a shorter TTL, and can later archive stale rows without
deleting the audit trail. `heal` supersedes conflicting current values and
writes correction evidence. `liveness-check` reads runner deadman rows and
writes loop tripwire evidence such as `heartbeat_starved` or
`no_ledger_writes`; it does not claim lanes or enqueue loop work.

## Verification

```bash
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . pytest -q
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . ruff check .
uv run --with-editable . python -m compileall src tests
```

## Principles

- Evidence before belief.
- Verified is not claimed.
- Memory is a view, not a store.
- Supersede/archive; do not overwrite in place.
- A derived object's privacy scope is the most restrictive linked source scope.
- Human gate before executable or prescriptive knowledge.
- Emit evidence; do not write durable knowledge directly from runtimes.
- Watch loops closely enough to tell done from wedged, without running them.
