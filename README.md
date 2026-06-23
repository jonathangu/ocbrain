# ocbrain

Lightweight shared memory for Codex, Claude, and OpenClaw.

`ocbrain` is the small, auditable OpenClawBrain runtime that turns runtime
evidence into source-backed current knowledge, serves it through MCP, and
records which retrieved knowledge actually helped. It is installed locally on
this Mac mini for Codex, Claude Code, and OpenClaw so each runtime can use the
same brain instead of living in separate memory silos.

The core model has two primitives:

- `evidence`: immutable, append-only, hash-pinned claims about what happened.
- `knowledge`: compiled current belief with a type, lifecycle, and gate.

Memory is a view over current injectable knowledge. Wiki/procedure pages are
`knowledge type='doc'`; skills are `knowledge type='capability'`. The hard gate
is readable versus executable or prescriptive: capabilities, high-risk items,
and prescriptive constraints stay human-gated.

## Status

Lightweight runtime V0 is installed and working locally.

- SQLite ledger with `evidence`, `knowledge`, `knowledge_evidence`, and `memory`
- safe historical ingest
- deterministic triage
- FTS search
- digest/counts
- read-only evaluation harness
- proposal markdown output
- managed native excerpt output
- stdio MCP server, read-only by default
- local retrieval-use feedback logging
- loop result envelope ingest into tagged evidence/knowledge rows

Installed local hosts:

- Codex: `~/.codex/config.toml`
- OpenClaw Codex ACP home: `~/.openclaw/acpx/codex-home/config.toml`
- Claude Code: user-scoped `claude mcp`
- OpenClaw: provider-safe MCP tools in `openclaw.json`

Current active work is still tracked by the long-running build loop in
[docs/BUILD_LOOP.md](docs/BUILD_LOOP.md). The repo is intentionally conservative:
capability and prescriptive knowledge remain human-gated, and the MCP server
does not expose write-capable proposal tools unless it is explicitly launched
with `--allow-writes`.

Runtime integration proof notes live in
[docs/RUNTIME_INTEGRATION.md](docs/RUNTIME_INTEGRATION.md).

## Quick Start

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ocbrain closeout --input ../artifacts/openclawbrain-lite-validated-synthesis-2026-06-21.md
ocbrain --pretty digest
pytest
```

## Historical Ingest

Build a local ignored history database:

```bash
mkdir -p data
ocbrain --db data/ocbrain.sqlite ingest --history-profile --workspace /Users/guclaw/.openclaw/workspace
ocbrain --db data/ocbrain.sqlite triage
ocbrain --db data/ocbrain.sqlite search OpenClawBrain --limit 5
ocbrain --db data/ocbrain.sqlite --pretty digest
```

The default historical profile reads safe text-like sources from workspace memory, artifacts, task artifacts/status, selected sessions, and memory-wiki. It excludes secret-bearing config names, databases, build directories, package caches, `.env`, keys, tokens, and credentials by default.

## Evaluation

Run the local quality and safety harness before trusting candidates:

```bash
ocbrain --db data/ocbrain.sqlite eval --per-target 40 --output-json reports/eval.json --output-md reports/eval.md
ocbrain --db data/ocbrain.sqlite eval --sample-size 10000 --fail-on-leak --output-json reports/leak-scan.json
```

The first harness is intentionally strict about duplicate candidate templates, generic candidate bodies, stale operational facts, search index consistency, and probable secret leakage.

## Proposal And Excerpt Output

```bash
ocbrain --db data/ocbrain.sqlite candidates --target wiki --limit 5
ocbrain --db data/ocbrain.sqlite review approve <candidate_id> --reason "source-backed"
ocbrain --db data/ocbrain.sqlite propose <candidate_id> --output-dir proposals
ocbrain --db data/ocbrain.sqlite excerpt --runtime codex --output /tmp/AGENTS.md --limit 10
```

## MCP

```bash
ocbrain --db data/ocbrain.sqlite mcp
```

For the local installed instance, use the stable launcher:

```bash
scripts/ocbrain-mcp
```

The MCP server currently exposes:

- `brain.search`
- `brain.digest`
- `brain.get`
- `brain.feedback`
- `brain.propose` with `--allow-writes`
- `brain.mark_stale` with `--allow-writes`
- `brain://digest/current`
- `brain://wiki/{slug}`
- `brain://loop/families`

`brain.get` serves current knowledge and reviewed legacy candidates by default;
candidate/private objects require explicit inspection flags. `brain.digest`
returns current injectable memory, values, documents, capabilities, and loop
family scores from `knowledge`. Write-capable tools stay hidden unless the
server is launched with `--allow-writes`.

`brain.search` and `brain.get` return `retrieval_use_id` values. Call
`brain.feedback` with one of `helpful`, `used`, `irrelevant`, `ignored`, or
`harmful` to record whether the retrieved knowledge helped the current runtime.

## Loop-Aware Ingest

`ocbrain` can read autonomous loop result envelopes without running the loop:

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --dry-run \
  --json
```

The envelope schema is `ocbrain.loop_result.v1` in each item `result.json`.
Dry-run ingest validates envelopes, reconstructs done/failed/kept/reverted
counts, summarizes the primary metric and experiment families, opens missing
artifact tripwires in the output, and stages candidates.

An explicit `--apply` mode writes loop-tagged `evidence` and `knowledge` rows
with stable IDs, refreshes the derivable `family_scores` rollup, and keeps
re-ingest idempotent:

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --apply \
  --json
```

It does not enqueue work, run loops, install capabilities, apply policy, or
turn human-gated knowledge current.

## Principles

- Evidence precedes belief.
- Knowledge compiles current belief from evidence.
- Memory is the injected view over current knowledge.
- Capabilities and prescriptive constraints are human-gated.
- MCP is the shared access layer.
- Native files are the high-adherence layer.
- Supersede and archive; do not overwrite or silently delete.
