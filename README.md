# ocbrain

Lightweight shared memory for Codex, Claude, and OpenClaw.

`ocbrain` is the small, auditable OpenClawBrain runtime that turns historical
agent work into source-backed knowledge, serves it through MCP, and records
which retrieved memories actually helped. It is installed locally on this Mac
mini for Codex, Claude Code, and OpenClaw so each runtime can search the same
historical ledger instead of living in separate memory silos.

The consolidation pipeline starts with one narrow job: take a completed work
artifact and produce structured dry-run candidates for the right durable surface:

- `memory`: evidence, facts, preferences, decisions
- `wiki`: compiled belief and stable synthesis
- `skill`: repeatable behavior, proposal-first
- `policy`: constraints, patch-suggestion only
- `ignore`: noise, duplicates, unsafe, or not future-useful

The intended shape is not a new chat agent. It is the shared knowledge layer
behind the agents: evidence in, compact retrieval out, feedback back into the
ledger, and proposal-first writes for durable memory/wiki/skill/policy changes.

## Status

Lightweight runtime V0 is installed and working locally.

- SQLite ledger
- safe historical ingest
- deterministic triage
- FTS search
- digest/counts
- read-only evaluation harness
- proposal markdown output
- managed native excerpt output
- stdio MCP server, read-only by default
- local retrieval-use feedback logging

Installed local hosts:

- Codex: `~/.codex/config.toml`
- OpenClaw Codex ACP home: `~/.openclaw/acpx/codex-home/config.toml`
- Claude Code: user-scoped `claude mcp`
- OpenClaw: provider-safe MCP tools in `openclaw.json`

Current active work is still tracked by the long-running build loop in
[docs/BUILD_LOOP.md](docs/BUILD_LOOP.md). The repo is intentionally conservative:
live memory/wiki/skill/policy mutation remains proposal-first, and the MCP server
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
- `brain://digest/current`

`brain.get` serves reviewed candidates by default; draft/private candidates require
explicit inspection flags. `brain.propose` is write-capable and hidden unless the
server is launched with `--allow-writes`.

`brain.search` and `brain.get` return `retrieval_use_id` values. Call
`brain.feedback` with one of `helpful`, `used`, `irrelevant`, `ignored`, or
`harmful` to record whether the retrieved knowledge helped the current runtime.

## Principles

- Memory stores evidence.
- Wiki compiles belief.
- Skills compile behavior.
- Policy constrains behavior.
- MCP is the shared access layer.
- Native files are the high-adherence layer.
- Skills and policy are proposal-first, never silent auto-mutation.
