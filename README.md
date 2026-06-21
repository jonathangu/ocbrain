# ocbrain

OpenClawBrain Lite: an OpenClaw-owned consolidation governor for shared agent knowledge.

`ocbrain` starts with one narrow job: take a completed work artifact and produce structured dry-run candidates for the right durable surface:

- `memory`: evidence, facts, preferences, decisions
- `wiki`: compiled belief and stable synthesis
- `skill`: repeatable behavior, proposal-first
- `policy`: constraints, patch-suggestion only
- `ignore`: noise, duplicates, unsafe, or not future-useful

The intended shape is not a new agent runtime. It is a small, auditable pipeline that preserves evidence, writes proposals, serves compact context through MCP, and compiles native excerpts for runtimes like Codex and Claude.

## Status

Baseline V0 is implemented. It is useful as a working seed, but it is not the finished
OpenClawBrain system.

- SQLite ledger
- safe historical ingest
- deterministic triage
- FTS search
- digest/counts
- read-only evaluation harness
- proposal markdown output
- managed native excerpt output
- stdio MCP server skeleton, read-only by default

Current active work is the long-running build loop in [docs/BUILD_LOOP.md](docs/BUILD_LOOP.md).
That loop must prove quality, runtime fit, reviewer ergonomics, and repeatable
consolidation before `ocbrain` is considered done.

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
ocbrain --db data/ocbrain.sqlite propose <candidate_id> --output-dir proposals
ocbrain --db data/ocbrain.sqlite excerpt --runtime codex --output /tmp/AGENTS.md --limit 10
```

## MCP

```bash
ocbrain --db data/ocbrain.sqlite mcp
```

The MCP server currently exposes:

- `brain.search`
- `brain.digest`
- `brain.get`
- `brain://digest/current`

`brain.propose` is write-capable and hidden unless the server is launched with `--allow-writes`.

## Principles

- Memory stores evidence.
- Wiki compiles belief.
- Skills compile behavior.
- Policy constrains behavior.
- MCP is the shared access layer.
- Native files are the high-adherence layer.
- Skills and policy are proposal-first, never silent auto-mutation.
