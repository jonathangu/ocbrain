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

V1 end-to-end local loop is implemented:

- SQLite ledger
- safe historical ingest
- deterministic triage
- FTS search
- digest/counts
- proposal markdown output
- managed native excerpt output
- stdio MCP server skeleton

## Quick Start

```bash
python -m venv .venv
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
- `brain.propose`
- `brain://digest/current`

## Principles

- Memory stores evidence.
- Wiki compiles belief.
- Skills compile behavior.
- Policy constrains behavior.
- MCP is the shared access layer.
- Native files are the high-adherence layer.
- Skills and policy are proposal-first, never silent auto-mutation.
