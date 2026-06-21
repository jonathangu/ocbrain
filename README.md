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

Brand-new repo. V1 is a dry-run classifier skeleton.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ocbrain-closeout --input ../artifacts/openclawbrain-lite-validated-synthesis-2026-06-21.md
pytest
```

## Principles

- Memory stores evidence.
- Wiki compiles belief.
- Skills compile behavior.
- Policy constrains behavior.
- MCP is the shared access layer.
- Native files are the high-adherence layer.
- Skills and policy are proposal-first, never silent auto-mutation.
