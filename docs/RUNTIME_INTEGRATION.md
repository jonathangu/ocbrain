# Runtime Integration Proof

`ocbrain` has two runtime-facing read surfaces:

- native managed excerpts for high-adherence instruction files
- a read-only stdio MCP server for search, digest, and reviewed candidate lookup

This document records the proof path. It is not an install request and does not enable
cron, live memory/wiki/skill/policy mutation, or write-capable MCP tools.

## Native Excerpts

Generate a compact managed block for a runtime:

```bash
ocbrain --db data/ocbrain.sqlite excerpt --runtime codex --output /tmp/ocbrain-AGENTS.md --limit 10
ocbrain --db data/ocbrain.sqlite excerpt --runtime claude --output /tmp/ocbrain-CLAUDE.md --limit 10
ocbrain --db data/ocbrain.sqlite excerpt --runtime openclaw --output /tmp/ocbrain-OPENCLAW.md --limit 10
```

Default excerpt output includes only reviewed states:

- `approved`
- `proposed`
- `applied`

Draft/deferred candidates require explicit inspection:

```bash
ocbrain --db data/ocbrain.sqlite excerpt --runtime codex --output /tmp/ocbrain-draft.md --include-draft
```

## MCP

Run the read-only server:

```bash
ocbrain --db data/ocbrain.sqlite mcp
```

Default tools:

- `brain.search`: source-backed event search over non-private scopes
- `brain.digest`: ledger counts
- `brain.get`: reviewed candidate lookup

`brain.get` blocks draft candidates unless `include_draft` is true. It blocks private
candidates unless `include_private` is true.

The write-capable `brain.propose` tool is hidden unless explicitly enabled:

```bash
ocbrain --db data/ocbrain.sqlite mcp --allow-writes
```

Do not enable `--allow-writes` for routine runtime integration. Use CLI review and
proposal workflows first.

## Proof Commands

Run these before any live runtime install:

```bash
PYTHONPATH=src uv run --no-project --with pytest --with ruff --python /opt/homebrew/bin/python3.13 python -m pytest
PYTHONPATH=src uv run --no-project --with ruff --python /opt/homebrew/bin/python3.13 ruff check .
PYTHONPATH=src /opt/homebrew/bin/python3.13 -m compileall -q src tests
PYTHONPATH=src /opt/homebrew/bin/python3.13 -m ocbrain.cli --db data/ocbrain.sqlite excerpt --runtime codex --output /tmp/ocbrain-AGENTS.md --limit 10
```

For MCP request/response smoke, use `ocbrain.mcp.handle_request` in a local fixture DB
or pipe JSON-RPC lines to `ocbrain --db ... mcp`.
