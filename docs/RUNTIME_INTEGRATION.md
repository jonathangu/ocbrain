# Runtime Integration Proof

`ocbrain` has two runtime-facing surfaces:

- native managed excerpts for high-adherence instruction files
- a read-mostly stdio MCP server for search, digest, reviewed candidate lookup,
  and retrieval-use feedback

This document records the proof path and the current local install. It does not
enable cron, live memory/wiki/skill/policy mutation, or write-capable proposal
tools.

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

Installed launcher:

```bash
/Users/guclaw/.openclaw/workspace/ocbrain/scripts/ocbrain-mcp
```

Default tools:

- `brain.search`: source-backed event search over non-private scopes
- `brain.digest`: ledger counts
- `brain.get`: reviewed candidate lookup
- `brain.feedback`: local retrieval-use outcome logging

`brain.get` blocks draft candidates unless `include_draft` is true. It blocks private
candidates unless `include_private` is true.

`brain.search` and `brain.get` return `retrieval_use_id` values. Feedback accepts
`helpful`, `used`, `irrelevant`, `ignored`, or `harmful` and updates only the
local `retrieval_uses` ledger row.

The write-capable `brain.propose` tool is hidden unless explicitly enabled:

```bash
ocbrain --db data/ocbrain.sqlite mcp --allow-writes
```

Do not enable `--allow-writes` for routine runtime integration. Use CLI review and
proposal workflows first.

## Local Runtime Install

The local lightweight runtime is installed for all three active agent surfaces:

```toml
[mcp_servers.ocbrain]
command = "/Users/guclaw/.openclaw/workspace/ocbrain/scripts/ocbrain-mcp"
args = []
startup_timeout_sec = 120
```

Installed locations:

- Codex: `/Users/guclaw/.codex/config.toml`
- Codex ACP home: `/Users/guclaw/.openclaw/acpx/codex-home/config.toml`
- Claude Code: user-scoped MCP entry added with `claude mcp add --scope user ocbrain -- ...`
- OpenClaw: `openclaw mcp add ocbrain --command ...` registered provider-safe tools

OpenClaw provider-safe tool names:

- `ocbrain__brain-search`
- `ocbrain__brain-digest`
- `ocbrain__brain-get`
- `ocbrain__brain-feedback`

No unattended cron or heartbeat loop is enabled by this install.

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

Current local verification also includes:

```bash
claude mcp list
openclaw mcp list
openclaw mcp probe ocbrain
```
