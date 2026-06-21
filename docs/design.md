# ocbrain Design Notes

## V1

V1 is `brain-closeout`: a dry-run command that reads one finished artifact and emits candidate JSON.

It does not write memory, wiki, skills, policy, cron jobs, or native excerpts.

## Candidate Contract

Each candidate carries:

- target surface
- title
- body
- confidence
- privacy scope
- risk
- evidence pointers
- duplicate or contradiction hints

## Later Phases

1. SQLite ledger for events, evidence, candidates, artifact links, retrieval uses, and invalidations.
2. Proposal writers for memory, wiki, Skill Workshop, and policy patch suggestions.
3. Read-mostly MCP server with tools and resources.
4. Native excerpt compiler for `AGENTS.md`, `CLAUDE.md`, and runtime-specific skill stubs.
5. OpenClaw cron dry-run consolidation.

## Non-Goals

- Custom agent runtime
- Proof engine
- Model routing
- Raw vector dump
- Automatic live skill or policy mutation
