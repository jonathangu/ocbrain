# ocbrain Final-Spec Plan

## Goal

Keep OCBrain as one shared brain over MCP: runtimes emit evidence, the brain
compiles current knowledge, and human gates protect executable or prescriptive
change.

## Done

- Final core tables: `evidence`, `knowledge`, `knowledge_evidence`, `memory`.
- Loop rows are tagged evidence/knowledge, not a parallel loop schema.
- MCP digest/search/get/feedback over final core rows.
- Write-gated `brain.propose` for human-gated knowledge.
- Write-gated `brain.mark_stale`.
- `brain.search` filters for project, type, status, loop id, and family.
- Legacy `events`/`candidates`/review-queue tables are dropped by startup.

## Next

- Prune job: mark stale/unreferenced current knowledge without hard deletion.
- Heal job: supersede contradictory values beyond threshold with evidence links.
- Liveness watcher: read runner ledger/deadman timestamps and emit tripwire
  evidence for silence or stale running items.
- Capability packaging: keep proposal-first until human approval.

## Verification

```bash
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . pytest -q
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . ruff check .
uv run --with-editable . python -m compileall src tests
```
