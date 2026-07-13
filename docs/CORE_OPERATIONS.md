# OCBrain core operations and v1 migration

Core operation is explicit, local, one-shot work plus an on-demand stdio MCP
process. No command in this guide installs a scheduler or authorizes hosted
calls, training, paging, or automatic activation.

## Inspect and reconcile

```bash
ocbrain --db /absolute/core.sqlite status
ocbrain --db /absolute/core.sqlite sync --max-events 1000 --time-budget 10
ocbrain --db /absolute/core.sqlite doctor
```

`status` is read-only. `sync` folds only bounded local event projection work and
rolls back if its declared event/time budget is exceeded. `doctor` checks the
database and negotiates initialize/ping/tools-list with a real child MCP
process.

The v1 runtime MCP should list exactly eight tools. Use
`ocbrain runtime-check` plus real fresh-client acceptance for the broader bridge;
configuration probes alone are not completion evidence.

## Backup and restore

Create a verified online backup at a fresh path:

```bash
ocbrain --db /absolute/source.sqlite backup --output /absolute/backup.sqlite
```

Restore only to a fresh path:

```bash
ocbrain restore --source /absolute/backup.sqlite --output /absolute/restored.sqlite
```

Neither operation silently overwrites a live database. Keep owner-only
permissions on private databases and manifests.

## Plan the v1 migration

```bash
ocbrain --db /absolute/v0.sqlite core-migrate-v1 \
  --core-db /absolute/v1/ocbrain-core-v1.sqlite \
  --archive-db /absolute/archive/ocbrain-v0.sqlite \
  --training-db /absolute/v1/ocbrain-training-v1.sqlite \
  --ops-db /absolute/v1/ocbrain-ops-v1.sqlite \
  --manifest /absolute/v1/ocbrain-v1-migration.json \
  --plan
```

Plan mode opens the source read-only and creates nothing. It refuses reused,
colliding, missing, or non-fresh paths.

## Build fresh outputs

Run the same command without `--plan`. Migration:

1. creates one coherent online archive snapshot;
2. verifies the source event chain;
3. copies the exact event sequence/rowids and event bytes into a strict core;
4. appends deterministic import events for relational evidence, knowledge,
   links, signals, and retrieval snapshots;
5. copies Shared Context source-handle, egress, and closeout receipts with their
   foreign-key links;
6. extracts training and operational tables into separate companion databases;
7. rebuilds semantic projections and FTS once;
8. checks chain, schema inventory, FTS integrity, foreign keys, counts, hashes,
   and table coverage;
9. publishes the five fresh artifacts only after every gate passes.

The outputs are:

- immutable pre-v1 archive;
- strict v1 core;
- training extract;
- ops extract;
- migration manifest.

On failure, owned temporary and partially published outputs are removed. The
source is untouched. A corrupt chain aborts; migration never invents a repaired
replacement history.

## Verify before activation

At minimum inspect:

```bash
ocbrain --db /absolute/v1/ocbrain-core-v1.sqlite status
ocbrain --db /absolute/v1/ocbrain-core-v1.sqlite doctor
```

Also verify from the manifest:

- exact legacy event-prefix count, maximum sequence, hash, and head;
- full event-chain verification;
- exact strict table inventory and absence of legacy/companion tables;
- source catalog and import/extract reconciliation;
- archive/core/training/ops file SHA-256 values and byte sizes;
- SQLite integrity, FTS integrity, and zero foreign-key violations;
- `automatic_activation=false`, `hosted_calls=0`, `network_calls=0`, and
  `schedulers_started=0`.

Run a full projection rebuild on a copy and compare semantic hashes plus MCP
responses. Runtime retrieval, feedback, source-handle, and closeout receipts
must survive that rebuild.

## Activate explicitly

All three registered clients use `scripts/ocbrain-mcp`. To activate a verified
candidate for new processes, write its absolute path to the ignored local
pointer:

```bash
printf '%s\n' '/absolute/v1/ocbrain-core-v1.sqlite' > data/active-core.path
chmod 600 data/active-core.path
```

Migration never performs this step. Start fresh clients afterward; already-open
tasks can retain their older MCP process.

Run a real `context → source → feedback → closeout` turn from Codex, Claude Code,
and OpenClaw. Confirm all three receipts landed in the same v1 core before
declaring activation complete.

## Roll back the launcher choice

Remove or replace `data/active-core.path`, then reconnect clients. This changes
only which existing database a new MCP child opens; it does not mutate the v1
candidate or archive.

Never use the archive as a hidden fallback behind v1 MCP. If v1 is not accepted,
make the rollback explicit and preserve the failed candidate for diagnosis.

## Optional companions

Install only when needed:

```bash
uv pip install ./packages/training
uv pip install ./packages/ops
```

- `ocbrain-training` defaults to `~/.ocbrain/training.sqlite`.
- `ocbrain-ops` and `ocbrain-watchdog` default to
  `~/.ocbrain/ops.sqlite`.
- Legacy mutating operations require an explicit `--legacy-db`.
- Neither package installs a recurring job.
- The core MCP imports and queries neither package.

The three old launchd labels remain retired and must stay disabled:

```text
com.jonathangu.ocbrain.autopilot.light
com.jonathangu.ocbrain.autopilot.heavy
com.jonathangu.ocbrain.stallcheck
```

## Training remains blocked

The clean named-human packet is a private Markdown handoff. The earlier AI
review is remediation evidence only and found 83 failures. Do not enable or run
pilot-v3 training until the pack is fixed, reminted, regraded, resampled,
reviewed by a named human, and separately authorized by the operator.
