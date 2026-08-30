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

The v1 runtime MCP should list exactly thirteen tools. Use
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
6. extracts the legacy source's training and operational tables into their own
   files, so every source table is accounted for rather than dropped;
7. rebuilds semantic projections and FTS once;
8. checks chain, schema inventory, FTS integrity, foreign keys, counts, hashes,
   and table coverage;
9. publishes the five fresh artifacts only after every gate passes.

The outputs are:

- immutable pre-v1 archive;
- strict v1 core;
- legacy training extract;
- legacy ops extract;
- migration manifest.

The two extracts exist so the manifest can reconcile every row in the source.
Nothing installs against them and no command reads them; the packages that once
did are gone.

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
- exact strict table inventory and absence of legacy training/ops tables;
- source catalog and import/extract reconciliation;
- archive/core/training/ops file SHA-256 values and byte sizes;
- SQLite integrity, FTS integrity, and zero foreign-key violations;
- `hosted_calls=0`, `network_calls=0`, and `schedulers_started=0` in the
  manifest's safety block.

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

Do not terminate a client-owned MCP child to make an already-open task pick up
the new core. Some stdio hosts retain a dead transport instead of reconnecting.
Use a fresh task or explicitly reconnect the client.

If a task reports `Transport closed`, stop after the first failure and preserve
the exact runtime-tool arguments. Run that request once through
`scripts/ocbrain-runtime-call`, which uses the normal runtime dispatcher and
rejects admin tools, then reconnect the client. Do not build a retry loop around
a dead stdio handle.

Run a real `context → source → feedback → closeout` turn from Codex, Claude Code,
and OpenClaw. Confirm all three receipts landed in the same v1 core before
declaring activation complete.

## Roll back the launcher choice

Remove or replace `data/active-core.path`, then reconnect clients. This changes
only which existing database a new MCP child opens; it does not mutate the v1
candidate or archive.

Never use the archive as a hidden fallback behind v1 MCP. If v1 is not accepted,
make the rollback explicit and preserve the failed candidate for diagnosis.

## Inspect the effective config

```bash
ocbrain --db /absolute/core.sqlite config
ocbrain --db /absolute/core.sqlite config --section curator --changed-only
```

The output labels every field `default`, `file`, or `env` and names the file it
resolved, so "why is the curator sending nothing" is one command rather than an
archaeology exercise. Resolution order is defaults, then
`~/.ocbrain/ocbrain.config.json` (or `$OCBRAIN_CONFIG`, or the legacy
checkout-relative `data/ocbrain.config.json` when only that exists), then
`OCBRAIN_<SECTION>_<FIELD>` environment variables.

There are four sections — `retrieval`, `scopes`, `curator`, `deslop` — holding
20 fields between them. There were 115 across seventeen sections; the thirteen
that are gone (`autopilot`, `review`, `correction`, `labels`, `quarantine`,
`promote`, `judge`, `dataset`, `dataset_grading`, `teacher`, `archive`, `embed`,
`excerpt_render`) configured code that is also gone. An unknown section or an
unknown key in an operator's config file is ignored rather than fatal, so a
config written for an older core still loads and the stale keys simply do
nothing.

## Retired launchd labels

The `com.jonathangu.ocbrain.*.plist` files are deleted from `ops/`. They named
the light autopilot, heavy autopilot, and stallcheck loops, all of which are
gone. An operator upgrading from a legacy install should unload and delete any
agent left behind:

```bash
launchctl print-disabled "gui/$(id -u)" | grep ocbrain
launchctl bootout "gui/$(id -u)/com.jonathangu.ocbrain.autopilot.light"
rm -f ~/Library/LaunchAgents/com.jonathangu.ocbrain.*.plist
```

For a scheduled loop you actually want, see
[SCHEDULED_MAINTENANCE.md](SCHEDULED_MAINTENANCE.md); it is an explicit opt-in
around `scripts/brain-sync.sh` and `scripts/brain-promote.sh`.
