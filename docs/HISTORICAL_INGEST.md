# Historical Ingest Notes

Date: 2026-06-21

## Run

Command:

```bash
PYTHONPATH=src python3 -m ocbrain.cli \
  --db data/ocbrain.sqlite \
  ingest \
  --history-profile \
  --workspace /Users/guclaw/.openclaw/workspace
```

Result:

```json
{"counts": {"by_target": {}, "candidates": 0, "events": 5202}, "inserted": 5202, "seen": 5330, "skipped": 128}
```

Triage:

```bash
PYTHONPATH=src python3 -m ocbrain.cli --db data/ocbrain.sqlite triage
```

Result:

```json
{
  "events_triaged": 5202,
  "candidates_inserted": 8453,
  "counts": {
    "events": 5202,
    "candidates": 8453,
    "by_target": {
      "ignore": 2236,
      "memory": 1476,
      "policy": 1655,
      "skill": 388,
      "wiki": 2698
    }
  }
}
```

## Safety

The historical profile is intentionally local and conservative.

Default exclusions include:

- `.git`, `.venv`, `node_modules`, build/cache dirs
- `.env`
- secret/credential/token/key named files
- OpenClaw config JSON names
- SQLite/database files
- large files over the configured byte limit

The ingester redacts common inline secret patterns before storing searchable body text.

## Current Limits

- Triage is deterministic and intentionally crude; it is useful for first-pass candidate routing, not final human-quality memory extraction.
- The DB is ignored by git under `data/*.sqlite`.
- This run did not create cron jobs, mutate live memory/wiki/skills/policy, or send data off-machine.
