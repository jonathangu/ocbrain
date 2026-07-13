# OCBrain Ops

Optional manual maintenance and watchdog tools for OCBrain. The package does
not install or enable a scheduler. Hosted operations remain disabled unless an
operator explicitly opts in through local configuration and invokes them.

Install beside the core distribution:

```sh
pip install ocbrain-ops
```

Operational state is written only to `~/.ocbrain/ops.sqlite` by default
(`OCBRAIN_OPS_DB` or `--ops-db` may override it). The watchdog is a one-shot,
explicit command:

```sh
ocbrain-watchdog --ops-db ./ops.sqlite --no-send
```

It may inspect a v1 core through explicit, read-only `--core-db`; findings,
deduplication, pager state, and heartbeat rows stay in the ops ledger.

The extracted v0.x maintenance engines remain available only as compatibility
commands and require an explicit `--legacy-db`. They never default to or mutate
the v1 core:

```sh
ocbrain-ops --legacy-db ./archived-v0.sqlite autopilot --dry-run
```

`ocbrain-ops`, `ocbrain-watchdog`, and `brain-loop-ingest` install no launchd,
cron, or hosted-call schedule.
