# OCBrain Training

Optional, explicitly invoked local dataset curation and training-pilot tools
for OCBrain. Installing this package does not add MCP tools, schedules, hosted
judgment, or background work to the core bridge.

Install beside the core distribution:

```sh
pip install ocbrain-training
```

The companion writes only its own ledger, `~/.ocbrain/training.sqlite` by
default (`OCBRAIN_TRAINING_DB` or `--training-db` may override it):

```sh
ocbrain-training dataset-stats
ocbrain-training --training-db ./training.sqlite dataset-mine --dataset sft
```

Pilot preparation and training-result recording remain disabled until the
operator explicitly enables local training. Retrieval benchmarks may inspect a
v1 core through the explicit, read-only `--core-db` option. They never inherit
the core CLI's database as a write target.

The initial v1 adapter mines SFT and persona examples from local source
artifacts. DPO/all-source mining fails closed until the event-source adapter is
available; it does not fall back to mutating a legacy monolith.
