# Public golden fixtures

`golden_context_v1.json` is a deterministic synthetic contract fixture for the
core Shared Context runtime. It is not harvested user data, an evaluation of a
person, or input to any training workflow.

The golden cases exercise real MCP `brain.context` and `brain.source` calls.
They intentionally assert semantic outputs—eligible IDs, scope and delivery
counts, contradictions, source hashes, and denial boundaries—without freezing
scores, timestamps, receipt IDs, latency, or entire packets.

Cross-scope retrieval is an explicit context-query opt-in. An issued foreign
source remains scoped: `brain.source` must receive context matching that
source's project rather than inheriting authority from the handle alone.

Run the focused gate with:

```bash
PYTHONPATH=src .venv/bin/pytest -q tests/test_golden_context_v1.py
```
