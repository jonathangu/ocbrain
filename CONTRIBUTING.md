# Contributing to OCBrain

Thanks for helping make local agent memory more useful and more accountable.
OCBrain welcomes focused bug fixes, documentation improvements, new MCP-client
setup proofs, retrieval and scope tests, and small proposals that preserve the
core's safety boundaries.

The canonical repository is
[jonathangu/ocbrain](https://github.com/jonathangu/ocbrain). The public
operator guide is [openclawbrain.ai](https://openclawbrain.ai/).

## Before you start

- Search the [issues](https://github.com/jonathangu/ocbrain/issues) and existing
  pull requests before opening a duplicate.
- For a large behavior or schema change, open an issue first so the contract and
  migration implications can be discussed before implementation.
- Never include a live OCBrain database, transcript corpus, access token,
  personal identifier, owner-specific configuration, or generated model
  artifact in an issue, fixture, commit, or pull request.

## Development setup

OCBrain currently supports Python 3.11+ on macOS and Linux. WSL is expected to
work but is not part of the dated release-acceptance proof. Native Windows is
not currently supported. The core has no third-party runtime dependencies.

```bash
git clone https://github.com/YOUR-USER/ocbrain.git
cd ocbrain
git remote add upstream https://github.com/jonathangu/ocbrain.git

python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pip install -e ./packages/ops
```

Create a focused branch:

```bash
git switch -c fix/short-description
```

## Make the change

- Keep the core on demand. Do not add a scheduler, timer, watchdog, hosted
  judgment call, training run, or background loop to the default install.
- Treat the event ledger, scope, provenance, corrections, retrieval receipts,
  source handles, and closeouts as durable; indexes and rankings are derived.
- Keep the default MCP surface narrow. New mutation or egress authority needs an
  explicit contract and safety review.
- Add or update tests for behavior changes.
- Update README, public docs, the changelog, and release notes when the user
  contract changes.
- Prefer the smallest change that proves the intended behavior.

## Run the local gate

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_golden_context_v1.py
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/python -m compileall -q src tests
git diff --check
```

The focused golden gate drives the real MCP `brain.context` and `brain.source`
path through deterministic synthetic relevance, scope, hosted-delivery,
contradiction, hash-verification, and denial cases. Golden fixtures must remain
public synthetic test data and must never be populated from a live brain or
used as training data.

The optional repository hook runs the public-safety scanner before a push:

```bash
scripts/install-hooks.sh
```

Installing the hook changes only this checkout. You can also run the same check
directly:

```bash
.venv/bin/ocbrain public-safety-check --root "$PWD"
```

If your change affects MCP startup or a client integration, also initialize a
fresh temporary database, launch `scripts/ocbrain-mcp`, and complete
`context → source → feedback → closeout` from a fresh client process. State
exactly which clients you tested; standard-protocol compatibility is not the
same as a dated client acceptance result.

## Open the pull request

Push your branch and open a pull request against `main`. Include:

- the problem and why the change is needed;
- the smallest useful summary of the implementation;
- privacy, scope, migration, and compatibility impact;
- commands and client flows used to verify it;
- documentation or release-note changes;
- any remaining uncertainty.

A maintainer may ask for a smaller patch, a migration plan, stronger verifier
evidence, or a fresh-client MCP round trip before merging.

## Security reports

Do not put a suspected vulnerability or private data exposure in a public issue.
Use the repository's
[private vulnerability reporting form](https://github.com/jonathangu/ocbrain/security/advisories/new).
Include a minimal reproduction with synthetic data and avoid attaching a real
brain database.
