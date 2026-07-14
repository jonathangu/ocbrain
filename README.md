# OCBrain

OCBrain is the local, source-backed context bridge shared by Codex, Claude Code,
OpenClaw, and compatible MCP clients. It retrieves a bounded dossier from an
effectively unbounded private history, expands exact sources on demand, records
whether the context mattered, and links the eventual outcome back to what the
agent saw.

Current core version: **v1.0.1**. License: Apache-2.0.

[Install](#quick-start) · [Connect a client](#connect-the-clients-you-use) ·
[Agent instructions](docs/RUNTIME_INTEGRATION.md#client-instruction-block) ·
[Contribute](CONTRIBUTING.md) · [Public guide](https://openclawbrain.ai/install/)

## What you need

**OpenClaw is optional.** OCBrain is a local stdio
[Model Context Protocol](https://modelcontextprotocol.io/) server. You can use
it with Codex, Claude Code, OpenClaw, or another compatible MCP client; install
and configure only the clients you actually use.

| Requirement | Current support |
|---|---|
| Python | 3.11 or newer |
| Operating system | macOS or Linux; WSL is expected to work but is not release-accepted |
| Source install | Git plus Python's built-in `venv` and `pip` |
| Agent client | At least one local stdio MCP client |
| Not required | OpenClaw, an API key, a hosted service, or a vector database |

The core declares no third-party runtime dependencies and stores its ledger in
local SQLite. The repository launcher is a Bash script, and parts of the
current file-locking implementation are POSIX-specific, so native Windows is
not currently supported. WSL has not yet been included in the dated acceptance
proof.

## Quick start

Clone the canonical repository and create a repository-local environment:

```bash
git clone https://github.com/jonathangu/ocbrain.git
cd ocbrain

python3 --version  # must be 3.11+
python3 -m venv .venv
.venv/bin/python -m pip install -e .

.venv/bin/ocbrain --version
.venv/bin/ocbrain --db data/ocbrain.sqlite init
.venv/bin/ocbrain --db data/ocbrain.sqlite status
.venv/bin/ocbrain --db data/ocbrain.sqlite doctor \
  --launcher scripts/ocbrain-mcp
```

This creates a new local brain. It does not import another person's history,
start a background process, or send anything to a hosted service. Runtime data
under `data/` is ignored by Git, and the database file is restricted to its
owner. The SQLite database is plaintext rather than encrypted at rest; use
full-disk encryption when the host or backup threat model requires it.

### A fresh brain starts empty

An empty `brain.context` result immediately after installation is honest and
expected. `brain.ingest` appends scoped evidence; it does not promote that
evidence directly into a durable serving belief.

To add a small set of reviewed starter facts, copy the synthetic
`examples/curated-memory` example, replace its source and facts, update the
source SHA-256, review the manifest, and apply it explicitly:

```bash
.venv/bin/ocbrain --db data/ocbrain.sqlite curated-apply \
  examples/curated-memory/manifest.json \
  --actor "human-curated:YOUR-NAME"
```

The command verifies every named source hash and appends evidence, proposal,
and approval events; it never writes a belief projection directly. Existing
v0.x users should follow the archive-first migration path instead.

The product is the evidence and outcome ledger, not a particular embedding
model, vector database, prompt, or training pipeline. Search indexes,
embeddings, rankings, summaries, and model-specific features are derived and
replaceable. Raw events, scope, provenance, corrections, retrieval receipts,
source handles, and closeouts remain durable.

## The runtime loop

```text
unbounded local evidence lake
          │
          ▼
scope-safe retrieval ──► ocbrain.context.v1
                              │
                              ├─► brain.source (bounded, hash-verified expansion)
                              │
                              ▼
                      agent performs work
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
            brain.feedback          brain.closeout
                                      │
                                      └─ actions, outcomes,
                                         artifacts, verifiers
```

This is retrieval plus bounded long context, not a choice between them.
Retrieval filters the data lake; the context packet supports reasoning across a
small coherent set; source expansion supplies the full few documents that
matter.

## Safety corrections

- The core is on-demand. It installs no light autopilot, heavy autopilot,
  stallcheck, timer, pager, or recurring maintenance job.
- Hosted judging, embedding, and teacher-package work is absent from the core
  MCP and disabled in the preserved v0.4.1 compatibility configuration.
- Training is paused. Local mining and audit tooling live in an optional
  companion, but no trainer is authorized until a genuine named-human audit and
  a separate operator decision both exist.
- `--allow-writes` is not a no-op. It is a deprecated alias for the explicit
  `--profile admin` MCP surface.
- Migration writes only fresh paths. It never replaces or repoints the live
  database automatically.

## One core, optional companions

| Distribution | Command | Default database | Responsibility |
|---|---|---|---|
| `ocbrain` | `ocbrain` | `~/.ocbrain/ocbrain.sqlite` | event ledger, projections, retrieval, MCP, receipts, backup/migration |
| `ocbrain-training` | `ocbrain-training` | `~/.ocbrain/training.sqlite` | optional local curation, grading, audit, and prepared training workflows |
| `ocbrain-ops` | `ocbrain-ops`, `ocbrain-watchdog` | `~/.ocbrain/ops.sqlite` | optional manual diagnostics and legacy operations |

The companion databases are not additional brains. The default MCP never
queries them. Legacy companion mutators require an explicit `--legacy-db`; they
never silently write the v1 core.

Install companions only when deliberately needed:

```bash
.venv/bin/python -m pip install -e ./packages/training
.venv/bin/python -m pip install -e ./packages/ops
```

For development:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

## MCP profiles

The default runtime profile has eight tools:

- `brain.context` — stable `ocbrain.context.v1` packet with coverage,
  exclusions, contradictions, and source handles;
- `brain.source` — bounded expansion of an issued handle with scope and content
  hash verification;
- `brain.search`, `brain.digest`, and `brain.get` — compact scoped lookup
  helpers;
- `brain.feedback` — retrieval usefulness only;
- `brain.ingest` — narrowly scoped evidence, never direct belief promotion;
- `brain.closeout` — append-only `ocbrain.closeout.v1` outcome receipt.

The admin profile adds preview, egress preview, durable correction, proposal
decision/listing, and tombstone controls. It does **not** add a hosted teacher,
training, scheduler, or stale-marking tool.

Run either profile explicitly:

```bash
ocbrain mcp --profile runtime
ocbrain mcp --profile admin
```

## Connect the clients you use

Resolve the launcher once from the repository root:

```bash
LAUNCHER="$PWD/scripts/ocbrain-mcp"
```

Register it with any installed clients.

### Codex

```bash
codex mcp add ocbrain -- "$LAUNCHER"
codex mcp get ocbrain
```

The ChatGPT desktop app, Codex CLI, and Codex IDE extension share the same
local Codex MCP configuration.

### Claude Code

```bash
claude mcp add --scope user ocbrain -- "$LAUNCHER"
claude mcp get ocbrain
```

### OpenClaw (optional)

If you also use OpenClaw:

```bash
openclaw mcp add ocbrain --command "$LAUNCHER"
openclaw mcp doctor ocbrain
openclaw mcp probe ocbrain
```

Registration is configuration, not acceptance. A fresh chat alone does not
activate OCBrain unless the client has the MCP server configured and the agent
is instructed to use it. A real acceptance turn in every configured client
should complete:

```text
brain.context → brain.source → brain.feedback → brain.closeout
```

Already-open chats may retain the MCP process they started before an upgrade.
Start a fresh task or restart/reconnect the client when testing a new core.
Copy the short policy from the
[runtime integration guide](docs/RUNTIME_INTEGRATION.md#client-instruction-block)
into `AGENTS.md`, `CLAUDE.md`, or the equivalent durable
instruction surface.

The July 13 v1 cutover passed this real gate against one activated core. Fresh
Codex, Claude Code, and OpenClaw processes each returned an
`ocbrain.context.v1` packet, expanded the same hash-verified source, recorded
feedback, and wrote a verified `ocbrain.closeout.v1` receipt. The owner-only
acceptance report remains beside the live database; the activation pointer is
retained.

## Fresh v1 databases and migration

`ocbrain init` creates the strict event-authoritative v1 schema on a fresh
database. An existing v0.x database remains compatibility-only until an
explicit archive-first migration.

Plan without writing outputs:

```bash
ocbrain --db /absolute/legacy.sqlite core-migrate-v1 \
  --core-db /absolute/v1/core.sqlite \
  --archive-db /absolute/archive/legacy.sqlite \
  --training-db /absolute/v1/training.sqlite \
  --ops-db /absolute/v1/ops.sqlite \
  --manifest /absolute/v1/migration.json \
  --plan
```

Run the same command without `--plan` to create fresh verified outputs. The
manifest accounts for preserved event-chain rows, imported semantic objects,
retrievals, companion rows, hashes, integrity checks, and anything intentionally
archive-only.

Activation is separate. `scripts/ocbrain-mcp` uses `OCBRAIN_DB` when set;
otherwise it reads the ignored local `data/active-core.path` when present, then
falls back to `data/ocbrain.sqlite`. The pointer must contain one absolute path.
The migration command never writes it.

## Explicit cross-machine evidence bundles

Bundle exchange is a manual file operation, never network sync or an MCP tool.
Export requires explicit evidence ids and applies the current scope, egress,
approval, size, and secret-redaction gates before publishing a fresh owner-only
file:

```bash
ocbrain --db /absolute/core.sqlite export-bundle \
  --evidence-id evd_example \
  --project source-project \
  --output /absolute/fresh.bundle.json
```

Evidence marked `local_only` or `prohibited` cannot be exported.
`approval_required` evidence additionally needs `--approve-egress`. Import is a
validation-only dry run unless `--apply` is supplied:

```bash
ocbrain import-bundle /absolute/fresh.bundle.json --project destination-project
ocbrain --db /absolute/core.sqlite import-bundle \
  /absolute/fresh.bundle.json --project destination-project --apply
```

Import ignores sender ids, derives local content ids, and appends evidence only.
Imported evidence is always `confidential` and `local_only` with explicit bundle
provenance; beliefs, retrieval receipts, and closeouts are never imported.

## What v1 stores

`brain_events` is the semantic authority. Evidence objects, current beliefs,
aliases, evidence links, and FTS are deterministic projections. Retrieval uses,
source-handle issuance, egress audits, and closeouts are append-only operational
receipts.

`ocbrain.closeout.v1` can retain two optional portable envelopes:

- `ocbrain.action.v1`: mechanism, local semantic role, target, pre-action
  context, policy/model, cost, provenance, and versioned features;
- `ocbrain.outcome.v1`: metric/value, role, unit, observation window, baseline,
  counterfactual, attribution, uncertainty, local interpretation, and versioned
  features.

This keeps a click or subscription meaningful within its own site and task.
One experiment may derive a scalar reward, but the ledger does not destroy the
components future models need for safer transfer.

## Training boundary

The earlier 150-item Opus fleet review is useful remediation evidence, but it is
not a named-human audit. It found 67 pass / 83 fail, so pilot-v3 remains blocked.
The main defects were persona sender envelopes, process chatter, routing tokens
and identifiers, and weak DPO contrasts.

The clean private handoff packet is generated separately and contains 150
pending decisions with no AI labels to anchor the reviewer. Completing it still
does not authorize training; remediation, reminting, local grading, a fresh
stratified audit, and explicit operator closeout remain separate gates.

## Documentation

- [Shared Context and v1 contract](docs/SHARED_CONTEXT_V1.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Agent use guide](docs/AGENT_USE_GUIDE.md)
- [Runtime integration](docs/RUNTIME_INTEGRATION.md)
- [Core operations and migration](docs/CORE_OPERATIONS.md)
- [Execution ledger](docs/CORE_ROADMAP_EXECUTION.md)
- [Release history](docs/releases/)

Files labeled historical preserve old decisions and evidence; they are not
current operating doctrine.

## Contributing

Bug reports, focused fixes, documentation improvements, new client setup
proofs, and scope/privacy tests are welcome. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), run the local test and lint gate, and open a
pull request against `main`. Never attach a live brain database, transcript
corpus, secret, or owner-specific runtime artifact to an issue or commit.
