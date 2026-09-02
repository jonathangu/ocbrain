# OCBrain

OCBrain is the local, source-backed context bridge shared by Codex, Claude Code,
Hermes, OpenClaw, and compatible MCP clients. It retrieves a bounded dossier from an
effectively unbounded private history, expands exact sources on demand, records
whether the context mattered, and links the eventual outcome back to what the
agent saw.

Current core version: **v1.1.0**. License: Apache-2.0.

The latest packaged release remains v1.1.0. Current `main` contains the newer
thirteen-tool briefing/ledger/goals/supersession contract documented below; do
not expect that unreleased surface from the older release artifact.

[Install](#quick-start) · [Connect a client](#connect-the-clients-you-use) ·
[Agent instructions](docs/RUNTIME_INTEGRATION.md#client-instruction-block) ·
[Contribute](CONTRIBUTING.md) · [Public guide](https://openclawbrain.ai/install/)

## What you need

**OpenClaw is optional.** OCBrain is a local stdio
[Model Context Protocol](https://modelcontextprotocol.io/) server. You can use
it with Codex, Claude Code, Hermes, OpenClaw, or another compatible MCP client; install
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

To prove a non-empty hosted `context -> source` round trip, first review the
four public facts in `examples/hosted-context-demo`, then apply them explicitly:

```bash
.venv/bin/ocbrain --db data/ocbrain.sqlite curated-apply \
  examples/hosted-context-demo/manifest.json \
  --allow-hosted-egress \
  --actor "human-curated:YOUR-NAME"
```

The acknowledgement is required because those exact fact bodies may be sent
to a hosted model. It does not authorize the database, full source file, or a
local path to leave the machine. Start a fresh client and query for `OCBrain
installation requirements and client constraints` with `project=ocbrain`; an
issued source should expand with `hash_verified=true`.

To add your own reviewed starter facts, copy the local-only synthetic
`examples/curated-memory` example, replace its source and facts, update the
source SHA-256, review the manifest, and apply it explicitly:

```bash
.venv/bin/ocbrain --db data/ocbrain.sqlite curated-apply \
  examples/curated-memory/manifest.json \
  --actor "human-curated:YOUR-NAME"
```

The command verifies every named source hash and appends evidence, proposal,
and approval events atomically after validating the entire manifest; it never
writes a belief projection directly. Existing
v0.x users should follow the archive-first migration path instead. A manifest
containing `hosted_ok` facts always requires `--allow-hosted-egress` and cannot
combine that policy with confidential or secret visibility.

The product is the evidence and outcome ledger, not a particular embedding
model, vector database, prompt, or training pipeline. Search indexes,
embeddings, rankings, summaries, and model-specific features are derived and
replaceable. Raw events, scope, provenance, corrections, retrieval receipts,
source handles, and closeouts remain durable.

### Optional sparse wiki compiler

`scripts/wiki-curator.py` can compile reviewed, high-signal evidence into
concise `wiki_fact` beliefs and a human-readable wiki. It never selects raw
transcript evidence, sends only bounded already-redacted evidence, rejects any
claim whose supporting quote cannot be found verbatim in its named evidence,
and makes no hosted call without explicit `--apply`. Project, visibility, and
egress gates apply to both evidence and existing wiki facts before prompt
construction. Local-only, prohibited, confidential, and secret objects are
never eligible; `--allow-hosted-egress` may add only bounded
`approval_required` objects.

Providers are pluggable — `anthropic` (default), `openai`, `moonshot` — and the
same gates and quote validation apply whichever model runs. The Anthropic
backend needs the optional extra: `pip install -e '.[curator]'`. Only the API
key's variable *name* is configured; the value is never persisted.

`--project` is repeatable and `--projects-from-config` curates every scope in
`curator.projects`. Each project is gated on its own digest inside one run, so a
scope whose evidence has not changed makes no call, and a scope with fewer than
`curator.min_evidence_per_project` eligible objects is reported as skipped rather
than billed. Do not loop the script over `--project` from a shell: `state.json`
holds one digest per project in a single file, so each invocation would discard
the previous project's cursor and every cycle would re-bill every project.

```bash
# Keep background harvesting in the evidence ledger, not current truth.
ocbrain --db /absolute/core.sqlite import-history /history/root \
  --project my-project --privacy-scope workspace --evidence-only

# Preview locally (no network call), then explicitly authorize one compilation.
.venv/bin/python scripts/wiki-curator.py \
  --db /absolute/core.sqlite --max-beliefs 12 --allow-hosted-egress
.venv/bin/python scripts/wiki-curator.py \
  --db /absolute/core.sqlite --max-beliefs 12 --allow-hosted-egress --apply

# Curate every configured project scope in one run.
.venv/bin/python scripts/wiki-curator.py \
  --db /absolute/core.sqlite --projects-from-config --apply
```

`scripts/kimi-wiki-curator.py` remains as a shim forwarding to
`--provider moonshot`.

The generated `wiki/index.md`, `wiki/pages/`, and append-only `wiki/log.md`
follow the raw-sources-plus-derived-wiki pattern. SQLite remains authoritative;
the Markdown wiki is a disposable current-truth materialization.

There is no automatic promotion path to disable. An `automatic_activation` flag
once compiled closeout summaries straight into serving beliefs; it produced 239
`auto_compiled` beliefs, all 239 were later retracted, and the flag has read
false ever since. It and the flag's CLI subcommand are gone. Closeout summaries
are still always recorded as evidence — that half was never the gated half —
and promotion into a serving belief remains a curation step a human or the
curator performs deliberately.

### Optional sealed-truth compiler

`scripts/compile-sealed-truth` validates an immutable Agent Control release and
its canonical closeout before compiling one sparse local-only wiki fact. A
closeout must declare `verification_status: verified` and include at least one
passed verifier reference; a legacy `verified: true` boolean is not sufficient.
The command is preview-only unless `--apply` is explicit:

```bash
# Verify hashes, closeout evidence, scope, and the proposed mutation.
scripts/compile-sealed-truth --seal /private/release/SEAL.json

# Apply only after reviewing the preview.
scripts/compile-sealed-truth --seal /private/release/SEAL.json --apply
```

Wiki pages carry lightweight freshness frontmatter (`valid_from`, plus
optional `valid_until` / `superseded_by` from belief attributes); `index.md`
renders a `**[stale: ...]**` marker for expired or superseded pages, and
`scripts/wiki-lint.py` flags expired/superseded pages, pages the ledger no
longer serves as current, pages older than the ledger's latest compilation,
and conflicting pages that share a key:

```bash
.venv/bin/python scripts/wiki-lint.py /path/to/wiki --db /absolute/core.sqlite

# Repair drifted pages and orphans instead of only reporting them.
.venv/bin/python scripts/wiki-lint.py /path/to/wiki --db /absolute/core.sqlite \
  --rematerialize
```

### Retiring knowledge

Compilation only grows the corpus. `ocbrain hygiene` retires beliefs that
expired (`valid_until` / `superseded_by`) and beliefs that restate a fact a
newer one already carries in the same scope. It reports by default and retires
only with `--apply`; every retirement is a soft retraction, undoable with
`--restore`.

```bash
ocbrain --db /absolute/core.sqlite hygiene                      # report
ocbrain --db /absolute/core.sqlite hygiene --class expired --apply
ocbrain --db /absolute/core.sqlite hygiene --restore belief_... # undo
```

Two further classes, `unused` and `unhelpful`, were removed in v2: neither ever
selected a belief across 155 scheduled runs, and `unhelpful` needed a feedback
watermark no operator ever set.

### Keeping knowledge well-written

Hygiene retires beliefs that stopped earning their place. A separate set of
mechanical rules catches the other failure: a belief that is well-formed,
sourced, schema-valid, and still fails to function as knowledge — several facts
fused into one body, a durable claim written in the present tense, a
present-state claim that can never age out.

Those rules are a **write-time gate**, not a cleanup pass. The curator rejects a
slop claim before it becomes a belief, a closeout receipt reports findings back
to whoever wrote it, and `scripts/wiki-lint.py` checks the materialized tree. A
post-hoc sweep also existed and found nothing in 155 consecutive runs, because
the gate had already done the work; it is gone. See
[docs/DESLOP.md](docs/DESLOP.md).

### Running it continuously

The core installs no scheduler. `scripts/brain-sync.sh` (harvest) and
`scripts/brain-promote.sh` (promote and retire) exist for operators who want
continuous operation — see [docs/SCHEDULED_MAINTENANCE.md](docs/SCHEDULED_MAINTENANCE.md).
Running only the harvester is the trap: it records evidence but promotes nothing,
so the serving corpus silently freezes.

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

- The core is on-demand. It installs no autopilot, timer, pager, or recurring
  maintenance job.
- Hosted judging, embedding, and teacher-package work is not in the core at all.
  The config sections that configured them are gone along with the code.
- `--allow-writes` is not a no-op. It is a deprecated alias for the explicit
  `--profile admin` MCP surface.
- Migration writes only fresh paths. It never replaces or repoints the live
  database automatically.

## One distribution

| Distribution | Commands | Default database | Responsibility |
|---|---|---|---|
| `ocbrain` | `ocbrain`, `ocbrain-closeout` | `~/.ocbrain/ocbrain.sqlite` | event ledger, projections, retrieval, MCP, receipts, backup/migration, public-safety scanning |

`pip install -e .` installs everything there is. Two optional companion
distributions, `ocbrain-training` and `ocbrain-ops`, were published alongside
the core and are gone: every table they owned in their own stores —
`autopilot_runs`, `judge_runs`, `embed_runs`, `signal_events`,
`harvest_watermarks`, `loop_liveness`, `family_scores`, `stall_pages`,
`watchdog_findings` — held zero rows. (Those stores also declared their own
`egress_audits`, empty like the rest. It is not the core table of the same
name: `src/ocbrain/egress.py` writes that one on every applied curation run,
and it is untouched.) There is no `pip install -e ./packages/...` step, no
`ocbrain-ops` / `ocbrain-training` / `ocbrain-watchdog` command, and no
entry-point mechanism for a companion to extend the CLI.

The one piece worth keeping moved into the core rather than dying with the
package: the public-repository safety scanner is now `src/ocbrain/publicsafety.py`,
and `ocbrain public-safety-check` and `ocbrain install-hooks` are ordinary
subcommands that need nothing installed beyond the core.

For development:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests/test_golden_context_v1.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The focused [golden Shared Context dataset](tests/fixtures/README.md) uses only
public synthetic facts. It tests the real MCP context/source contract,
including hosted filtering, scope isolation, contradictions, and source hash
verification; it is not training data.

## MCP profiles

The default runtime profile has thirteen tools:

- `brain.briefing` — deterministic, bounded fresh-context orientation;
- `brain.ledger` — stable task-ref projection of verified, failed, and in-flight
  work;
- `brain.goal_open` / `brain.goal_close` — repository-spec pointers with
  executable finish lines and verifier-backed lifecycle events;
- `brain.context` — stable `ocbrain.context.v1` packet with coverage,
  exclusions, contradictions, and source handles;
- `brain.source` — bounded expansion of an issued handle with scope and content
  hash verification;
- `brain.search`, `brain.digest`, and `brain.get` — compact scoped lookup
  helpers;
- `brain.feedback` — retrieval usefulness only;
- `brain.ingest` — narrowly scoped evidence, never direct belief promotion;
- `brain.closeout` — append-only `ocbrain.closeout.v1` outcome receipt;
- `brain.supersede` — replace one serving belief with a corrected one, in its
  own scope, atomically. The only runtime write that changes what is served,
  and the only correction shape that does not subtract from the corpus.

The admin profile adds six more: preview, egress preview, durable correction,
proposal decision, proposal listing, and the tombstone control. Nineteen tools
is the whole surface. There is no hosted teacher, training, or scheduler tool,
and `brain.mark_stale` is gone — it was published for two years and could never
be dispatched, because `tools_for_profile` returned it for no profile.

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

### Hermes

```bash
hermes mcp add ocbrain --command "$LAUNCHER"
hermes mcp test ocbrain
```

Hermes gateways may multiplex sessions over one MCP child. When Hermes cannot
provide its real runtime session id, omit `context.session`; OCBrain records the
server connection identity instead. Never invent a session label, because the
identity is what joins closeouts to tool-call traces.

### OpenClaw (optional)

If you also use OpenClaw:

```bash
openclaw mcp add ocbrain --command "$LAUNCHER"
openclaw mcp doctor ocbrain
openclaw mcp probe ocbrain
```

Registration is configuration, not acceptance. A fresh chat alone does not
activate OCBrain unless the client has the MCP server configured and the agent
is instructed to use it. A real acceptance turn in every configured client on
current `main` should complete:

```text
brain.briefing → brain.ledger → brain.context → brain.source
               → brain.feedback → brain.closeout
```

`brain.source` is conditional on the context packet issuing a handle. Pass the
runtime's real session id, or omit the field rather than inventing one. A saved
configuration or successful transport probe is not a model-driven acceptance.

Already-open chats may retain the MCP process they started before an upgrade.
Start a fresh task or restart/reconnect the client when testing a new core.
Never terminate an individual client-owned MCP child to force an upgrade:
stdio hosts can retain the dead transport without reconnecting it.

If a runtime call returns `Transport closed`, do not retry the dead connection.
Save the exact tool arguments in a private JSON file and execute that normal
runtime call once:

```bash
scripts/ocbrain-runtime-call brain.closeout \
  --arguments-file /private/path/closeout-arguments.json
```

This one-shot path uses the same runtime dispatcher and runtime-only tool
allowlist, then exits. It is not another server and cannot invoke admin tools.
Reconnect or start a fresh client task afterward so later calls use a new MCP
transport.

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
retrievals, hashes, integrity checks, and anything intentionally archive-only.

`--training-db` and `--ops-db` are still accepted, and both are about the *legacy
source*: a v0.x database carries training and operational tables that have no
home in the strict v1 inventory, so migration extracts them to their own files
rather than dropping them or smuggling them into the core. They do not install
anything and there is nothing left that reads them.

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

There is no training code in this repository. The dataset mining, grading, and
pilot-packet tooling lived in the `ocbrain-training` companion and was deleted
with it; the `dataset-*` and `pilot-*` subcommands, `scripts/generate-pilot-candidates.py`,
and `scripts/grade-pilot-blind.py` are gone.

It never shipped a model. The one prepared pack, pilot-v3, was blocked by a
150-item review that returned 67 pass / 83 fail — persona sender envelopes,
process chatter, routing tokens, and weak DPO contrasts — and an AI review was
never the named-human audit the gate required anyway. Keeping a blocked pipeline
installed so it could stay blocked was the part that made no sense.

## Documentation

- [Shared Context and v1 contract](docs/SHARED_CONTEXT_V1.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Agent use guide](docs/AGENT_USE_GUIDE.md)
- [Runtime integration](docs/RUNTIME_INTEGRATION.md)
- [Core operations and migration](docs/CORE_OPERATIONS.md)
- [Deslop: knowledge quality](docs/DESLOP.md)
- [Code quality](docs/CODE_QUALITY.md)
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
