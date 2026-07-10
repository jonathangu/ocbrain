# ocbrain

Current release: **v0.3.2**. Licensed under Apache-2.0.

Lightweight shared brain for ChatGPT/Codex, Claude Code, OpenClaw, and future
runtimes.
It is one local/on-prem source-backed ledger with scope as a first-class
dimension, not federated silos and not one undifferentiated memory pool.

`ocbrain` follows the final OpenClawBrain spec: immutable evidence goes in,
compiled current knowledge comes out. The brain itself never enqueues agent
work or pushes irreversible change. Separate local autopilot and stallcheck
processes compile, maintain, export, and observe the store on bounded schedules.

For a full architecture walkthrough — the two-plane store, the five-movement
pipeline, the 14 dispatched autopilot stages plus lock/finalize, the signal
taxonomy, the safeguards, the privacy model, and the dataset factory — read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Core Model

- `evidence`: immutable, append-only, hash-pinned claims about what happened.
- `knowledge`: current belief compiled from evidence, with `type`, lifecycle,
  privacy scope, and gate.
- `knowledge_evidence`: support/contradiction/derivation links.
- `memory`: a SQLite view over current injectable knowledge.

Knowledge types:

- `value`: facts and metrics.
- `doc`: readable wiki/procedure pages.
- `capability`: executable or loadable skills/procedures.

The bright line is readable versus executable or prescriptive. Capabilities,
high-risk knowledge, and prescriptive constraints pass automatic tripwires,
quarantine, provenance, and verifier checks before they can become injectable;
there is no human approval queue in the autonomous knowledge path.

The scope line is just as important. Default ingest uses the narrowest known
runtime/repo/task context. Global doctrine can surface everywhere, but promotion
from project-scoped fact to global knowledge is deliberate and gated.

## Status

The legacy `events`/`candidates`/review-queue model has been removed from the
active schema and CLI. Startup drops those old tables if they exist.

Current surfaces:

- SQLite ledger: `evidence`, `knowledge`, `knowledge_evidence`, `retrieval_uses`,
  `loop_liveness`, `family_scores`, `brain_events`, `current_beliefs`,
  `egress_audits`, and `memory`.
- CLI: `init`, `evidence`, `value`, `knowledge`, `import-memory`,
  `import-history`, `search`, `preview`, `event-ingest`, `event-compile`,
  `egress-preview`, `event-correct`, `event-forget`, `event-dream`,
  `event-proposals`, `event-decide`, `event-digest`, `event-teacher-request`,
  `event-backfill`, `digest`, `loop-ingest`, `mark-stale`, `prune`, `heal`,
  `liveness-check`, `mcp`, `autopilot`, `quarantine`, `label`, `dataset-mine`,
  `dataset-grade`, `dataset-export`, `dataset-stats`, `dataset-pilot-prepare`,
  `dataset-pilot-record-training`, `dataset-pilot-blind`,
  `dataset-pilot-score`, `public-safety-check`, and `install-hooks`.
- MCP: `brain.search`, `brain.preview`, `brain.egress_preview`, `brain.get`,
  `brain.teacher_request`, `brain.digest`, `brain.feedback`, write-gated
  `brain.ingest`, write-gated `brain.proposals`, write-gated `brain.forget`,
  and write-gated `brain.mark_stale`.
- Resources: `brain://digest/current`, `brain://wiki/{slug}`,
  `brain://loop/families`.

For agent runtime behavior, read
[`docs/AGENT_USE_GUIDE.md`](docs/AGENT_USE_GUIDE.md). For the current product and
engineering walkthrough, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
For a pickup guide for another agent, read
[`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md).

For the public install, upgrade, and model-driven verification path, use
[openclawbrain.ai/install](https://openclawbrain.ai/install/).

## Quick Start

```bash
uv run --with-editable . ocbrain init
uv run --with-editable . ocbrain evidence --claim "Codex emitted evidence."
uv run --with-editable . ocbrain value \
  --subject runtime:codex \
  --predicate shared_brain \
  --bool true \
  --status current \
  --inject
uv run --with-editable . ocbrain import-memory MEMORY.md memory/
uv run --with-editable . ocbrain --pretty digest
```

`import-memory` converts markdown memory files into final-spec immutable
evidence plus current `doc` knowledge, then indexes redacted document text so
`search`, `digest`, and MCP tools can return source-backed context.

To harvest local runtime transcript stores:

```bash
uv run --with-editable . ocbrain import-history \
  ~/.openclaw/agents ~/.openclaw/commitments ~/.openclaw/media/inbound \
  ~/.codex/sessions ~/.codex/archived_sessions \
  ~/.claude/projects ~/.claude/sessions ~/.claude/tasks
```

`import-history` catalogs every matched history file as evidence plus current
`doc` knowledge. It records a source path, file-size/mtime fingerprint, and a
bounded redacted head/tail text window so large transcript trees stay usable.
Repeated imports skip already-harvested source paths before reading excerpts.
The ChatGPT desktop app's Codex mode continues to write conversation rollouts
under `~/.codex/sessions`; `session_index.jsonl` and `history.jsonl` are treated
as bookkeeping rather than conversations. OpenClaw's current Codex/ACP agents
also keep isolated rollouts below
`~/.openclaw/agents/<agent>/agent/codex-home/sessions`; those are detected and
attributed to the producing Codex runtime even though the outer container is
OpenClaw. Current Codex `agent_message` records are retained only as injected
inter-agent context, never persona voice. Structured OpenClaw `isError` and
Claude Code `is_error` tool-result flags are honored in addition to text-based
error detection.

## Dataset grading and eval-first pilot

Dataset grading is deliberately local-only. The command rejects every
non-loopback endpoint before reading an example, applies stream-specific
rubrics, stores the normalized grade in `metadata.llm_grade`, and records a
bounded `dataset_grade_runs` ledger row. The default transport is Ollama's local
chat endpoint; provide the installed local model explicitly. A DB-adjacent file
lock prevents two graders from running together. Grades and run progress commit
per item, deterministic model/response failures stay skipped until `--force`,
and SQLite infrastructure failures remain retryable. If a scheduled writer is
active, grading returns `blocked` instead of claiming success.

```bash
uv run --with-editable . ocbrain dataset-persona-curate \
  --input data/curation/private-voice-examples.jsonl
uv run --with-editable . ocbrain dataset-grade \
  --model your-local-grader-model \
  --limit 100
uv run --with-editable . ocbrain dataset-export --min-grade 0.8
```

The first pilot is eval-before-train by construction. Preparation refuses to
write training files unless it can first reserve at least twenty graded persona
prompts, writes the prompts/references/rubric, and excludes every held-out
content hash from all training streams:

```bash
PILOT=data/datasets/pilot-v1
uv run --with-editable . ocbrain dataset-pilot-prepare \
  --min-grade 0.8 \
  --base-model /absolute/path/to/local-4bit-model \
  --base-model-source upstream/model-repo \
  --base-model-revision pinned-commit
uv run --with-editable . ocbrain dataset-pilot-record-training \
  --pilot-dir "${PILOT}" \
  --iterations 25 \
  --train-loss 2.218 \
  --validation-loss 4.592 \
  --exit-code 0
uv run --with-editable . ocbrain dataset-pilot-blind \
  --pilot-dir "${PILOT}" \
  --candidate-responses "${PILOT}/eval/candidate-responses.jsonl"
uv run --with-editable . ocbrain dataset-pilot-score \
  --pilot-dir "${PILOT}" \
  --ratings "${PILOT}/eval/blind-ratings.jsonl"
```

A later pilot can improve the training corpus without moving the evaluation
bar. `--eval-from` copies the earlier prompts, references, and rubric
byte-for-byte, verifies the held-out source hashes, and excludes those same
examples again:

```bash
uv run --with-editable . ocbrain dataset-pilot-prepare \
  --output-dir data/datasets/pilot-v2 \
  --eval-from data/datasets/pilot-v1 \
  --seed ocbrain-voice-pilot-v2-training \
  --training-iterations 50 \
  --min-grade 0.8 \
  --base-model /absolute/path/to/local-4bit-model \
  --base-model-source upstream/model-repo \
  --base-model-revision pinned-commit
```

After verified local training is recorded, generate the candidate side with the
same pinned MLX-LM environment used for training:

```bash
uvx --from 'mlx-lm[train] @ git+https://github.com/ml-explore/mlx-lm.git@PIN' \
  python scripts/generate-pilot-candidates.py \
  --pilot-dir "${PILOT}"
PYTHONPATH=src uv run python scripts/grade-pilot-blind.py \
  --pilot-dir "${PILOT}" \
  --calibration data/curation/private-judge-calibration.jsonl \
  --calibration-labels data/curation/private-human-labels.jsonl \
  --model your-local-grader-model
```

The rating helper refuses non-loopback endpoints, then requires at least six
calibration cases and 80% agreement with a separate, complete set of
human-provenance labels before it opens the blind pair file. Embedded
machine-authored winners are ignored. Use `--calibration-only` to test the judge
without reading blind pairs. The helper checkpoints each completed rating
atomically and never reads the separate A/B key. `dataset-pilot-score` is the
only step that unblinds results.

The dated July 10 human gate contains eight private operator labels. After the
judge prompt was aligned to those labels, it passed 7/8 (87.5%); the preserved
miss prefers a longer answer with reasons over the operator's concise answer
that needs one reason added. No blind rating was rerun after this calibration.

The dated v0.3.0 second-pilot result is intentionally modest. Against the exact
same 20-item blind set, the candidate improved from 2 preferences to 7; the
reference still won 13. The pipeline passed. The model-quality bar did not.

All generated corpus, reference, candidate, key, and rating files stay under
the gitignored local dataset tree. The pilot also writes deterministic MLX-LM
`train.jsonl`/optional `valid.jsonl` chat files from SFT + persona rows. DPO
stays as a separate preference artifact because MLX-LM LoRA is the SFT trainer
for this first pilot. The manifest records the exact pinned trainer argv; it
does not claim training started until that command actually runs.

Dataset mining also keeps the shared brain responsive. SFT, DPO, and persona
writes commit after at most 50 mutating units or two seconds, whichever comes
first, and each stage reports SQLite writer-lock wait, total hold time, and the
longest hold. Redaction, serialization, quality scoring, and dedup reads happen
before the writer transaction; each final example insert commits before the
next candidate is parsed or scored. After mining has committed, autopilot truncates a WAL larger than
64 MiB; a live reader produces an honest `busy` result for a later retry.
Autolabel releases the writer slot between source miners and before every
expensive FTS attribution query, then batches the fast label-fold updates under
the same measured bounds. Post-turn review now applies the same 50-operation or
two-second limit inside a session, then also commits the session watermark
before the lazy iterator parses the next transcript. Embedding
and the hosted judge commit their egress audits before network dispatch, then
store results in a second per-provider-batch transaction.
The passive stall watcher also commits findings before optional Telegram paging.
FTS attribution consumes the autolabel stage's remaining time budget.
Tripwire and promotion scans do their expensive reads outside write transactions.
The heavy harvester commits each imported file before reading the next one.

To use the scoped event-sourced core directly:

```bash
uv run --with-editable . ocbrain event-ingest \
  --body "Never weaken rules to clear red." \
  --global-doctrine
uv run --with-editable . ocbrain event-compile \
  --belief-id belief:red-rule \
  --body "Never weaken rules to clear red." \
  --evidence-id evd:red-rule \
  --global-doctrine \
  --confidence 0.9 \
  --approve
uv run --with-editable . ocbrain --pretty preview "rules red" --project bountiful
uv run --with-editable . ocbrain --pretty egress-preview \
  --target hosted_teacher \
  --project bountiful
uv run --with-editable . ocbrain event-correct \
  --target-layer belief \
  --target-id belief:red-rule \
  --op pin \
  --hard
uv run --with-editable . ocbrain event-forget \
  --target belief:red-rule \
  --mode soft \
  --reason "no longer serve"
uv run --with-editable . ocbrain --pretty event-dream \
  --project bountiful \
  --target local_model \
  --record-egress
uv run --with-editable . ocbrain --pretty event-proposals --project bountiful
uv run --with-editable . ocbrain --pretty event-proposals --project bountiful --approval-packet
uv run --with-editable . ocbrain event-decide \
  --proposal-event-id evt_... \
  --decision approve
uv run --with-editable . ocbrain --pretty event-digest --project bountiful
```

Unscoped event writes are quarantined as `legacy_unscoped` with
`egress_policy=local_only`; they are never silently promoted to global doctrine.
Compiled beliefs require at least one evidence id. Hard corrections are durable
constraints: once a hard `mark_wrong`, `retract`, or `demote` correction targets
a belief, the teacher path cannot re-derive that same belief id.
`event-dream` is a local deterministic consolidation pass that writes pending
compilation proposals only. It does not call a hosted model and does not approve
beliefs. `event-proposals` and `event-decide` are the CLI gate: decisions append
`compilation_decided` events and then rebuild the projection. Proposed
compilations carry teacher rationale plus a reward band (`discard`, `weak`,
`moderate`, or `strong`) rather than a fragile decimal reward.
Pass `event-proposals --approval-packet` to include a local, Telegram-ready gate
packet with `/ocbrain_gate ...` approval text, MCP `brain.feedback` arguments,
and exact `event-decide` argv actions. It never sends the packet; transport is
an outer OpenClaw concern.
`event-teacher-request` is the hosted-teacher bridge: it packages only
hosted-eligible, redacted scoped evidence plus the required JSON response schema,
records the egress audit, and returns `approval_required` without dispatching a
hosted call.
Scoped `preview` also returns a ranked `contradictions` list for visible belief
pairs that share claim terms but disagree through explicit negation; foreign
confidential scopes remain excluded from that ranking.
`event-backfill` migrates existing current legacy knowledge into the scoped event
core with deterministic scope classification. Use bounded slices while testing,
or `--all` for the remaining corpus after taking a DB backup. Large outputs are
sampled while preserving total counts.

```bash
uv run --with-editable . ocbrain event-backfill --project workspace --type doc --limit 25
uv run --with-editable . ocbrain event-backfill --all --sample-limit 25
```

`event-forget --mode shred` appends a cryptographic tombstone receipt and stops
serving the projected body/evidence ids for the belief. It does not destructively
rewrite the append-only ledger; destructive deletion remains an outer, explicitly
approved operational step.

## MCP

```bash
uv run --with-editable . ocbrain --db data/ocbrain.sqlite mcp
```

Installed launcher:

```bash
scripts/ocbrain-mcp
```

The launcher is portable: it resolves the repository from its own location,
uses `.venv/bin/python` when present, and falls back to `python3`. Override
`OCBRAIN_ROOT`, `OCBRAIN_DB`, or `OCBRAIN_PYTHON` only when the default local
layout is not the one you want.

Routine MCP is read-first. Write-capable tools are hidden unless the server is
launched with `--allow-writes`:

```bash
uv run --with-editable . ocbrain --db data/ocbrain.sqlite mcp --allow-writes
```

`brain.search` supports final-spec filters:

```json
{
  "query": "typecheck narrowing failures",
  "filters": {
    "loop_id": "repo-quality-loop",
    "family": "typecheck_narrowing",
    "project": "ocbrain"
  }
}
```

When `brain.search` receives a `context` object or `cross_scope=true`, it uses
the same scoped event-core retrieval path as `brain.preview`:

```json
{
  "query": "rules red Bountiful",
  "context": { "project": "bountiful" },
  "cross_scope": false
}
```

`brain.egress_preview` shows which event evidence would be included or rejected
before a local model, hosted teacher, or human export payload is assembled. Large
included/rejected sets are sampled and include total counts, so hosted-teacher
dry runs stay readable after full backfills.

With `--allow-writes`, the connector exposes `brain.ingest` for scoped evidence
appends, `brain.proposals` for gate review, and `brain.forget` for gated
tombstones. `brain.feedback` can append durable corrections with `layer`,
`target`, `op`, `body`, and `hard`, or append a compilation decision with
`proposal_event_id` and `decision`; it does not write beliefs directly.

`brain.digest` returns the legacy digest by default. Pass `event_core=true`, a
`context`, or `since` to include event-core counts, pending proposals, scoped
current beliefs, and a falsifiable quiet-loop surface. The event-core digest also
reports runtime health as the last useful ledger write per writer/session, not
transport availability.

`brain.search` and legacy `brain.get` rows return `retrieval_use_id` values. Call
`brain.feedback` with `helpful`, `used`, `irrelevant`, `ignored`, or `harmful`
to record usefulness. With `--allow-writes`, `brain.feedback` can also record a
decision on a legacy candidate row:

```json
{ "id": "know_...", "decision": "approve", "actor": "jon" }
```

During a long SQLite writer window, scoped retrieval remains available. If its
audit row cannot be written, the response reports
`retrieval_use_status=database_busy` with a null handle. Use the successful
search result and do not retry solely to manufacture feedback evidence.

## Loop Ingest

`ocbrain` observes loop result envelopes as evidence; it does not run the loop.

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --dry-run \
  --json
```

Apply mode writes loop-tagged evidence/knowledge rows and refreshes
`family_scores`:

```bash
brain-loop-ingest \
  --loop-id repo-quality-loop \
  --run-id 2026-06-23-nightly \
  --artifacts loops/artifacts/repo-quality-loop/2026-06-23-nightly \
  --apply \
  --json
```

Kept loop results require verifier evidence whose target hash matches the
changed artifact hash. Mismatches fail ingest and write tripwire evidence.
Failed loop results must include `failure_class` as `approach`, `precondition`,
`infra`, `safety`, or `unknown`. Only `approach` failures count toward an
`exhausted` family; `precondition` and `infra` failures mark the family
`blocked` and stage repair context instead of suppressing the family.
Forced-exploration envelopes can set `forced_exploration=true`; ingest records
whether those attempts found improvement.

## Maintenance

Maintenance commands are designed for OpenClaw cron/heartbeat lanes, but no cron
is installed by this repo.

```bash
uv run --with-editable . ocbrain prune \
  --ttl-days 30 \
  --unhelpful-ttl-days 14 \
  --archive-stale-days 90
uv run --with-editable . ocbrain heal --numeric-threshold 0.01
uv run --with-editable . ocbrain liveness-check --runner-ledger loops/runner.sqlite
```

`prune` marks unreferenced expired knowledge `stale`, decays served-but-never
useful knowledge on a shorter TTL, and can later archive stale rows without
deleting the audit trail. `heal` supersedes conflicting current values and
writes correction evidence. `liveness-check` reads runner deadman rows and
writes loop tripwire evidence such as `heartbeat_starved` or
`no_ledger_writes`; it does not claim lanes or enqueue loop work.

`ocbrain.stallcheck` is a separate, passive companion process on its own
15-minute launchd timer, independent of the autopilot lock. It reads agent
transcripts and runner tables (read-only) for a parked-and-forgotten turn — an
`end_turn` left with a still-pending monitor, or a task output file opened but
never closed — writes the finding as `loop_liveness` + tripwire evidence in
the same ledger the liveness sweep reads, and can send a single deduplicated
Telegram digest of new stalls when a local pager config is present. It also
writes its own heartbeat row so a weekly review would notice if the watchdog
itself stopped running. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for detail.

## Public safety

This is a **public** repository. The default is enforced, not just intended:
code, tests, and docs are public; runtime data is not. Nothing under `data/` or
`logs/`, no `*.sqlite` / `*.jsonl` dataset artifact, and no private memories,
transcripts, or local config may ever be tracked. `data/` is fully gitignored.

A tracked-tree scanner enforces this. It reads git only (never the runtime
brain DB) and checks the outgoing commit range for four violation classes:

- **placement** — no tracked file under `data/` or `logs/`, no dataset artifact.
- **denylist** — no hits against a local, gitignored list of private
  identifiers (`data/public-safety-denylist.txt`). The list is never committed;
  if it is absent the scanner warns and continues with built-in secret patterns.
  Findings report counts and locations only — never the matched value.
- **new secrets** — no high-entropy tokens or known secret patterns in
  newly-added diff lines (reuses the shared `text.py` scanners).
- **private paths** — no absolute `/Users/` paths that reveal a repo outside a
  small allowlist (this repo plus orchestrator infra).

```bash
# scan the whole tracked tree
uv run --with-editable . ocbrain public-safety-check
# scan an outgoing range (what the pre-push hook runs)
uv run --with-editable . ocbrain public-safety-check --diff-range origin/main..HEAD
# install the pre-push hook (symlinks ops/hooks/pre-push into .git/hooks)
uv run --with-editable . ocbrain install-hooks
```

The `pre-push` hook (tracked at `ops/hooks/`) runs the check on every push and
blocks it on any finding; a human can override with `git push --no-verify`.

## Verification

```bash
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . pytest -q
PYTHONPATH=. uv run --with pytest --with ruff --with-editable . ruff check .
uv run --with-editable . python -m compileall src tests
```

## Principles

- Evidence before belief.
- Verified is not claimed.
- Memory is a view, not a store.
- Supersede/archive; do not overwrite in place.
- A derived object's privacy scope is the most restrictive linked source scope.
- Automatic safeguards before executable or prescriptive knowledge can serve.
- Emit evidence; do not write durable knowledge directly from runtimes.
- Watch loops closely enough to tell done from wedged, without running them.

## License

Apache License 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
