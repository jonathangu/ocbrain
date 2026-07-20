# Changelog

## Unreleased

- Keep stdio MCP transports alive by default instead of imposing a two-hour
  launcher idle exit. Hosts that do not reconnect treated that intentional
  exit as `Transport closed`; orphan cleanup remains available through an
  explicit positive `OCBRAIN_MCP_IDLE_TIMEOUT_SECONDS` value.
- Default the local stdio MCP server to `local_model` delivery so local coding
  agents retrieve their own memory at full fidelity, and add a
  `--delivery-target` flag plus `OCBRAIN_DELIVERY_TARGET` env to select
  `hosted_model` (egress-filtered) delivery when feeding a hosted teacher.
  Restores the pre-1.1.0 local default while keeping hosted delivery explicit.
- Add opt-in unattended promotion (`automatic_activation`, off by default,
  toggled with `ocbrain automatic-activation --enable/--disable`). When enabled,
  `brain.ingest` and `brain.closeout` auto-compile evidence and closeout
  summaries into served beliefs with no human review, so continuity accrues
  automatically. Promotion is idempotent, scopes to the shared project so any
  client on it can recall the belief, and never widens egress beyond
  `local_only`. Off, promotion stays human-gated exactly as before.
- Reconcile the published v1 MCP tool schemas with the dispatcher so every
  advertised property is callable. The v1 core no longer advertises the
  `at_ts` (as-of time-travel) parameter it cannot serve, and a null or blank
  `at_ts` from a provider that eagerly populates every schema field is treated
  as omitted instead of rejected; only a meaningful timestamp is refused.
  Legacy v0.x cores continue to advertise and honor `at_ts`.
- Accept a double-encoded `context`, `scope`, or `filters` argument — a JSON
  string that decodes to an object — at the single argument-parsing seam, so a
  client that stringifies a nested object is not rejected when its fields are
  correct. A string that is not a JSON object is still refused.
- Document `brain.closeout`'s conditionally required `task_ref` (required
  unless supplied through `context.task`) and its required `summary` directly
  in the tool schema.
- Report `coverage.feedback_needed` on context and search packets and instruct
  agents not to file feedback on, or re-poll, a retrieval that returned no
  items; `brain.context` is not a task-state store.
- Add a schema/validator consistency contract test asserting, for every
  `brain.*` tool, that the fields the dispatcher requires are exactly the ones
  the published schema marks non-nullable.

## 1.1.0 — 2026-07-17

- Add optional hybrid lexical/dense retrieval with an explicit local vector
  sidecar and deterministic lexical fallback when the sidecar is absent,
  stale, or incompatible.
- Add source-hash-verified curated-memory manifests for explicitly reviewed
  starter beliefs; relative source paths are portable and the public example
  contains synthetic data only.
- Restrict core database files to owner read/write permissions and document
  that SQLite remains plaintext at rest.
- Add a clone-to-first-smoke quick start, explicit empty-brain behavior,
  contribution and security policies, issue/PR templates, code ownership, and
  a public CI gate.
- Clarify that OpenClaw is optional and that each compatible MCP client must be
  configured and instructed before a fresh chat can use OCBrain.
- Enforce server-controlled hosted-model delivery, bounded 32 KB context
  packets, bounded excerpts and source handles, and local-path redaction.
- Partition current serving inventory into eligible, scope-excluded, and
  delivery-excluded counts without listing excluded hosted IDs or content;
  these exact, query-independent counts disclose category cardinalities.
- Require an explicit hosted-egress acknowledgement for curated manifests and
  apply fully prevalidated manifests atomically; add a public, source-backed
  hosted-context demonstration.
- Add a deterministic public golden retrieval dataset covering relevance,
  scope, delivery policy, source hashes, contradictions, and negative queries.

## 1.0.1 — 2026-07-13

- Add explicit, owner-only evidence bundles for manual cross-machine exchange;
  imports are database-free dry runs by default, derive local content ids, and
  cannot import beliefs, receipts, or companion state.
- Redact credential-shaped material before truncation, reject credential files
  and directory-sweep symlink escapes, and stream bounded history windows
  instead of loading an unbounded transcript into memory.
- Reject malformed MCP frames and stale active-database pointers while keeping
  the default runtime contract at exactly eight tools.
- Prevent late proposal decisions and projection rebuilds from reviving
  tombstoned, retracted, or subsequently corrected beliefs.
- Preserve confidentiality and `local_only` egress across scope refinement,
  restrict legacy-table cleanup to exact retired OCBrain schemas, and make
  insecure local pointer/config permissions fail runtime diagnostics.

## 1.0.0 — 2026-07-13

- Make the append-only event chain the single semantic authority and keep
  evidence, beliefs, links, aliases, and full-text search as deterministic
  projections.
- Add archive-first, fresh-path-only migration with exact legacy event-prefix
  preservation, corruption refusal, strict schema inventory, projection
  rebuild verification, and separate activation.
- Split local training and legacy operations into optional `ocbrain-training`
  and `ocbrain-ops` packages with independent databases and no recurring jobs.
- Preserve action and outcome feature envelopes in closeout receipts so later
  models can reinterpret local metrics without assuming that clicks,
  subscriptions, deploys, and tests mean the same thing everywhere.
- Pass fresh Codex, Claude Code, and OpenClaw context, source, feedback, and
  closeout turns against the same verified core.

## 0.5.0 — 2026-07-13

- Add the stable `ocbrain.context.v1` packet with resolved scope, ranked
  beliefs, contradictions, coverage, exclusions, and bounded source handles.
- Add hash-verified `brain.source` expansion and append-only
  `ocbrain.closeout.v1` receipts linked to retrieval feedback and decision
  impact.
- Separate the eight-tool runtime MCP profile from protected administrative
  mutation and correct `--allow-writes` into a deprecated admin-profile alias.

## 0.4.1 — 2026-07-13

- Default hosted judging, embedding, and teacher authority off even when
  credentials exist.
- Block pilot training until the stratified named-human audit and a separate
  local training opt-in are both complete.
- Retire and disable the light-autopilot, heavy-autopilot, and stallcheck
  schedules; retained plist labels are inert uninstall markers.
- Keep MCP on demand and require a fresh client process after upgrades.

## 0.4.0 — 2026-07-10

- Add a frozen 100-case retrieval benchmark across Codex, ChatGPT, Claude Code,
  and OpenClaw, including negative, injection, citation, scope, and latency
  checks; improve repo-section ranking and demote raw catalog stubs.
- Instrument retrieval uses with queries, runtimes, sessions, and served ids;
  preserve explicit feedback provenance and conservatively infer later
  same-session or exact-reference outcomes.
- Make MCP tool schemas provider-safe with required-but-nullable optional
  fields, closed object shapes, and one null-stripping dispatch seam so eager
  tool callers cannot turn invented defaults into intended scope or flags.
- Classify corpus rows into `train_voice`, `train_judgment`, `train_skill`,
  `retrieval_only`, or `exclude`, with adversarial persona-author and injection
  contamination guards.
- Select a deterministic bounded training pack, locally grade only that pack,
  and refuse pilot preparation until it is fully graded and meets unchanged
  skill/voice/judgment/evaluation minimums.
- Preserve the original 20-case evaluation as a byte-frozen sentinel and add
  four-way blind base/tuned/Jonathan/frontier preparation and scoring.
- Anchor the local judge to separate human labels, reasons, and ideal responses;
  keep the 90% calibration bar fixed while teaching the rubric concise reasons,
  quantified uncertainty, and explicit fictional assumptions.
- Prepare dataset rows outside SQLite transactions and commit ordered bounded
  batches; retry harvest locks and hosted-judge timeouts only within stage
  deadlines.
- Page partial/failed/stale autopilot runs and judge failure streaks, add an
  optional pager canary, and require an explicitly human actor for quarantine
  release.
- Add `docs/CONTRACT.md` as the canonical autonomy, authority, and privacy
  boundary.

## 0.3.3 — 2026-07-10

- Replace the tripwire scan's timestamp-only watermark with a composite
  `(updated_at, id)` cursor, so equal-timestamp rows cannot be skipped at a batch
  or time-budget boundary.
- Replace per-row full event-log deserialization for hard corrections with an
  indexed JSON target lookup. On the live backlog, a clean 1,000-row tripwire
  page fell from 301.2534 seconds to 0.0351 seconds.
- Avoid a full FTS delete scan when inserting a brand-new evidence or knowledge
  row; existing parents still replace their exact index row.
- Retire newly discovered stalls outside the paging backlog window so they do
  not remain `new` and rewrite their ledger every 15 minutes.
- Run liveness checking from independent autopilot maintenance, make unchanged
  deadman evidence idempotent, and record malformed deadlines explicitly.
- Commit an autopilot `running` row and profile deadman before work, checkpoint
  both after every stage, and have the independent stallcheck process page an
  overdue producer deadline, including in read-only dry-run inspection.
- Preserve polymorphic legacy retrieval ids as `task_ref` provenance instead of
  invalid knowledge foreign keys; repair existing orphan references without
  deleting retrieval history.
- Repair a missing event-projection cursor even when compilation has no new
  proposals.
- Exclude runtime logs, data, local caches, build output, and the untracked lock
  file explicitly from source distributions; `logs/` is now gitignored too.


## 0.3.2 — 2026-07-10

- Bound post-turn review inside large sessions at 50 mutating units or two
  seconds, while preserving the commit-before-next-lazy-session boundary.
- Replace conservative per-session lock estimates with measured writer-lock
  wait, total, and maximum telemetry from explicit transactions.
- Move persona/SFT/DPO redaction, serialization, quality scoring, and dedup reads
  outside SQLite writer transactions; evidence and each final example insert
  commit before parsing or scoring the next candidate.
- Require a separate, complete human-label file with named provenance before a
  local judge can pass calibration; embedded machine-authored winners are
  ignored.
- Add a calibration-only mode that cannot open blind pairs, preserve judge
  explanations for audit, and align the evaluator to concise reasoning,
  evidence-aware optionality, and quantified uncertainty without fake precision.
- Record the dated human-grounded gate honestly: 7/8 (87.5%), with the remaining
  concision-versus-reason miss preserved. The v0.3.0 blind result was not rerun.
- Refresh runtime documentation against installed OpenClaw 2026.6.11, Claude
  Code 2.1.206, and Codex CLI 0.144.1 command surfaces.

## 0.3.1 — 2026-07-09

- Commit post-turn review work at each fully processed session before the lazy
  transcript iterator parses the next file.
- Report review's per-session transaction count and conservative total/maximum
  writer-lock upper bounds.
- Add a concurrency regression test that acquires SQLite's writer slot between
  two lazily yielded sessions.
- Commit judge and embedding egress audits before hosted network I/O, then
  commit verdict/vector results per completed provider batch.
- Commit stall findings before optional Telegram paging so notification latency
  never owns the brain database's writer slot.
- Release persona-mining evidence writes before running the next Git subprocess.
- Apply the shared autolabel stage budget to FTS attribution instead of letting
  that substage overrun the light profile indefinitely.
- Finish promotion scoring/eligibility reads before opening bounded score-update
  batches, and commit each tripwire quarantine before evaluating the next row.
- Commit history and doctrine harvests per imported file before reading the next.

## 0.3.0 — 2026-07-09

This is the first licensed release of ocbrain. It turns the earlier public
source into an Apache-2.0 open-source project and aligns the shared brain with
current ChatGPT/Codex, Claude Code, and OpenClaw transcript and MCP surfaces.

### Added

- Named light and heavy autopilot profiles, a shared overlap lock, stage
  budgets, run ledgers, stall detection, and managed runtime excerpts.
- Local-only SFT, DPO, and persona mining; loopback-only dataset grading; and an
  eval-before-train MLX-LM pilot with pinned trainer provenance.
- Frozen-evaluation reuse for later pilots, explicit private persona curation,
  and a judge-calibration gate that runs before blind material is opened.
- Cross-runtime MCP search, feedback, digest, preview, and guarded write tools.
- A tracked-tree privacy scanner and pre-push hook for public releases.

### Changed

- Dataset mining commits bounded batches instead of holding one writer
  transaction across corpus parsing. Autolabel also releases the writer slot
  between source miners and expensive FTS attribution.
- Writer-lock wait, total hold time, and maximum hold time are recorded in stage
  results. Large WAL files are checkpointed only after the dataset writer has
  committed; a blocking reader is reported honestly as busy.
- Current Codex agent-message records are treated as injected context rather
  than persona voice, and OpenClaw-hosted Codex/Claude transcripts retain their
  producing runtime attribution.

### Privacy

Corpus text, references, ratings, local config, database files, and model
weights remain under the gitignored `data/` tree. Public release artifacts
contain source, tests, documentation, and aggregate results only.

### Dated model evidence

The second local voice pilot reused the first pilot's 20 prompts, references,
rubric, held-out hashes, and blind randomization exactly. After adding eleven
canonical first-party examples, ten of which cleared the unchanged local grade
threshold, the candidate improved from 2/20 preferences to 7/20. The reference
still won 13/20, so v0.3.0 treats this as corpus progress and a model-quality
failure, not as a voice-model release.
