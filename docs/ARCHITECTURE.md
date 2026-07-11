# ocbrain Architecture

A repo-native map of how ocbrain turns the history your agents already
produce into source-backed memory — and, as a byproduct, a fine-tuning dataset
built from your own decisions. It runs unattended on two cadences — a fast
`light` cycle every 15 minutes and a full `heavy` cycle hourly. Private corpus
text stays local; configured non-private judge/embedding calls are redacted,
scope-checked, budget-capped, and egress-audited.

This document is the engineering companion to the public explainers:

- **How it works** — <https://openclawbrain.ai/how-it-works/>
- **Safe & inspectable** — <https://openclawbrain.ai/proof/>

and to the in-repo build spec [`docs/V2_AUTONOMY_SPEC.md`](V2_AUTONOMY_SPEC.md)
(the approved v0.2 implementation contract, kept as a historical design
record) and the top-level [`README.md`](../README.md). For v0.2-era detail —
exact schema migration, module inventory, signal-fold math — the spec is
authoritative. Everything the shipped system has added since v0.2 (targeted
judge, embeddings, the light/heavy split, `excerpt_render`, the
human-bootstrap pin, the stallcheck watchdog) is documented only here.
The canonical authority boundary is [`CONTRACT.md`](CONTRACT.md).

---

## 1. The one-sentence version

Immutable **evidence** goes in; a compiler distills it into current
**knowledge**; the brain reads its own outcomes to **label** that knowledge
good/bad/neutral; only the trustworthy, scanned, corroborated rows are promoted
into an injectable **memory** view; and the same evidence compiles into
local-only fine-tuning **datasets** — all driven by a single idempotent
autopilot loop that runs as two profiles (a fast `light` cycle and a full
`heavy` cycle, v0.3), guarded by six always-on safeguards that replaced the
old human approval queue.

One direction. Nothing overwritten. Every step skippable-and-safe.

Learning quality is measured rather than inferred from corpus size. The
selected pack is divided into `train_voice`, `train_judgment`, `train_skill`,
`retrieval_only`, and `exclude`; only locally graded passing examples can enter
the v0.4 pilot. The original blind evaluation remains frozen as a sentinel.

Stallcheck treats a `partial` or failed autopilot run as unfinished work and
pages it, detects a stuck `running` ledger row, and watches repeated judge
sub-stage failures. The reciprocal deadman and optional daily pager canary make
grit a structural heartbeat contract rather than motivational prompting.

The MCP surface also normalizes provider behavior. Optional schema properties
are exposed as required-but-nullable, unknown object keys are rejected, and
explicit nulls are stripped at the dispatcher seam. This prevents an eager
tool caller from turning an invented default into intended scope, privacy, or
limit input.

---

## 2. The two-plane store

ocbrain is a single local SQLite database with two cooperating cores. They are
not alternatives; they are two planes over the same evidence.

### 2.1 Relational core (the librarian/compiler)

The durable, queryable spine:

| Table | Role |
|---|---|
| `evidence` | immutable, append-only, hash-pinned records of what happened |
| `knowledge` | the *current* compiled belief, with type / lifecycle / scope / labels |
| `knowledge_evidence` | support / contradiction / derivation links (belief → its sources) |
| `retrieval_uses` | did a served memory turn out helpful, used, ignored, irrelevant, harmful |
| `memory` (VIEW) | the read-only projection of injectable knowledge — **not a store** |
| `signal_events` | dated, weighted good/bad/neutral votes attached to knowledge (v0.2) |
| `dataset_examples` / `dataset_sources` / `dataset_exports` | the dataset factory ledger (v0.2) |
| `autopilot_runs` / `judge_runs` / `harvest_watermarks` | run telemetry + incremental cursors (v0.2) |
| `embed_runs` | vector-embedding run telemetry + budget tracking (v0.3) |

Knowledge has three types, split on one bright line — **readable vs. executable/prescriptive**:

- `value` — facts and metrics (queryable truths).
- `doc` — readable wiki / procedure prose.
- `capability` — executable or loadable skills/procedures (the risky class).

`memory` is a projection, never a second store to keep in sync:

```sql
CREATE VIEW memory AS
  SELECT * FROM knowledge
  WHERE status = 'current'
    AND inject = 1
    AND quarantine_reason IS NULL;
```

### 2.2 Event core (the hash-chained audit trail)

`brain_events` is an append-only, hash-chained ledger of every decision:
compilations decided, corrections recorded, quarantines applied, clobbers
refused. Beliefs in the event core carry a first-class `ScopeTag`
(visibility × scope-type × egress policy), and `rebuild_projection()` folds the
event stream into `current_beliefs`. Corrections are durable: once a hard
`mark_wrong` / `retract` / `demote` targets a belief, the teacher path can never
re-derive that belief id.

**Why two cores.** The relational plane answers "what do we believe *now*, and
what's the source?" fast. The event plane answers "how did we get here, and what
did we decline to do?" tamper-evidently. Supersede/archive, never overwrite:
the relational `knowledge` row moves forward while the event chain preserves the
full trail. Two scope systems (relational `privacy_scope` ladder + event-core
`ScopeTag`) both persist and map into each other at the dataset boundary.

---

## 3. The five-movement pipeline

Raw agent history becomes trustworthy injectable memory — and a byproduct
dataset — in one direction, without any step overwriting history. Every arrow
passes the automatic safeguards (§5).

```
                        ┌───────────────── automatic safeguards on every arrow ─────────────────┐
                        │  injection-scan · no-clobber · secret-scrub · privacy-ratchet · quarantine │
                        └───────────────────────────────────────────────────────────────────────┘

   agent           ┌──────────┐      ┌───────────┐      ┌──────────┐      ┌──────────┐
   history  ─────▶ │ EVIDENCE │ ───▶ │ KNOWLEDGE │ ───▶ │  LABELS  │ ───▶ │  MEMORY  │
  transcripts      │ immutable│      │ compiled  │      │ good/bad │      │ injectable│
  corrections      │ hash-pin │      │  belief   │      │ /neutral │      │   view    │
  test/deploy      └──────────┘      └───────────┘      └──────────┘      └──────────┘
  verifier out          │                                                       │
  loop results          │                                                       │ (read-only byproduct)
                        │                                                       ▼
                        │                                                 ┌──────────┐
                        └────────────────────────────────────────────────│ DATASETS │
                                (compiled from the same evidence)         │ SFT/DPO/ │
                                                                          │ persona  │
                                                                          └──────────┘
                                            datasets NEVER feed back into what an agent sees
```

1. **Evidence — what actually happened.** Everything starts immutable and
   hash-pinned. A correction is a *new* record; the original stays as written.
   Harvested third-party text (web pages, other tools' output, pasted logs) is
   injection-scanned at the evidence layer before it can influence anything —
   external content is data, never instruction.
2. **Knowledge — the current belief.** A compiler distills evidence into
   knowledge. A belief with no evidence is not a valid shape. Knowledge
   supersedes rather than overwrites.
3. **Labels — good / bad / neutral.** The brain reads its own outcomes and
   decides which knowledge earned trust, via the signal taxonomy (§4). Each
   signal is a small, dated, weighted vote; they fold with time-decay and a mass
   threshold into a label + confidence.
4. **Memory — only the trustworthy parts.** A *scored* promotion (§6) lets only
   current, good-labeled, scan-clean, budget-fitting rows become `inject = 1`.
5. **Datasets — your decisions become training data.** The same evidence
   compiles into SFT/DPO/persona corpora (§7). Drawn with a dashed edge on the
   public diagram because it is a **read-only byproduct**: datasets are compiled
   *from* evidence but never feed back into what an agent sees at runtime.

---

## 4. Signal taxonomy & founder-weighting

### 4.1 The 19-signal taxonomy (four families)

Labeling is the heart of v0.2. Signals are mined from history that already
exists — no extra instrumentation — and persisted as frozen `Signal` rows in
`signal_events` (stable id → `INSERT OR IGNORE`, idempotent). They group into
four families:

- **Human feedback** — corrections, thanks, approvals, reverts. Correcting an
  agent is a strong negative on whatever it just relied on; "perfect, ship it"
  is a positive.
- **Verification** — tests passing/failing, deploys succeeding/rolling back,
  verifier evidence. *Verified beats claimed.*
- **Retrieval outcomes** — whether a served memory turned out helpful, used,
  ignored, irrelevant, or actively harmful.
- **Operational & maintenance** — commitment outcomes, scheduled-job results,
  error-recovery arcs, learning-DB gate rules, and conflict-heal supersessions
  the brain resolved on its own.

Representative kinds (full table in [spec §5.2](V2_AUTONOMY_SPEC.md)):
`user_correction`, `user_thanks`, `user_approval`, `task_closeout_success`,
`test_pass`/`test_fail`, `deploy_success`/`deploy_failure`, `revert`,
`error_recovery`, `retrieval_feedback`, `learning_gate_rule`, `gate_violation`,
`commitment_outcome`, `cron_run`, `hard_correction_event`, `verifier_result`,
`heal_superseded`, `clobber_refused`, and the optional `llm_judge`.

### 4.2 The fold

`fold_labels` computes, per knowledge row with new signals since its watermark,
a decayed signed score and mass:

```
S = Σ sign(polarity) · weight · 0.5^(age_days / half_life)      # signed score
M = Σ weight · 0.5^(age_days / half_life)                       # mass
```

- **Hard-bad precedence** — any bad signal at or above the hard-bad weight makes
  the label `bad` outright. The optional judge can *never* override this.
- Else `good` if `S/M ≥ 0.35` and `M ≥ min_mass`; `bad` if `S/M ≤ −0.35` with
  the same mass; else `neutral`. Confidence `= min(0.95, |S|/M · n/(n+1))`.

Time-decay means recent evidence outweighs stale; the mass threshold means a
single weak vote can't flip a belief.

### 4.3 Founder-weighting (generic)

Not every human vote carries equal weight. A small, **locally-configured** set
of high-trust author ids (e.g. a co-founder or operator whose feedback is the
highest-value taste signal) can be assigned an attribution weight. When one of
those authors corrects, thanks, or approves, the `user_correction` /
`user_thanks` / `user_approval` signal is scaled by that weight (so a founder
correction can reach hard-bad on its own), and the author provenance is recorded
in the signal details and carried onto any mined DPO pair.

Two invariants keep this honest and safe:

- **Attribution ≠ persona.** Weighting a founder's feedback does **not** admit
  their messages into the persona/voice stream — that stream stays strictly the
  single operator's. An identified non-persona sender never enters persona.
- **Ids live only in local, gitignored config.** No real identifier — telegram
  id/username, email, or name — is ever committed; the code matches on
  configured ids, never on hardcoded names.

---

## 5. The six automatic safeguards

v0.1 staged every risky belief as a candidate and waited for a human to promote
it. Safe, but it didn't scale — the brain grew only as fast as someone worked a
queue, and a busy queue gets skipped. v0.2 removes the approval queue and
enforces the *same safety property* — risky content can't reach the prompt
unchecked — with fixed invariants that run on every write. **Gates don't scale.
Invariants do.**

| # | Safeguard | What it enforces |
|---|---|---|
| 1 | **Injection scanning** | Harvested third-party text is scanned for prompt-injection (role hijacks, "ignore previous," exfil links, hidden characters, encoded blobs) at the evidence layer, at write time, and again before rendering into a prompt. |
| 2 | **Provenance no-clobber** | An automatic write can never silently overwrite a human-authored row. It is refused and logged as a `clobber_refused` breadcrumb; first writer wins, later passes add evidence not authorship. |
| 3 | **Secret scrubbing** | Bodies are scanned for credential/key patterns; a leaky row is quarantined out of memory, and secrets are stripped before any body reaches the judge or a dataset. |
| 4 | **Privacy ratchet** | Scope only tightens. Derived knowledge inherits the most restrictive scope of its sources; private can never widen through derivation. (See §8.) |
| 5 | **Tripwire quarantine** | Injection, secret leaks, repeated harmful feedback, hard corrections, belief thrash, or unverified risky content each trip a wire that pulls the row out of memory instantly — a status change, not a queue. |
| 6 | **Hash-chained audit** | Decisions, quarantines, and refusals append to the tamper-evident event log. Nothing is hard-deleted; the record of what the brain did *and declined to do* stays intact. |

### 5.1 Tripwire registry

Quarantine is encoded additively — `quarantine_reason IS NOT NULL` drops a row
from the memory view, from retrieval, and from promotion the instant it fires.
It writes its own tripwire evidence + a `correction_recorded` (op `demote`)
audit event, and there is exactly one path back out: an explicit release.

| tripwire | fires when |
|---|---|
| `injection_suspected` | a linked third-party source or the body itself trips the injection scanner |
| `secret_leak` | value/title looks like it contains a credential or key |
| `bad_feedback_spike` | repeated harmful/failed retrieval outcomes in a short window |
| `hard_correction` | an explicit durable "this is wrong" targets the row |
| `contradiction_thrash` | the belief keeps flipping — superseded and re-asserted repeatedly |
| `prescriptive_unverified` | risky knowledge serving with no passed verifier and no approval signal |

### 5.2 The risky class still gets a higher bar

Removing the queue did **not** flatten the risk model. Executable / prescriptive
/ high-risk knowledge must carry **passed-verifier evidence or an explicit human
approval signal** before it can ever reach `inject = 1` — the automatic
replacement for the old human gate, aimed at exactly the dangerous class. If it
serves without either, `prescriptive_unverified` quarantines it.

---

## 6. Promotion & decay

Memory is a scored projection, not automatic-by-default. A row becomes
injectable only when **all** hold: (1) current and not quarantined; (2) labeled
`good` with confidence above threshold; (3) injection-scan clean (body + every
linked source); (4) the risky class additionally carries verifier/approval
evidence. A sparse-signal **bootstrap** path (the prod memory view starts empty)
admits very-high-confidence rows backed by passed-verifier evidence.

Promotion is ranked by a score blending confidence, freshness, and how useful a
memory has actually been when served:

```
promote_score = 0.4·confidence + 0.25·freshness_decay + 0.2·use_rate + 0.15·scope_bonus
```

The injectable set is **budget-bounded** — a fixed cap on rows and on rendered
characters (a `build_excerpt` dry-run enforces the char budget, demoting
lowest-score-first on overflow). A label flip to bad, a confidence drop below
threshold, or a quarantine demotes immediately. `origin='human'` injected rows
are pinned and never auto-demoted by score.

**Human-bootstrap pin (v0.3).** A row admitted through the sparse-signal
bootstrap path is stamped `origin='human_bootstrap'` and inherits the same
exemption from *score* and *label-decay* demotion (a soft judge-bad label, low
confidence, or use-rate decay). The pin is narrower than full `origin='human'`
immunity, though: quarantine, a hard-bad fold (a hard human or founder
correction), or an injection/secret-scan failure still eject a bootstrap row
immediately — the pin never resurrects an ejected row. Each held exemption
records a neutral `pin_demotion_exempt` breadcrumb, so the audit trail shows
when the judge disagreed but the pin held.

---

## 7. The dataset factory

The same evidence that feeds memory compiles into fine-tuning datasets — the
long game toward a model trained on how you actually work. Three streams, each
idempotent and carrying full provenance:

- **SFT** (`format: chat`) — supervised examples: context (≤ N non-injected
  turns, char-bounded, head-trimmed) plus a good final assistant answer, mined
  from exchanges that ended well (affirmation, clean multi-step success, error
  recovery). Bad exchanges are retained but never exported to SFT — they feed DPO.
- **DPO** (`format: openai-preference`) — preference pairs mined from
  corrections: a corrected first attempt (rejected) vs. the later accepted
  attempt (chosen), plus event-core pairs (edit decisions, corrections, heal
  supersessions). Your corrections are the training signal. A relaxed
  structural pair gate (`dpo_relaxed_gate`, on by default in v0.3) admits a
  pair on softer structural evidence where the strict gate would otherwise
  reject a real correction. A separate, narrow `--founder-rescan` mode
  bypasses — without ever clearing — the normal DPO watermark to re-mine
  sessions containing a configured founder id under that relaxed gate;
  content-hash and dedup-key dedup keep repeated rescans from producing
  duplicate pairs.
- **Persona** (`format: chat`) — voice examples where the single operator's own
  verified messages and authored commits are the assistant *target*, so a
  fine-tuned model learns your style, not a generic one. Founders and other
  identified senders are excluded from this stream by construction.

### 7.1 Provenance, dedup, idempotency

Every example carries: `evidence_ids` (≥ 1, enforced) tracing back to the
transcript/commit/event rows; `privacy_scope` = the most restrictive scope over
all linked evidence; a `quality_label` + confidence + the rule names that fired.
A `content_hash` over the canonical-JSON of the messages/pair *only* (stable
across re-mines) backs `UNIQUE(dataset, content_hash)`; a `dedup_key` drives a
separate near-duplicate pass (dataset dedup is deliberately its own pass, not
reliant on DB uniqueness).

A **quality scrub** runs on every candidate before storage; failures are marked
`excluded`: secret residue (post-redaction), high-entropy blobs, length bounds,
near-dup, refusal-only targets, error-dumps, injected-memory-block leakage,
telegram-envelope residue, and prompt-injection inside the target.

### 7.2 Idempotent manifests & export

Export is **byte-deterministic**: deterministic ordering + canonical JSON means
an unchanged corpus produces identical bytes, and the writer skips the write when
the new `payload_hash` matches the last export. Every export writes a
`dataset_exports` row and a signed `manifest.json` (per-dataset count, bytes,
sha256, label/scope breakdowns, excluded count) plus an egress-audit row.
Sources are tracked incrementally in `dataset_sources` (path+size+mtime
fingerprints), so re-runs only re-parse changed files.

### 7.3 Local LLM grading and the eval-before-train pilot

`dataset-grade` grades SFT, DPO, and persona examples against different rubrics.
The transport boundary accepts loopback HTTP only, so corpus text cannot be sent
to a hosted endpoint through this lane. Runs are capped by per-run and daily
item budgets, recorded in `dataset_grade_runs`, and normalized grades live both
in additive columns and `example_json.metadata.llm_grade`. Export can require a
minimum grade; when it does, ungraded examples fail closed.
One DB-adjacent file lock admits a single grader. Each successful example and
its progress counter commit before the next local inference call, so inference
does not hold a SQLite write transaction. Deterministic response failures are
remembered for that model/prompt version; SQLite lock failures are retryable.
An autopilot overlap returns `blocked`, and the next owned invocation repairs a
run row left `running` by an interrupted process.

`dataset-pilot-prepare` creates a private training pack only after it
has deterministically reserved at least twenty graded persona prompts. It writes
prompts, private references, and the voice/taste rubric before any training
file, then removes the held-out content hashes from every training stream. The
manifest records `eval_built_before_train=true` and `training_started=false`.
Candidate responses are randomized against the real references by
`dataset-pilot-blind`; `dataset-pilot-score` resolves completed ratings through
the separate blind key. The supplied local rating helper rejects non-loopback
endpoints before opening the pair file, checkpoints each rating, and never reads
that key. Verified adapter weights and losses are recorded separately by
`dataset-pilot-record-training`; a trainer exit alone is not accepted as proof.
For a later pilot, `--eval-from` reuses the prior prompts, references, and rubric
byte-for-byte, verifies the original held-out hash, and excludes those same
source examples from training. The local judge must pass an explicit calibration
set against a separate, complete human-label file before the rating helper opens
any blind pair; the default gate is 80% across at least six cases. Labels require
named human provenance, and machine-authored winners embedded in the case file
are ignored. `--calibration-only` verifies the gate without reading blind pairs.
The pack also writes MLX-LM chat `train.jsonl` plus an optional deterministic
validation split from eligible SFT and persona rows. DPO remains separate: the
first local LoRA pilot is supervised chat tuning, while preference training
needs a DPO-capable trainer. The manifest contains an argv array pinned to a
verified MLX-LM Git commit and keeps `training_started=false` until execution.

---

## 8. Privacy model

Four scopes, least to most restrictive: `private` → `workspace` → `project` →
`public`.

- **Scope ratchet (one-way).** Derived knowledge inherits the *most restrictive*
  scope of everything it links. A public doc that cites a private source becomes
  private. Scope can only tighten through derivation — this is the invariant that
  lets everything else run unattended.
- **Private never exports.** `private`-scope rows never leave in a dataset,
  regardless of any flag; the export `min_scope` floor cannot widen past it.
- **The dataset never leaves the machine.** There is *no hosted export path in
  the code* — the export target class is `local_model` by construction. Datasets
  are written to a local directory only.
- **The optional judge is the only outward call, and it is fenced.** It is off
  unless an API key env var is present; it operates on *knowledge rows only*
  (never raw data or datasets); private rows are dropped before it runs; every
  body is secret-scrubbed; it is daily-budget-capped; and every batch writes an
  `egress_audits` row. Its verdict is just one more signal and can never override
  a hard human correction. **Targeted (v0.3):** the judge no longer grades the
  whole backlog. `judge.targeting` whitelists the knowledge origins it looks at
  (retrieval-touched, lessons, session-derived) and excludes the large
  catalog-doc backlog, so spend lands on rows a decision actually depends on
  rather than on inert file-catalog rows that no retrieval ever touches.
- **Public-safety pre-push gate.** This is a public repo; runtime data is not.
  A tracked-tree scanner (`ocbrain public-safety-check`) reads git only — never
  the runtime DB — and blocks four violation classes on the outgoing commit
  range: tracked files under `data/`/`logs/` or any dataset artifact; hits
  against a local gitignored denylist of private identifiers; new high-entropy
  tokens/secret patterns in added diff lines; and absolute private paths outside
  a small allowlist. Findings report counts and locations only, never the matched
  value. The tracked `pre-push` hook runs it on every push and blocks on any
  finding (`--no-verify` is the human override). `data/` is fully gitignored.

---

## 9. The autopilot state machine

Everything above runs as one idempotent, single-instance state machine,
scheduled locally (launchd). Any stage can be killed and safely re-run:
watermarks, stable ids, and uniqueness constraints mean nothing is
double-counted and no committed progress is lost. State lives in the stages
themselves; `run_autopilot` only *sequences* them and commits after each success.

**Two profiles (v0.3).** The old single full cycle ran every 30 minutes but took
34–45 minutes end-to-end — it overran its own timer. The stages are now split
into two named sequences in `cfg.autopilot.profiles`, each on its own launchd
timer:

- **`light` — every 15 min** (`StartInterval 900`): migrate → review → autolabel
  → embed → tripwires → promote → excerpt_render → maintain. The fast
  keep-current loop; skips snapshot, harvest, compile, and the dataset
  mine/export.
- **`heavy` — hourly** (`StartInterval 3600`): the full sequence below, including
  snapshot, harvest, compile, and the expensive fold/mine/export pass.

Both plists run through one shared env wrapper that sources the API key from a
`chmod 600` file outside the repo and passes the profile as an argument; the key
never appears in a plist. Because `cfg.autopilot.profile_locks == "shared"`, both
profiles contend for the *same* autopilot lock, so an overlapping light/heavy
fire finds the lock held and exits cleanly instead of double-running.

| # | Stage | What it does | Idempotency |
|---|---|---|---|
| 0 | **lock** | `fcntl.flock` single-instance; a slow run never collides with the next tick — the second invocation just exits | single-instance |
| 1 | **snapshot** | daily rotated copy through SQLite's online backup API before anything is touched | date-named file |
| 2 | **migrate** | additive-only schema migration (`IF NOT EXISTS` / `_ensure_column`) — never a destructive rebuild; self-heals drift | conditional DDL |
| 3 | **harvest** | fingerprint-gated import of new history into evidence rows | source fingerprints |
| 4 | **injection-scan** | scan new third-party evidence before it can reach anything injectable | rowid watermark |
| 5 | **review** | post-turn review of *settled* sessions — task successes, error recoveries, corrections, novel workflows become candidate knowledge + signals | path watermark + stable ids |
| 6 | **compile** | turn undecided proposals into knowledge; **one** `rebuild_projection` at the end (never per-item) | decision check |
| 7 | **autolabel** | mine signals → attribute to knowledge → fold → optionally consult the judge → fold again | stable signal ids + watermarks |
| 7b | **embed** (v0.3) | embed pending knowledge rows into vectors for semantic attribution (was FTS-only); self-skips when disabled / keyless / over the daily USD cap; audited egress | `embed_runs` + stored vectors |
| 8 | **tripwires** | run the tripwire registry and auto-quarantine anything that fires | knowledge watermark |
| 9 | **promote** | re-score, promote trustworthy rows into memory, demote what slipped, enforce the budget | deterministic re-rank |
| 9b | **excerpt_render** (v0.3) | render the just-promoted injectable set into the `BEGIN/END OCBRAIN MANAGED BLOCK` of each configured runtime memory file — content outside the markers is untouched, and an unchanged block is not rewritten | idempotent block write; unchanged block skipped |
| 10 | **maintain** | prune stale knowledge, heal conflicts (emitting supersession signals), and **archive never-referenced stale catalog docs** (v0.3) out of the working set | existing TTL logic + reversible status flip |
| 11 | **dataset-mine** | mine SFT/DPO/persona from newly-settled history in bounded, measured writer batches; checkpoint a large WAL after commit | `dataset_sources` fingerprints + `UNIQUE` |
| 12 | **dataset-export** | deterministically write the JSONL corpora + manifest, skipping if unchanged | `payload_hash` |
| 13 | **finalize** | finalize the already-visible `autopilot_runs` row, clear the profile deadman, release the lock | durable stage checkpoints |

The `light` profile runs stages 2, 5, 7, 7b, 8, 9, 9b, 10; the `heavy` profile
runs the full table. The `embed` stage always runs immediately after
`autolabel`, and `excerpt_render` always runs immediately after `promote`, in
every profile.

Before the first stage, the runner commits a `running` row and a profile
deadman. It checkpoints both after each completed stage and clears the deadline
only when the run finalizes. A hard kill therefore leaves visible partial
progress plus an overdue deadline instead of erasing the run from the ledger.

### 9.1 Abort vs. partial

Each independent stage runs in its own `try/except`. A failure records the error
in `stages_json`, downgrades the run to **`partial`**, and **continues** to later
independent stages. Only the two foundational stages — **snapshot (1)** and
**migrate (2)** — **abort** the whole run with status `error`, because every
later stage assumes a snapshotted, migrated DB.

### 9.2 Budgets, flock, snapshot-first

- **Snapshot-first.** Before any write run, autopilot takes a daily rotated
  SQLite snapshot through the online backup API, with no source-WAL checkpoint,
  rotating to keep a small number. If snapshot or migrate fails, nothing
  downstream runs.
- **Per-stage time budgets.** Each stage carries a wall-clock budget; miners
  accept a `time_budget_seconds` and return early with their watermark advanced
  only past *fully-processed* items, so a cycle fits its timer. The `light`/`heavy`
  split (v0.3) is the coarse-grained version of the same discipline: the fast loop
  stays inside its 15-minute tick by leaving the expensive fold/mine/export to the
  hourly `heavy` cycle.
- **Flock overlap safety.** The stage-0 file lock makes the launchd interval +
  a slow run safe: a second invocation exits 0 immediately rather than
  double-processing. Because `light` and `heavy` share one lock, a still-running
  `heavy` cycle also makes the next `light`/`heavy` fire exit cleanly.
- **Watermarks in-transaction.** Each watermark is written in the same
  transaction as the work it covers, so a kill mid-run loses no committed
  progress and re-runs resume cleanly.
- **Measured writer windows.** Dataset mining and post-turn review commit after
  50 mutating units or two seconds, whichever comes first, and record writer-lock
  wait, total hold, and longest hold. Autolabel commits between source miners and
  before each FTS attribution. Review also commits each processed session and its
  watermark before lazily parsing the next transcript. Dataset redaction,
  serialization, quality scoring, and dedup reads happen before the writer is
  acquired; each final example insert commits before the next candidate. The
  hosted judge and embedding lanes
  commit each egress audit before network I/O and their results after each
  provider batch. Once dataset mining has committed, a WAL above 64 MiB is
  checkpointed with `TRUNCATE`; a blocking reader is reported as `busy` and is
  retried by a later run.
- **Budget carries through substages.** Autolabel passes its remaining wall-clock
  budget into FTS attribution, so the light profile can stop and resume instead
  of overrunning its own timer on a large unattributed backlog.
- **Network outside transactions.** Hosted judge/embedding audits and stall
  findings are committed before provider or pager I/O. Verdicts, vectors, and
  delivery status land in bounded follow-up transactions.
- **Read first, write second.** Promotion finishes scoring and eligibility reads
  before bounded score updates. Tripwires commit a fired quarantine before
  evaluating the next row. Harvest commits each imported file before reading the
  next source.

### 9.3 The stall watchdog (companion process, v0.3)

`ocbrain.stallcheck` is a separate, passive process on its own launchd timer
(every 15 minutes) — it does not share the autopilot lock or its cadence, and
it never promotes knowledge or claims work. Each pass reads, read-only:
agent-workflow transcripts, for an `end_turn` left with a still-pending
monitor/background-task call, or a task output file opened but never closed;
and the runner ledger, for lost or stalled task runs and failed inbound
message handling. It also reads overdue producer deadmen from the brain itself,
including the autopilot profile deadline. A finding upserts a `loop_liveness` row and a
`loop_tripwire` evidence row into the same brain DB the liveness sweep and
weekly review already read, and — if a local pager configuration is present —
sends a single, deduplicated Telegram digest of any *new* stalls so a
persistent stall pages once, not repeatedly. The checker also writes its own
heartbeat row (`loop_id='stallcheck'`). The independently scheduled autopilot
`maintain` stage consumes that deadman row, while stallcheck consumes the
autopilot runner's stage-checkpointed deadman. Watchdog death and a stuck
maintenance loop therefore have different observers. Findings first discovered
after the backlog window are marked retired, not repeatedly re-counted as new.

---

## 10. What stays true by construction

- **Evidence before belief** — no durable knowledge without a source that traces
  back to something that happened.
- **Verified is not claimed** — passing tests and verifier evidence outrank
  confident-sounding text.
- **Memory is a view, not a store** — update the knowledge, the view follows.
- **Supersede, never overwrite** — history is appended; you can always ask what
  was once believed.
- **External content is data** — harvested third-party text is scanned and
  quarantined, never trusted as instruction.
- **Local by construction** — the datasets have no hosted export path; private
  scope never leaves; the machine is the boundary.

---

## 11. Where to go next

- [`docs/V2_AUTONOMY_SPEC.md`](V2_AUTONOMY_SPEC.md) — the approved v0.2
  implementation contract: exact schema migration, module inventory, config
  surface, signal-fold math, test plan, and lane breakdown.
- [`docs/ULTIMATE_GUIDE.md`](ULTIMATE_GUIDE.md) — historical pre-autonomy
  product + engineering walkthrough.
- [`docs/AGENT_USE_GUIDE.md`](AGENT_USE_GUIDE.md) — runtime behavior for agents.
- [`README.md`](../README.md) — quick start, CLI, MCP surface, public-safety.
- Public explainers: [how-it-works](https://openclawbrain.ai/how-it-works/) ·
  [proof](https://openclawbrain.ai/proof/).
