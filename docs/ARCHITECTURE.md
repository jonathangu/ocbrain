# OCBrain v1 architecture

OCBrain is one local, scope-aware evidence and context layer shared by Codex,
Claude Code, OpenClaw, and future MCP clients. It is not a design handbook, an
agent scheduler, or a prompt-sized copy of the user's data lake.

The architectural rule is simple:

> Keep durable observations and outcomes in an open, source-backed ledger;
> derive disposable retrieval features; give each model only the bounded,
> coherent context needed for the current decision.

## 1. Retrieval and long context are complementary

OCBrain has more history than any useful prompt can hold. Pure long context
would repeatedly reread the lake, dilute attention, and still fail at scale.
Pure top-k retrieval can miss omissions, cross-document comparisons, or the one
sentence needed to interpret a result.

The runtime path combines them:

1. Scope and lexical retrieval filter the unbounded lake.
2. `brain.context` creates a stable `ocbrain.context.v1` dossier with coverage,
   exclusions, contradictions, evidence support, and source handles.
3. `brain.source` expands the few exact sources needed for comparison or global
   reasoning. The handle is scope-bound, size-bounded, and content-hash checked.
4. The model performs the work.
5. `brain.feedback` records whether the retrieval helped.
6. `brain.closeout` links task state, artifacts, verification, decisions,
   actions, and outcomes back to the retrievals used.

The retrieval index is replaceable infrastructure. The evidence and outcome
ledger is the product.

## 2. One semantic authority

The v1 core has one semantic plane. `brain_events` is an append-only,
hash-chained authority. Existing v0.x events retain their sequence, body bytes,
hashes, and links during migration; new events continue from that verified head.

Derived semantic projections are rebuilt from the event chain:

- `evidence_objects` — normalized source-backed observations. Transcript
  windows (`*_history_file`) are stored by reference: `body` is empty,
  `body_head` holds the recorded excerpt, and `metadata_json.event_body.body_ref`
  names the file and the hashes that let `brain.source` rebuild the window and
  prove it rebuilt correctly (`src/ocbrain/history_window.py`);
- `current_beliefs` — the current serving interpretation, never an independent
  second truth;
- `belief_evidence` — explicit support/contradiction provenance;
- `object_aliases` — stable resolution for legacy and collision-safe IDs;
- `search_index` — derived FTS content;
- `projection_cursor` — the verified fold boundary.

Operational append-only receipts are intentionally separate from semantic
belief projection:

- `retrieval_uses` and `retrieval_items`;
- `context_source_handles` and issuance history;
- `task_closeouts` and their retrieval links;
- `egress_audits`.

They survive a full semantic projection rebuild because they describe what a
runtime actually received or reported, not what the brain currently believes.

The strict inventory contains no legacy `knowledge`, relational `evidence`,
`memory`, dataset, trainer, judge, embedding, autopilot, scheduler, deadman,
watchdog, or paging table.

## 3. Events preserve raw meaning

Every semantic change is a later event, not a silent rewrite:

- evidence recording;
- compilation proposal and decision;
- correction or hard constraint;
- tombstone;
- explicit scope promotion;
- deterministic legacy-import events used by migration.

The event body keeps the original payload, event schema, actor/runtime, session,
time, scope, and provenance. Derived classifiers and embeddings are never the
only copy of an observation.

A correction targeting an old `know_*` ID resolves through aliases and is
replayed after deterministic import mapping. A restrictive relational lifecycle
state cannot resurrect a belief that an event already demoted, retracted, or
tombstoned.

## 4. Scope is mechanical

Scope is part of every evidence object and belief. `SCOPE_TYPES` is:

- global doctrine;
- project;
- repository;
- client;
- task;
- session;
- conservatively quarantined legacy-unscoped data.

`personal_finance` used to be an eighth type. No row ever carried it, so it is
gone from `SCOPE_TYPES` and from the published MCP enums. `session` was a
candidate for the same treatment and was deliberately kept: one stored evidence
event in the shipped ledger carries `scope_type="session"`, and dropping the
value would make `ScopeTag` refuse that event and break projection replay. No
client can request a session scope.

`VISIBILITIES` is `internal`, `confidential`, `secret`. `public` was published
for two years and never once written, so it went with `personal_finance`.
`EGRESS_POLICIES` is unchanged: `hosted_ok`, `local_only`, `approval_required`,
`prohibited`.

Visibility and egress policy are separate dimensions. Matching scope controls
local retrieval; egress policy controls whether content may leave the machine.
Confidential foreign scopes are excluded before ranking. `brain.get` and
`brain.source` enforce the same scope and lifecycle boundary as search rather
than becoming ID-based bypasses.

Migration does not infer sensitive scope from prose. Only explicit legacy
metadata or already-scoped events may establish a broader meaning. The old
placeholder project value `workspace` becomes `legacy_unscoped`, not a real
project claim.

## 5. Stable Shared Context contract

`ocbrain.context.v1` is deliberately model-neutral. It returns:

- query plus resolved runtime, project, repository, client, task, and session context;
- ranked current serving beliefs with evidence IDs, an evidence-support count,
  and the recording time of their newest supporting evidence. The authored
  `confidence` / `confidence_band` pair is deliberately *not* served: measured
  against recorded feedback it ran backwards, so a reader weighting on it was
  being steered toward the rows readers liked least;
- visible contradictions;
- bounded OCBrain-issued source handles;
- returned counts plus separate scope- and delivery-excluded serving-inventory
  counts; these exact query-independent counts disclose category cardinalities,
  but hosted packets never include excluded IDs, content, or an object sample;
- estimated token cost and unavailable-source reasons;
- a retrieval-use ID for feedback and closeout linkage.

Source handles store the exact body hash, source locator, scope, issuance time,
origin retrieval, and reissuance history. Expansion refuses a changed source or
mismatched scope and requires a fresh context call.

Historical `at_ts` retrieval is not silently approximated in v1. The MCP
explicitly rejects that argument until an event-sequence-correct implementation
exists.

## 6. Actions and outcomes retain local semantics

Surface events are not universal rewards. A click can be exploration on one
site and conversion on another; a subscription can be the primary objective,
an accidental proxy, or a guardrail failure.

`ocbrain.action.v1` therefore separates:

- physical mechanism;
- local semantic role;
- target identity;
- state and alternatives visible beforehand;
- policy/model/version and context consumed;
- latency, tokens, money, human attention, and reversibility;
- provenance and optional versioned features.

`ocbrain.outcome.v1` preserves a vector:

- metric, JSON value, role, and unit;
- observation window and delay;
- baseline and counterfactual;
- attribution and uncertainty;
- explicit local interpretation;
- optional versioned features and receipt-level verifier evidence.

Both are optional arrays inside the hash-addressed append-only
`ocbrain.closeout.v1` receipt. A later experiment may derive a scalar reward,
but the durable record keeps the components so a future model can reinterpret
them.

## 7. Authority and MCP profiles

The runtime profile is the ordinary agent surface:

```text
brain.context  brain.source  brain.search     brain.digest
brain.get      brain.feedback brain.ingest    brain.closeout
brain.supersede
```

Runtime writes are narrow and append-only: evidence, retrieval feedback, task
outcome receipts, and one correction shape. They do not directly promote a
durable belief.

`brain.supersede` is the exception that proves the rule. It compiles a
replacement belief, so it looks like promotion — but it can only replace a
belief that is already serving, only inside that belief's own scope, only at
`min(old_confidence, 0.7)`, and only a bounded number of times per caller per
day. Doctrine, pinned beliefs, and rate-cap overflow route to an undecided
proposal instead, which only the admin profile can decide. There is no path
through it that widens scope or creates a belief where none existed.

The admin profile adds local preview, egress preview, proposal listing/decision,
correction, and tombstone tools — fifteen tools in total. `--allow-writes` is a
deprecated alias for this profile, not a no-op. Hosted teacher, training, and
watchdog controls are not core MCP tools.

`brain.mark_stale` was published in the tool list and is gone. It was
undispatchable: `tools_for_profile` returns `RUNTIME_TOOLS` for the runtime
profile and `RUNTIME_TOOLS | ADMIN_ONLY_TOOLS` for admin, and `brain.mark_stale`
was in neither set, so every call reached the unknown-tool branch.

## 8. One distribution

The source tree produces exactly one distribution, `ocbrain`, with two console
scripts that both dispatch to `ocbrain.cli:main`. Nothing extends the CLI from
outside the wheel; there is no companion entry-point mechanism.

Two companion distributions used to exist. `ocbrain-training` held dataset
mining, grading, and pilot-packet preparation; `ocbrain-ops` held autopilot,
stallcheck, the judge, the embedder daemon, loop ingest, and legacy maintenance.
Both are deleted. The decisive evidence was their own stores: every ops table —
`autopilot_runs`, `judge_runs`, `embed_runs`, `signal_events`,
`harvest_watermarks`, `loop_liveness`, `family_scores`, `stall_pages`,
`watchdog_findings` — held zero rows.

The companion stores declared an `egress_audits` table of their own, and that
one is empty too. It is not the `egress_audits` listed in section 2: the core
owns a table by the same name, `src/ocbrain/egress.py` writes it on every
applied curation run, and it holds real rows. Only the companion copy is
deleted.

One capability was worth rescuing rather than deleting. The public-repository
safety scanner moved into the core as `src/ocbrain/publicsafety.py`, so
`ocbrain public-safety-check` and `ocbrain install-hooks` are ordinary
subcommands and `ops/hooks/pre-push` runs
`python -m ocbrain.cli public-safety-check` against a plain checkout. The three
`com.jonathangu.ocbrain.*.plist` files that named the autopilot and stallcheck
jobs are deleted with the code they pointed at.

## 9. Archive-first migration

Migration follows five non-destructive steps:

1. Open the live v0.x source read-only and verify its event chain.
2. Create a coherent fresh archive snapshot and hash it.
3. Preserve the exact event prefix, then append deterministic import events for
   relational evidence, beliefs, links, lifecycle constraints, and retrieval
   snapshots.
4. Extract the legacy source's training and operational rows into their own
   fresh files and account for every source table in the manifest. Those tables
   have no place in the strict v1 inventory, and a migration that silently
   dropped them would not be accountable.
5. Verify schema inventory, event chain, projection rebuild, FTS integrity,
   counts/hashes, foreign keys, and MCP behavior before publishing fresh paths.

Failure leaves the source untouched and does not publish partial outputs. The
migration command never changes `OCBRAIN_DB` or the ignored activation pointer.

Activation is a distinct, reversible operator action after migration
verification. Fresh clients then exercise that provisionally activated core;
the pointer is retained only after three-client acceptance, or rolled back on
failure. The old archive remains available for audit, never as a hidden MCP
fallback.

## 10. Acceptance

The v1 core is accepted only when:

- its database has the exact strict inventory and no legacy training or
  operational tables;
- the legacy event prefix is byte-for-byte preserved and the full chain verifies;
- every imported row is represented or explicitly accounted for;
- full projection rebuild reproduces semantic hashes and MCP responses while
  preserving runtime receipts;
- no hosted call, timer, watchdog, training run, or automatic activation occurs;
- fresh Codex, Claude Code, and OpenClaw processes each complete
  `context → source → feedback → closeout` against the same activated core.

See [SHARED_CONTEXT_V1.md](SHARED_CONTEXT_V1.md) for the contract and
[CORE_OPERATIONS.md](CORE_OPERATIONS.md) for commands and recovery procedures.
