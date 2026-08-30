# Changelog

## Unreleased

- Correct the client-facing runtime contract to the current thirteen-tool v1
  surface, document Hermes as a first-class stdio MCP client, and put the
  deterministic briefing/ledger harness steps ahead of ranked retrieval for a
  fresh context.

- Stop a restatement becoming a new key, by fixing what the existing cascade
  could not see rather than by adding a threshold to it. A supersession can only
  ever replace a belief filed under the **same** key — the transaction copies the
  predecessor's key onto its successor, and the curator's own rationale says it
  "recompiled key `<k>`" — so a fact reworded under a new slug was uncollapsible
  by construction. The live corpus is the proof, measured on the
  **2026-08-28 morning snapshot (347 serving beliefs)**: **344 serving wiki facts
  carrying 344 distinct keys**, perfect uniqueness, with **35 same-scope
  near-duplicate clusters at cosine 0.88 covering 98 of those 347**, five of them
  Plane-1 recency beliefs compiled on one day under five keys, two differing by a
  single hyphen. A reader asking one question got one answer five times and read
  it as five corroborations: `0.0166` appears in **13** serving beliefs and `65.4`
  in **12**.

  The surprise, and the reason this is not a threshold change: replaying the
  actual mint of each cluster's second member against a copy of the live core —
  its real key, its real body, the real local embedder — the existing cascade
  catches **all 34** of them when the vector sidecar is freshly built. It is the
  *availability* of that cascade that fails, and it fails almost always. The
  hourly maintenance pass curates first and rebuilds the sidecar last, so the
  first belief a cycle writes invalidates the whole-corpus fingerprint and every
  later claim in that cycle reads `vector_sidecar_stale` — which
  `conflict_neighbor` treats as "no conflict". Same replay, same 34 clusters, with
  the sidecar in the state a cycle actually leaves it in: **23 of 34 mint a second
  serving belief before this change, 2 after**. With a fresh sidecar, 0 before and
  0 after; nothing regresses.

  Three arms, in the order they cost anything:
  - **The folded key**, which needs no embedding at all.
    `plane1-recency-gate-result` and `plane-1-recency-gate-result` are one key.
    Folding is separators-only and was checked against the corpus before it
    shipped: on the 2026-08-28 morning snapshot, 344 exact wiki keys fold to 343,
    collapsing exactly that pair and merging nothing else. Over the ledger's
    history it catches 2 of 1,213 keyed proposals — small, and it is the arm that
    cannot break. Re-measured at 2026-08-28T19:36Z: 300 wiki facts, 300 exact
    keys, 300 folded — this arm currently collapses nothing, because the pair it
    was built for has since been retired by an operator compaction.
  - **Document-to-document cosine** at `NEAR_DUPLICATE_COSINE = 0.88`, verifying
    each candidate's vector by that candidate's own `content_hash` instead of by a
    corpus fingerprint the corpus keeps moving, and embedding on demand whatever
    is left. Measured on a copy of the live core after one belief is written:
    `semantic_neighbors` → `vector_sidecar_stale`, this reader → answered, **203
    vectors reused, 1 embedded, 0 uncovered** (a 204-belief working copy; the
    adversarial reviewer reproduced the same shape at 301/300/1/0). The floor is pinned equal to
    `compact.DEFAULT_COSINE_FLOOR` by a test, because a claim this gate admits and
    the compactor then proposes retiring is a gate that moved the work rather than
    doing it. It is a different scale from the query-side
    `CONTRADICTION_COSINE_FLOOR = 0.60`, and the two are documented as two scales.
  - **Fail-closed** when it still cannot see. A claim the gate could not check is
    recorded as an undecided proposal (`pended_unverified`) rather than minted,
    and an identical re-derivation next cycle writes nothing — the pend path is a
    producer too, and it gets the same dedup that stopped the supersede ledger
    growing without bound. Two exemptions are **declared**, rather than the
    fail-open list being everything nobody enumerated: no core path, and no
    sidecar at all. An install that never opted into semantic dedup keeps
    compiling on the two lexical arms; an install that did opt in does not get to
    quietly lose the third because Ollama died at 03:00.
    `curator.duplicate_gate_fallback=admit` restores the previous behaviour.

  One thing this does not fix, measured rather than assumed: the 0.60 conflict arm
  is still stale-blind, which is why 2 of the 34 replayed clusters still mint. Both
  are cases a fresh sidecar would have caught on the query-side scale and that sit
  below the document-side floor. Giving that arm the same resilience needs a
  document-scale floor calibrated on more than two data points, so it is left
  named rather than guessed at.

- Make the curator's egress gate falsifiable. Read-only at **2026-08-28T19:34Z**,
  `egress_audits` holds **347 rows** spanning 2026-08-04T22:11 to
  2026-08-28T19:27, and `SELECT DISTINCT rejected_json` returns exactly one value
  on all 347: the literal string `'[]'`, across **25,958 transmitted items**. (The
  same structure at an earlier snapshot the same day read 240 rows / 25,106 items;
  the gate has never refused anything at either.) That was structural, not lucky, and it had two causes.
  `select_evidence` filtered on the allow-list inside its own SQL, so the audit
  could only ever be handed rows that had already passed; and the declared
  allow-list named `approval_required`, `hosted_ok` and `local_only` — every
  policy an operator may declare — so no input could fail it. Selection now
  *partitions* instead of filtering and the audit records what was refused, with
  `declared_egress_policies` beside `present_egress_policies` so a later reader
  can tell a gate that had nothing to reject from a gate that could not reject
  anything. On a copy of the live core the `example` project's next audit carries
  **3 refusals** where every previous audit carried 0 — three `confidential` +
  `prohibited` rows that the code floor had always been dropping invisibly — and
  the other two projects record 0 refusals beside `allowlist_vacuous: true`. The
  policy itself is unchanged and `select_evidence` returns exactly what it always
  did: whether `local_only` evidence may lawfully reach a hosted curator is not a
  question this change answers.
- Report a vacuous allow-list as a defect rather than as 240 clean audits.
  `egress_refusable_policies` counts how many policies present in the eligible
  evidence the *declaration* would refuse, with the code floor subtracted so an
  everything-admitting allow-list cannot borrow credit for `prohibited`. Measured
  0 on the live core, which alarms. `curator.egress_allowlist_ack` downgrades it
  to `watch` and carries the operator's stated reason into the scorecard —
  declared, not enumerated away.
- Key a belief's TTL on how fast its subject moves, not on which of two lifecycle
  words the model picked. `lifecycle=current` bought a flat ~90 days (158 beliefs,
  all expiring in November, none expired) and `lifecycle=durable` bought no expiry
  at all (185 beliefs, zero with a `valid_until`), so one serving belief still
  stated which ClickHouse host was live "as of 2026-07-24" **35 days later**,
  under a `valid_until` running to November, with a second belief repeating it.
  A claim is now classified `volatile` / `measured` / `doctrine`, mechanically
  where a body dates itself, pins a version, names a host or an access path, or
  asserts what is running right now — 33 of the 347 serving beliefs in the
  2026-08-28 morning snapshot, 11 of them `durable`-marked; re-run read-only at
  2026-08-28T19:36Z on the 303 that remain, 29 (9.6%), 10 `durable`-marked. The model may declare a class, and the declaration can only
  shorten a claim's life, never extend it, or `durable` becomes a way to opt out
  of expiry. Existing beliefs are **not** swept: `ocbrain wiki-volatility` prints
  the plan and needs `--apply --yes` to write, because on a copy of the live core
  it re-dates 173 beliefs and **7 are already expired** under the new scheme.
- Record why a supersede proposal is pending in the ledger, not only in the return
  value. The pending queue's reasons — the confidence-margin rule, a pinned or
  doctrine target, a rate cap — were legible to the caller and invisible to the
  operator reading the queue a week later.
- Give the curator per-cadence model profiles and an optional independent critic,
  both inert by default. `wiki-curator.py --cadence {hourly,nightly}` selects the
  profile, and `--provider` no longer carries an argparse default, which had made
  `curator.provider` unreachable: a flag beats the cadence profile beats the
  configured pair. `curator.nightly_provider`/`nightly_model` ship empty so both
  cadences resolve to the one configured pair; a cadence that names its own
  provider does not inherit the other's model id, because that posts one
  provider's model to another provider's endpoint.
  `curator.critic_provider` ships empty, and when set it gates only the changes
  where being wrong costs most — a supersession of a pinned belief or of anything
  in `global:*`. A critic configured to the curator's own provider family is
  refused rather than run: two calls to one family is one opinion counted twice,
  and correlated error is the whole reason for a second one. Anything other than
  an explicit approval — a refusal, a missing credential, a provider error — routes
  the change to the pending ledger with the critic's reason attached.
- Test the instrument the duplicate gate rests on, and give an operator control
  that was left standing after its rule moved somewhere to go. Both defects came
  out of an adversarial review of the four changes above; both are the classes
  this work exists to remove.

  `document_neighbors` — the ~140-line document-side reader the whole
  availability fix depends on — had **no test**. Every test of the gate's
  positive path monkeypatched it out, so reverting it to exactly the blindness
  the entry above names as the defect left all 949 tests green. Three
  independently killed mutations now cover it: requiring a fresh whole-corpus
  fingerprint (the query-side arm's blindness), dropping the per-candidate
  `content_hash` check, and a default embed budget of 0. The third is checked by
  score, not by a count: a candidate rewritten to be a near-copy must come back
  at cosine 1.0, which it cannot do from its dead stored vector.

  `--current-ttl-days` had become silently inert, and its help text stated two
  things that were false. Measured directly:
  `claim_ttl_days({'lifecycle':'current'}, current_ttl_days=0, volatility_ttl=True)`
  returned 45, and `current_ttl_days=30` also returned 45; the help said "0
  disables expiry. Durable claims never expire" while a durable body naming a
  rotating host got 14 days. `current_ttl_days <= 0` now means no expiry under
  either scheme, `--no-volatility-ttl` makes the lifecycle rule (and the number)
  reachable again, both flags default to the config field instead of to an
  argparse literal, and the run's rollup line reports which rule it compiled
  under. A test drives the whole chain from the flag to the stored `valid_until`.
  `ocbrain wiki-volatility` reads the same setting, because a switch honoured on
  the compile path and ignored by the sweep is the same defect one layer down.

  Three smaller things from the same review. `DEFAULT_DOCUMENT_EMBED_BUDGET = 32`
  is an availability cliff — past it the gate reports `candidates_uncovered:N`
  and the fail-closed default pends every remaining claim in the cycle — so it is
  now `curator.document_embed_budget`, documented as a cliff rather than a dial in
  `docs/THRESHOLDS.md`, and held above `--max-beliefs`. The fail-closed literal
  that applies when the config file is unreadable is now asserted whole, so the
  next default drifting there cannot pass. And the retired-model check no longer
  inspects only `PROVIDER_DEFAULTS`: it scans every text file in the repo, with
  three **declared** exemptions carrying their reasons, and has its own test that
  it can report dirty. It found what the first fix had left standing — a
  `gpt-5-mini` config example in `docs/V2_AUTONOMY_SPEC.md` and both retired ids
  as test parameters.

- Correct two provider defaults that had gone stale. `moonshot-v1-32k` named a
  series that sunsets **2026-08-31**, i.e. a shipped default that stops answering
  three days from now, and `gpt-5-mini` is a legacy tier beside the current gpt-5.6
  line. A default is a dated fact about somebody else's catalogue; a test now
  refuses any `moonshot-v1*` id.
- Stop `brain.context` answering an id it does not hold. `brain.search` has
  short-circuited on a locator-shaped query since the exact-lookup pre-pass
  landed; `brain.context` never did, and a locator shares no lexical terms with
  any body, so the query fell through to a dense arm that always has a nearest
  neighbour to offer. Recorded `harmful` feedback names the case: the
  nonexistent, exactly well-formed `belief_ffffffffffffffff` came back as an
  unrelated fact, and it survived a vector rebuild because the shape of the
  failure has nothing to do with the vectors. Reproduced on a `mode=ro` copy of
  the live core: that locator returned **two** unrelated beliefs at cosine
  0.5603 and 0.6134. A locator-shaped query is now resolved by equality against
  the serving projection and a miss is empty. Over 100 well-formed absent belief
  ids on a copy of the live core: **17 of 100 returned an unrelated belief
  before, 0 of 100 after**. The same run in the other direction is the reason
  this is a fix and not just a refusal — asking for a belief *by its own id*
  resolved to that belief **0 of 100 times before, 99 of 100 after**, the
  hundredth being a `goal`, which is task state and has never been retrievable
  as knowledge. The visibility gate is the ranker's own: holding an id is not
  authorisation, so confidential material whose scope the caller did not name
  stays unreachable by locator. `looks_like_exact_locator` now has one
  definition, in `core_v1`, shared with the `brain.search` pre-pass — two copies
  is how the two surfaces came to disagree.
- Stop serving `confidence` and `confidence_band` in the context packet. The
  field is authored, not measured: 345 of the live core's 347 serving beliefs
  carry one, every one inside `[0.65, 1.0]`, and 116 of them are the same round
  0.85. Joined to recorded feedback it points the wrong way — moderate-band
  items drew 68 `irrelevant` and 0 `harmful` of 1,061 judged (6.41%), strong-band
  items drew 463 `irrelevant` and 23 `harmful` of 1,331 (36.51%). That per-item
  ratio is not identified, because an outcome is recorded per *retrieval* and
  one verdict tars every item in the packet; re-measured at the packet grain,
  one vote each over 470 judged retrievals, the direction survives intact:
  packets judged irrelevant or harmful held items averaging **0.8707**
  confidence, packets judged used or helpful **0.7263**. A reader weighting on
  that field was being steered toward the rows readers liked least. In its place
  the packet carries `evidence_count` and `evidence_latest_at` — how many
  evidence objects back the belief and when the newest was recorded. These are
  not offered as better predictors of usefulness; at the packet grain they
  barely separate the two verdict classes at all (1.06 vs 1.02 evidence objects,
  32.0 vs 34.4 days old). They are offered because they are facts about the
  record that a reader can go and check, which the authored score never was.
- Put the confidence term of `ranking_prior` behind
  `retrieval.confidence_prior_enabled`, **defaulting to on**, and report the
  flag on every packet. Whether that term should go, or `confidence` should be
  re-derived from evidence count, recency and verifier status, is an operator
  decision and not one a bug fix gets to make, so the default reproduces every
  packet built to date. Replaying the 200 most recent distinct recorded queries
  against a copy of the live core, switching it off moved the served set on
  **30 of 200** queries (31 items in, 31 out), reordered another 113, and changed
  the top-1 item on 13, out of 2,236 items served either way. Provenance in
  `docs/THRESHOLDS.md`.
- Golden context contract: no expected value in
  `tests/fixtures/golden_context_v1.json` changed. Served ids, `scope_mix`,
  `eligible_count` and `source_handle_count` are byte-identical on all eleven
  cases. `serialized_bytes` moved on all eleven — 1947→1977 and 1891→1921 for
  the single-item cases, 2823→2883 and 2944→3004 for the two-item cases, and
  +32/+33 on the six empty ones for the new `confidence_prior_enabled` ranking
  key — but that value is asserted against a re-serialization of the packet, not
  against a pinned literal, so it needed no edit. The contract was tightened
  rather than relaxed: the packet item key set is now pinned outright, and every
  case asserts the envelope contains no `confidence` field anywhere. Both new
  gates are mutation-proved, as are the locator short-circuit, its visibility
  gate, and the evidence-recency lookup.
Every live-corpus figure below is read from one `.backup` snapshot frozen at
**2026-08-28T19:28:58Z**, not from the running database. The corpus is live —
the hourly curator mints and two operator compactions ran the same afternoon —
and the retrieval count moved four times while these were being written, so a
number here without that instant attached would not reproduce by evening.

- Stop an empty retrieval being filed as a bad one. `retrieval_uses.outcome`
  carries both "the corpus had nothing for this query" and "the corpus served
  the wrong thing", and feedback is the only ranking signal the brain has. Of
  the snapshot's **2,048 retrievals (2026-07-15 to 2026-08-28), 1,086 (53.0%)
  served zero items** — genuinely nothing, not a logging gap, and stable across
  months (537/983 July, 549/1,065 August). **183 of those zero-item reads carry
  a relevance verdict anyway** (174 `irrelevant`, 5 `ignored`, 4 `used`), filed
  against the server's own written instruction not to file one; the agents' own
  notes say so verbatim ("No prior items were returned"). Restricted to the 962
  retrievals that did serve something, the picture inverts: **360 of 481 judged
  verdicts are positive (74.8%)**. An instruction 183 rows ignore is not a rule,
  so the server enforces it: `brain.feedback` refuses every relevance outcome on
  a zero-item receipt and says what to write instead, and the zero-item case is
  recorded as a new `no_coverage` outcome. That value is **server-derived, not
  caller-supplied** — the item count is observed in the same statement that
  writes the receipt, and the population a voluntary flag would describe is
  exactly the one that goes unreported when reporting is voluntary.
- Say plainly what that fix does *not* do. A zero-item receipt has no
  `retrieval_items` rows, so those 183 verdicts were never attributable to any
  belief: reclassifying all of them against the snapshot moved **0 of 191
  feedback boosts**, while blanking the 104 `irrelevant` verdicts that *did*
  have items moved 124 — the same instrument, reporting dirty. What the 183
  polluted is the corpus-level read of whether retrieved context helps, which is
  what a retirement rule keyed on outcome counts would have used. The existing
  rows are live history, so rewriting them is an operator's call and not a
  migration: `ocbrain feedback-repair` reports by default and rewrites under
  `--apply`, keeping the prior verdict in the row's note. Dry run on the
  snapshot: **183 candidates, 174 `irrelevant` / 5 `ignored` / 4 `used`, 0
  written**. The live database is unchanged.
- Stop retrieval and feedback history dying at every recompile. Each curator
  pass mints a new `belief_id`, and `retrieval_items` keeps pointing at the old
  one, so retirement eligibility measured a belief's *age* rather than its
  usefulness: **390 of 587 ever-retrieved ids are now retracted**, and **106 of
  303 serving beliefs had never been retrieved at all** — almost entirely
  recency, not rot, since the never-retrieved share by compile day runs 2/39 on
  08-04, 13/57 on 08-25, 34/45 on 08-28. A belief now ranks on its whole
  lineage's record. The lineage is *derived* from the `superseded_by` era pointer
  the projector already stamps on each predecessor, never copied forward at
  supersede time: a chain therefore accumulates by construction (generation
  three walks back through two to one), a copy cannot go stale because there is
  no copy, and because the walk yields a set of ids folded per
  `(belief, retrieval_use)` pair, one retrieval that served two members of a
  lineage is still one verdict. Reading `superseded_by` rather than the
  successor's `supersedes` is what makes the curator's key-collision cascade
  visible: **274 era closures against 64**. On the snapshot, **47 serving
  beliefs inherit 307 verdicts**, **18 of the 106 never-retrieved beliefs gain a
  judged record**, and beliefs carrying a feedback boost rise from **170 to
  191**, 45 of them changing value.
- Walk that lineage in Python off one read of the era pointers. Expressed as a
  recursive CTE, each step re-scans `current_beliefs` evaluating `json_extract`
  per row, because no index covers that expression: **85 ms per ranked
  retrieval, against 1.5 ms** for the same answer, on a path that runs on every
  `brain.context`. The walk is deliberately *unbounded* in depth, where the
  forward walk over the same pointer stops at `MAX_RESOLUTION_HOPS = 10`: that
  bound pays a belief read per hop and needs one answer, this one loads the
  pointer map once and needs every generation, and the snapshot's deepest
  serving lineage is **12 generations** (widest, 22 members; 112 of 303
  serving beliefs have more than one), so a ten-hop cap here would drop two
  generations of verdicts out of ranking today. Both walks now say so at their
  own constant, and a test pins the divergence.
- Do not let the instruction block promise what the open core does not do. The
  refusal above lives in the v1 feedback path, but the `initialize` instructions
  and the `brain.feedback` description were served to **every** connection — so
  a legacy v0 core described itself refusing zero-item feedback and recording
  `no_coverage`, and did neither: filing `irrelevant` on a zero-item legacy
  receipt was accepted, and the row read `irrelevant` afterwards. The guard does
  not port. A legacy `retrieval_uses.outcome` `CHECK` has no `no_coverage` value
  (the `UPDATE` raises `IntegrityError`), and a legacy receipt cannot prove a
  read served nothing: `brain.get` of a belief and `brain.digest` both write
  `knowledge_id` NULL with `served_ids_json` `[]` *having served an item*, so a
  served-count refusal there would refuse feedback on reads that did serve. Both
  texts are therefore chosen from the core actually open, and the legacy wording
  keeps the instruction it has always carried. Two siblings of the same defect
  fell out of sweeping for it: the legacy wording also named
  `coverage.feedback_needed`, a key the legacy `coverage` envelope has never
  emitted (7 keys, confirmed through `brain.context` on a legacy core), and the
  outcome vocabulary was spelled by hand in **three** places — the v1 validator,
  the legacy validator, and the `inputSchema` `enum` clients actually act on. All
  three now read `RELEVANCE_OUTCOMES`, and each is pinned by narrowing that tuple
  underneath the running server rather than by comparing one copy to another.
- Enforce the closeout's session identity at write time instead of asking for it
  in a docstring. `context.session` is the join that makes a closeout's tool-call
  trace minable, and on the live core **only 211 of 1,239 receipts (17.0%) carry
  a runtime-shaped id**: 431 are absent and 597 are a human typing something
  descriptive — `example_cleanup_audit`, `2026-07-22`, `2026-07-21 release
  checklist`, and one filesystem path in an id field. Guidance had six weeks and
  moved the rate from 15.1% in July to 19.6% in August, so it now lives in the
  server. What makes this a shape question rather than a taste question: **all 94
  closeouts that join a Claude Code transcript are UUIDs, and zero of the 597
  hand-written ids join one** — admitting exactly the machine-minted shapes keeps
  94 of 94 and costs none. (Every corpus figure in this entry comes from one
  read-only backup taken 2026-08-28 12:30:51 PDT; the live core gained rows while
  it was being written, and a set of numbers assembled from several reads of a
  moving table is not a measurement.) Three parts:
  - **The column is filled by the most trustworthy witness available**, in
    descending order: the harness-attested `client_session_hint` the server read
    from its own environment, then the caller's claim if it is runtime-shaped,
    then the server's own connection id under a `conn:` prefix so no later join
    can mistake it for a transcript. `session_id_source` records which, beside
    the value. Two runtime-shaped ids that disagree — the Claude Code subagent
    case — keep both, with `session_id_conflict` set.
  - **Omitting the field is legal**, and that is what makes the refusal
    satisfiable: a client with no session id of its own is never unable to file a
    closeout. Replaying the 77 post-capture live closeouts through the new path:
    **5 of 77 carried no identity before, 0 of 77 after.**
  - **The claim is never destroyed.** It stays verbatim in `context.session` in
    the receipt and is echoed as `session_id_claim` whenever the column stores
    something else.
- Give `context.runtime` an enum. It was free text and arrived as **160 distinct
  spellings across 1,239 closeouts** — five spellings of "local mac", four of
  "codex desktop", and `local macOS + analytics ClickHouse` (13 rows) with an
  environment description welded onto the client name. Nothing could be grouped
  by that column. A `runtime_family` of `claude-code`/`codex`/`cursor`/`hermes`/
  `mcp`/`cli`/`unknown` is derived at write time from the server-observed
  `client_runtime_key` first and the model's claim second; the raw spelling stays
  verbatim beside it, and `runtime_detail` is where the environment goes.
  Applied to the 160 historical spellings: **codex 426, unknown 438, hermes 123,
  mcp 95, claude-code 84, cursor 57, cli 16** — 1,239 rows, every one placed. The mapping is a pure function
  precisely because `task_closeouts` is append-only under a trigger and history
  can never be rewritten in place. `unknown` is a real answer, not a failure:
  "local", "desktop" and "macOS" name the machine, and inventing a client for
  them would be guessing. Install-specific labels go in `closeout.runtime_aliases`,
  which ships empty.
  - Matching is on whole segments, not substrings. A substring rule classified
    those 13 `ClickHouse` rows as the `cli` family, and 16 more besides, because
    "ClickHouse" contains "cli". A normaliser that guesses is worse than a column
    nobody can group.
- Make a closeout that is not a clean success say what did not work. The live
  status distribution is completed 1,052, partial 148, blocked 38, **failed 1** —
  one failure in six weeks of unattended agent work, which is a reporting gap, not
  a failure rate, and `brain.ledger` exists to surface exactly what is missing.
  Sampling the 148 `partial` receipts says where it went: they are overwhelmingly
  progress heartbeats ("Main acquisition is healthy at 129/284 receipts"), not
  partial failures. The evidence that is already in the corpus is elsewhere:
  **95 receipts claim `completed` while carrying a verifier that reports
  `failed`**, and not one of them has a field saying which check failed. A new
  required `unresolved` closes it, triggered by the status **or** by the verifier
  evidence independently — status alone would miss all 95. Over the live core
  that is **282 of 1,239 receipts (22.8%) that carry evidence something did not
  work and no field naming it; after, none of them can be written without one.**
  - `brain.ledger` reads it, which is what makes the charge honest. `unresolved`
    rides every `failed_attempts` row and the entry's `latest_unresolved`, and
    the briefing's FAILED line prints it in place of the summary — "what is still
    broken" rather than "what the session did" — falling back to the summary for
    every pre-gate row, which is all 1,239 of them. A required field no reader
    serves is a toll, not a gate.
  - Deliberately not a status override. Of the twelve closeouts whose verifiers
    all failed, **seven are read-only audits where the FAIL verdict is the
    deliverable** — "Read-only re-review found remaining blockers; verdict FAIL".
    Deriving `failed` from the evidence would relabel successful work as failure.
    The caller keeps the verdict and owes a sentence.
- Report both write-time refusals together. A caller with a hand-written session
  id *and* an unexplained failure used to learn about one, retry, and learn about
  the other; one retry now clears both.
- Both gates are configurable under a new `closeout` config section, and
  `session_id_policy: "quarantine"` keeps every receipt while still keeping junk
  out of the identity column. Replaying all 1,239 live closeouts through the
  shipped defaults refuses **747 (60.3%)** — 597 for the session shape and 150
  more for a missing `unresolved` — every one satisfiable on retry, and the number
  an operator should see before deciding which policy they want.
- Apply the same identity resolution to `retrieval_uses.session_id`, which
  carried the identical defect at four times the scale: of 2,048 read receipts,
  1,115 have a session id, **967 of them hand-written and joining nothing**, 148
  machine-shaped and 18 joining. Both writers — `core_v1.record_core_v1_retrieval`
  and the legacy `db.log_retrieval_use` — now resolve it under the `quarantine`
  policy, never `enforce`: a closeout is a write the agent chose and can retry,
  while this receipt is a side effect of a *read*, and refusing retrieval to fix
  a join is not a trade worth making. The claim is kept in `provenance_json` as
  `session_identity.session_id_claim`, so a resolver that replaces a value never
  destroys the one it replaced. A matching nullable `session_id_source` column is
  added to both cores.
- Reconcile the repo's three runtime folders instead of adding a fourth.
  `scripts/procmine/episodes.normalize_runtime` now asks
  `closeout.runtime_family` first and falls through to its own install-specific
  rules only when that abstains, so the two can differ by abstention but never by
  contradiction — asserted over the frozen live census rather than described in a
  comment. Two consequences: `ocbrain-runtime-call`, this repository's own
  runtime, is no longer `unknown` to its own normaliser (matched as an exact
  spelling, because the token `ocbrain` also appears in `local-agent-mode-ocbrain`
  on 66 rows); and the miner stops matching family tokens as substrings, the
  defect the write-time rules were written to avoid and then left standing next
  door. Measured across the 3,287 closeout and retrieval rows both have ever
  seen, the reconciliation moves 8 rows — the spelling `cli`, from `unknown` to
  `host-batch` — and regresses none.
- `procmine` labels a server-minted `conn:` session id `server_observed` rather
  than `model_reported`. It was inferring the answer from whether a
  `client_session_hint` was present; the write path records who filled the column,
  so the miner reads that and keeps the old heuristic only for the rows written
  before the column existed.
- Operator `runtime_aliases` keys are folded the way candidates are, so an entry
  written `"claude code desktop"` can actually match. The shipped table is empty,
  so nothing was broken — but a config key that can never fire is a trap with a
  config file in front of it.
- Pin the boundary between the two rankers this repo still carries. `core_v1.py`
  serves the live path — FTS5 bm25 with tuned column weights, a 1024-dim dense
  sidecar, weighted RRF at k=60, and a multiplicative
  scope, confidence, quality, recency and feedback prior. `retrieve.py` independently
  implements a retired blend: a flat `relevance * scope_weight * confidence *
  pinned * catalog_stub` product with a repo-FTS fallback. `mcp.py`, `cli.py`
  and `shared_context.py` all still import it, each behind an `is_core_v1` early
  return, and **nothing enforced that**. A single `retrieve(...)` added outside
  one of those guards would rank live beliefs by the legacy product, and all 907
  tests would still pass — the two formulas would just quietly average.
  Tracing `sys.settrace` over `retrieve.py` while dispatching the complete
  advertised admin surface against `pre-compaction-20260828-claude.sqlite`, a
  frozen 208,285,696-byte copy of the live core holding 1,247 current-belief
  rows and 6,492 evidence objects: **0 of its 22 top-level functions run**. Those
  two counts describe that file and nothing else — the live core reads 1,258 /
  6,651 at 19:20Z the same day, because it is compacted and re-minted hourly.
  Two drivers now assert that zero, one over `call_tool` and one over the
  read-side CLI, and each carries its own positive control on a legacy core
  where the count **must** be non-zero (it runs 9 distinct functions), so
  blinding either driver fails rather than passing on an empty set. A static
  scanner freezes the call sites at `{mcp.py: 2, cli.py: 1,
  shared_context.py: 1}`, because a dynamic driver can only see call sites it
  dispatches. Planting one v1-reachable `retrieve(...)` call in `call_tool`
  above the `is_core_v1` return: **both gates fail**; removed, green.
- Fix the containment gate's own two blind spots, found by adversarial review of
  the commit above. Neither was hypothetical; both were demonstrated with a live
  legacy call executing on a v1 core while the whole suite stayed green.
  **The static scanner was an allow-list by import form.** It skipped any module
  without a line *beginning* `from ocbrain.retrieve import retrieve`, and its
  regex excluded aliased and attribute-qualified calls — so a function-local
  import, a relative import, an alias, or `ocbrain.retrieve.retrieve(...)` was
  invisible, and so was the parenthesized multi-line form this repo already uses
  elsewhere. Against nine planted call sites written in nine legal binding
  forms, it found **0 of 9**. It is now an AST binding resolver with no
  import-form filter, counting every *reference* to the bound name rather than
  only a direct `retrieve(` — and the scanner's evasive-binding self-test plants
  all nine and requires all nine. Mutating the resolver back three ways
  (top-level-only import scan, dropped attribute branch, calls-only counting)
  fails that test each time.
  **The dynamic gate had zero coverage of `cli.py`.** The driver dispatched
  `mcp.call_tool` only, so the CLI trace cited as evidence was a one-off
  analysis nothing re-ran — and `cmd_preview` is where the reviewer's planted
  call lived. `READ_SIDE_CLI` now drives `status`, `briefing`, `digest`,
  `search` and `preview` (with `--project` and `--cross-scope`) under the same
  tracer, with the same legacy-core positive control; deleting `preview` from
  that table fails the control.
  Both of the reviewer's probes — a shim with a function-local import, and the
  same shim with a relative one — now fail the static gate, and the first also
  fails the new CLI gate.
- Give `build_context` the guard its docstring was borrowing. It called the
  legacy ranker unconditionally; the only `is_core_v1` check was one frame up in
  `mcp.py`. That made the containment a property of today's call graph rather
  than of the function — a second caller would have opened a live unguarded path
  without changing the frozen call-site table, so neither gate would have seen
  it. `build_context(v1_conn, ...)` did not raise; it returned a packet ranked by
  the retired blend, because `current_beliefs` is a table name both schemas
  share and nothing else fails. It now refuses a v1 core and points at
  `build_context_v1`. `retrieve.py`'s other three call sites were re-read: all
  three already refuse in their own frame.
- Pin the tracer to the module under test. `_RETRIEVE_SOURCE` was built from the
  repo layout, but under a bare interpreter `ocbrain` resolves through the
  editable install instead of the checkout, no frame matches, and every count in
  the gate reads a clean zero for the wrong reason — the reviewer hit exactly
  this and caught it only because the positive control was empty too. It is now
  derived from the imported module and asserted to be this worktree's file.
- Nothing in the legacy dual path was deleted, and the reason is a measurement.
  **Every** core database in `~/.ocbrain/backups/` is v1-shaped —
  `schema_meta.core_schema = ocbrain.core.v1`, zero of `db.py`'s
  `evidence`/`knowledge` tables — including the ones whose names imply otherwise
  (`pre-v2-20260825`, `pre-v22-20260825`, `pre-brainfix-20260804`,
  `ocbrain-sparse-wiki-20260722-1115`, and the oldest, `ocbrain-2026-07-15`).
  The claim is "all of them", not a count: that directory gains a file per
  compaction, and re-running the sweep read-only over all 24 core files present
  at 19:20Z on 2026-08-28 returned 24 v1-shaped, 0 legacy-shaped. The one
  non-core file, `ops-2026-07-17-pre-v1.1.0.sqlite`, does come back
  legacy-shaped, which is the sweep's positive control.
  No legacy-shaped core exists on this install. But `v1_migration.py` is still
  reached by the shipped `ocbrain core-migrate-v1` command and is the only
  reader for a *downstream* legacy archive, and `db.py` exports `connect`,
  `now_iso`, `DEFAULT_DB_PATH`, `DB_BUSY_TIMEOUT_MS` and `PUBLIC_SCOPES` to
  eleven other modules — its legacy `init_db` already refuses a v1 core by
  design, which is why the live core has none of its tables.
  And `retrieve.py` is imported at module scope by `cli.py`, `mcp.py` and
  `shared_context.py`, so removing the file does not fail 25 tests — it takes
  the suite down at collection: **39 collection errors, 0 tests run**. A change
  that can only go green by deleting the tests that would have judged it is not
  verified. Whether to drop legacy-core support is a policy call, not a
  dead-code call; the containment gate above is what makes taking it later
  cheap.

- Stop the pending supersede ledger growing without bound. The first unattended
  night gave it producers and no consumer: the per-caller rate cap
  (`supersede.direct_cap`, default 8/24h) is sized for a runtime agent, so past
  its eighth correction the scheduled curator pended everything — and because a
  *proposal* does not change the input that produced it, the next hourly cycle
  re-derived the same claims and pended them again. The live core reached **312
  undecided proposals against 33 beliefs in nineteen hours**, one pair carrying
  twelve identical copies, growing at roughly 17/hour with nothing to stop it.
  Two changes close it, and it takes both:
  - **Dedup at proposal time.** A supersession the ledger already carries
    undecided is not minted twice. The successor id is content-and-scope
    addressed, so an identical re-derivation is an exact `(target, successor)`
    match and writes nothing at all — not the proposal, not the rationale
    evidence row — while a different replacement body for the same belief is a
    different pair and still mints, because two people disagreeing about one
    fact is something an operator has to see. Reported as `pending_deduped`.
  - **Curator direct authority** for an ordinary belief — unpinned, outside
    `global:*` — under `supersede.curator_direct` (default on). Dedup alone
    collapses the backlog but does not stop the growth: the model rewords the
    same fact between cycles, so ~4 genuinely-new pairs per hour survive pair
    matching. Landing the supersession is what ends the loop, because the next
    cycle then sees its own claim already serving. The agent-facing cap is
    untouched, and the margin rule and digest gate still bound the curator.
  Replaying two consecutive cycles against a copy of the live core: **+24
  proposals per cycle before, +0 after**.
- Stop the curator ratcheting the corpus toward 0.7. `min(old, claim, 0.7)` is
  right for a contested correction — a replacement must not gain authority by
  replacing — and wrong for the curator restating its own fact from better
  evidence. Approving the live core's 33 pending proposals as-proposed would
  have **dropped confidence on 30 of them, mean −0.09**, every one landing on
  0.65 or 0.70; run hourly that walks the whole corpus down. A curator-authored
  supersession that keeps the predecessor's `key` now inherits the
  predecessor's confidence instead. It is no-gain as well as no-loss, and
  cross-key curator supersessions and every agent-issued supersession keep the
  ceiling. Replayed over the live backlog: 24 facts replaced, **0 dropped, 24
  held, mean +0.0000**.
- Report the pending supersede queue as distinct targets with the raw proposal
  count beside it (`33 distinct (312 proposals)`, previously a bare `312`). That
  number was the operator's only window onto this queue and it read as ordinary
  backlog while a loop grew it without bound. A metric that can hide unbounded
  growth is worse than no metric.
- Give "undecided proposal" one definition. The proposal listing, the queue
  depth, the selftest metric and the new dedup guard now share a single SQL
  predicate, so a dedup guard cannot drift away from the count that is supposed
  to prove it works.

- Stop the curator from silently overwriting a fact it has already compiled. A
  claim on a key the corpus already served used to be an update in place: the
  body was replaced and the confidence overwritten with whatever the hosted
  model returned, with no event marking the fact as changed and nothing saying
  what it used to say. On the live corpus that had happened **80 times across 50
  distinct beliefs**, out of 324 approved curator proposals — the single largest
  source of pollution in the brain, run hourly, unattended. A key collision
  carrying a different statement now routes through the same supersession
  transaction `brain.supersede` uses: the old copy is era-closed with
  `superseded_by`, the replacement keeps the key under its own id, the
  confidence is capped at `min(old, claim, 0.7)` instead of taken on trust, and
  a paired `correction_recorded` says the fact changed. Replaying the 43
  recoverable collisions on a copy of that corpus: **43 in-place overwrites
  before, 0 after** — 8 supersessions landed and 35 were deferred to the pending
  ledger, because the transaction is shared and so is its routing.
- Give new-key claims a cheap-then-escalate cascade instead of a single
  similarity test. A cosine pre-filter over the local sidecar ends it for the
  overwhelming majority of claims, which are about something nothing else in the
  corpus mentions; above the floor, `is_restatement` separates an elaboration
  (updates in place, unchanged) from a claim that is about the same subject and
  says something else. Only that remainder escalates. With no vector sidecar the
  cascade stands down silently and the corpus keeps its previous behaviour, the
  same way retrieval degrades to lexical.
- Let the curator model adjudicate the ambiguous tail by **index selection**, in
  the hosted call that already runs. `CLAIMS_SCHEMA` gains an optional
  `conflicts_with: [{index, resolution}]` selecting rows out of the advisory
  existing-beliefs list the prompt already carried, now explicitly numbered.
  `validate_claims` range-checks every index against the list actually supplied,
  so an invented index produces no action — and, unlike an invented citation,
  does not cost the claim, which is a separate judgement. `supersede` runs the
  transaction; `coexist` writes `attributes.contradicts` on **both** beliefs and
  leaves both serving. Free-form contradiction generation is the first thing to
  collapse on a small model, and this curator is meant to be able to run on one.
- Populate `contradictions[]` for the first time. The packet builder has read
  `attributes.contradicts` since the packet existed and nothing had ever written
  it, so every packet the brain has ever served carried an empty array while
  handing the reader two answers to one question. The coexist path is its
  writer. Zero beliefs in the live corpus carried the attribute before this.
- Never let recency alone resolve a conflict, and never let stale evidence
  resurrect a correction. A claim more than 0.05 below the confidence of the
  belief it would retire is **deferred** — recorded as an undecided proposal in
  the pending ledger, with the standing belief still serving — rather than
  enacted. A claim whose newest supporting evidence predates the newest content
  correction on its target is **blocked**: a scheduled curator reads a window of
  evidence rather than a diff, so without this the Wednesday run quietly
  restores what a human corrected on Tuesday. `annotate` is deliberately not a
  content correction, or the coexist path would block itself next cycle.
- Add `correction` and `gotcha` to `ELIGIBLE_KINDS`, closing a write-only hole:
  agents could already ingest both kinds and the curator could never read one,
  so they were hash-chained and invisible forever. This is also what closes the
  supersession loop — the rationale evidence a supersession records comes back
  round for re-confirmation or challenge on the next cycle, on any install whose
  `curator.egress_policies` admits the scope it was written in.
- `apply_claims` returns `superseded`, `coexist_marked`, and `deferred` beside
  `applied`/`unchanged`/`blocked`, and `wiki-curator.py` reports all six per
  project and in the rollup. A cycle that corrects the corpus has to be legible
  as one.
- Add `brain.supersede`: correct a belief by replacing it, not by deleting it.
  The ledger held 719 correction events and 718 of them were retracts, because
  retracting was the only correction the store could express. All 11 corrections
  an agent had ever issued took the same shape — soft-retract the wrong belief,
  then type the replacement fact into the correction's `body`, a column nothing
  indexes and nothing serves. So correcting the brain *subtracted* from it: the
  next agent asked the same question and got the same wrong answer, or nothing.
  One runtime call now retires the old belief, closes its era with `valid_until`,
  compiles the replacement in the **same scope**, and leaves the old id pointing
  forward, in one transaction with one commit. Nothing is deleted — the retired
  belief keeps its body, evidence, feedback, and retrieval history; only its
  service stops.
- Bound that authority in the primitive rather than behind a profile gate. The
  replacement's scope is copied from the superseded belief byte-for-byte, so a
  supersession can never widen reach; confidence is capped at
  `min(old, 0.7)`, so a claim does not gain authority by arriving later (the
  deliberate rejection of recency-always-wins); a replacement that restates the
  original after whitespace and case folding is refused; and each caller has a
  bounded number of direct supersessions per 24 hours, matched on the
  harness-attested session hint. Doctrine (`global:*`), pinned targets, and
  rate-cap overflow are **routed, never refused**: they become an undecided
  `compilation_proposed` carrying `attributes.supersedes`, which is the pending
  ledger — no new table, no new status. `brain.proposal_decide` completes the
  pair atomically, recording the deciding admin as author and preserving the
  requesting agent as `requested_by`; a rejection changes nothing and leaves the
  agent's rationale in the corpus as curatable evidence. `brain.digest` reports
  the queue depth as `pending_corrections`. `OCBRAIN_SUPERSEDE_TIER=pending_all`
  routes every supersession to review; `OCBRAIN_SUPERSEDE_DIRECT_CAP` (default 8)
  sets the cap. Both code paths exist under either setting — the flag only picks
  the routing predicate.
- Add two correction ops the projector understands. `supersede` sets
  `attributes.superseded_by`, stamps `valid_until` from the **event's own
  timestamp** (not the wall clock, so a replay reproduces it), and retires the
  belief; removal from the FTS index is free, since `_write_belief` already
  deletes the search row in the same statement that writes a non-serving belief.
  `annotate` merges an `attributes_patch` and touches nothing else — no status,
  serve, body, or confidence — with a `null` value deleting its key, so a mined
  statistic is republished by recompute-and-replace rather than incremented in
  place. `annotate` is the writer `attributes.contradicts` never had.
  **Caveat:** an *older* binary doing a full reprojection treats `op="supersede"`
  as an unknown no-op and would resurrect every superseded belief. Roll the code
  back and the corpus is fine until something calls `project_core_v1(full=True)`;
  do not run an old binary's full rebuild against a core that has superseded
  beliefs in it.
- Record who issued a correction. `correct_v1` now carries the connection's
  `Provenance` and writes it into the event body along with the harness-attested
  session id, which every one of the 719 existing correction events lacks
  (`session_id` NULL on all of them, `author` defaulting to the literal string
  "human"). The most consequential events in the ledger were the least
  attributable.
- Stop unpinning beliefs on every recompilation. `_project_compilation_decision`
  hardcoded `pinned=False`, so an approved proposal silently dropped whatever an
  operator had pinned — and a scheduled curator recompiles constantly. That is
  why one real corpus held exactly one pinned belief. The stored value is now
  carried forward, which is deterministic under replay because the `pin`
  correction that set it is an earlier event.
- Refuse to restore a belief whose successor is currently serving. A restore
  that put a superseded fact back beside its replacement would serve both halves
  of a conflict the ledger has already resolved. Once the successor is itself
  retired, the original is restorable again — the block is about serving a
  contradiction, not about permanence. Joins `tombstoned` and `hard-corrected`
  as the third `_restore_blocked` reason.
- `ocbrain hygiene --supersede` retires the old belief immediately instead of
  stamping an attribute and waiting for the next `expired` sweep. Between the
  two, the corpus served the fact an operator had just declared wrong alongside
  its replacement — up to a day on the scheduled cadence. The CLI interface is
  unchanged, and the `expired` class still retires anything carrying
  `superseded_by` from before this release.
- `brain.get` gains `mode`. `resolve` (the default) follows `superseded_by`
  forward to whatever serves now, reporting `resolved_from` and
  `resolution_hops`, so an agent holding an id from an old transcript gets the
  current fact instead of a refusal; the walk is cycle-guarded and bounded at 10
  hops. `as_stored` returns the retired belief itself, labelled `invalidated`
  with its `valid_from`/`valid_until` era, which is how drift is measured. A
  retracted belief with **no** successor stays refused in both modes: filtering
  invalidated facts by default is the point, not an oversight.
- Flag conflicting beliefs at serve time. `contradictions` has always been in the
  context packet and has always been empty, because `attributes.contradicts` had
  no writer. A packet-local advisory pass now flags two same-`attributes.key`
  beliefs as `duplicate_key` (the key is a wiki fact's identity, so two of them
  in one packet is two answers to one question) and near-identical pairs as
  `embedding_similarity` at cosine >= 0.90 from the local vector sidecar. Bounded
  by the packet and not the corpus — at most 12 items is at most 66 comparisons —
  it makes no network call, and it stands down silently when the sidecar is
  missing, stale, or unreadable.
- Add the `supersede` config section (`tier`, `direct_cap`), bringing the
  surface to five sections.

- Store transcript evidence by reference instead of by value. Measured on a
  177 MB core: `*_history_file` bodies were 53.3 MB (99.13% of all evidence body
  bytes), their `evidence_recorded` events 73.9 MB (82.4% of the ledger), and
  the beliefs compiled from them another 20.9 MB across `current_beliefs` and
  `compilation_proposed` — 148 MB of 177 MB, growing ~106 MB a month. Import now
  emits `body: ""` plus a `body_ref` naming the source file and the hashes that
  rebuild its window, a 2,000-character head excerpt in `evidence_objects
  .body_head`, and the `source_content_hash` that was empty on every history row
  ever written. Replaying the same corpus under the new scheme: 19.9 MB, and
  ~14 MB a month. **New events only — no migration**; existing rows keep their
  bodies. `brain.source` re-reads the file, rebuilds the window, and verifies it
  against the recorded hash before serving a byte; a transcript that has grown,
  rotated, or been deleted (12.4% of recorded source URIs already dangle) comes
  back as a typed `content_unavailable` with a reason and the recorded head
  excerpt, never as a raised tool error. The evidence id is still derived from
  the full window text, so no identity moves. The belief compiled from a
  transcript now carries the head rather than the whole window — required, not
  incidental: a re-windowed transcript has to compare unchanged, and it cannot
  compare against a body the store no longer holds. The import-time re-window
  gate reads `COALESCE(NULLIF(body, ''), body_head)`, and bundle export sends
  the head excerpt instead of refusing a pointer row as `invalid_body_size`.
- Capture caller identity at the server rather than accepting the model's word
  for it. Runtime and session were free text a model typed into `context`: 129
  distinct `served_to_runtime` spellings, no session id at all on 44.7% of
  retrievals and 37.4% of closeouts, and 13 closeouts that identity-join to a
  transcript. `initialize` now mints a per-connection UUID and reads the MCP
  child's own environment, and that value is threaded through `tools/call` and
  `resources/read` — which never received the session state at all — down to
  every write path. Three identities are stored separately, because only one of
  them can be trusted and the row should say which: `server_connection_id`
  (server-minted, authoritative, names a connection and not a conversation),
  `client_session_hint` (`OCBRAIN_SESSION_ID` or `CLAUDE_CODE_SESSION_ID`;
  harness-attested and never server-verified — its stability across `/resume`,
  `/clear` and compaction is unverified, and a Claude Code subagent inherits its
  parent's value), and `client_runtime_key` (`OCBRAIN_CLIENT`, else `AI_AGENT`,
  else the handshake `clientInfo.name`, with the winning source recorded). The
  model-supplied `context.session` and `context.runtime` keep their columns and
  are now recorded verbatim: `canonical_runtime` and the
  `context_json["runtime_raw"]` side-channel are gone from the write path, and
  the folder moves to `scripts/procmine/runtimes.py` where the historical corpus
  still needs it. Provenance never joins `ScopeContext`, whose `to_dict()` feeds
  the retrieval `stable_id`. Columns are additive and nullable, applied wherever
  a core is opened including `serve()`; `task_closeouts` is append-only, so
  there is no backfill and existing rows are untouched. Two connections writing
  a byte-identical closeout now produce two receipts instead of colliding on the
  UNIQUE `content_hash`.

- Delete the code that never ran: 31,600 lines out, with nothing that serves
  changing behaviour. `packages/ops` and `packages/training` are gone; every
  table in their own stores (`autopilot_runs`, `judge_runs`, `embed_runs`,
  `signal_events`, `harvest_watermarks`, `loop_liveness`, `family_scores`,
  `stall_pages`, `watchdog_findings`) held zero rows, so OCBrain is now one
  distribution with one console script and no optional companion mechanism.
  Those stores also declared an `egress_audits` table, which is a different
  table from the core's `egress_audits`: the core one is written by
  `src/ocbrain/egress.py` on every applied curation run, holds 30 rows in the
  live brain, and is untouched here. Their one live module, the public-safety scanner, moved
  first: it is `src/ocbrain/publicsafety.py`, and `ocbrain public-safety-check`
  and `ocbrain install-hooks` are ordinary core subcommands, so the CI gate that
  keeps private paths out of this repo no longer depends on a companion install.
  Also gone: auto-compile (it produced 239 beliefs and all 239 were retracted);
  the `ocbrain deslop` sweep, judged pass, repair, doctrine install and volume
  eviction (155 consecutive hourly runs reported `actionable: 0`); hygiene's
  `unused` and `unhelpful` classes and the feedback-watermark subsystem that
  existed only to make `unhelpful` safe (neither class ever selected a belief,
  and no operator ever set a watermark); `brain.mark_stale`, which
  `tools_for_profile` could never return; and `_build_legacy_parser`, 682 lines
  of v0 argument parsing no caller reached.
- Keep the deslop rules, drop the sweep. The mechanical rules were never the
  dead part: they fire as `curator.validate_claims`' write-time gate, which
  rejected 34 unverified quotes, 8 fused claims and 7 temporal-in-durable claims
  before they were stored. `find_slop`, `RULES` and `ENFORCED_RULE_IDS` stay,
  along with `closeout_v1`'s `slop_findings` receipt and `wiki-lint.py`.
  `rewindowed_evidence_id` stays too: volume mode used it, but so does
  `import_source_v1`, which is the live transcript-import dedupe.
- Remove four vocabulary values the ledger proves were never used: the
  `personal_finance` scope type and the `public` visibility (0 rows each),
  `reward_band` (written 1,325 times, read by nothing in the v1 core), and
  `evidence_objects.verifier_status` (all 3,987 rows read `unknown`). The
  `session` scope type stays despite a plan to remove it — one stored
  `evidence_recorded` event carries it, and dropping it from `SCOPE_TYPES` would
  make `ScopeTag` refuse that event and break projection replay.
- Cut the config surface from 115 fields to 20. Only `retrieval`, `scopes`,
  `curator` and `deslop` are ever read; thirteen other sections configured
  deleted code or nothing at all. A section or key an operator's file names but
  this build does not define is now explicitly skipped rather than fatal, which
  is what lets a config written against the old surface keep working.
- Curate every configured project scope, not one pinned project. The curator
  compiled whichever single project the promote script pinned, which on one real
  brain reached 5 of 574 curator-eligible objects while 535 closeout summaries
  sat in ~40 other project scopes; every hourly run for six days logged
  `unchanged_no_api_call` against a wiki that had frozen. `curator.projects` now
  names the scopes to compile, and on that corpus three configured scopes reach
  509 of 568 eligible rows. The loop lives inside `wiki-curator.py` rather than
  in the caller because `state.json` is one file per wiki directory: a shell loop
  over `--project` would overwrite the previous project's digest and make every
  cycle re-bill a hosted call for every project. Each project carries its own
  digest under `projects`, a legacy flat `input_digest` migrates onto the
  `workspace` entry so an existing install keeps its short-circuit, and a project
  with fewer than `curator.min_evidence_per_project` eligible objects is reported
  as `skipped_thin_project` instead of billed. One JSON line per project plus a
  roll-up; the wiki is materialized once, after the loop.
- Keep a doctrine fact at doctrine scope when a curator run rewords it. An
  approved proposal writes its scope onto the belief, so a claim the curator
  typed as project-scoped demoted the `global:doctrine` fact it restated. Only a
  `scope_promoted` event with a named approver may move a belief between tiers;
  a rewording now carries the matched belief's own scope through unchanged.
- Add `ocbrain scope-promote`, the missing emitter for the `scope_promoted`
  event. The kind, its projection, and its rebuild path all shipped; nothing ever
  wrote one, which is why a real brain holds zero global beliefs. `--approved-by`
  is required — widening a belief's audience is a human decision the ledger has
  to be able to name — and `--select-durable-preferences` picks the current,
  served workspace `wiki_fact` rows whose lifecycle is `durable` and whose
  category is preference/decision/workflow/system. A promotion widens reach and
  never egress: each belief carries its own visibility and egress policy through,
  so a `local_only` belief promoted to `global:doctrine` becomes recallable from
  every project on this machine and is still refused for hosted delivery.
- Stamp curator claim scope mechanically instead of always using the running
  project. A durable `preference` claim is doctrine — as true in one project as
  the next — and stamping it into `project:<whatever ran the curator>` is what
  leaves a workspace scope nobody else can reach. The model never chooses this;
  it supplies a category and lifecycle that are already range-checked, and
  letting it name a scope would make the visibility boundary an injection
  target. The dedup lookup now also searches `global:doctrine`, so a promoted
  fact is updated rather than re-minted once per project.
- Retry a retrieval across scopes when the scoped pass returns nothing at all,
  and declare it in `coverage.scope_fallback`. `cross_scope` is an opt-in almost
  no caller sends, so a question the brain could answer from a neighbouring
  project abstained on a technicality instead. The retry is reach only: it is the
  same primitive with the same dense floors, multi-term lexical bar, redundancy
  filter, and dedup, so a question nothing answers still comes back empty. There
  is no merge — a scoped pass that returned anything is returned untouched, and a
  cross-scope item carries scope_weight 0.15 against a scoped item's 1.25, so it
  could not displace a scoped hit even if there were. The top-level `cross_scope`
  keeps reporting what the caller asked for. `retrieval.scope_fallback_enabled`
  turns it off for strict isolation.
- Match scopes by canonical spelling instead of exact string equality. Callers
  name their own scope, so one project arrived as `coframe-brain`,
  `coframe_brain__v2`, `Coframe Brain`, and five other spellings, each of which
  reached nothing. `project`, `repo`, and `client` are now folded (lowercased,
  separator runs collapsed to `-`) once in `ScopeContext.__post_init__`, which
  every entry point already goes through, and an operator `scopes.aliases` table
  maps the remaining genuinely-different names onto one id. Stored rows are never
  rewritten: the search prefilter widens its `IN (...)` list to the stored
  spellings that resolve to a scope the caller already names, and `scope_match`
  folds the same way so a row the SQL admitted is not zeroed by the ranker.
  Task and session ids stay trim-only — they are machine-minted and
  high-cardinality, so folding them risks collapsing two distinct ids into one —
  and a path-shaped component is left alone because `repo` is routinely a
  filesystem path that `Path(...).resolve()` has to still find. `ScopeTag` itself
  never folds: it runs during projection replay and over stored handle scopes,
  where an alias-dependent rewrite would make a ledger refold depend on today's
  config. The alias table ships empty, so a fresh install behaves exactly as
  before, and an alias may rename a scope but never re-type one — promotion to
  `global` stays a `scope_promoted` event with a named approver.

- Preserve the last curator input digest when a non-curator wiki rematerialization
  replaces `state.json`. The promote script rematerializes after every run; losing
  that digest made an unchanged next cycle call the hosted model again instead of
  taking the promised free no-op path.
- Stop the curator minting a second belief when a later run restates a fact it
  already holds. A belief is keyed by the topic name the model chose, so a
  reworded claim under a new key created a duplicate and exact-body dedup never
  saw it — every scheduled run added another phrasing. One real brain reached 44
  served beliefs carrying 33 distinct facts, and each copy costs a result slot.
  A claim that restates a served fact now updates that belief instead.
- Add a `redundant` hygiene class that retires older restatements, keeping the
  newest wiki fact in each exact delivery scope. Pinned beliefs, non-wiki facts,
  and equivalent text in different scope/visibility/egress boundaries are never
  collapsed. The similarity threshold is configurable (`--restatement-threshold`,
  default 0.80) because it runs unattended: under-retiring leaves a little
  redundancy, over-retiring loses knowledge.
- Add `body_similarity` / `is_restatement` to `ocbrain.text`. Token-set overlap is
  deliberately crude — deterministic, dependency-free, explainable — since the
  decision it feeds is a soft, reversible retirement.
- Make `doctor` and the optional ops stall checker use the same durable config
  resolver as the core, so the migrated user config is permission-checked and
  optional operations do not silently fall back to the checkout path.
- Resolve configuration from `~/.ocbrain/ocbrain.config.json` before the
  checkout-relative `data/ocbrain.config.json`. The old default made resolution
  depend on the working directory, let a `git clean -xfd` or fresh clone silently
  discard operator settings, and let a test suite inherit whatever a checkout
  happened to have — a curator egress-boundary test passed in CI, which has no
  such file, and failed on a machine that did. A brain that loses its curator
  policy that way keeps exiting 0 while promoting nothing. The legacy path is
  still honored when it exists and the new one does not; `brain-sync.sh` and
  `brain-promote.sh` no longer pin the config into the checkout.
- Add `ocbrain config`, which prints the effective configuration, the file it
  resolved, and whether each value came from a default, the file, or the
  environment. `--section` and `--changed-only` narrow it. A layered config is
  only usable if you can see which layer won.
- Raise the enforced ruff selection to `E, F, I, UP, B, C4, RET, PIE`, all at zero
  across the repository, and add `scripts/code-quality.sh` plus
  `docs/CODE_QUALITY.md` for the advisory layer (complexity, maintainability,
  duplication, slop patterns). `C90` and `S` stay out of the gate: both carry a
  large standing count that is not incrementally fixable, and a gate nobody can
  pass is a gate people learn to skip.
- Replace a best-effort rollback's bare `except: pass` with
  `contextlib.suppress` and a note on why masking it is deliberate; give the
  tri-state bool-to-SQLite conversion a named helper instead of a nested
  conditional; name the three branches of the belief-provenance filter.

- Use OpenAI's `max_completion_tokens` field for curator requests while keeping
  Moonshot on its compatible endpoint's `max_tokens` field. The live
  `gpt-5-mini` endpoint rejects the legacy field.
- Make the curator's evidence eligibility an operator declaration instead of a
  hardcoded rule. `curator.egress_policies` ships as `hosted_ok` only, so a fresh
  install still sends nothing it was not given, but a brain whose evidence is all
  `local_only` — the default for anything written through a client — can now let
  its own curator read its own notes. `prohibited` egress and `secret` visibility
  are refused in code and cannot be configured on; a policy admitting nothing is
  an error rather than a silent empty selection.
  Without this the curator was unrunnable on a real brain: 0 of 2,545 evidence
  objects qualified, so the promote loop reported success while promoting nothing.
- Record an `egress_audits` row on every applied curation run, before the send,
  naming each evidence id, kind, scope, policy and size plus a payload hash. That
  table existed and had never been written to. Widening what the curator may read
  is only defensible if every send is accountable afterwards, and the audit
  deliberately stores identity and size rather than the bodies themselves.
- Always record a closeout summary as evidence, and gate only the promotion step
  on `automatic_activation`. Both used to sit behind that flag, which conflated
  *recording evidence* with *promoting it to a served belief* — so turning the
  flag off to stop unattended promotion also stopped closeout summaries becoming
  evidence at all. Closeout summaries are the single largest supply of
  curator-eligible evidence; one real brain silently lost 567 of 799 closeouts
  (71%) this way over two weeks, which no amount of fixing the curator would have
  recovered. The receipt now returns `evidence_id`.
- Give the lexical retrieval arm a relevance floor. The dense-quality gate was
  guarded by "not already found lexically", so any FTS hit bypassed it entirely,
  and the redundancy filter only ran when more than one lexical row came back. A
  long, specific query sharing one generic token with an unrelated belief
  therefore served that belief — and because the lexical arm scores by unweighted
  RRF while the dense arm is scaled by similarity, the filler outranked
  well-matched dense results. A lexical hit is now held to the same dense floor
  when the dense arm is healthy, unless the query names the belief outright or
  quotes its body; the multi-term bar applies to a lone row; and when nothing
  clears that bar the rows are dropped only if dense retrieval can answer
  instead, so a stale sidecar still degrades to lexical rather than to silence.
- Make the retrieval gates configurable through the existing
  `OCBRAIN_<SECTION>_<FIELD>` mechanism. They were hardcoded constants, so tuning
  serving precision meant editing source. Defaults are unchanged.
- Widen the retrieval-feedback boost from a range that could never reorder
  adjacent results to a damped ±0.25, and attribute alias-recorded feedback to
  the canonical belief instead of silently dropping it.
- Add `ocbrain hygiene`: retire beliefs that expired, were never once retrieved,
  or are consistently judged unhelpful. Retirement is a soft retraction, never
  touches pinned or curated wiki facts outside the unambiguous class, is bounded
  by a batch cap that reports its remainder, and refuses to act on feedback
  gathered before a watermark — verdicts collected while a ranker was mis-serving
  a belief describe the ranker, not the belief.
- Add a `restore` correction op and `ocbrain hygiene --restore`, the inverse of a
  soft retraction, so an unattended sweep is undoable. Tombstoned and
  hard-corrected beliefs stay terminal: the projector refuses to honour a restore
  for them even if the event is forged, under incremental folding and full
  rebuild alike.
- Make the wiki curator provider-pluggable (Anthropic by default, plus OpenAI and
  Moonshot) with the same evidence gates and local verbatim-quote validation
  whichever model runs. The Anthropic SDK is a lazy import behind a new `curator`
  extra, so the core package stays dependency-free.
  `scripts/kimi-wiki-curator.py` is now a forwarding shim.
- Stamp `valid_until` on `current`-lifecycle wiki facts. The wiki's freshness
  markers previously had readers and no writer, so nothing ever aged out and two
  lint checks were unreachable against live data.
- Give `wiki-lint.py` a `--rematerialize` actuator so a detector run can repair
  drifted pages and orphans instead of only reporting them.
- Add `scripts/brain-promote.sh` and `docs/SCHEDULED_MAINTENANCE.md`. The
  harvester only records evidence, so a brain running it alone accumulates
  knowledge no retrieval can return; this is the other half of the loop. Still
  not installed by default — OCBrain ships no scheduler.
- Canonicalize `served_to_runtime` at write time, keeping the operator's exact
  string in the retrieval context. One deployment held over 100 spellings of
  about eight clients, which made per-client analysis and feedback aggregation
  impossible.
- Stop `brain.ingest` and `brain.closeout` failing when an auto-recompile is
  blocked. Auto-belief ids are content-addressed, so re-ingesting text matching a
  hard-retracted belief raised out of the write and lost the evidence too.
- Stop duplicating the evidence body into the projection's metadata. The text was
  already in the same row and authoritatively in the event ledger; the third copy
  was ~23% of one real 125MB core. Bundle provenance in that metadata is kept.
- Bound the legacy source-expansion path's issuance list and report a total,
  matching the v1 path; the underlying audit table grows once per retrieval
  forever.
- Count only genuine duplicates in `deduplicated_candidates`, which was folding
  in `limit` truncation.
- Treat exact-shaped but missing object IDs, SHA-256 hashes, and artifact URIs
  as terminal empty exact lookups instead of feeding them into semantic search;
  raise the dense-only relevance floor and suppress weak single-token FTS
  matches when a multi-term query has a much stronger lexical hit.
- Normalize empty closeout feature maps as absent while preserving the required
  schema contract for non-empty features, preventing valid runtime closeouts
  from failing on provider-generated empty objects.
- Attribute future Codex, Claude, Hermes, and OpenClaw memory imports to the
  runtime inferred from their source path instead of labeling every import as
  OpenClaw.
- Make `runtime-check` verify Claude's actual `ocbrain` registration instead
  of accepting any successful `claude mcp list`, and treat an absent optional
  OpenClaw installation as skipped while still failing broken installed probes.
- Refuse irrelevant same-scope filler in hybrid retrieval: expand FTS stopword
  filtering, raise the general dense-candidate floor, and require stronger
  cosine evidence for dense-only results so unrelated queries return an honest
  empty packet without weakening strong semantic-only recall.
- Add evidence-only history imports, sparse source-backed wiki materialization,
  deterministic sealed-release truth compilation, a recent verified-closeout
  lane in `brain.digest`, a runtime-only one-shot transport recovery command,
  corpus-based vector freshness, and the smaller local
  `qwen3-embedding:0.6b` default.
- Bound local dense-index inputs to a deterministic 1,800-byte head/tail view
  so Ollama runners do not terminate on long or token-dense transcript beliefs,
  and preserve the endpoint's structured HTTP error when a local rebuild fails.
- Reuse content-hash-identical vectors during an atomic rebuild so a small
  corpus delta embeds only changed or new beliefs instead of recomputing the
  entire disposable sidecar.
- Harden the optional compilers before activation: hosted curator prompts now
  enforce exact project, visibility, and egress gates for both evidence and
  existing wiki facts; `local_only`, prohibited, confidential, and secret
  objects can never be overridden into hosted delivery; sealed truth requires
  an explicit verified status plus passed verifier evidence and is preview-only
  unless `--apply` is supplied; curator defaults are installation-neutral.
- Accept dot-free MCP tool names (`brain_context`) that clients such as Cursor
  substitute for the canonical dotted names (`brain.context`). The profile
  gate previously rejected every tools/call from those clients as "not
  available". Unknown or ambiguous spellings still raise PermissionError.
- Exact-locator retrieval: `brain.search` on the v1 core now runs an
  exact-match pre-pass before semantic ranking. Queries that are locators —
  event ids, evidence ids, belief ids, closeout ids, retrieval-use ids,
  artifact URIs, SHA-256 hashes, or an exact `task_ref` on a recorded closeout — return
  `match_mode: "exact"` with metadata-only, scope-gated `exact_matches`
  instead of unrelated ranked beliefs. Auto-derived
  `retrieval_uses.task_ref` values are never matched, so a repeated query
  cannot hijack itself. Closeout and retrieval receipts remain local-only,
  underspecified contexts fail closed, and hosted evidence/event metadata
  redacts local locators and writers.
- Wiki freshness and supersession: materialized pages carry `valid_from`
  frontmatter (derived from `last_compiled_at`) plus optional `valid_until` /
  `superseded_by` from belief attributes; `index.md` renders
  `**[stale: ...]**` markers, and stale pages show a notice under their
  title. Validity timestamps are normalized to UTC before comparison. New
  `scripts/wiki-lint.py` flags expired/superseded pages, pages the ledger no
  longer serves as current, pages older than the ledger's latest compilation,
  and conflicting pages that share a key.
- Skill-usage telemetry convention (`docs/SKILL_TELEMETRY.md`): a metadata-only
  event envelope (`ocbrain.skill_telemetry.v1`) for `skill_build`,
  `skill_install`, `skill_load`, `skill_outcome`,
  `skill_correction_candidate`, and `skill_retirement` evidence — hashes,
  URIs, and ids only, never skill bodies or transcripts. Constants and
  `validate_skill_telemetry()` live in `ocbrain.events`; both legacy and v1
  ingest paths enforce the envelope, and automatic activation never promotes
  telemetry evidence into current truth.
- Bound SQLite writer-lock windows so concurrent agents stop seeing "database
  is locked": `import-history` now commits after every file instead of holding
  one implicit write transaction across up to `--batch-size` slow redactions
  (the same rationale as `DatasetWriteBatch` for dataset miners; the flag is
  now deprecated), and the default `busy_timeout` rises 5s -> 30s
  (`OCBRAIN_BUSY_TIMEOUT_MS` still overrides) so MCP ingest/closeout/feedback
  writes from Codex/Claude/Cursor/Hermes queue instead of failing. WAL +
  per-file commits + a generous busy timeout is the local queuing model.
- Harvest Cursor AI chat history: `scripts/export-cursor-chats.py` renders each
  Cursor workspace's `state.vscdb` (prompts, generations, and composer bubbles)
  to secret-redacted, content-compared JSONL under `~/.ocbrain/exports/cursor/`,
  and `history_runtime()` now attributes those exports (and `.cursor` paths) to
  the `cursor` runtime. `brain-sync.sh` includes the export step and sweeps the
  export directory, so Cursor sessions accrue to the shared core alongside
  Codex, Claude Code, and Hermes.
- Harden `brain-sync.sh` for macOS operators: replace the non-portable
  `flock(1)` single-instance guard (absent on macOS, which made the script
  exit silently before every harvest) with an atomic mkdir lock that reclaims
  stale locks via PID liveness, and bound the harvest with a hard time budget
  (`OCBRAIN_SYNC_BUDGET_SECONDS`, default 45 minutes) so a stuck import can
  never block the launchd schedule — partial batches stay committed and the
  next run resumes. The script now logs `brain_events` counts before and after
  each run for operator visibility.
- Persist a stat-fingerprint gate for v1 history imports (`schema_meta` key
  `history_file_fingerprints_v1`). Previously the v1 path re-read and
  re-redacted every history file on every run to make its unchanged decision,
  so multi-GB transcript corpora never converged on a recurring schedule.
  Unchanged files now skip in O(stat); `import_source_v1` remains the
  authoritative changed/unchanged judge for any file whose fingerprint moved.
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
