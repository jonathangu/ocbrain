# Selftest thresholds and where their numbers come from

`ocbrain selftest` gives every metric a verdict — `ok`, `watch`, `alarm`, or
`not_measured` — and exits non-zero if anything alarmed. That only means
something if the boundaries can be traced. This document is the provenance for
every number in `THRESHOLDS` (`src/ocbrain/selftest.py`).

Each entry says which of three kinds it is:

- **Measured** — taken from this install, with a date. A measurement without a
  date becomes folklore.
- **Convention** — adopted from somewhere named, deliberately, rather than
  derived here.
- **Judgement** — a guess. Said plainly, because a threshold that pretends to
  authority it does not have is worse than no threshold at all. These are
  starting lines to calibrate away from.

`watch` never fails the gate. It exists so a human gets a chance to look before
cron starts failing, which is the difference between a gate people act on and a
gate people mute.

## How to run it

```bash
ocbrain --db ~/.ocbrain/ocbrain.sqlite selftest --pretty          # human table
ocbrain --db ~/.ocbrain/ocbrain.sqlite selftest                   # JSON scorecard
ocbrain selftest --out today.json                                 # save a baseline
ocbrain selftest --baseline today.json --pretty                   # diff against one
```

The core is opened `mode=ro` with `PRAGMA query_only`, and the command refuses
to run if it cannot. It is safe to point at the live core an MCP server is
serving from. Wall clock on the 180 MB live core is **~1.6 s**, comfortably
inside an hourly cron slot.

---

## Section A — retrieval health

### `answer_rate` — ok ≥ 0.40, watch ≥ 0.25 (higher is better)

Fraction of retrievals in the window that served at least one item, split by
whether the caller's scope could reach the corpus at all.

**Measured**, 2026-08-25 on the live core: 45.8% all-time (917/2001) and 45.1%
over the trailing 30 days. Bands sit just under the measured floor so ordinary
variation does not fire. **Judgement anchored on measurement** — there is no
principled correct answer rate, only a level this install is known to sustain.

A note on the source: `retrieval_items` and `served_ids_json` agree on all 2001
rows of the live core. The metric reads `served_ids_json`, because it is written
in the same statement as the retrieval row and therefore survives a core whose
normalized item rows were never backfilled.

### `scope_reachability` — ok ≥ 0.85, watch ≥ 0.70 (higher is better)

Fraction of retrievals whose passed scope, after folding and alias resolution,
names a scope some serving belief actually occupies.

**Measured**, as a regression guard on the 2026-08-24 alias-table repair, which
was recorded at **25.1% before and 95.2% after**.

Those two figures are the reason this metric exists, and this module does not
reproduce them exactly. Measured here on the same corpus:

| definition | before (no alias table) | after |
| --- | --- | --- |
| this module, any named scope, all-time | 22.5% | 87.2% |
| this module, any named scope, 30 days | — | 91.8% |
| project component only, 30 days | — | 94.0% |

The gap has two causes, both worth stating rather than papering over. The
original figures were taken on 2026-08-24 against a smaller corpus, and they
appear to have counted only the `project` component, whereas this module counts
any non-global scope the caller named — `repo`, `task`, `session` included. A
caller who named only a task scope genuinely cannot reach the corpus by scope,
so it belongs in the denominator; that choice costs about two points and buys a
metric that means what it says.

**The global scope is deliberately excluded.** Every caller's compatible set
contains `global:doctrine`, and serving beliefs occupy it, so counting it would
make this metric 100% forever and measure nothing. `tests/test_selftest.py`
seeds a serving global belief precisely so that a regression to counting it
fails a test.

The alarm band sits far below the repaired floor and far above the broken one,
so only a real regression crosses it.

### `zero_result_rate` — ok ≤ 0.60, watch ≤ 0.75 (lower is better)

The complement of `answer_rate`, stated separately so the census of which
project strings came back empty carries a verdict of its own. Same measurement,
same date.

---

## Section B — corpus quality

These are the TARL metrics, adopted deliberately.

### `pollution_rate` — ok ≤ 0.15, watch ≤ 0.30 (lower is better)

Of beliefs approved in the window, the fraction retracted or tombstoned within
14 days of mint.

The 14-day horizon is **convention** (TARL), adopted rather than derived. The
bands are **judgement**: no pre-v2.2 measurement of this exists, because
supersession did not exist to distinguish replacement from deletion.

Measured 14.6% (47/321) on 2026-08-25 — inside the band, but only just, and the
cause is worth knowing: the auto-compiler mints receipt and transcript beliefs
that the hygiene pass sweeps at a **median of 1.5 days** after mint.

### `structured_removal_share` — ok ≥ 0.80, watch ≥ 0.50 (higher is better)

Of those horizon removals, the share carrying a `superseded_by` pointer.

**Judgement**, set to the same bar as `correction_adoption` because they measure
the same shift from opposite ends. Should trend to 1.0 now that supersession
exists. Measured 12.8% (6/47) on 2026-08-25 — a true baseline reading, expected
to climb.

### `conflict_preservation` — ok ≥ 1.0, watch ≥ 0.99 (higher is better)

Of detected conflicts — supersede corrections plus `annotate`-written
`contradicts` pairs — the fraction whose losing side is still reachable.

The target is exactly **100%**, which is not a judgement but the invariant the
correction pathway is supposed to hold: nothing it touches may become
unreachable. A supersession's loser is preserved when its row survives *and*
carries the validity window that says which era it belonged to, which is what
`brain.get mode=as_stored` needs to answer "what did we believe then". A
`contradicts` pair asserts disagreement without retiring either side, so both
must simply still be there.

The `watch` band at 0.99 exists only so one in-flight row does not page before a
human looks. Any sustained loss is an alarm.

Measured 100% (6/6) on 2026-08-25.

### `calibration_gap` — ok ≤ 0.20, watch ≤ 0.35 (lower is better)

Largest gap across confidence bands between the mean *stated* confidence and the
observed 30-day survival rate.

The comparison uses the mean stored confidence rather than an invented band
midpoint: the stored number is what the brain actually claimed, and scoring it
against a midpoint nobody wrote down would be measuring an artefact.

**Judgement** — no prior calibration measurement exists for this corpus.

**This currently alarms at 0.668 (2026-08-25) and the cause is known.** The
`moderate` band's survival is **0.0 across 658 beliefs**: auto-compiled receipt
and transcript beliefs are minted at moderate confidence and swept by the
hygiene pass within days. This is a real finding about the auto-compile pipeline,
not a threshold that needs widening — the fix is upstream (mint receipts at low
confidence, or do not mint them as beliefs at all). The per-band `removed_by`
breakdown in the JSON names the pathway that retired each cohort, so the
diagnosis is one command away rather than one investigation away.

### `duplicate_key_clusters` — ok ≤ 0, watch ≤ 3 (lower is better)

Serving beliefs sharing an `attributes.key`.

**Derived from the invariant**, not guessed: `attributes.key` is a wiki fact's
identity and is meant to be unique across serving beliefs, so the correct count
is zero by construction. The `watch` band tolerates a curator cascade caught
mid-flight.

Measured 1 cluster on 2026-08-25.

### `near_duplicate_clusters` — informational, never gated

Connected components of serving beliefs whose embedding cosine is ≥ **0.88**.

The threshold is deliberately **not** `mcp_v1.ADVISORY_COSINE_THRESHOLD` (0.90),
which governs what a single served packet warns a reader about. This is the
lower, more sensitive bar for a standing corpus sweep: a pair at 0.88 is not
worth interrupting a read for, but is worth counting when asking how much of the
corpus says the same thing twice.

No verdict is attached, because no calibrated bar exists for how much semantic
overlap a healthy corpus carries and inventing one would be authority this
measurement has not earned. Track it against a saved `--baseline`.

Stands down to `not_measured` when the sidecar is absent, stale, or on a
different schema version. Reports `embedding_coverage`, because 40 clusters
found across 100% of serving beliefs and 40 found across 30% are different facts.

---

## Section C — the correction pathway

### `correction_adoption` — ok ≥ 0.80, watch ≥ 0.50 (higher is better)

Agent-issued corrections that are structured supersedes, versus the old broken
shape: a bare retract with the replacement fact typed into the correction body,
a field nothing indexes and nothing serves.

The 0.80 target is **brief-specified**. The baseline is **measured** and is the
sharpest number in this document: on 2026-08-25 the live core held **11
agent-issued corrections across its entire life, all `op=retract`, zero
supersedes** — 0 of 11.

"Agent-issued" excludes writers prefixed `maintenance:`, `operator-approved:`,
and `deterministic:`. Without that filter, 707 machine retractions drown the
signal this metric exists to carry.

Measured 0.0% (0/10) over the trailing 30 days on 2026-08-25. A true alarm: the
structured pathway exists and agents are not yet using it.

### `lossy_supersession_share` — ok ≤ 0.15, watch ≤ 0.40 (lower is better)

Share of machine-authored supersessions (curator refreshes, compactor merges)
whose successor drops a checkable token — an issue ref, backticked literal,
path, flag, identifier, or figure — that the predecessor carried. A machine
rewording claims to restate, not to correct; when it loses the tokens a reader
could look up, the corpus got smoother and knows less. Agent supersessions are
excluded: a correction is supposed to drop the tokens of the fact it refutes.

The 2026-08-26 backlog triage found 15 of 28 curator-proposed supersessions
would have silently destroyed checkable facts, and the landed population
measures 46/82 (56%) with this extractor (2026-08-26) — the common case, not
the exception, and an honest alarm at ship. A refresh that legitimately updates
a count (473 sessions → 500 sessions) moves this metric too, which is why the
ok band is not zero. Both bands are judgement.

### `pending_supersede_depth` — informational, never gated

Undecided compilation proposals carrying `attributes.supersedes`. An undecided
proposal carrying that attribute *is* the pending correction — there is no
second table and no new status.

The value counts **distinct targets** and the display carries the raw proposal
count beside it (`33 distinct (283 proposals)`). Raw depth alone was the
operator's only window onto this queue, and it read as ordinary backlog while a
proposal loop grew it without bound: 283 proposals were 33 beliefs, one of them
proposed twelve times. A number that can hide unbounded growth is worse than no
number, so neither figure is reported without the other.

Depth alone is not a fault; the cost is carried by the age metric below.

### `pending_supersede_age_hours` — ok ≤ 72, watch ≤ 168 (lower is better)

Age of the oldest undecided supersede proposal.

**Judgement.** The stale belief keeps serving until an admin decides, so age is
the real cost of a pending queue: three days to notice, a week to alarm.

### `contradictions_nonempty_rate` — ok ≥ 0.01, watch ≥ 0.0 (higher is better)

How many served packets in the window carried at least one contradiction entry.

**Measured baseline: provably 0.0 for this core's entire life before v2.2.** The
declared pass had no writer, so every packet ever served shipped an empty
`contradictions` list while sometimes carrying visibly conflicting items. Any
non-zero value is the single clearest proof the writer now exists.

The bar is deliberately just above zero. Most packets legitimately carry no
conflict, and a *high* rate would itself be bad news — this metric is a
liveness check on the writer, not a quality target.

**This metric is reconstructed, and the JSON says so in its `basis` field.**
`contradictions[]` is computed at serve time and never persisted — there is no
column holding it — so the two declared signals are recomputed over each
packet's stored `served_ids_json` against the corpus as it stands now. It
answers "does the writer produce anything", which is the question that matters
while the pathway is new. It is not a replay of what each packet literally
shipped, and it should not be read as one.

The signals and their caps are imported from `ocbrain.mcp_v1` rather than
restated, so the measurement cannot drift away from the thing it measures. The
comparison is batched — keys and vectors loaded once for the whole corpus rather
than once per packet — because per-packet sidecar opens would put the command
well past its time budget.

Measured 2.7% (34/1246) on 2026-08-25.

---

## Section D — plumbing

### `provenance_coverage` — ok ≥ 0.90, watch ≥ 0.50 (higher is better)

Retrievals and closeouts carrying `server_connection_id`, broken down by
`client_runtime_key`.

**The denominator is rows written since provenance capture began on this core,
not the whole window.** Server-captured provenance landed with PR #33 on
2026-08-24; no earlier row could ever have carried it, however well-behaved the
client. Measuring over the full window would report the migration date rather
than whether clients are stamping — 0.7% (12/1818) with a 30-day window, versus
44.4% (12/27) over rows that could actually have been stamped. The second number
is the one that says something.

Capture start is discovered from the data: the earliest row anywhere carrying a
`server_connection_id`. When no row has ever carried one, the metric reports
`not_measured` rather than zero, because "capture has never run here" and
"clients are not stamping" are different faults.

Below **20** post-capture rows the metric abstains: a percentage of eleven is
not a verdict.

**Judgement** — once every client has reconnected, near-total coverage is the
expectation, so the `ok` band is set where a healthy post-migration install
should sit.

### `closeout_trace_join_rate` — ok ≥ 0.50, watch ≥ 0.20 (higher is better)

Closeouts whose recorded session identity is byte-identical to a transcript
filename. This join is what makes a closeout's tool-call trace minable later.

**Measured baseline: 3.7%** — 13 closeouts identity-joining to a transcript,
recorded in `ocbrain.provenance`'s module docstring. Reproduced here at 3.8%
(22/572) over the trailing 30 days on 2026-08-25, which is the same number a
larger corpus later.

The brief is explicit that this climbs *for new rows only*, so the gated figure
uses the same post-capture boundary as `provenance_coverage`. The window and
all-time figures ride along in the detail unbadged, because the historical 3.7%
is the baseline this is measured against and hiding it would make the
improvement unfalsifiable. Early reading since capture began: **7/11 = 63.6%**,
below the 20-row floor and therefore reported as `not_measured` until the sample
is real.

`not_measured` when no transcript root exists on the machine, which is the
correct answer on a server that has never hosted a Claude Code session.

**Judgement** for the bands themselves.

### `harvest_silence_hours` — ok ≤ 48, watch ≤ 48 (lower is better)

Longest silence among **live** harvest streams.

The 48-hour figure is the brief's, and is **judgement**. The Hermes fleet once
went dark for 14 days and nothing noticed; this row is that alarm.

**Liveness filtering is what makes this usable.** The live core carries **90
distinct `source_runtime` spellings**, most of them one-shot lane labels
(`cursor-opus5-lane-r2`, one row, last seen in July). A naive per-label freshness
check alarms on 51 of them, permanently, which makes the row permanently red and
therefore permanently ignored — the failure mode where a gate nobody can pass
becomes a gate people learn to skip.

A stream is **live**, and therefore something whose silence is news, when it
wrote **≥ 3 rows in the last 7 days**. The alarm band is then the gap between
48 hours and that 7-day window: *was harvesting this week, has not in two days*,
which is exactly the shape of a fleet going dark. On the live core this reduces
90 labels to **14 live streams and exactly one alarm** — `cursor`, silent 121
hours — which is a true finding.

The known limit: a stream dark for longer than 7 days drops out of the live set
and stops alarming. By then it has alarmed for five days, which is enough. Both
constants are named (`HARVEST_LIVE_DAYS`, `HARVEST_MIN_ROWS`) rather than
inlined, so this trade-off is adjustable rather than buried.

### `db_size_mb` — informational, never gated

`page_count * page_size`. A core has no correct size; the signal is the trend
against a saved `--baseline`.

### `rows_added_in_window` — informational, never gated

Rows added per tracked table over the window, with per-table detail.

**This deliberately reports rows, not an estimated megabyte figure.** Prorating
size by row share reads as authoritative and is not: evidence bodies are orders
of magnitude larger than retrieval receipts, and an earlier draft of this metric
estimated ~98 MB of 30-day growth against the known post-PR-#33 figure of ~14
MB/month — wrong by a factor of seven, and wrong in a way that looked precise. A
real byte delta needs two snapshots, which is what `--baseline` is for.

PR #33 cut projected growth from ~106 MB/month to ~14 MB/month by storing
transcript evidence as verified file pointers rather than inline copies.

### `egress_refusable_policies` — ok ≥ 1, watch ≥ 1 (higher is better)

How many egress policies present on this brain's curation-eligible evidence the
operator's declared allow-list would refuse. Not "how many did it refuse" — that
number is in the audits, and it is a different question.

**Measured**, 2026-08-28 on the live core: **0**. `egress_audits` holds 240 rows
covering 2026-08-04 to 2026-08-28, and `SELECT DISTINCT rejected_json` returns
exactly one value: the literal string `'[]'`. Across those audits 25,106 evidence
items were transmitted and none was ever refused. The reason is structural rather
than lucky — the declared allow-list named `approval_required`, `hosted_ok`, and
`local_only`, which is every policy an operator may declare, so no input could
fail the check.

Binary by construction, which is why `ok` and `watch` are the same number: either
a refusal is reachable or the gate is a transmission log. `prohibited` is
subtracted before counting, because it is refused in code whatever an operator
declares, and crediting the declaration for the floor would let an
everything-admitting allow-list look like one with teeth.

Set `curator.egress_allowlist_ack` to a sentence saying why an all-admitting
allow-list is intended on this install. The acknowledgement downgrades `alarm` to
`watch` and is recorded in the metric's detail — declared, not enumerated away.

### `vector_sidecar_lag_events` — ok ≤ 500, watch ≤ 5000 (lower is better)

Events appended to the core since the sidecar recorded its `core_event_seq`.

**Judgement.** The sidecar indexes only serving beliefs, and the large majority
of events are evidence rather than compilations, so a lag of a few hundred
events is normal drift rather than staleness. Measured 2 events behind on
2026-08-25.

`not_measured` when the sidecar is absent, unreadable, on a different schema
version, or records no `core_event_seq`.

### `integrity` — ok ≥ 1.0 (binary)

`PRAGMA quick_check` plus `PRAGMA foreign_key_check`. Either the core is
structurally sound or it is not.

`quick_check` rather than `integrity_check` is a deliberate trade: it skips the
exhaustive per-row index cross-check, which is what makes the full check take
tens of seconds on a 180 MB core. Everything this command exists to catch — a
torn page, a corrupt b-tree, a broken foreign key — it still catches, and a
check too slow to run hourly does not get run.

---

## Section E — harness surface

### `briefing_determinism` — ok ≥ 1.0 (binary)

`brain.briefing` promises byte-identical output for the same scope and corpus
state, and a harness that reorients through it every iteration inherits that
promise. Anything below 1.0 means the promise is already broken, so there is no
watch band to sit in.

### `briefing_budget_compliance` — ok ≥ 1.0 (binary)

Also binary. The budget is a hard ceiling the renderer enforces before it
spends anything on items; a briefing over budget means the skeleton reservation
is wrong, not that a scope got busy.

### `goal_pointer_resolution` — ok ≥ 1.0, watch ≥ 0.80 (higher is better)

Share of open goals whose `source_pointer` still resolves on this machine. A
goal is a pointer to a spec in the repo; when the spec moves and the goal does
not, the goal is pointing at nothing. Watch rather than alarm at 80% because a
pointer written on another machine can legitimately not resolve on this one.
Judgement.

### `goal_open_age_days` — ok ≤ 14, watch ≤ 45 (lower is better)

Age of the oldest open goal. Goal drift is a distinct failure that pass/fail
benchmarks cannot see, and an objective nobody has closed or abandoned in six
weeks is the observable form of it. Judgement: no measurement exists yet
because goals ship with this release.

---

## Constants outside the scorecard

Not every number that decides something is a selftest threshold. These sit in
`src/ocbrain/curator.py`, on the compile path, and get the same treatment.

### `NEAR_DUPLICATE_COSINE = 0.88`

Document-to-document cosine at or above which a new-key claim is treated as a
restatement of a belief already serving in its scope, and routed to supersession
rather than minted under its own key.

**Measured**, and deliberately not an independent number: it is pinned equal to
`compact.DEFAULT_COSINE_FLOOR`, which v2.2 Phase 7 calibrated by finding 38
same-scope clusters on the live corpus and proposing 18 merges retiring 25
beliefs. The duplicate-gate/compactor floor equality test holds them equal. A
claim the gate admits and the compactor then proposes
retiring would be a gate that moved the work rather than doing it.

Re-measured on a copy of the live core, 2026-08-28, with this repo's own
`compact.find_clusters`: 35 same-scope clusters at 0.88 covering 98 of 347
serving beliefs, 63 of which are excess copies. At 0.85 the same corpus gives 51
clusters over 145 beliefs, and at 0.90, 28 over 77.

What the number is *not* doing is worth stating, because it would be easy to
claim credit for it. Replaying each cluster's second member as a fresh claim
through `apply_claims` — real key, real body, real local embedder — the existing
0.60 query-side cascade already collapses **34 of 34** replayed clusters when the
sidecar is freshly built. This floor earns its place only in the state a curation
cycle actually leaves the sidecar in, where the query-side arm is blind: **23 of
34 grow a second serving belief before this gate, 2 after**. Choosing 0.85 or
0.90 instead would move that "2" a little; keeping the compactor's number keeps
the two stages agreeing about what one fact is, which matters more.

The scale matters: this is a **document-side** score, comparable with the
sidecar's stored vectors, and it is not the same scale as
`CONTRADICTION_COSINE_FLOOR = 0.60`, which scores a *query*-side embedding
against those documents. Two floors, two scales, one reason each.

### `NEAR_DUPLICATE_NEIGHBORS = 5`

How many of the scored candidates the gate looks at before deciding. The list is
sorted by descending similarity, so this only ever matters when the top match is
below the floor and a lower one is not — which cannot happen. It is a bound on
the returned list, not a decision: the decision is the first neighbour's score
against `NEAR_DUPLICATE_COSINE`.

**Judgement**, matched deliberately to `CONTRADICTION_NEIGHBORS = 5` so the two
readers of the same sidecar ask for the same shape of answer. Nothing measured
argues for 5 over 3 or 10; what would change behaviour is the floor, not this.

### `DEFAULT_DOCUMENT_EMBED_BUDGET = 32` (`src/ocbrain/hybrid.py`)

How many candidate bodies one duplicate-gate call may embed on demand. The
candidates that need it are exactly the beliefs written since the last sidecar
build, because those are the rows whose stored vector no longer matches their
body. Past the budget the extra candidates come back as `uncovered`, the gate
returns `candidates_uncovered:N`, and that reason is **not** in
`DUPLICATE_GATE_EXEMPT_REASONS` — so on the shipped `pend` fallback every
remaining claim in that cycle is pended rather than compiled.

So this is an **availability cliff, not a performance dial**, and it is stated
that way in `curator.document_embed_budget` because the failure it produces
looks like the curator quietly doing nothing.

**Judgement, bounded by a measured quantity.** The number that has to fit under
it is one cycle's own output: `wiki-curator.py --max-beliefs` defaults to 24 and
is capped at 40, and `scripts/brain-promote.sh` rebuilds the sidecar (line 214)
only after curation has finished, so every belief a cycle writes stays uncovered
for the rest of that cycle. 32 sits above the default and below the cap. The
shipped embed-budget cycle test holds it above 24 and equal to the config
default.

The cliff was measured by this change's adversarial reviewer, not by its author,
on a `.backup` copy of the live core with a live embedder (2026-08-28): with 40
serving beliefs' bodies no longer matching their stored vectors,
`document_neighbors` answered with coverage `{candidates 301, reused 261,
embedded 32, uncovered 8}` and the gate returned `candidates_uncovered:8`.
Recorded here as a second-hand measurement because it is one.

Raise the config field if your cycles are larger; each extra candidate is one
local embedding call. Lowering it does not buy a cheaper gate, it buys a gate
that refuses to answer.

### `DEFAULT_VOLATILE_TTL_DAYS = 14`, `DEFAULT_MEASURED_TTL_DAYS = 45`

TTL by how fast a claim's subject moves, replacing a flat 90 days for `current`
and no expiry at all for `durable`.

**Judgement, anchored on one measured incident.** On 2026-08-28 a serving belief
still named which ClickHouse host was live "as of 2026-07-24" — 35 days after the
state it describes — under a `valid_until` running to 2026-11-02, with a second
belief repeating it. 14 days is
chosen to be shorter than that observed staleness by a factor of about 2.5, so a
fact of that kind expires while it is still true rather than long after; 45 days
is half the previous flat figure, on the reasoning that a measurement is re-run on
roughly a monthly cadence.

An honest note on what could not be measured: the obvious provenance would be the
observed interval between a volatile belief being compiled and being corrected.
On this corpus that is **1.87 days median across 364 corrected volatile beliefs**
— and 0.94 days for doctrine and 0.99 for measurements, which is the hourly
curator rewriting its own output rather than the world changing. The measurement
does not discriminate between the classes, so it is not used as the source.

The mechanical detectors (`VOLATILITY_PATTERNS`) are tuned for precision, not
recall: a false positive puts a two-week clock on a durable truth and stops it
serving, a false negative leaves today's behaviour. Measured on the 347 serving
beliefs, 2026-08-28: they fire on **33 (9.5%)**, of which 11 are `durable`-marked
beliefs carrying a dated, versioned, host-named, credential-named or live-state
statement.

Re-dating the existing corpus is a separate decision and a separate command.
`ocbrain wiki-volatility` prints the plan; it needs both `--apply` and `--yes` to
write. On a copy of the live core: 174 doctrine / 140 measured / 33 volatile, 15
beliefs gain a TTL they never had, 158 move to a shorter one, and **7 are already
expired** under the new scheme — which is the number to read before running it,
because those seven stop serving at the next hygiene sweep.

Every corpus figure in this section was taken from a 347-serving-belief snapshot
on 2026-08-28. Two operator compactions ran later the same day (347 → 303
serving, 51 retired via supersession), so the date alone no longer identifies the
snapshot; the rates are what travel, not the counts. Re-measured read-only at
2026-08-28T19:34Z: 303 serving beliefs, 164 `durable` of which **0** carry a
`valid_until`, and 139 `current`. The premise — durable buys no expiry at all —
is unchanged.

### Turning expiry off

`--current-ttl-days 0` (or `curator.current_ttl_days = 0`) means no expiry at
all, under **either** scheme. Re-keying TTL on volatility initially left that
number read and then ignored, so a run started with 0 still stamped 14 days on a
volatile claim. `--no-volatility-ttl` restores the lifecycle rule, where a
positive `--current-ttl-days` is again the number and `durable` claims never
expire; the wiki-curator operator-control tests hold the chain from the flag to
the stored `valid_until`.

`ocbrain wiki-volatility` reads the same `curator.current_ttl_days`, so a brain
that has turned expiry off plans and applies zero rewrites. The compile path and
the sweep honouring one switch differently is the shape of defect this whole
section exists to record.
## Section F — serving policy dials

Not selftest thresholds: these are constants in the *served read path* that an
operator can change. They are documented here because the same rule applies —
a number nobody can trace is a number nobody should trust.

### `confidence_prior_enabled` — ships `True`

`retrieval.confidence_prior_enabled` (`src/ocbrain/config.py`, default mirrored
by `CONFIDENCE_PRIOR_ENABLED` in `src/ocbrain/core_v1.py`) decides whether
`ranking_prior` keeps its `0.85 + 0.15 * confidence` factor.

**Measured**, 2026-08-28, on a `mode=ro` backup copy of the 208 MB live core
(347 serving beliefs):

- 345 of 347 serving beliefs carry a `confidence`, every one inside
  `[0.65, 1.0]`, clustered on authored round numbers — 0.85 ×116, 0.80 ×85,
  0.75 ×64, 0.70 ×26, 0.90 ×24, 0.99 ×11. Bands: strong 317, moderate 28,
  unknown 2.
- Joining `retrieval_items` to `retrieval_uses.outcome` over judged outcomes
  (`used`, `helpful`, `irrelevant`, `harmful`): moderate-band items drew 68
  `irrelevant` and 0 `harmful` of 1,061 (6.41%); strong-band items drew 463
  `irrelevant` and 23 `harmful` of 1,331 (36.51%). Ratio 5.70x.
- `outcome` is recorded per *retrieval*, not per item, so one verdict tars every
  item in that packet and the 5.70x is not identified. Re-measured with one vote
  per packet — 470 judged retrievals — the direction holds: packets judged
  irrelevant/harmful held items averaging **0.8707** confidence and 89.63%
  strong-band, against **0.7263** and 40.90% for packets judged used/helpful.
- Replaying the 200 most recent distinct recorded queries against that copy,
  switching the term off moved the served *set* on 30 of 200 queries (31 items
  in, 31 out), reordered another 113, and changed the top-1 item on 13. Total
  items served was 2,236 either way.

**Judgement, deferred.** The default is `True` because that is the behaviour
every packet ever served was built with, and because the honest reading of the
measurement is "this field does not mean what it says", which is an argument for
re-deriving it, not automatically for deleting its ranking weight. Whether the
term should go, or `confidence` should be re-derived from evidence count,
evidence recency, and verifier status, is an operator decision. Disable it by
setting the config key `retrieval.confidence_prior_enabled` to `false`, and read
`ranking.confidence_prior_enabled` on any packet to see which way it ran.
## Section G — write-time closeout gates

Not selftest metrics: these are constants in `src/ocbrain/closeout.py` that
decide whether a `brain.closeout` call is accepted. They are documented here
because this file is where a constant's provenance lives, and a gate that
refuses a client's work has to justify its number at least as hard as a gate
that colours a row amber.

All three are configurable under the `closeout` config section, so an operator
who disagrees changes one line rather than forking the write path.

### `RUNTIME_SESSION_SHAPES` — `{runtime_uuid, runtime_hex}`

Which shapes of `context.session` reach `task_closeouts.session_id`. A UUID
(8-4-4-4-12 hex, which is what Claude Code and Codex mint) or a bare 32/40-char
hex id. Everything else is refused under the default policy.

**Measured** against one read-only backup of the live core taken 2026-08-28
12:30:51 PDT, 1,239 closeouts from 2026-07-15: 211 (17.0%) runtime-shaped, 431
absent, and 597 hand-written — `example_cleanup_audit`, `2026-07-22`,
`2026-07-21 release checklist`, `/srv/example/receipt`. **All 94
closeouts that join a Claude Code transcript are UUIDs; zero of the 597
hand-written ids join one.** So the boundary is not a strictness dial: admitting
exactly the machine-minted shapes keeps 94 of 94 joinable rows and costs none.

Every number in this section comes from that one backup. The live core gained
three rows during the afternoon this was written; a section assembled from
several reads of a moving table would be arithmetically inconsistent with
itself, which is why it is measured from a fixed copy and the copy is dated.

`runtime_hex` is admitted on the same reasoning rather than on observation —
zero live `session_id` values carry it, but a 32/40-char hex id is machine-minted
and high-cardinality, and refusing a real runtime's real id would make the gate
unsatisfiable for that client. **Judgement**, and the weaker half of this entry.

Prose alone had six weeks to fix this: the UUID rate went 15.1% in July to 19.6%
in August. That is why the shape is enforced in the server rather than asked for
in a docstring.

### `closeout.session_id_policy` — default `enforce`

`enforce` refuses a non-runtime-shaped id; `quarantine` keeps the claim in the
receipt and out of the identity column; `off` restores pre-2026-08-28 behaviour.

**Judgement.** The case for `enforce`: the gate is always satisfiable, because
omitting `context.session` is legal and the server then fills the column from its
own connection id, so no client can be unable to file a closeout. The error is
also the only channel that has ever reached an agent, and it names
`$CLAUDE_CODE_SESSION_ID` and `$OCBRAIN_SESSION_ID` — a quarantine fixes the
column's type but can never produce a joinable id, because nothing tells the
agent to go and find one.

**The cost, measured:** replaying all 1,239 backed-up closeouts through the
shipped defaults refuses **747 (60.3%)** — 597 for the session shape, 150 more
for a missing `unresolved` (282 trip that gate, 132 of them already refused for
their session id). Every one is satisfiable on retry and both refusals are
reported together so one retry clears both, but a cron job that does not retry
loses that receipt. An operator who would rather keep every receipt sets
`{"closeout": {"session_id_policy": "quarantine"}}`.

### `SERVER_CONNECTION_SESSION_PREFIX` — `conn:`

**Convention.** A server connection id names a *connection*, not a conversation,
and joins no transcript. The prefix makes that structural rather than something
a later miner has to remember: a `conn:` value can never be mistaken for a
transcript filename by a string comparison.

### `CLEAN_SUCCESS_STATUSES` — `{completed}`

A closeout owes an `unresolved` unless it is `completed` **and** no verifier it
filed reports `failed`.

**Measured**, same backup: 282 of 1,239 (22.8%) trip this — 187 by status, and
**95 more that claim `completed` while carrying a failed verifier**. Gating on
status alone would miss all 95, which is why the verifier evidence is a second,
independent trigger. Live status distribution was completed 1,052 / partial 148 /
blocked 38 / **failed 1**; one failure in six weeks of agent work is not a
failure rate, it is a reporting gap, and `brain.ledger`'s only job is surfacing
the attempts that did not work.

Deliberately **not** a status override. Of the twelve closeouts whose verifiers
all failed, seven are read-only audits where the FAIL verdict is the deliverable
("Read-only re-review found remaining blockers; verdict FAIL"). Deriving `failed`
from the evidence would relabel successful work. The caller keeps the verdict and
owes a sentence.

### `RUNTIME_FAMILY_RULES` — seven families

**Measured from the listing**, not invented: every token is a substring of a
spelling in the live `SELECT runtime, COUNT(*)` output. That column held **160
distinct spellings across 1,239 rows** — five of "local mac", four of "codex
desktop", and `local macOS + analytics ClickHouse` (13 rows) with an
environment welded onto the client name. Nothing could be grouped by it.

The token list is **not** exhaustive of the corpus and does not claim to be. One
exact spelling is carried beside it in `RUNTIME_FAMILY_EXACT` —
`ocbrain-runtime-call`, which this repository itself emits — because its token
`ocbrain` also appears in `local-agent-mode-ocbrain`, a Claude Code client key on
66 rows. Everything else install-specific belongs in `closeout.runtime_aliases`
or in `scripts/procmine`, which is a mining taxonomy for one operator's history
and is allowed to know a Hermes profile hash.

Matching is on whole hyphen-delimited **segments**. A substring match put those
13 ClickHouse rows in the `cli` family, because "ClickHouse" contains "cli" —
a normaliser that guesses is worse than a column nobody can group.

`unknown` is a real member and covers 438 of 1,239 rows: `local`, `desktop`,
`macOS` name the machine, not the client, and inventing one for them would be
guessing. `closeout.runtime_aliases` is where an install's own labels go; it
ships empty for the same reason `scopes.aliases` does. The environment detail
those spellings were carrying now has `runtime_detail`.

The function is pure and history-independent, because `task_closeouts` is
append-only under a trigger and the 160 historical spellings can never be
rewritten in place. Applied to them at read time, server-observed key first, it
yields codex 426, unknown 438, hermes 123, mcp 95, claude-code 84, cursor 57,
cli 16 — 1,239 rows, every one placed.

`scripts/procmine/episodes.normalize_runtime` asks this function first and adds
only install-specific rules on top, so the repo has one folder and one superset
of it rather than two rivals. Over the 3,287 closeout and retrieval rows both
have ever seen, that reconciliation moves 8 rows (the spelling `cli`, from
`unknown` to `host-batch`) and regresses none.
## Section H — dual-path containment constants

These are not selftest metrics. They are the hand-typed numbers in the
legacy-retriever gate test, which pins the boundary between the two rankers this
repo still carries: the live v1 path in `core_v1.py` (FTS5 bm25 with tuned
column weights, a 1024-dim dense sidecar, weighted RRF at k=60, and a
multiplicative scope, confidence, quality, recency and feedback prior) and the
retired legacy blend in `retrieve.py` (a flat `relevance * scope_weight *
confidence * pinned * catalog_stub` product with a repo-FTS fallback).

### `retrieve.py` functions executing on a v1 core — must be exactly 0

Measured 2026-08-28 by tracing `sys.settrace` over `src/ocbrain/retrieve.py`
while dispatching the complete advertised admin tool surface against
`pre-compaction-20260828-claude.sqlite` — a frozen 208,285,696-byte
copy of the live core holding 1,247 current-belief rows and 6,492 evidence
objects, `schema_meta.core_schema = ocbrain.core.v1`. Those two counts describe
**that file**, not the live corpus: the live core is compacted and re-minted
hourly and read 1,258 / 6,651 at 19:20Z the same day. Cite the filename, not the
figures, and re-read it with `?immutable=1` rather than correcting it.
**Zero** of `retrieve.py`'s 22 top-level functions ran.

The read-side CLI is driven by the same gate, on the same terms — see the
dispatch counts below. It is not a separate one-off trace, because a trace
nothing re-runs is provenance, not a gate.

The number is 0 rather than a band because the two formulas must never mix. A
single unguarded call would rank live beliefs by the legacy product and every
other test would still pass.

### `retrieve.py` functions executing on a legacy core — floor of 5, measured 9

The same driver on a freshly initialized legacy core runs `retrieve`,
`belief_rows`, `visible_belief_rows`, `rank_contradictions`,
`contradiction_candidate_rows`, `terms`, `meaningful_terms`, `estimate_tokens`
and one genexpr — 9 distinct functions.

**This is the gate's own mutation proof, not a performance figure.** Blinding the
tracer (pointing it at a path that does not exist) makes the v1 assertion above
pass on an empty set; this assertion is what fails instead. The floor is 5 rather
than 9 so that refactoring inside the legacy ranker does not produce a spurious
failure, but it is far enough above 0 that a silent instrument cannot clear it.

### `LEGACY_RETRIEVE_CALL_SITES` — `{mcp.py: 2, cli.py: 1, shared_context.py: 1}`

Counted 2026-08-28 across `src/ocbrain/**/*.py`: `mcp.py:736` (scoped
`brain.search`) and `mcp.py:778` (`brain.preview`), `cli.py:1241`
(`cmd_preview`), and `shared_context.py:52` (`build_context`). Each of the four
sits below an `is_core_v1(conn)` refusal **in its own frame** — not merely in a
caller's. `build_context` acquired its own guard on 2026-08-28 for exactly that
reason: its only check was one frame up in `mcp.py`, which made the containment
a property of the call graph rather than of the function, so a second caller
could open a live unguarded path without changing this table.

The dynamic gate can only see call sites the tool driver reaches, so a fourth
call added in a module the driver never dispatches would slip past it. This
count is the backstop, and it is an AST binding resolver rather than a line
scan: it resolves function-local, relative, aliased, parenthesized and
attribute-qualified imports, and it counts every *reference* to the bound name,
not only a direct `retrieve(` call. The first version of this gate matched
import lines by prefix and found **0 of 9** planted evasive call sites; the
evasive-binding self-test is what holds the replacement to all nine. Adding a
call site fails this test on purpose: the
new guard has to be proved by hand before the table is updated.

### Dispatch counts — 23 MCP tool calls, 8 CLI commands

19 tools in `tools_for_profile(ADMIN_PROFILE)` plus 4 scoped/cross-scope
variants, and 8 read-side CLI invocations (`status`, `briefing`, `digest`,
`search` and `preview`, including `--project` and `--cross-scope` variants).
Both are asserted so that an emptied or drifted driver cannot report a clean
zero by dispatching nothing; `test_tool_coverage_is_complete` separately
requires the tool table to equal the advertised surface exactly, so a tool added
to the server fails the suite until it is driven here.

The CLI count exists because the MCP driver reaches `call_tool` only. `cli.py`
hosts its own legacy call site, and until 2026-08-28 nothing regression-guarded
it: a call planted in `cmd_preview` above the `is_core_v1` return executed the
legacy ranker on a v1 core with the whole suite green. Each driver carries its
own positive control on a legacy core, so blinding either one fails rather than
passing on an empty set.

### `_RETRIEVE_SOURCE` must resolve inside the worktree

The tracer keys on a filename. Under a bare interpreter `ocbrain` resolves
through the editable install rather than the checkout under test, no frame ever
matches, and every count in this section reads a clean zero for the wrong
reason. `_RETRIEVE_SOURCE` is therefore derived from the imported module and
pinned to this worktree's path by the tracer-keying self-test.

---

## Changing a threshold

Change the number in `THRESHOLDS` and change its `source` in the same commit,
then update the corresponding section here. A threshold whose source no longer
describes it is worse than one with no source, because it invites trust it has
not earned.

`tests/test_selftest.py::test_thresholds_all_carry_a_source` enforces that every
entry has a non-trivial source string, and
`test_every_metric_has_a_threshold_entry` enforces that no metric is emitted
without one.

Section H's constants are not in `selftest.THRESHOLDS`; they are module-level
names in the legacy-retriever gate test, so neither test above governs them. The
same rule applies by hand: change the constant and its
Section H paragraph in the same commit, and re-run the mutation proof the
paragraph names. A constant that only a test file holds is the easiest kind to
edit until green, which is why its provenance is written down here rather than
left in a comment.

If a threshold is being widened to make a red row go green, that is the moment
to check whether the row is telling the truth. `calibration_gap` on this install
is the worked example: it alarms, the alarm is correct, and the fix is upstream
of the measurement.
