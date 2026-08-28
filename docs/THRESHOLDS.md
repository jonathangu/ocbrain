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

## Changing a threshold

Change the number in `THRESHOLDS` and change its `source` in the same commit,
then update the corresponding section here. A threshold whose source no longer
describes it is worse than one with no source, because it invites trust it has
not earned.

`tests/test_selftest.py::test_thresholds_all_carry_a_source` enforces that every
entry has a non-trivial source string, and
`test_every_metric_has_a_threshold_entry` enforces that no metric is emitted
without one.

If a threshold is being widened to make a red row go green, that is the moment
to check whether the row is telling the truth. `calibration_gap` on this install
is the worked example: it alarms, the alarm is correct, and the fix is upstream
of the measurement.
