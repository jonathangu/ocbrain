# Running a loop against OCBrain

How to point a ralph-style loop, a spec-driven flow, or a long-horizon ops agent
at OCBrain, and what each of them gets from it.

**The brain owns no loop.** No scheduler, no queue, no run state, no watchdog.
That is not a missing feature; it is the boundary. Execution state belongs in
the harness, and OCBrain's own instructions forbid it starting hosted judgment,
a timer, or a loop. What OCBrain provides is three answers a fresh context window
needs before it can do anything useful, and it provides them the same way every
time.

## The three objects

| Object | Question | Not |
|---|---|---|
| `brain.briefing` | Where was I? | A search. It takes no query. |
| `brain.ledger` | Has this already been tried? | A write path. Pure projection. |
| goals | What am I trying to finish, and how will I know? | The home of the spec. |

`brain.context` is still the tool for "what do I know about X". It ranks, it
takes free text, and two calls can legitimately differ. That is right for
knowledge and wrong for reorientation.

## The loop shape

```
loop:
  1. fresh context window
  2. brain.briefing --project <scope>        # deterministic, <=1500 chars
  3. read the spec the open goal points at   # from the repo, at its git ref
  4. brain.ledger --task-ref <task>          # already done? already failed?
  5. do one unit of work
  6. run the goal's finish_line              # the executable verifier
  7. brain.closeout ... --verifier-refs      # pass or fail, both get filed
  8. brain.goal_close when the finish line passes
  9. exit; the harness starts the next iteration
```

Step 1 is not optional and step 2 has to be first. Anthropic's memory-tool
guidance is blunt about it — view your memory before doing anything else, and
assume interruption — and Ralph's requirement is that specs be "deterministically
allocated the same way every loop". A loop whose first act is a ranked query
reorients differently every iteration and calls the difference progress.

## What each iteration gets

The briefing has four sections, always in this order, each marked rather than
dropped when empty:

```
OCBRAIN BRIEFING · project:<scope>
A. OPEN GOALS            objective | verify: <finish line> | spec: <path>@<ref>
B. DONE LEDGER           verified done, then failed attempts, with the failure text
C. LATEST CLOSEOUT CHAIN the most recent multi-closeout task and its parentage
D. GOTCHAS               gotcha-typed beliefs, then pinned ones
E. NOTHING KNOWN ...     only when A-D are all genuinely empty
```

Properties you can build on:

- **Deterministic.** Same scope, same corpus state, same bytes. No similarity
  ranking appears anywhere in it. Order and selection are rules, not scores.
- **Bounded.** 1500 characters by default (~300-400 tokens), a hard ceiling the
  renderer reserves the skeleton against before it spends anything on items.
  Truncation is counted in `truncation.items_omitted` and printed as a line;
  it is never silent.
- **Identical across clients.** Claude Code, Codex, Cursor and Hermes get the
  same payload. A briefing two agents cannot hand to each other is not shared
  memory.

The budget is deliberately small. Chroma's context-rot study (2025-07-14) found
that a single distractor measurably degrades output, and that
coherent-but-irrelevant text hurts *more* than random filler. A comprehensive
briefing is a worse briefing.

## Where specs live: the repo, not the brain

A goal stores `{objective, finish_line, source_pointer{path, git_ref}, status}`.
The `source_pointer` is a pointer and stays one.

```
brain.goal_open
  objective:      "Edge assign lane serves scored variants at 900 RPS"
  finish_line:    "repo://coframe-core/pytest -k edge_assign"
  source_path:    "docs/EDGE-ASSIGN-HANDOFF.md"
  source_git_ref: "6e3fded77"
```

Spec-kit, Kiro, AGENTS.md and Ralph all land in the same place: requirements are
files, reviewed by humans, versioned by git. If the brain became the editable
home of a spec, the spec would stop being reviewable in a pull request, and two
agents editing "the goal" would be editing a row nobody diffs.

A goal whose `source_pointer` no longer resolves does **not** disappear. It comes
back with `warning.type = "source_pointer_unresolved"`, rendered inline in the
briefing right after the goal id so it survives line truncation. A goal pointing
at a spec that moved is the most interesting goal in the corpus.

Goals are retrieved **by scope and status only**. Never by embedding similarity;
`tests/test_briefing.py` mutation-tests that, and goals are excluded from the
hybrid candidate pool in `core_v1._servable_knowledge_sql` so they cannot leak
into a ranked packet either.

Status transitions are new events. `brain.goal_close` appends an `annotate`
correction carrying an attributes patch — metadata-only by construction, replay
stable — so "when did this close and who said so" survives a full rebuild.
Closing requires naming the verifier evidence.

## How goals and the ledger stop double-implementation

The documented ralph failure is specific: a ripgrep false negative convinced the
agent a feature was missing, so it built it a second time. Search missing
something is not a bug you can fix by searching harder; the fix is a projection
that does not depend on phrasing.

`brain.ledger` groups every closeout by its folded `task_ref` and types each
group:

- `verified_done` — latest closeout completed **with a passing verifier**;
- `attempted_failed` — latest closeout failed, blocked, or cancelled;
- `in_flight` — anything else, including `completed` with no passing verifier.

That last rule matters. An agent-reported completion is a claim, and the ledger
reports claims as claims. Calling an unverified completion "done" is how a loop
stops checking.

Failures are first-class. `failed_attempts` lists every attempt that did not
land — including the ones that came *before* a later success — with the summary
text, not just a count. `COFASC-292` on one real core is `verified_done` after
14 closeouts and still reports 5 failed attempts, which is exactly the shape a
next iteration needs: the work is done, and here is what did not work on the way.

Each attempt also carries `unresolved`, and the entry carries
`latest_unresolved`: the filer's own sentence about what is *still* not working,
which is a different question from what the session did. The write-time gate in
`ocbrain.closeout` charges every non-clean closeout for that sentence, so this is
the reader that makes the charge honest — a required field nothing serves is a
toll, not a gate. It is null on every receipt written before 2026-08-28, and both
the ledger entry and the briefing line degrade to the summary when it is.

Two grouping notes:

- Grouping folds `task_ref` at read time rather than trusting the stored
  `task_ref_norm` column. The column is written at closeout time and history is
  never rewritten, so on the install this was built against it is NULL for 1161
  of 1171 rows. A ledger that grouped on the column alone would report zero
  attempts for a task with fourteen.
- A ledger call **with** `task_ref` ignores scope. "Has anyone, anywhere, tried
  this" must not be answered "not in your project".

## Wiring it up

**Claude Code SessionStart hook.** `examples/harness/ocbrain-session-start.sh`,
with install instructions in its header. It prints the briefing text and exits 0
silently if the brain is unreachable — a hook that fails a session start because
a database is locked is a hook that gets uninstalled. Nothing in this repo writes
to `~/.claude`; you install it.

**Codex, Hermes, and AGENTS.md.** `examples/harness/AGENTS-snippet.md`. Paste it
into `AGENTS.md` or a profile `SOUL.md`.

**CLI.** `ocbrain briefing --project X --text` is what the hook calls;
`ocbrain briefing --project X --pretty` gives the full payload with the
truncation accounting and typed warnings. `ocbrain ledger --task-ref COFASC-292`
reads one chain.

## What the selftest watches

Section E of `ocbrain selftest`:

- `briefing_determinism` — the briefing is built twice per sampled scope and the
  bytes compared. Binary: there is no watch band, because a broken promise is
  broken.
- `briefing_budget_compliance` — rendered characters against the declared budget.
- `goal_pointer_resolution` — share of open goals whose spec pointer still
  resolves.
- `goal_open_age_days` — the oldest open goal. Goal drift is a distinct failure
  that pass/fail benchmarks cannot see (arXiv 2608.06663), and a six-week-old
  open objective is what it looks like from outside.

Two precision-side numbers the harness literature asks for are deliberately
*not* added here, because the scorecard already carries them under other names:
`pollution_rate` (section B) is the false-positive injection measure — beliefs
approved and then removed inside a horizon — and `zero_result_rate` with
`calibration_gap` are the abstention calibration. A second name for the same
number is how a scorecard stops being read.

## What this does not do

- No scheduling, no retries, no concurrency control. Twelve-factor agents puts
  execution state in the harness and that is where it stays.
- No editing of specs. The brain pins pointers.
- No ranked reorientation. If you find yourself wanting a query parameter on
  `brain.briefing`, you want `brain.context`.
