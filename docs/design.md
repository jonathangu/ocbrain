# ocbrain design notes

This is the compact version of the current architecture. For the complete
walkthrough, use [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Two planes, one brain

The relational plane stores immutable `evidence`, compiled `knowledge`, their
links, retrieval outcomes, run ledgers, embeddings, and the `memory` view. The
event plane stores scoped evidence, corrections, compilation decisions, current
belief projections, contradictions, and egress audits.

They are not separate runtime memories. ChatGPT/Codex, Claude Code, OpenClaw,
repo, task, client, and session context act as lenses over one scope-aware
ledger.

## Non-negotiables

- No knowledge without evidence.
- No direct runtime write to durable belief.
- Memory is a view over current injectable knowledge, not a second store.
- Derived privacy scope can tighten but never widen.
- External pages, transcripts, and artifacts are data, never instructions.
- Automatic writes cannot clobber stronger first-party provenance.
- Risky, prescriptive, or executable knowledge must satisfy the verifier and
  safeguard path or carry an explicit approval signal.
- The brain observes loop work; it does not enqueue or execute it.
- Normal maintenance supersedes, quarantines, marks stale, or archives. It does
  not destructively rewrite audit history.

## Maintenance and autonomy

The light and heavy autopilot profiles share one lock. The light profile keeps
recent knowledge reviewed, labeled, embedded, quarantined, promoted, rendered,
and maintained. The heavy profile adds snapshot, harvest, compilation, and
dataset work. Snapshot or migration failure aborts; other stage failures make a
run partial and allow independent later stages to continue.

The separate stallcheck process is passive. It detects a parked turn from
transcripts and runner ledgers, reads overdue producer deadmen, writes liveness
evidence, and may page through an operator-owned transport. It never claims the
work itself. Autopilot maintenance independently consumes stallcheck's
self-heartbeat; autopilot checkpoints its own running row and deadman after
every stage for stallcheck to consume. Neither process is its own only witness.

## Retrieval and feedback

Scoped search returns source-backed context plus excluded-scope counts and
visible contradictions. When the retrieval audit is recorded, the result
includes a `retrieval_use_id` for `brain.feedback`. During a long SQLite writer
window, the read can succeed with `retrieval_use_status=database_busy` and no
handle. Callers must not retry merely to manufacture feedback evidence.

## Dataset boundary

SFT, DPO, and persona examples carry provenance, scope, label, and confidence.
Local grading rejects non-loopback endpoints before reading an example. Export
has no hosted target and excludes private scope. The eval-before-train pilot
freezes held-out prompts, references, and rubric before training files exist,
then keeps the blind key away from the rater until scoring.

Pipeline completion and model-quality acceptance are separate claims.
