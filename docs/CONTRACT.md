# ocbrain Product Contract

This is the canonical boundary for current ocbrain behavior. Historical design
documents are useful context, but they do not override this contract or the
shipped source.

## Purpose

ocbrain closes one private learning loop across ChatGPT/Codex, Claude Code,
OpenClaw, and compatible future runtimes:

> runtime evidence → scoped current knowledge → better work → outcome feedback
> → curated datasets → an evaluated personal model

Model weights learn stable voice, judgment, taste, and durable skills. The
brain retains changing facts, projects, identities, and private context so they
remain correctable and scope-controlled.

## Autonomous knowledge path

Routine evidence ingestion, compilation, labeling, promotion, decay, dataset
mining, and excerpt rendering are autonomous. They are protected by mechanical
invariants: provenance, scope composition, injection scanning, quarantine,
quality thresholds, verifier evidence for risky rows, bounded budgets, and an
append-only audit trail. There is no routine human approval queue in this path.

Autonomy stops at four authority boundaries:

1. External egress beyond explicitly configured, capped, scope-checked calls.
2. Widening private or project-scoped material into a broader scope.
3. Releasing a quarantined row back into the autonomous path.
4. Destructive, irreversible, or externally mutating action.

## Execution boundary

ocbrain does not enqueue work, run agent loops, install skills, apply policy, or
mutate external systems. Separate runtimes do work. ocbrain observes their
evidence, compiles memory, serves source-backed context, measures outcomes, and
reports stalled or failed work.

## Privacy boundary

The ledger, raw transcripts, dataset examples, calibration text, model weights,
and private evaluations remain local. Private-scope content is never sent to a
hosted judge or embedding provider. Eligible non-private text may leave only
through the configured scope/egress checks, after secret redaction, with a
bounded budget and an egress-audit record. Training-data egress requires an
explicit separate authorization.

## Learning-quality boundary

Retrieval volume and dataset row counts are not success. Success is measured by
relevant scoped retrieval, explicit or clearly labeled inferred feedback,
fully graded selected training packs, contamination-resistant author
verification, and blind evaluation against a frozen bar.

No reliability fix may weaken a safeguard merely to make a failing check pass.
