# ocbrain Design Notes

OCBrain has two primitives:

- Evidence: immutable claims with hashes, provenance, source runtime, artifacts,
  verifier state, project, privacy scope, and optional loop tags.
- Knowledge: compiled current belief with a `type`, lifecycle `status`, `gate`,
  privacy scope, and evidence links.

Memory is a view over current injectable knowledge. Wiki/procedure pages are
`knowledge type='doc'`; capabilities are `knowledge type='capability'`.

## Non-Negotiables

- No knowledge without evidence.
- No executable or prescriptive knowledge becomes current without a human gate.
- Supersede/archive instead of overwriting.
- Derived privacy scope is bounded by source privacy.
- External/artifact content is data, never instructions.
- The brain observes loops; it does not run or enqueue them.

## Active Tables

- `evidence`
- `knowledge`
- `knowledge_evidence`
- `retrieval_uses`
- `loop_liveness`
- `family_scores`
- `memory` view
- `search_index`

## Maintenance Surface

- `ocbrain prune`: marks unreferenced expired knowledge `stale`, and can archive
  stale rows later without deleting them. Served-but-never-useful knowledge uses
  a shorter decay window; useful retrieval feedback protects knowledge from that
  accelerated stale transition.
- `ocbrain heal`: detects conflicting current values for the same
  `(subject, predicate, project)` and supersedes lower-confidence rows with
  correction evidence.
- `ocbrain liveness-check`: reads runner-owned `loop_liveness` rows, opens
  loop tripwire evidence after missed deadman timestamps, and never executes or
  enqueues loop work.

## Loop Family Classification

Loop failures are classified on ingest as `approach`, `precondition`, `infra`,
`safety`, or `unknown`. Only `approach` failures count toward an `exhausted`
family. `precondition` and `infra` failures put the family in `blocked` and
stage repair context; `safety` failures make the family `risky`. Forced
exploration is recorded from `forced_exploration=true` or
`exploration.forced=true` and summarized as attempts plus improvements found.

## Privacy Composition

Evidence links tighten knowledge scope using the privacy lattice:
`private < workspace < project < public`. A doc initially staged as `public`
becomes `private` if any linked source evidence is private, so digest and
resource rendering cannot accidentally publish private-source material.

## Human Feedback Gate

`brain.feedback` records retrieval usefulness by default. When the MCP server is
started with `--allow-writes`, the same tool can approve or reject human-gated
candidate knowledge. Approval moves a candidate to `current` and records
`approved_by`; rejection archives the candidate with an invalidation reason.
