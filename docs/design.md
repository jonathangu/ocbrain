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
