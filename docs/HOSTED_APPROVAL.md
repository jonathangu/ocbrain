# Hosted Approval

How evidence that is not yet allowed to reach a hosted model becomes a belief
that is — and why it can only happen through a human at the CLI.

## The problem

Default ingest stamps an `approval_required` egress policy on ordinary
internal evidence: it may leave this machine only after someone decides it
may. Until that decision, the evidence is invisible to hosted delivery —
`_delivery_sql('hosted_model')` requires `egress_policy='hosted_ok'`, and no
runtime write path can grant it. That is the design property: **an unattended
write can never reach hosted-model delivery.** It can only narrow, or ask.

Before this change, "ask" was informal. Now it is an event.

## Two verbs

Two human-gated verbs now move material toward hosted delivery, for two
different shapes of pending work:

- **`egress-promote`** lifts a belief that is **already compiled**: it selects
  current beliefs (by id, or in bulk with `--scope-id … [--provenance P]`) and
  writes one `egress_promoted` event per belief — egress only, scope and body
  untouched, refusing confidential/secret exactly like `curated-apply`
  (added 2026-09-04).
- **`hosted-approve`** compiles **evidence that has no belief yet** through the
  approved-compile path documented below.

The daily loop for new evidence is `hosted-queue` → `hosted-approve`.
`egress-promote --scope-id …` is for bulk lifts of curated beliefs that are
already compiled and known-safe.

## The verbs (CLI-only)

    ocbrain hosted-queue [--project PROJECT] [--writer WRITER] [--since ISO] [--limit N] [--pretty]
    ocbrain hosted-approve (--all-from-queue | EVIDENCE_ID...)
                          --approved-by human:NAME [--reason TEXT]
                          [--project PROJECT] [--dry-run]

CLI-only means CLI-only: neither verb is in `RUNTIME_TOOLS`, neither appears
in any MCP tool catalogue, and no MCP tool can stamp an egress policy. The
approval requires `--approved-by` spelled `human:NAME` — the same spelling
`scope-promote` and `event-forget` use — and that string is recorded as the
actor on the decision event.

### hosted-queue (read-only)

Lists, in one JSON payload:

- `queue`: evidence rows with `egress_policy='approval_required'`, visibility
  not `confidential`/`secret`, and **no current belief** already covering them
  (via `belief_evidence` → `current_beliefs`). One entry per row:
  `evidence_id`, `kind`, `writer`, `scope_id`, `recorded_at`, and a 120-char
  `body_head`.
- `proposals`: pending `hosted_egress_proposal` events (see below), pending
  exactly as long as the evidence they name is still uncompiled.

The verb opens the core without migrating or committing anything, so it is
safe to point at the live operator database. Filters: `--project` matches
`scope_id='project:<name>'` (proposals: the `requested_scope` they ask for),
`--writer` matches the recording writer, `--since` a recorded-at timestamp.

### hosted-approve (human-gated)

For each selected evidence row the verb runs a gauntlet, refusing with a typed
reason and writing nothing for that row when any check fails:

| Refusal | Condition |
| --- | --- |
| `visibility_confidential` / `visibility_secret` | confidential or secret evidence is never eligible, even with `--project` |
| `egress_not_approval_required` | the queue only promotes `approval_required` rows; `local_only` rows need their own review (the local-only audit) |
| `already_compiled` | a current belief already links the evidence |
| `client_scope_requires_project` | client-scoped evidence must be retargeted by an explicitly named project |
| `secret_leak_body` | the body trips the same secret scanner `public-safety-check` uses on tracked files |

Eligible rows are compiled through the **existing approved-compile path** — a
`compilation_proposed` event followed by `decide_proposal_v1` approving it —
so the resulting belief is built by exactly the machinery `event-compile
--approve` uses: belief row served from `current_beliefs`, `belief_evidence`
link, decision event written by the `human:NAME` actor. The belief is scoped
`project:<name>` (from `--project`, or the evidence's own project scope),
`visibility='internal'`, `egress_policy='hosted_ok'`,
`provenance='human_approved_hosted'`, confidence 0.8. When no project is
derivable the belief keeps the evidence's own scope: reach never widens past
what the row already had.

The target scope resolves in that order: `--project` wins outright; with no
`--project`, the evidence's own pending widening proposal names the project its
request asked for (approving the queue is answering that request, so a
task-scoped ingest's `project:hosted_ok` request lands where it asked instead
of at a task id nobody queries); only when neither applies does the belief keep
the evidence row's scope. Visibility is never taken from a request — it is
always `internal`. The per-row result's `scope_source` says which rule applied:
`cli_project`, `requested_by_proposal`, or `evidence_row` (with
`answers_proposal` naming the proposal event when one was answered).

`--dry-run` runs the whole gauntlet and reports what would happen without
writing a single event. Exit codes: `0` when at least one row was approved
(or the queue was empty), `2` when nothing was approved and there were
refusals, or the invocation was malformed (no ids selected, or an
`--approved-by` that is not `human:NAME`).

## The widening proposal (D2)

`brain.ingest` has always advertised a `scope` argument in its tool schema;
the v1 dispatcher silently dropped it. Now:

- A requested scope that keeps the **same canonical scope identity** and only
  narrows visibility and/or egress is honored with `provenance='explicit'`.
  Scope IDs are not hierarchical, so lateral sibling IDs and unprovable
  cross-family retargeting are proposals rather than unattended writes.
- A request that changes reach or widens visibility/egress is never applied.
  The evidence is stored under the inferred scope, and a
  `hosted_egress_proposal` event
  records the request: requested scope, inferred scope, writer, evidence id,
  and a 160-char body excerpt. The ingest receipt names the proposal event
  (`scope_decision: "hosted_egress_proposal"`) and the proposal appears in
  `event-proposals` output with no new verb — it is a request on the ledger,
  not a decision.

So the loop closes: an agent that wants hosted reach for a fact can only file
the request; `hosted-queue` shows it next to the row it names; a human closes
it by approving the evidence (the proposal drops out of the queue once the
evidence is compiled) or by leaving it.

## Provenance chain

    evidence (approval_required, inferred)
        └─ ingestion: brain.ingest (narrowing honored / widening proposed)
        └─ human:     ocbrain hosted-approve --approved-by human:NAME
        └─ events:    compilation_proposed → compilation_decided (actor human:NAME)
        └─ belief:    hosted_ok, project-scoped, human_approved_hosted

Every hop is an event; nothing is stamped in place.
