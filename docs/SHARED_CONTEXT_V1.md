# Shared Context and the OCBrain v1 Core

Status: implementation contract for the v1 core.

## The product boundary

OCBrain is not a design handbook and it is not a prompt-sized scrapbook. It is
the local, source-backed bridge that lets Codex, Claude Code, OpenClaw, and
future runtimes begin from the same relevant state and return evidence about
what happened.

The scarce resource is model attention, not stored information. The durable
system therefore keeps the full evidence lake outside the prompt, retrieves a
small candidate set, enforces scope and provenance mechanically, gives the
model a bounded context packet, and expands exact sources only on demand.

Long context and retrieval are complementary here:

1. Retrieval filters an effectively unbounded, changing local history.
2. `ocbrain.context.v1` assembles a bounded dossier with coverage and
   contradiction metadata.
3. `brain.source` lets the model inspect the few full sources needed for global
   reasoning, omission detection, or comparison.
4. The runtime does the work.
5. `ocbrain.closeout.v1` links the result, artifacts, verification, and decision
   impact back to the retrievals that informed it.

The retrieval index is replaceable infrastructure. The evidence and outcome
ledger is the product.

## What remains useful to future models

Store raw observations and versioned interpretations separately. A future
model must be able to reinterpret old activity without trusting today's
embedding model, taxonomy, or compiler.

Every durable semantic object needs:

- an immutable source or event payload and content hash;
- time, actor/runtime, session, task, project/repo/client, and privacy scope;
- a stable object id plus aliases for earlier ids;
- explicit provenance and whether a field was observed, asserted, inferred, or
  verified;
- links between evidence, belief, action, retrieval, and outcome;
- versioned derived features whose producer and schema version are recorded;
- corrections and lifecycle changes as later events, never silent rewrites.

Embeddings, FTS rows, rankings, summaries, labels, and projections are derived
views. They must be rebuildable or replaceable without losing the underlying
record.

## Actions and outcomes are not one universal number

A click, signup, subscription, deployment, test pass, or human correction can
mean different things in different environments. OCBrain must preserve both
the common shape and the local meaning.

An action record should separate:

- **mechanism** — what physically happened, such as click, form submit, code
  edit, deploy, or message;
- **semantic role** — exploration, commitment, conversion, correction,
  verification, rollback, or another versioned category;
- **target** — the object acted on, with page/component/repo/task identity;
- **context** — the state and alternatives visible before the action;
- **policy** — which agent/model/version chose it and what prior context it
  consumed;
- **cost** — latency, money, tokens, human attention, and reversibility;
- **scope/provenance** — where the record may be used and how it was observed.

An outcome should remain a vector, not be prematurely collapsed into reward:

- primary metric and unit;
- guardrail metrics and harms;
- observation window and delay;
- baseline or counterfactual when one exists;
- attribution method and uncertainty;
- verifier evidence;
- local interpretation, such as whether a subscription is the intended goal on
  this site or a misleading proxy.

Learning systems may derive a scalar reward for one experiment, but the ledger
keeps the components. That is how a later model can transfer a useful prior
without assuming that the same surface event means the same thing everywhere.

`ocbrain.closeout.v1` implements this as two optional, versioned arrays. Each
`ocbrain.action.v1` entry requires a mechanism, local semantic role, and target,
and can retain pre-action context, policy/model identity, cost, provenance, and
versioned features. Each `ocbrain.outcome.v1` entry requires a metric, JSON
value, and explicit local interpretation, and can retain its role, unit,
observation window, baseline, counterfactual, attribution, uncertainty, and
versioned features. These envelopes remain embedded in the hash-addressed,
append-only closeout receipt; a later feature pipeline may derive a task-specific
reward without destroying the original components.

## The v1 database boundary

The event ledger is the single semantic authority. Evidence objects, current
beliefs, belief-to-evidence links, aliases, FTS, and retrieval items are
projections of it. The default MCP reads only those projections and never falls
back to the legacy archive.

The core owns:

- the append-only event chain and projections;
- retrieval uses and normalized served items;
- source-handle issuance;
- closeout receipts;
- egress audits;
- local backup, restore, bounded sync, and runtime diagnostics.

Training and watchdog systems are optional companions with their own operational
stores. They may read a core snapshot or submit evidence through the core API;
they are not additional brains, are never queried by the default MCP, and
install no recurring schedule through the core package.

Migration is archive-first. The live v0.x database is opened read-only, copied
to a verified immutable archive, transformed into fresh outputs, and never
replaced or repointed automatically. A candidate is provisionally activated
only after schema, chain, projection, and retrieval-shadow checks; fresh
three-client acceptance decides whether that reversible pointer is retained.

## Acceptance contract

This contract passed locally on 2026-07-13. The three fresh-client closeouts
are `close_e10efadbd024d638` (Codex), `close_5fb53a5e30362451` (Claude Code),
and `close_016a635c9facf310` (OpenClaw); all are linked to scoped v1 retrievals
in the same activated core.

The v1 change is complete only when all of these are true:

- the core database has one semantic plane and no training/watchdog tables;
- the old database and event-chain prefix are preserved and verified;
- every imported evidence, belief, source link, retrieval, and correction is
  represented or explicitly accounted for in the migration manifest;
- wiping and rebuilding projections yields the same hashes and MCP responses;
- the core wheel contains and imports no companion implementation;
- fresh Codex, Claude Code, and OpenClaw processes each complete
  context → source → feedback → closeout against the same core;
- no hosted call, timer, training run, or automatic live-database activation is
  part of those checks.
