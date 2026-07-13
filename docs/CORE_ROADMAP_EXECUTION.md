# OCBrain Core Roadmap Execution Ledger

Started: 2026-07-13 PDT

This ledger records the additive, archive-first implementation of the OCBrain
core roadmap. It is progress evidence, not a substitute for tests or runtime
acceptance.

## Safety Baseline

- Source baseline: `v0.4.0`; Git commit `c5e73ec6cdb149d8c4fd1730bbde0f21e5aea1ab`.
- Existing uncommitted dataset/grader work was preserved in place before edits.
- Light autopilot, heavy autopilot, and stallcheck launchd jobs were confirmed
  unloaded.
- A pre-change database snapshot was stored in the ignored, owner-only data
  directory.
- Backup permissions: owner read/write only.
- Backup `PRAGMA integrity_check`: `ok`.
- No live table will be dropped or vacuumed in place.

## Workstream Status

| Workstream | Status | Evidence |
|---|---|---|
| v0.4.1 safety patch | environment-verified | 81 focused safety tests; inert plists; live hosted/training gates off; Ruff clean |
| v0.5 Shared Context | environment-verified | 20 focused legacy/v1 context and MCP tests; runtime/admin gate enforced |
| v1 core migration/extraction | environment-verified | strict core gate 24/24; full 1.9 GB archive-first rehearsal; exact clone refold |
| Three-runtime acceptance | environment-verified | fresh Codex, Claude Code, and OpenClaw round trips; DB-linked receipts |

## Verified Corrections

- Correcting: the former no-op `--allow-writes` behavior is now an enforced,
  deprecated alias for the admin MCP profile. The default runtime profile keeps
  append-only ingest/closeout and retrieval feedback but hides and rejects
  administrative/destructive tools.
- The three scheduled maintenance/watchdog jobs are currently disabled.
- Hosted knowledge judging and embedding were enabled in the ignored live
  config at baseline; the v0.4.1 local safety change disabled both.

## 2026-07-13 Progress

- Applied the local safety change: both `judge.enabled` and `embed.enabled` are
  now `false` in the ignored operator config, and `load_config` verified both
  values as disabled.
- The three retired launchd labels are persistently disabled in the user domain,
  and their installed `~/Library/LaunchAgents` plist copies are absent. Repository
  copies remain only until the deprecation/archive patch lands.
- Baseline runtime configuration is healthy: Codex and Claude Code point to the
  same `scripts/ocbrain-mcp` launcher; OpenClaw doctor and probe pass against that
  launcher.
- Baseline OpenClaw exposes eleven MCP tools. The post-change runtime profile
  must expose only the bounded context/evidence surface; admin mutation will be
  separately gated.
- Implemented `ocbrain.context.v1`, bounded hash-verified `brain.source`
  expansion, and append-only `ocbrain.closeout.v1` receipts linked to retrieval
  outcomes. The focused safety/context/core-operations suite currently passes
  56 tests.
- Correcting: `brain.teacher_request` now checks the default-disabled
  `teacher.enabled` authority gate even when invoked through the admin profile.
- Correcting: the archive-first v1 migration is now genuinely
  event-authoritative. The fresh core contains the immutable event chain and
  rebuildable projections only; legacy relational state exists solely in the
  immutable archive, while training and watchdog/operations tables exist solely
  in their optional companion databases.
- Integrated v0.4.1/v0.5 baseline verification: all 587 repository tests pass,
  Ruff reports no findings, and `git diff --check` passes.
- Re-verified the combined legacy/v1 MCP and Shared Context contracts after the
  strict-core integration: all 20 focused tests pass.
- Exported a fresh, unbiased named-human Markdown audit packet to the ignored,
  owner-only task-artifact area. It contains all 150 frozen examples, omits the prior AI triage to avoid
  anchoring, and explicitly leaves both the human gate and training authorization
  false.
- The physical v1 split is in progress: core MCP imports no training or watchdog
  companion modules; companion defaults are isolated to
  `~/.ocbrain/training.sqlite` and `~/.ocbrain/ops.sqlite`.
- Extended the closeout contract with optional `ocbrain.action.v1` and
  `ocbrain.outcome.v1` envelopes. Actions retain mechanism plus local semantic
  role/context/policy/cost; outcomes retain metric vectors, counterfactual,
  uncertainty, and explicit local interpretation. The focused context/MCP suite
  remains 20/20 and Ruff/diff checks pass.
- Corrected the installed Codex, Claude Code, and OpenClaw instruction surfaces:
  they now require context → source → feedback → closeout and no longer claim a
  v0.3 autonomous promoter, 15-minute memory rewrite, or paging stall checker.
- Added an ignored, explicit launcher activation pointer. Migration never writes
  it; absent that pointer and `OCBRAIN_DB`, the existing repository database
  remains selected.
- The strict v1 fixture gate passes 24 tests: exact gapped event-prefix
  preservation, corrupt-chain refusal, alias-ordered correction replay,
  conservative legacy scope, body/source hash separation, strict inventory,
  isolated companion artifacts, append-only triggers, and receipt survival
  across full projection rebuild.
- The physically split repository currently passes all 608 regression tests and
  Ruff. Core help is core-only, core MCP eagerly imports no companion module,
  and training/ops default to independent databases.
- Full 1.9 GB archive-first rehearsal produced five owner-only, verified, and
  unactivated artifacts under `data/v1-rehearsal-20260713/`. The strict core has
  671,580 chain-verified events (exact 307,285-event legacy prefix plus 364,295
  deterministic imports), 232,677 evidence objects, 137,530 beliefs, 241,185
  evidence links, 108,222 serving/search documents, 1,725 retrievals, and four
  preserved closeouts.
- Independently rechecked all manifest file hashes, all four database integrity
  checks, zero foreign-key violations, the event-chain head, exact prefix hash,
  exact schema inventory, and equal FTS/source-document counts. The live source
  size and nanosecond mtime remain identical to the manifest finish state; the
  activation pointer is still absent.
- On an APFS clone, wiped and refolded all projections from all 671,580 events.
  Before/after row-stream hashes match for evidence, beliefs, links, aliases,
  search documents, the projection cursor, retrieval/audit ledgers, and all four
  closeout receipts. A representative five-item `OCBrain` context packet and
  its eight source handles also match exactly. SQLite, foreign-key, and FTS
  integrity checks pass. Evidence is preserved in the ignored, owner-only
  rehearsal report; the migration candidate was not modified or activated.
- Created a separate owner-only prospective live core. Its pre-activation SHA-256 is
  byte-identical to the verified immutable candidate and its SQLite integrity
  and foreign-key checks pass. The activation pointer remains absent while the
  clean-install package gate completes.
- Added a 14-test adversarial v1 gate. It caught and fixed two pre-activation
  defects: concurrent appenders could read the same chain head before either
  reserved SQLite's writer slot, and a colliding legacy alias could shadow an
  event-authored canonical object. Appends now reserve before reading the head;
  direct event objects win over aliases in both import orders. The combined
  core/migration/MCP/adversarial suite passes 31/31. The verified rehearsal has
  zero affected collision imports or shadowing aliases, so its projection and
  manifest hashes remain valid.
- Final source/package gate: all 632 tests pass; the adversarial subset is
  14/14; Ruff, diff, and compilation checks pass. Fresh Python 3.12 installs
  verified the core-only CLI, exact eight-tool stdio MCP, an actual
  archive-first migration, isolated training/ops stores, lazy companion
  dispatch, and implicit-v1-mutation refusal. Final wheel digests are retained
  in the ignored, owner-only release evidence.
- Provisionally activated the separate v1 live copy for fresh-client
  acceptance at 2026-07-13T09:10:39Z. Migration did not perform this step; the
  ignored owner-only pointer did. Retention remains conditional on successful
  Codex, Claude Code, and OpenClaw round trips.
- Corrected the MCP initialization instructions themselves: fresh clients now
  receive the Shared Context contract (`context → source → feedback → closeout`)
  plus the on-demand safety boundary, rather than the retired search-first and
  loop-family wording. The final suite passes 632 tests; the final core-wheel
  digest is retained in the ignored, owner-only release evidence.
- Three-runtime acceptance passed against the same activated core. Fresh Codex
  (`ret_3272313222e5c7bb` / `close_e10efadbd024d638`), Claude Code
  (`ret_596c6bc80cbc774b` / `close_5fb53a5e30362451`), and OpenClaw
  (`ret_5a86ca6be307d173` / `close_016a635c9facf310`) each completed context,
  hash-verified source expansion, feedback, and a verified closeout. OpenClaw
  additionally marked and linked the distinct source-expansion retrieval.
  Post-acceptance integrity and foreign-key checks pass, the pointer is
  retained, and the receipt inventory is recorded beside the owner-only live
  database.
