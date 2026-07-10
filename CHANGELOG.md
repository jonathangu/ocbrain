# Changelog

## 0.3.3 — 2026-07-10

- Replace the tripwire scan's timestamp-only watermark with a composite
  `(updated_at, id)` cursor, so equal-timestamp rows cannot be skipped at a batch
  or time-budget boundary.
- Replace per-row full event-log deserialization for hard corrections with an
  indexed JSON target lookup. On the live backlog, a clean 1,000-row tripwire
  page fell from 301.2534 seconds to 0.0351 seconds.
- Avoid a full FTS delete scan when inserting a brand-new evidence or knowledge
  row; existing parents still replace their exact index row.
- Retire newly discovered stalls outside the paging backlog window so they do
  not remain `new` and rewrite their ledger every 15 minutes.
- Run liveness checking from independent autopilot maintenance, make unchanged
  deadman evidence idempotent, and record malformed deadlines explicitly.
- Commit an autopilot `running` row and profile deadman before work, checkpoint
  both after every stage, and have the independent stallcheck process page an
  overdue producer deadline, including in read-only dry-run inspection.
- Preserve polymorphic legacy retrieval ids as `task_ref` provenance instead of
  invalid knowledge foreign keys; repair existing orphan references without
  deleting retrieval history.
- Repair a missing event-projection cursor even when compilation has no new
  proposals.
- Exclude runtime logs, data, local caches, build output, and the untracked lock
  file explicitly from source distributions; `logs/` is now gitignored too.


## 0.3.2 — 2026-07-10

- Bound post-turn review inside large sessions at 50 mutating units or two
  seconds, while preserving the commit-before-next-lazy-session boundary.
- Replace conservative per-session lock estimates with measured writer-lock
  wait, total, and maximum telemetry from explicit transactions.
- Move persona/SFT/DPO redaction, serialization, quality scoring, and dedup reads
  outside SQLite writer transactions; evidence and each final example insert
  commit before parsing or scoring the next candidate.
- Require a separate, complete human-label file with named provenance before a
  local judge can pass calibration; embedded machine-authored winners are
  ignored.
- Add a calibration-only mode that cannot open blind pairs, preserve judge
  explanations for audit, and align the evaluator to concise reasoning,
  evidence-aware optionality, and quantified uncertainty without fake precision.
- Record the dated human-grounded gate honestly: 7/8 (87.5%), with the remaining
  concision-versus-reason miss preserved. The v0.3.0 blind result was not rerun.
- Refresh runtime documentation against installed OpenClaw 2026.6.11, Claude
  Code 2.1.206, and Codex CLI 0.144.1 command surfaces.

## 0.3.1 — 2026-07-09

- Commit post-turn review work at each fully processed session before the lazy
  transcript iterator parses the next file.
- Report review's per-session transaction count and conservative total/maximum
  writer-lock upper bounds.
- Add a concurrency regression test that acquires SQLite's writer slot between
  two lazily yielded sessions.
- Commit judge and embedding egress audits before hosted network I/O, then
  commit verdict/vector results per completed provider batch.
- Commit stall findings before optional Telegram paging so notification latency
  never owns the brain database's writer slot.
- Release persona-mining evidence writes before running the next Git subprocess.
- Apply the shared autolabel stage budget to FTS attribution instead of letting
  that substage overrun the light profile indefinitely.
- Finish promotion scoring/eligibility reads before opening bounded score-update
  batches, and commit each tripwire quarantine before evaluating the next row.
- Commit history and doctrine harvests per imported file before reading the next.

## 0.3.0 — 2026-07-09

This is the first licensed release of ocbrain. It turns the earlier public
source into an Apache-2.0 open-source project and aligns the shared brain with
current ChatGPT/Codex, Claude Code, and OpenClaw transcript and MCP surfaces.

### Added

- Named light and heavy autopilot profiles, a shared overlap lock, stage
  budgets, run ledgers, stall detection, and managed runtime excerpts.
- Local-only SFT, DPO, and persona mining; loopback-only dataset grading; and an
  eval-before-train MLX-LM pilot with pinned trainer provenance.
- Frozen-evaluation reuse for later pilots, explicit private persona curation,
  and a judge-calibration gate that runs before blind material is opened.
- Cross-runtime MCP search, feedback, digest, preview, and guarded write tools.
- A tracked-tree privacy scanner and pre-push hook for public releases.

### Changed

- Dataset mining commits bounded batches instead of holding one writer
  transaction across corpus parsing. Autolabel also releases the writer slot
  between source miners and expensive FTS attribution.
- Writer-lock wait, total hold time, and maximum hold time are recorded in stage
  results. Large WAL files are checkpointed only after the dataset writer has
  committed; a blocking reader is reported honestly as busy.
- Current Codex agent-message records are treated as injected context rather
  than persona voice, and OpenClaw-hosted Codex/Claude transcripts retain their
  producing runtime attribution.

### Privacy

Corpus text, references, ratings, local config, database files, and model
weights remain under the gitignored `data/` tree. Public release artifacts
contain source, tests, documentation, and aggregate results only.

### Dated model evidence

The second local voice pilot reused the first pilot's 20 prompts, references,
rubric, held-out hashes, and blind randomization exactly. After adding eleven
canonical first-party examples, ten of which cleared the unchanged local grade
threshold, the candidate improved from 2/20 preferences to 7/20. The reference
still won 13/20, so v0.3.0 treats this as corpus progress and a model-quality
failure, not as a voice-model release.
