# ocbrain v0.2 — Autonomy + Dataset Factory Implementation Spec

> **Historical design doc.** This is the approved v0.2 build spec, preserved as
> a record of that implementation contract. The shipped system has since moved
> to v0.3 (targeted judge, embeddings, the light/heavy autopilot split,
> `excerpt_render`, the human-bootstrap pin, the stallcheck watchdog); for
> current behavior read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md).

**Status:** APPROVED FOR BUILD (synthesized 2026-07-08 from Design A "autonomy+labeling" and Design B "dataset factory")
**Owner mandate (Jonathan Gu, 2026-07-08, overrides prior human-gated doctrine):** ocbrain is dramatically upgraded, NOT human gated. It automatically compiles knowledge, labels good/bad, and over time builds the ultimate dataset for a fine-tuned "Jonathan Gu agent" LLM.

**Non-negotiables (orchestrator constraints):**

- Repo `/Users/guclaw/.openclaw/workspace/ocbrain`, Python 3.13, uv, stdlib-only style (plain `sqlite3`, `argparse`, frozen dataclasses, `MaintenanceResult`-shaped returns).
- Live DB `data/ocbrain.sqlite` is 834 MB + 337 MB WAL. **Snapshot-first before any write run** (checkpoint or copy db+wal+shm together).
- **Additive schema migration only.** Keep evidence / knowledge / knowledge_evidence / memory view / retrieval_uses / brain_events core.
- Keep the hash-chained `brain_events` audit trail and the privacy scope composition ratchet (`private < workspace < project < public`). These are automatic invariants, not human gates.
- Human approval gates are **removed entirely**, replaced with automatic safeguards: tripwire → auto-quarantine (a status change, never a human queue), injection-scan on harvested third-party content before it can reach the injectable memory view, egress audit retained. Everything else auto-approves, including capability/prescriptive knowledge.
- New `ocbrain autopilot` command: harvest → compile → auto-label → promote → prune/heal → dataset-export; idempotent; safe every 30 min via launchd.
- Dataset factory outputs to `data/datasets/` (JSONL + manifest): SFT chat examples, DPO preference pairs from corrections, persona/voice set with Jonathan's own messages as assistant targets. Every example carries provenance evidence ids + privacy scope + quality label + confidence. Content-hash dedup. **Dataset never leaves the machine.**
- Hermes lessons honored: post-turn background review (triggers: 5+ tool-call successes, error recovery, user corrections, novel workflows); decay + consolidation against knowledge rot; provenance no-clobber against self-edit clobbering; injection-scan + quarantine tripwires against unvalidated self-writes (arXiv 2605.13471 "sleeper channels").

---

## 0. Conflict resolutions (merge decisions)

The two half-designs collide in six places. Resolutions, binding for all lanes:

| # | Collision | Resolution |
|---|-----------|------------|
| R1 | **Config**: Design A used ad-hoc `cfg.judge.*`/`cfg.promote.*`; Design B had `dataset/config.py:DatasetConfig.from_env()`. Duplicate key: A's correction cue heuristics vs B's `correction_threshold`. | ONE config module `src/ocbrain/config.py` with `OcbrainConfig` (frozen dataclass of section dataclasses: `judge`, `labels`, `promote`, `quarantine`, `review`, `dataset`, `autopilot`). Loaded from optional JSON at `$OCBRAIN_CONFIG` (default `data/ocbrain.config.json`) + env overrides. `dataset/config.py` is **not created**; `DatasetConfig` is the `dataset` section. Single `correction.threshold = 0.6` key shared by the review signal miner and DPO mining. |
| R2 | **Correction detection duplicated**: A's `user_correction` signal detector vs B's `mine_dpo.correction_score`. | ONE implementation: `correction_score(text) -> float` and `AFFIRMATION_RE` live in `src/ocbrain/text.py` (shared text-utility home). `review.py` (signals) and `dataset/mine_dpo.py` (pairs) both import it. Transcript parsing is owned solely by `dataset/transcripts.py`; `review.py` consumes its `Session`/`Turn` DTOs — no second parser. |
| R3 | **Watermark ledgers**: A's `harvest_watermarks` (rowid/time watermarks) vs B's `dataset_sources` (file fingerprints). | Both exist, different jobs. `harvest_watermarks(domain, stream)` for monotonic rowid/timestamp cursors over DB tables and JSON state files; `dataset_sources(source_uri, dataset)` for on-disk transcript fingerprints (`file_fingerprint` = path+size+mtime_ns). Never mix. |
| R4 | **brain_events kinds**: both designs need "autopilot did X" events, but `EVENT_KINDS` is mirrored in a CHECK constraint (db.py:143-152) and adding kinds forces a table rebuild. | **No new event kinds in v0.2.** Auto-decisions reuse `compilation_decided` (actor `ocbrain-autopilot`); quarantine reuses `correction_recorded` (op `demote`); run telemetry goes to a new `autopilot_runs` table. Event-kind expansion deferred to v0.3 with the `_migrate_retrieval_uses` rebuild pattern. |
| R5 | **Quality-label vocabularies**: knowledge labels (A) vs dataset example labels (B). | `knowledge.quality_label IN ('good','bad','neutral')`. `dataset_examples.quality_label IN ('good','neutral','bad','excluded')` — `excluded` exists only at the dataset layer (scrub/dedup rejects). Enforced in code, not CHECK, for the knowledge columns (additive ALTER cannot add CHECKs). |
| R6 | **Egress tension**: "dataset never leaves the machine" vs A's hosted LLM judge. | Dataset examples are NEVER sent to any hosted endpoint — export target class is `local_model` only. The judge operates on *knowledge rows* only, gated per row by composed scope (`private` never leaves) + `egress_allowed(scope, ctx, 'hosted_teacher')` + `redact_secrets`, every batch writing an `egress_audits` row. Judge is optional (disabled when API key env is unset); autopilot runs fine without it. |
| R7 | **Shared text scanners**: A adds `INJECTION_PATTERNS`; B adds an entropy scanner "shared with the injection-scan half". | Both land in `src/ocbrain/text.py`: `find_probable_injection()` and `find_high_entropy_spans()`. Quarantine tripwires, the injectable guard, and dataset scrubbing all import from there. |
| R8 | **Two scope systems** (relational `privacy_scope` ladder vs event-core `ScopeTag`). | Unchanged and both kept. Dataset rows and knowledge rows filter on composed `privacy_scope` (via `most_restrictive_scope` over ALL linked evidence, db.py:682). Event-core-sourced dataset pairs map `ScopeTag` → `privacy_scope`: `confidential`/`secret` visibility, `client` scope type, or non-`hosted_ok` egress ⇒ `'private'`; else `'workspace'`. `egress_allowed()` (scope.py:177) remains the enforcement primitive for judge and export target checks. |

---

## 1. Unified additive SQL migration

Applied by `_migrate_schema(conn)` in `db.py` (called from `init_db`, db.py:223). New tables use `CREATE TABLE IF NOT EXISTS` appended to `SCHEMA`; new columns are applied conditionally via a new helper:

```python
def _ensure_column(conn, table: str, column: str, decl: str) -> None:
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
```

**No table rebuilds. No CHECK-constraint changes to existing tables.** The `memory` VIEW is dropped and recreated (views hold no data — this is additive-safe). The existing `DROP TABLE IF EXISTS` legacy block at the top of `SCHEMA` is untouched.

### 1.1 New columns on existing tables

```sql
-- knowledge (labels enforced in code, not CHECK)
ALTER TABLE knowledge ADD COLUMN origin TEXT;               -- 'human'|'autopilot'|'loop'|'harvest'|'backfill'; NULL (all 101,641 existing rows) treated as non-human
ALTER TABLE knowledge ADD COLUMN quarantine_reason TEXT;    -- NULL = not quarantined; else tripwire slug
ALTER TABLE knowledge ADD COLUMN quality_label TEXT;        -- 'good'|'bad'|'neutral'|NULL(unlabeled)
ALTER TABLE knowledge ADD COLUMN quality_confidence REAL;
ALTER TABLE knowledge ADD COLUMN quality_updated_at TEXT;
ALTER TABLE knowledge ADD COLUMN promote_score REAL;

-- evidence
ALTER TABLE evidence ADD COLUMN injection_scan_status TEXT; -- NULL=unscanned | 'clean' | 'flagged'
ALTER TABLE evidence ADD COLUMN injection_scan_hits TEXT;   -- JSON array of pattern names
```

### 1.2 New tables

```sql
CREATE TABLE IF NOT EXISTS signal_events (
  id TEXT PRIMARY KEY,                 -- stable_id('sig', source, source_ref, kind, content_hash(canonical_json(details)))
  kind TEXT NOT NULL,                  -- taxonomy slug (§5.2)
  polarity TEXT NOT NULL,              -- 'good'|'bad'|'neutral'
  weight REAL NOT NULL,                -- 0..1
  source TEXT NOT NULL,                -- 'session'|'retrieval'|'learning_db'|'commitments'|'cron'|'events'|'maintenance'|'judge'
  source_ref TEXT NOT NULL,            -- file path+offset or table:rowid
  session_key TEXT,
  knowledge_id TEXT REFERENCES knowledge(id),
  evidence_id TEXT REFERENCES evidence(id),
  details TEXT,                        -- JSON
  occurred_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sig_knowledge ON signal_events(knowledge_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sig_kind ON signal_events(kind, created_at);
CREATE INDEX IF NOT EXISTS idx_sig_session ON signal_events(session_key);

CREATE TABLE IF NOT EXISTS harvest_watermarks (
  domain TEXT NOT NULL,                -- 'autolabel'|'tripwires'|'review'|'injection_scan'|...
  stream TEXT NOT NULL,                -- 'retrieval_uses'|'brain_events'|'knowledge'|'learning.db:learnings'|file path...
  watermark TEXT NOT NULL,             -- stringified rowid / ISO ts / content hash / per-job ms
  updated_at TEXT NOT NULL,
  PRIMARY KEY (domain, stream)
);

CREATE TABLE IF NOT EXISTS judge_runs (
  id TEXT PRIMARY KEY,                 -- stable_id('judge', request_hash, ts)
  ts TEXT NOT NULL,
  model TEXT NOT NULL,
  status TEXT NOT NULL,                -- 'ok'|'error'|'skipped_budget'|'skipped_egress'
  item_count INTEGER NOT NULL DEFAULT 0,
  request_hash TEXT,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  cost_usd REAL NOT NULL DEFAULT 0,
  response_json TEXT,                  -- verdicts ONLY; never the dispatched bodies
  egress_audit_id TEXT                 -- egress_audits.id
);
CREATE INDEX IF NOT EXISTS idx_judge_ts ON judge_runs(ts);

CREATE TABLE IF NOT EXISTS autopilot_runs (
  id TEXT PRIMARY KEY,                 -- stable_id('run', started_at)
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'running',  -- 'running'|'ok'|'partial'|'error'
  stages_json TEXT,                    -- {stage: MaintenanceResult-dict|error}
  error TEXT
);

CREATE TABLE IF NOT EXISTS dataset_examples (
  id TEXT PRIMARY KEY,                 -- stable_id('dsx', dataset, content_hash)
  dataset TEXT CHECK (dataset IN ('sft','dpo','persona')) NOT NULL,
  content_hash TEXT NOT NULL,          -- ids.content_hash over canonical_json of messages/pair ONLY (not metadata)
  dedup_key TEXT NOT NULL,             -- text.claim_key near-dup key
  source_kind TEXT CHECK (source_kind IN
    ('openclaw_session','claude_session','codex_session',
     'correction_event','git_commit','authored_doc')) NOT NULL,
  source_uri TEXT,                     -- transcript path | 'git://<repo>#<sha>' | brain_events id
  source_span TEXT,                    -- JSON: message ids / line offsets
  evidence_ids TEXT NOT NULL,          -- JSON array, >=1 (provenance)
  privacy_scope TEXT CHECK (privacy_scope IN ('private','workspace','project','public'))
    NOT NULL DEFAULT 'workspace',      -- most_restrictive_scope over all linked evidence
  quality_label TEXT CHECK (quality_label IN ('good','neutral','bad','excluded')) NOT NULL,
  quality_confidence REAL,
  quality_reasons TEXT,                -- JSON array of fired rule names
  n_turns INTEGER,
  n_chars INTEGER,
  example_json TEXT NOT NULL,          -- full scrubbed JSONL record (metadata included)
  session_id TEXT,
  occurred_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(dataset, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_dsx_dataset_label ON dataset_examples(dataset, quality_label);
CREATE INDEX IF NOT EXISTS idx_dsx_dedup ON dataset_examples(dataset, dedup_key);
CREATE INDEX IF NOT EXISTS idx_dsx_occurred ON dataset_examples(dataset, occurred_at);

CREATE TABLE IF NOT EXISTS dataset_sources (
  source_uri TEXT NOT NULL,
  dataset TEXT NOT NULL,
  fingerprint TEXT NOT NULL,           -- fsutil.file_fingerprint() at mining time
  mined_at TEXT NOT NULL,
  examples_emitted INTEGER NOT NULL DEFAULT 0,
  status TEXT CHECK (status IN ('mined','skipped','error')) NOT NULL DEFAULT 'mined',
  detail TEXT,
  PRIMARY KEY (source_uri, dataset)
);

CREATE TABLE IF NOT EXISTS dataset_exports (
  id TEXT PRIMARY KEY,                 -- stable_id('dsexp', dataset, payload_hash, ts)
  ts TEXT NOT NULL,
  dataset TEXT NOT NULL,
  path TEXT NOT NULL,
  min_scope TEXT NOT NULL,
  min_label TEXT NOT NULL,
  format TEXT NOT NULL,                -- 'chat' | 'openai-preference'
  example_count INTEGER NOT NULL,
  excluded_count INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  payload_hash TEXT NOT NULL,          -- sha256 of the JSONL bytes
  manifest_json TEXT NOT NULL,
  egress_audit_id TEXT
);
```

### 1.3 Memory view rebuild (quarantine exclusion)

```sql
DROP VIEW IF EXISTS memory;
CREATE VIEW memory AS
  SELECT * FROM knowledge
  WHERE status = 'current' AND inject = 1 AND quarantine_reason IS NULL;
```

`list_current_knowledge` (db.py:808) gains `AND quarantine_reason IS NULL` and new ordering `ORDER BY inject DESC, promote_score DESC, confidence DESC, updated_at DESC` (NULL promote_score sorts last via `COALESCE(promote_score, -1)`).

### 1.4 Migration ops procedure

1. `PRAGMA wal_checkpoint(TRUNCATE)` on the live DB, then copy `data/ocbrain.sqlite` → `data/ocbrain-pre-v2-autonomy-<ts>.sqlite` (matches existing snapshot naming in `data/`). If checkpoint is blocked by readers, copy db+wal+shm together.
2. Run `init_db` (new SCHEMA + `_migrate_schema`) once via `ocbrain init` against a **copy** first; verify `PRAGMA integrity_check`, row counts unchanged, `memory` view returns 0 rows (expected — no `inject=1` rows exist in prod).
3. Apply to live. Migration is idempotent; autopilot also calls `init_db` on every run, so drift self-heals.
4. Optional (deferred decision, see Open Questions): one-time `UPDATE knowledge SET origin='backfill' WHERE origin IS NULL` — code treats NULL as non-human either way.

---

## 2. Module inventory

### 2.1 New files

| File | Responsibility | Lane |
|------|---------------|------|
| `src/ocbrain/config.py` | `OcbrainConfig` + section dataclasses + `load_config()` (JSON file + env overrides). Single source of every tunable in §3. | 1 |
| `src/ocbrain/fsutil.py` | `file_fingerprint(path)`, `history_runtime(path)` moved out of `cli.py` (re-exported from cli for back-compat) so dataset/review code never imports the CLI module. | 1 |
| `src/ocbrain/safeguards.py` | `quarantine_knowledge`, `release_quarantine`, `scan_evidence_for_injection`, `run_tripwires` + `TRIPWIRES` registry, `auto_decide_compilations`, `THIRD_PARTY_SOURCE_TYPES`. | 2 |
| `src/ocbrain/autolabel.py` | `Signal` dataclass, `record_signal`, taxonomy miners (`mine_retrieval_signals`, `mine_event_signals`, `mine_learning_db`, `mine_commitments`, `mine_cron_state`), `attribute_signals`, `fold_labels`, `label_from_signals`, top-level `autolabel()`. | 3 |
| `src/ocbrain/judge.py` | `judge_ambiguous`, `eligible_rows`, `build_judge_batch`, `call_openai` (stdlib urllib), `spent_today`; budget + egress + redaction gating. | 3 |
| `src/ocbrain/review.py` | Hermes-style post-turn background review: iterate settled sessions (via `dataset.transcripts`), fire harvest triggers (5+ tool-call success, error recovery, user corrections, novel workflow), emit `Candidate`s → `upsert_knowledge(origin='harvest', gate='auto')` + session `Signal`s. | 3 |
| `src/ocbrain/promote.py` | `promote_to_memory`, `promotion_eligible`, `promote_score`, `demote_and_decay`; context-budget simulation against `build_excerpt`. | 3 |
| `src/ocbrain/autopilot.py` | `run_autopilot(conn, cfg)` — stage orchestration (§4), flock, daily snapshot, `autopilot_runs` ledger, per-stage time budgets. | 5 |
| `src/ocbrain/dataset/__init__.py` | Re-exports (`mine_all`, `export_all`, DTOs). | 4 |
| `src/ocbrain/dataset/transcripts.py` | `Turn`/`Session` DTOs; `parse_openclaw_session`, `parse_claude_session`, `parse_codex_session`, `parse_transcript`; `is_conversation_transcript`; `classify_user_text` (telegram envelope / injected / media / bare); `iter_unmined_transcripts`; `INJECTED_PREFIXES`. | 4 |
| `src/ocbrain/dataset/mine_sft.py` | `mine_sft`, `segment_exchanges`, `label_exchange`, `sft_example`. | 4 |
| `src/ocbrain/dataset/mine_dpo.py` | `mine_dpo`, `DpoPair`, `find_transcript_pairs`, `find_event_pairs` (edit-decisions, corrections, heal supersessions). | 4 |
| `src/ocbrain/dataset/mine_persona.py` | `mine_persona`, `telegram_examples`, `commit_examples`, `doc_examples`, `is_style_admissible`. | 4 |
| `src/ocbrain/dataset/quality.py` | Scrub/exclusion rules (§7.4): secret residue, entropy blobs, length bounds, near-dup, refusal-only, error-dump, managed-block leak, envelope residue. | 4 |
| `src/ocbrain/dataset/export.py` | `export_all`, deterministic byte-idempotent JSONL writers, manifest, `dataset_exports` rows, egress audit, `--min-scope/--min-label/--verified-only` filters. | 5 |
| `src/ocbrain/dataset/stats.py` | Growth stats: per-dataset counts by label/scope/source_kind/week, dedup rate, export history. | 5 |
| `ops/com.jonathangu.ocbrain.autopilot.plist` | launchd job (§9). | 5 |

### 2.2 Modified files

| File | Change | Lane |
|------|--------|------|
| `src/ocbrain/db.py` | §1 migration + `_ensure_column`; **delete gate-forcing block db.py:533-535**; `upsert_knowledge` gains `origin=`, `actor=` params, human no-clobber guard, injectable injection-scan guard, `origin=COALESCE(knowledge.origin, excluded.origin)` and `quarantine_reason=knowledge.quarantine_reason` in the ON CONFLICT SET; new `admit_knowledge`; `approve_knowledge`/`reject_knowledge` become thin deprecated wrappers (gate predicates dropped); `list_current_knowledge` quarantine filter + ordering. | 1 |
| `src/ocbrain/text.py` | Add `INJECTION_PATTERNS` + `find_probable_injection`; `find_high_entropy_spans`; `correction_score` + `AFFIRMATION_RE` (shared by review + DPO, per R2). | 1 |
| `src/ocbrain/events.py` | **Delete** `approval_packet`, `approval_packet_item`, `telegram_approval_text` (events.py:513-598). KEEP `decide_compilation` and `hard_blocked_belief`. | 2 |
| `src/ocbrain/excerpt.py` | Belt-and-suspenders: query already excludes quarantined via `list_current_knowledge`; additionally scan each rendered line with `find_probable_injection` + `find_probable_secret_leaks` and drop on hit. | 2 |
| `src/ocbrain/maintenance.py` | `heal_conflicts` emits `heal_superseded` signals (winner good 0.3 / loser bad 0.4) and hands winner/loser pairs to a callback the dataset lane consumes; `check_loop_liveness` unchanged (its tripwire-as-evidence pattern is the template for quarantine evidence). | 2 |
| `src/ocbrain/loops.py` | `knowledge_from_lesson` (loops.py:249) + `knowledge_from_candidate_summary` (loops.py:315): `gate="human"` → `gate="auto"`; add `origin="loop"` to every `upsert_knowledge` call. | 2 |
| `src/ocbrain/mcp.py` | Delete `brain.propose` tool + `write_proposal` import; kill `allow_writes` asymmetry — `serve(db_path)` only, all `PermissionError` branches and the `tool_list(allow_writes)` split removed; `brain.proposals` stays read-only listing. | 2 |
| `src/ocbrain/proposals.py` | **DELETED** (proposal-markdown human queue). | 2 |
| `src/ocbrain/cli.py` | Delete `propose` subparser (cli.py:338) + `cmd_propose` (cli.py:1354) + `--approval-packet` flag (cli.py:662-663); `--allow-writes` kept as documented no-op; new subcommands: `autopilot`, `quarantine list|release`, `label` (manual signal), `dataset-mine`, `dataset-export`, `dataset-stats`; `file_fingerprint`/`history_runtime` re-imported from `fsutil`. | 5 |
| `pyproject.toml` | version 0.2.0. No new dependencies (stdlib only). | 5 |

Lane 2 must also grep-audit `dream.py`, `teacher.py`, `retrieve.py` for `gate=`/`approve` couplings and route them through `admit_knowledge`.

---

## 3. Config surface

All keys in `OcbrainConfig` (`src/ocbrain/config.py`), JSON file `data/ocbrain.config.json` (optional) + `OCBRAIN_*` env overrides. Defaults:

```
[autopilot]
lock_path            = data/autopilot.lock
snapshot_dir         = data/snapshots/          # daily snapshot, rotate keep 3
snapshot_keep        = 3
stage_budget_seconds = 300                       # per stage; total must fit 30-min cadence
runtimes_excerpt     = []                        # write_excerpt targets (opt-in)

[review]
settle_minutes         = 30      # session file must be idle this long before review
min_tool_calls_success = 5       # Hermes task-success trigger
session_roots          = [~/.openclaw/agents, ~/.claude/projects, ~/.codex]

[correction]
threshold = 0.6                  # shared: review user_correction signal + DPO pair mining

[labels]
half_life_days  = 30
good_threshold  = 0.35
bad_threshold   = -0.35
min_mass        = 0.6
hard_bad_weight = 0.9            # >= this weight bad signal wins outright

[quarantine]
bad_feedback_count       = 2
bad_feedback_window_days = 7
thrash_count             = 3
thrash_window_days       = 14

[promote]
min_confidence = 0.6
max_injected   = 40
max_chars      = 6000
decay_days     = 30
bootstrap_min_confidence = 0.85  # zero-signal bootstrap path (memory view empty in prod)

[judge]                          # optional; disabled when api key env unset
enabled          = true
api_key_env      = OPENAI_API_KEY   # value NEVER persisted, logged, or printed
model            = gpt-5-mini
daily_usd_cap    = 0.50
batch_size       = 20
per_run_item_cap = 100
signal_weight    = 0.4
price_per_mtok   = {model: {prompt: ..., completion: ...}}

[dataset]
sft_min_assistant_chars = 80
sft_max_context_turns   = 12
sft_max_context_chars   = 16000
dpo_side_chars          = [40, 8000]
include_tool_turns      = false
tool_result_truncate    = 500
persona_author_ids      = ["<TELEGRAM_SENDER_ID>", "<TELEGRAM_USERNAME>"]   # telegram sender_id / username — config, never name-matching
persona_direct_agents   = ["main"]
persona_git_repos       = []     # empty = discover */.git one level under ~/.openclaw/workspace
persona_git_authors     = ["Jonathan Gu", "<GIT_AUTHOR_EMAIL>"]
persona_authored_globs  = []     # docs source OFF by default
persona_system_prompt   = "You are Jonathan Gu. Reply as Jonathan would."
export_dir              = data/datasets
export_min_scope        = workspace   # 'private' rows never export
export_min_label        = good
learning_db             = ~/.openclaw/learning.db
commitments_path        = ~/.openclaw/commitments/commitments.json
cron_state_path         = ~/.openclaw/cron/jobs-state.json
```

---

## 4. Autopilot pipeline

`ocbrain autopilot` → `autopilot.run_autopilot(conn, cfg)`. Designed for launchd every 30 min; every stage idempotent; safe to kill at any point.

### 4.1 Stage order

| # | Stage | Function | Idempotency mechanism |
|---|-------|----------|----------------------|
| 0 | lock | `fcntl.flock` on `data/autopilot.lock`; exit 0 silently if held | single-instance |
| 1 | snapshot | daily: `PRAGMA wal_checkpoint(TRUNCATE)` then copy db → `data/snapshots/ocbrain-YYYYMMDD.sqlite`; rotate keep-3; skip if today's exists | date-named file |
| 2 | migrate | `init_db(conn)` (SCHEMA + `_migrate_schema`) | `IF NOT EXISTS` / `_ensure_column` |
| 3 | harvest | existing history import path (fingerprint-gated, `import_history_file`) → new evidence rows | `UNIQUE(source_uri,content_hash)` + fingerprints |
| 4 | injection-scan | `scan_evidence_for_injection` over new third-party evidence | `harvest_watermarks('injection_scan','evidence')` rowid |
| 5 | review | post-turn review of settled sessions → Candidates (`upsert_knowledge(origin='harvest')`) + session Signals | `harvest_watermarks('review', <path>)` + stable ids |
| 6 | compile | `auto_decide_compilations` over undecided proposals; ONE `rebuild_projection` at end (never per-item — fold is O(304,923 events)) | `check_existing` in `decide_compilation` |
| 7 | autolabel | miners → `record_signal` → `attribute_signals` → `fold_labels` → `judge_ambiguous` → fold again | stable signal ids (`INSERT OR IGNORE`) + watermarks |
| 8 | tripwires | `run_tripwires` → auto-quarantine | `harvest_watermarks('tripwires','knowledge')` |
| 9 | promote | `promote_to_memory` + `demote_and_decay`; then `write_excerpt` for `cfg.autopilot.runtimes_excerpt` | deterministic re-rank; excerpt block splice is idempotent |
| 10 | maintain | `prune_knowledge` + `heal_conflicts` (emits signals + supersede pairs) | existing TTL logic |
| 11 | dataset-mine | `mine_sft`, `mine_dpo`, `mine_persona` (time-budgeted) | `dataset_sources` fingerprints + `UNIQUE(dataset, content_hash)` |
| 12 | dataset-export | `export_all` — skip-if-unchanged via `payload_hash` | byte-deterministic writer |
| 13 | finalize | write `autopilot_runs` row (per-stage MaintenanceResult dicts), release lock | — |

### 4.2 Failure semantics

Each stage runs in try/except; a stage failure records the error in `stages_json`, sets run status `partial`, and **continues** to independent later stages — except stages 1-2 (snapshot/migrate), whose failure aborts the run with status `error`. Per-stage wall-clock budget `cfg.autopilot.stage_budget_seconds`; miners accept `time_budget_seconds` and return early with their watermark advanced only past fully-processed items.

### 4.3 Watermark design

- DB-table streams: max rowid processed, stored as string in `harvest_watermarks`.
- File streams (commitments/cron JSON): content hash or per-job `updatedAtMs`; stable signal ids make re-emission harmless.
- Transcript files: `dataset_sources.fingerprint` (path+size+mtime_ns); session files are append-only, so a changed fingerprint re-parses the file — `UNIQUE(dataset, content_hash)` dedups previously-mined examples, and `review` re-emission dedups on stable signal/candidate ids.
- Watermarks are written in the same transaction as the work they cover.

---

## 5. Autonomy: gate removal, safeguards, labeling

### 5.1 Gate removal — exact cut points

1. **db.py:533-535** — delete the forcing block (`if prescriptive or knowledge_type=='capability' or risk in {'high','critical'}: gate='human'; ...`). `gate` becomes vestigial (column kept — additive-only; all new writes pass `gate='auto'`; nothing consults it).
2. **db.py `upsert_knowledge`** — new signature params `origin='autopilot'`, `actor='ocbrain'`. **No-clobber guard** (Hermes self-edit-clobbering fix): if the existing row has `origin='human'` and `actor` does not start with `human`, skip the write, insert a `signal_events` row (`kind='clobber_refused'`, polarity neutral), return the id unchanged. ON CONFLICT SET adds `origin=COALESCE(knowledge.origin, excluded.origin)` (first writer wins) and `quarantine_reason=knowledge.quarantine_reason` (upserts never clear quarantine). **Injectable guard**: caller passes `inject=True` → run `find_probable_injection` + `find_probable_secret_leaks` over `value_text or title or ''`; any hit forces `inject=0` and stamps `quarantine_reason='injection_scan:<hits>'`.
3. **db.py `approve_knowledge`/`reject_knowledge`** — drop `gate='human' AND status='candidate'` predicates; new canonical `admit_knowledge(conn, id, *, actor='ocbrain-autopilot')` — candidate → current iff `quarantine_reason IS NULL`, stamps `approved_by=actor`. Old functions become deprecated wrappers.
4. **proposals.py deleted**; `cli.py` `propose` subparser + `cmd_propose` deleted; `mcp.py` `brain.propose` deleted.
5. **events.py:513-598 deleted** (`approval_packet`, `approval_packet_item`, `telegram_approval_text`); `--approval-packet` removed from cli. **KEEP** `decide_compilation` (the mechanism) and `hard_blocked_belief` (automatic tripwire).
6. **safeguards.auto_decide_compilations(conn, *, actor='ocbrain-autopilot', limit=500)** — for each undecided proposal: decision `'shadow'` (the ready-made quarantine analog) if `find_probable_injection(body)` hits, `hard_blocked_belief` is true, or `reward_band=='discard'`; else `'approve'`. All with `rebuild=False`; one `rebuild_projection` at the end.
7. **mcp.py allow_writes asymmetry removed** — all tools always listed; `scripts/ocbrain-mcp` and launchd invocations keep working because `--allow-writes` remains a CLI no-op.
8. **loops.py** — `gate='auto'` + `origin='loop'` everywhere (see §2.2).

### 5.2 Signal taxonomy (unified table)

Signals are frozen `Signal` dataclasses persisted to `signal_events` via `record_signal` (stable id → `INSERT OR IGNORE`, idempotent).

| kind | polarity | base weight | source | detector |
|---|---|---|---|---|
| `user_correction` | bad | 0.8 | session (review.py) | `correction_score(text) >= correction.threshold` on user msg within 3 messages after an assistant action; **also feeds DPO pair mining (same detector, R2)** |
| `user_thanks` | good | 0.6 | session | `AFFIRMATION_RE` (`thanks|perfect|great|nice|love it|ship it|beautiful`) |
| `user_approval` | good | 0.5 | session | `yes|approved|lgtm|go ahead|do it` directly after an assistant proposal |
| `task_closeout_success` | good | 0.7 | session | closeout w/ verification evidence, no correction before session settle |
| `test_pass` / `test_fail` | good 0.4 / bad 0.6 | session | toolResult matching `(\d+) passed`, `FAILED`, `✕`, `AssertionError` |
| `deploy_success` / `deploy_failure` | good 0.6 / bad 0.6 | session | fly/CI result strings in toolResults |
| `revert` | bad | 0.7 | session | `git revert` / `git reset --hard` / "rolling back" after a change |
| `error_recovery` | good | 0.6 | session | toolResult error → different-approach success (also a harvest trigger) |
| `retrieval_feedback` | improved 0.5 / helpful 0.4 / used 0.4 good; harmful 0.9 / failed 0.6 bad; irrelevant 0.2 / ignored 0.1 neutral | retrieval | `retrieval_uses` rowid watermark; carries `knowledge_id` natively |
| `learning_gate_rule` | bad | `conf * min(1, 0.5 + 0.1*recurrence)` | learning_db | `learning.db learnings` active GATE/CORRECTION rows (pre-labeled negatives); each `prevention_rule` also emitted as prescriptive Candidate (origin='harvest') |
| `gate_violation` | bad | 0.7 | learning_db | `learning.db gate_violations`; `content_snippet` kept in details for DPO |
| `commitment_outcome` | completed/fulfilled good 0.5; expired/missed bad 0.5 | commitments | `commitments.json` status |
| `cron_run` | ok+delivered good 0.3; error bad 0.5 | cron | `jobs-state.json` per-job state, watermark per-job `updatedAtMs` |
| `hard_correction_event` | bad | 1.0 | events | `brain_events` kind=`correction_recorded`, body.hard=true |
| `verifier_result` | passed good 0.5 / failed bad 0.6 | events | `evidence.verifier_status` joined through `knowledge_evidence` |
| `heal_superseded` | loser bad 0.4 / winner good 0.3 | maintenance | emitted by `heal_conflicts` |
| `clobber_refused` | neutral | 0.1 | events | no-clobber guard fired (audit breadcrumb) |
| `llm_judge` | verdict | `judge.signal_weight` (0.4) | judge | §5.4; can never override hard human signals |

`attribute_signals(conn, limit=2000)`: for `knowledge_id IS NULL` rows, match via `claim_key` + FTS `search()` top-1 with score cutoff; session-only signals stay unattributed and still label dataset examples by `session_key`.

### 5.3 Label fold

`fold_labels(conn, cfg)`: per knowledge row with new signals since watermark, decayed signed score `S = Σ sign(polarity)·weight·0.5^(age_days/half_life)`, mass `M = Σ weight·decay`.

- **Hard-bad precedence**: any bad signal with weight ≥ `labels.hard_bad_weight` (0.9) ⇒ label `bad`, confidence = that weight. The LLM judge can never override this.
- Else `good` if `S/M ≥ 0.35` and `M ≥ 0.6`; `bad` if `S/M ≤ -0.35` same mass; else `neutral`.
- Confidence `min(0.95, |S|/M · n/(n+1))`.
- Writes `quality_label`, `quality_confidence`, `quality_updated_at`. Human-origin rows get labels (they inform the dataset) but labels never mutate their status/inject. A label flip to `bad` on an injected row triggers immediate demotion + tripwire check.

### 5.4 LLM judge (optional, budget-capped, egress-audited)

Eligibility: `quality_label='neutral'` with mass ≥ 0.3 (conflicting evidence) plus zero-signal promotion candidates, ordered by `promote_score` desc, capped `judge.per_run_item_cap`. Per row: compose scope via `most_restrictive_scope` over row + all linked evidence; **drop `private`**; drop rows failing `egress_allowed(scope_tag, ctx, 'hosted_teacher')`; `redact_secrets` every body. Batch (20/call), strict-JSON verdict prompt `[{"id","label","confidence","rationale"}]`. Skip run with `judge_runs.status='skipped_budget'` when `spent_today ≥ judge.daily_usd_cap`. Every dispatched batch writes `egress_audits` via `record_egress_audit` (included/rejected ids + payload hash). `judge_runs.response_json` stores verdicts only, never dispatched bodies. API key from `os.environ[judge.api_key_env]` — never persisted, logged, or printed. Verdicts fold as `llm_judge` signals.

### 5.5 Quarantine tripwires

Quarantine is encoded additively: `quarantine_reason IS NOT NULL` ⇒ out of the memory view, out of `list_current_knowledge`, out of promotion. `quarantine_knowledge` sets `inject=0`, demotes `current→candidate`, stamps reasons, writes an `autopilot_tripwire` evidence row (the `check_loop_liveness` pattern, maintenance.py:272) linked `relation='contradicts'`, and records a `correction_recorded` event (op `demote`, existing kind — audit chain intact). `release_quarantine(conn, id, *, actor, reason)` is the only path back.

`run_tripwires` registry (rows in `('candidate','current')`, not quarantined, touched since watermark):

| tripwire | fires when |
|---|---|
| `injection_suspected` | any linked third-party evidence flagged by scan, or body flags (`THIRD_PARTY_SOURCE_TYPES = {openclaw_history_file, codex_history_file, claude_history_file, unknown_history_file, session_harvest, web}`) |
| `secret_leak` | `find_probable_secret_leaks` on value_text/title |
| `bad_feedback_spike` | ≥ 2 `harmful`/`failed` retrieval outcomes within 7 days |
| `hard_correction` | `correction_recorded` event with hard=1 targeting this id (mirrors `hard_blocked_belief`) |
| `contradiction_thrash` | superseded/re-upserted ≥ 3× in 14 days |
| `prescriptive_unverified_serving` | `prescriptive=1 OR type='capability'` serving with `inject=1`, zero `verifier_status='passed'` evidence, no good approval signal — the automatic replacement for the old human gate on the risky class |

### 5.6 Injection scan

`text.INJECTION_PATTERNS` (named regex list): `ignore_previous`, `role_hijack`, `tool_coax`, `exfil_link`, `base64_blob`, `invisible_chars`, `prompt_leak_probe` (exact patterns per Design A §1.2). `find_probable_injection(text) -> list[str]` mirrors `find_probable_secret_leaks` (text.py:51). Three enforcement choke points:

1. `promote.py` — mandatory scan of body + ALL linked third-party evidence claims before `inject=1`.
2. `db.upsert_knowledge` injectable guard (§5.1-2).
3. `excerpt.build_excerpt` — rendered lines scanned; hits dropped (belt-and-suspenders).

### 5.7 Promotion / decay

`inject=1` requires ALL: (1) `status='current'`, not quarantined; (2) `quality_label='good'` with `quality_confidence ≥ 0.6` — bootstrap exception while signals are sparse (prod memory view is empty): `confidence ≥ 0.85` AND ≥1 `verifier_status='passed'` evidence; (3) injection scan clean; (4) risky class (`prescriptive OR capability OR risk high/critical`) additionally requires passed-verifier evidence OR a `user_approval`/`user_thanks` signal.

`promote_score = 0.4·quality_confidence + 0.25·0.5^(age_days/30) + 0.2·use_rate + 0.15·scope_rank_bonus`, `use_rate = (improved+helpful+used)/max(1, served)`. Top `promote.max_injected` (40) win; `build_excerpt` dry-run enforces `promote.max_chars` (6000), overflow demoted lowest-score-first. Immediate demotion on label flip to bad/neutral, confidence < 0.4, quarantine, or budget overflow. Served-but-never-useful within 30 days ⇒ `promote_score *= 0.5`. `origin='human'` rows with `inject=1` are pinned — never auto-demoted by score.

---

## 6. Post-turn review (Hermes pattern)

`review.py` runs inside autopilot stage 5 over **settled** sessions (file idle ≥ `review.settle_minutes`), parsed by `dataset.transcripts` (single parser, R2). Triggers → outputs:

| trigger | condition | output |
|---|---|---|
| task success | ≥ `review.min_tool_calls_success` (5) tool calls, zero trailing errors, no correction | `task_closeout_success` signal + `Candidate(target=SKILL/WIKI)` "what worked" summary |
| error recovery | error → different approach → success | `error_recovery` signal + `Candidate` capturing the recovery recipe |
| user correction | `correction_score ≥ correction.threshold` | `user_correction` signal (DPO pair minted later by `mine_dpo` from the same detector) |
| novel workflow | tool-sequence claim_key unseen in `signal_events` history | low-confidence `Candidate(target=WIKI)` |

Candidates flow through the existing `schema.Candidate` DTO → `upsert_knowledge(origin='harvest', gate='auto', status='candidate')` + `link_knowledge_evidence` to the session's evidence rows (privacy ratchet applies automatically) → admitted by `admit_knowledge` once labeled good, or auto-pruned. Review never overwrites human rows (no-clobber guard is in `upsert_knowledge` itself).

---

## 7. Dataset factory

### 7.1 Corpus + parsing (verified formats)

History evidence rows are file-level fingerprints; the factory re-parses transcripts from disk via `evidence.source_uri` (81,847 of 96,892 openclaw rows are real `/sessions/` transcripts; the predicate excludes plugin/config junk, `*.trajectory.jsonl`, `*.trajectory-path.json`, `*.codex-app-server.json`, `sessions.json`, `codex-home/.tmp/`). Three parsers (openclaw session lines, claude-code project JSONL, codex `rollout-*.jsonl`) normalize to `Session`/`Turn`. Rules: consecutive same-role lines collapse; `thinking`/`reasoning` blocks NEVER enter text (other models' CoT is not Jonathan-agent signal); tool results become role='tool' turns truncated to 500 chars, excluded from exported messages in v0.2 (`include_tool_turns=false`), tool activity summarized in metadata; `tool_errors` via `ERROR_RESULT_RE`.

`classify_user_text` kinds: `telegram_envelope` (parse `Conversation info (untrusted metadata):` fenced JSON; `authored_by` set iff sender_id/username ∈ `persona_author_ids`), `injected` (`INJECTED_PREFIXES` table — subagent context, boot checks, heartbeats, cron, compaction flushes, ~70% of main-agent user turns), `media`, `bare` (human-authored when session agent ∈ `persona_direct_agents`, but `sender_verified=false`).

### 7.2 Mining

**SFT** (`mine_sft.py`): exchange = context (≤12 non-injected turns, ≤16,000 chars, head-trimmed) + final assistant text turn (≥80 chars). Labels: **good** — affirmation follow-up (0.9), Hermes task-success trigger ≥5 tool calls clean (0.7), error-recovery arc (0.8); **bad** — correction follow-up, refusal/apology, terminal tool failure, abandonment (retained, never exported to SFT; feeds DPO); **neutral** — else (0.5); **excluded** — quality.py rejects. Cross-ref boost: linked `retrieval_uses` good outcomes +0.1, harmful/failed ⇒ bad. Sessions whose only user turns are injected yield nothing (kills orchestrator→subagent lanes as SFT — correct).

**DPO** (`mine_dpo.py`): (A) transcript pairs A1 → U(correction ≥ 0.6) → …An: prompt = context through the ORIGINAL request (correction excluded); rejected = A1; chosen = final accepted attempt (walk forward while corrections continue; accept on affirmation/topic-change/clean end). Require `claim_key(chosen) != claim_key(rejected)`, both sides in `dpo_side_chars`, both survive scrub. conf = correction score (cap 0.9; 0.95 when U states the right answer). (B) event-core pairs (0 today, grows as autopilot runs): `compilation_decided decision='edit'` (rejected=proposal body, chosen=edited_body, prompt synthesized from evidence excerpts, conf 0.85); `correction_recorded` op edit/reframe (hard ⇒ conf 0.95; mark_wrong/retract without replacement ⇒ no pair — that's a hard-block, not a preference); `heal_conflicts` supersessions (loser=rejected, winner=chosen, value-type only). Event-pair scope: ScopeTag → privacy_scope per R8.

**Persona** (`mine_persona.py`): (1) Telegram his-side — envelope-verified turns become assistant TARGETS (`messages = [system: persona_system_prompt, user: preceding assistant turn ≤4000 chars, assistant: Jonathan's message verbatim post-scrub]`); openers skipped by default; bare unverified texts admitted with `sender_verified=false` metadata and −0.2 confidence (`--verified-only` excludes). (2) Git commits — `git log --author` per configured repo, EXCLUDING agent-authored (`Co-Authored-By: Claude`, `🤖 Generated with`); prompt = `git show --stat` (≤2000 chars), target = subject+body; each commit upserts a `git_commit` evidence row so persona examples carry real provenance ids. (3) Authored docs — OFF by default (`persona_authored_globs` empty; the 376 memory files are mostly agent-written and would poison voice).

### 7.3 Provenance, scope, dedup

Every example: `evidence_ids` = the transcript/commit/event evidence row ids (≥1, enforced); `privacy_scope = most_restrictive_scope(*evidence_scopes)` (db.py:682 — defaults 'workspace'); `content_hash` over canonical-JSON of messages/pair ONLY (stable across re-mines) with `UNIQUE(dataset, content_hash)`; `dedup_key = claim_key(...)` for the near-dup pass (later duplicates marked `excluded` reason `near_dup`, keeping earliest-highest-confidence). Note: ~4.3% of knowledge/evidence rows are content-duplicates that survive ID-level dedup — dataset dedup is deliberately its own pass and does not rely on DB uniqueness.

### 7.4 Quality scrub/exclusion rules (`dataset/quality.py`)

Applied to every candidate example before storage; failures ⇒ `quality_label='excluded'` + reason:

1. `secret_residue` — `redact_secrets` first; if `find_probable_secret_leaks` still hits post-redaction ⇒ exclude.
2. `entropy_blob` — `find_high_entropy_spans` (long base64/hex runs) not redactable ⇒ exclude.
3. `length` — target < 40 chars or example > 32,000 chars.
4. `near_dup` — dedup_key already stored for this dataset.
5. `refusal_only` — assistant target is only an apology/refusal.
6. `error_dump` — target is mostly stack trace/tool noise.
7. `managed_block_leak` — text contains `BEGIN/END OCBRAIN MANAGED BLOCK` (never train on injected memory).
8. `envelope_residue` — unparsed `Conversation info (untrusted metadata)` fragments remain.
9. `injection_flagged` — `find_probable_injection` hits inside the target text.

### 7.5 JSONL schemas

All records are single-line JSON, UTF-8, deterministic key order (`canonical_json`), ordered by `(occurred_at, id)`.

**SFT + persona (`format: chat`)**:

```json
{"messages": [
   {"role": "system", "content": "..."},
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "..."}
 ],
 "metadata": {
   "id": "dsx_...", "dataset": "sft|persona",
   "source_kind": "openclaw_session", "source_uri": "...", "session_id": "...",
   "evidence_ids": ["evd_..."], "privacy_scope": "workspace",
   "quality_label": "good", "quality_confidence": 0.9,
   "quality_reasons": ["affirmation"], "n_tool_calls": 7,
   "sender_verified": true, "occurred_at": "2026-07-01T...", "content_hash": "..."}}
```

**DPO (`format: openai-preference`)**:

```json
{"input": {"messages": [{"role": "user", "content": "..."}]},
 "preferred_output":     [{"role": "assistant", "content": "<chosen>"}],
 "non_preferred_output": [{"role": "assistant", "content": "<rejected>"}],
 "metadata": {"id": "dsx_...", "dataset": "dpo",
   "correction_kind": "transcript|event_edit|event_correction|supersedes",
   "hard": false, "confidence": 0.85, "source_uri": "...",
   "evidence_ids": ["evd_..."], "privacy_scope": "workspace",
   "occurred_at": "...", "content_hash": "..."}}
```

### 7.6 Export (`dataset/export.py`)

- Stable paths: `data/datasets/sft.jsonl`, `data/datasets/dpo.jsonl`, `data/datasets/persona.jsonl`, `data/datasets/manifest.json`.
- Filters: `min_label` (default `good`; `--min-label neutral` widens), `min_scope` (default `workspace`; **`private` rows never export regardless of flags**), `--verified-only` (persona).
- Byte-idempotent: deterministic ordering + canonical JSON ⇒ identical corpus produces identical bytes; if new `payload_hash` equals the last `dataset_exports` row's, skip the write.
- Every export writes a `dataset_exports` row and an `egress_audits` row via `record_egress_audit` (target `local_model` — audit trail even for local writes; included/excluded counts + payload hash). Export target class is hard-coded `local_model`; there is no hosted export path. **The dataset never leaves the machine.**
- Manifest: `{generated_at, config_hash, datasets: {sft: {path, count, bytes, sha256, label_counts, scope_counts, excluded_count}, dpo: {...}, persona: {...}}}`.
- `dataset-stats` (`dataset/stats.py`): per-dataset counts by label/scope/source_kind/ISO-week of `occurred_at`, dedup/exclusion rates, export history — the growth curve toward "the ultimate dataset".

---

## 8. CLI additions (`cli.py`, lane 5)

```
ocbrain autopilot [--stage <name>] [--dry-run]
ocbrain quarantine list | release <knowledge_id> --actor human:jonathan --reason "..."
ocbrain label <knowledge_id> --outcome good|bad --note "..."     # manual Signal, weight 0.9, source 'session'
ocbrain dataset-mine  [--dataset sft|dpo|persona] [--limit N] [--time-budget S]
ocbrain dataset-export [--dataset ...] [--min-scope workspace] [--min-label good] [--verified-only]
ocbrain dataset-stats
```

Removed: `propose` (+ `cmd_propose`), `event-proposals --approval-packet`. Kept as no-op: `--allow-writes` (help text notes deprecation) so `scripts/ocbrain-mcp` and existing plists don't break.

---

## 9. launchd

`ops/com.jonathangu.ocbrain.autopilot.plist` (install: `cp` to `~/Library/LaunchAgents/` then `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.jonathangu.ocbrain.autopilot.plist`). Verify the `uv` path with `which uv` at install time (assumed `/opt/homebrew/bin/uv`).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.jonathangu.ocbrain.autopilot</string>
  <key>ProgramArguments</key>
  <array>
    <string>/opt/homebrew/bin/uv</string>
    <string>run</string>
    <string>--directory</string>
    <string>/Users/guclaw/.openclaw/workspace/ocbrain</string>
    <string>ocbrain</string>
    <string>autopilot</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>OCBRAIN_DB</key>
    <string>/Users/guclaw/.openclaw/workspace/ocbrain/data/ocbrain.sqlite</string>
  </dict>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityBackgroundIO</key><true/>
  <key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key>
  <string>/Users/guclaw/.openclaw/workspace/ocbrain/data/logs/autopilot.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/guclaw/.openclaw/workspace/ocbrain/data/logs/autopilot.err.log</string>
</dict>
</plist>
```

**Secrets:** the judge API key is NOT placed in the plist. `config.load_config()` reads `judge.api_key_env` from the process environment; when absent the judge stage records `skipped` and the run proceeds. If Jonathan wants the judge under launchd, the key goes in a `chmod 600` env file (e.g. `data/secrets/judge.env`, gitignored) sourced by a small wrapper script — never committed, never logged, never echoed.

Overlap safety: the flock (§4.1 stage 0) makes StartInterval + a slow run safe — the second invocation exits 0 immediately.

---

## 10. Test plan

New/updated test files (pytest, in-memory or tmp-path sqlite via `connect()`+`init_db`; judge HTTP mocked with a stub `call_openai`; git tested against a throwaway repo in tmp_path). Existing `test_db_flow.py` updated for the wrapper semantics of approve/reject.

| Test file | Covers | ~tests | Lane |
|---|---|---|---|
| `tests/test_migration_v2.py` | fresh init vs migrate-existing parity; `_ensure_column` idempotency; memory view excludes quarantined; re-running init_db is a no-op; legacy DROP block untouched; counts preserved on a seeded old-schema DB | 10 | 1 |
| `tests/test_text_scanners.py` | each INJECTION_PATTERN fires + clean-text negatives; `find_high_entropy_spans`; `correction_score` lexicon incl. affirmation-zeroing; existing secret patterns still pass | 12 | 1 |
| `tests/test_db_noclobber.py` | human-origin no-clobber (skip + `clobber_refused` signal); first-writer-wins origin; upsert never clears quarantine; injectable guard forces inject=0 + quarantine_reason; `admit_knowledge` paths; deprecated wrappers | 10 | 1 |
| `tests/test_safeguards.py` | quarantine/release round-trip incl. tripwire evidence + correction event; each of 6 tripwires fires and watermark advances; `auto_decide_compilations` approve/shadow branches + single rebuild + hard-block respect | 14 | 2 |
| `tests/test_mcp.py` (update) | `brain.propose` gone; all tools listed without allow_writes; writes work | 4 | 2 |
| `tests/test_loops.py` (update) | loop knowledge lands gate='auto' origin='loop'; excerpt line-scan drop | 4 | 2 |
| `tests/test_autolabel.py` | 5 miners with fixture stores; stable-id idempotency (double-run emits once); attribution via claim_key/FTS; fold thresholds/mass/decay; hard-bad precedence over judge; watermark advance | 14 | 3 |
| `tests/test_judge.py` | eligibility; private-scope + egress drops; redaction before dispatch; daily budget skip; cost accounting; verdict→signal fold; response stores verdicts only | 8 | 3 |
| `tests/test_review.py` | settle gating; 4 triggers; Candidate → upsert(origin='harvest'); signal emission dedup on re-review | 8 | 3 |
| `tests/test_promote.py` | 4 eligibility rules; bootstrap exception; risky-class verifier/approval requirement; top-N + char-budget overflow demotion; label-flip demotion; human-pin; score decay | 10 | 3 |
| `tests/test_transcripts.py` | 3 parsers on verified-format fixtures; sidecar/junk exclusion predicate; envelope classification + author verification; INJECTED_PREFIXES with timestamp strip; thinking dropped; tool truncation; fingerprint re-mine on append | 12 | 4 |
| `tests/test_mine_sft.py` | segmentation bounds; good/bad/neutral labels; injected-only session yields nothing; retrieval cross-ref; provenance + composed scope | 10 | 4 |
| `tests/test_mine_dpo.py` | correction detection; multi-correction walk-forward; claim_key inequality; side-length bounds; event edit/correction/supersede pairs; hard mapping; ScopeTag→privacy_scope map | 10 | 4 |
| `tests/test_mine_persona.py` | envelope-verified targets; opener skip; unverified −0.2 + flag; commit mining excl. agent-authored; git evidence rows; docs off by default | 8 | 4 |
| `tests/test_dataset_quality.py` | all 9 exclusion rules fire; redaction-then-pass path; near-dup keeps first | 8 | 4 |
| `tests/test_dataset_export.py` | byte-determinism (two runs identical); skip-if-unchanged; min_scope/min_label/verified-only filters; private never exports; manifest contents; dataset_exports + egress_audits rows; DPO format shape | 10 | 5 |
| `tests/test_autopilot.py` | stage order; flock single-instance; snapshot daily-skip + rotation; stage failure ⇒ partial + later stages run; snapshot/migrate failure aborts; autopilot_runs ledger; stage time budget | 8 | 5 |
| `tests/test_cli_v2.py` | new subcommands parse + dispatch; propose gone; --allow-writes no-op; dataset-stats output | 6 | 5 |

**Target: ~166 tests total (≥150 gate).** All existing tests must stay green (`test_scope_core.py` untouched).

---

## 11. Work breakdown — 5 parallel lanes, disjoint file ownership

Every file appears in EXACTLY one lane. Lanes build against this spec's interfaces, not each other's branches.

### Lane 1 — foundations (schema, db, shared text/config)
**Files:** `src/ocbrain/db.py`, `src/ocbrain/text.py`, `src/ocbrain/config.py` (new), `src/ocbrain/fsutil.py` (new), `tests/test_migration_v2.py`, `tests/test_text_scanners.py`, `tests/test_db_noclobber.py`, `tests/test_db_flow.py` (update).
**Brief:** §1 migration + `_ensure_column`; §5.1 items 1-3 (gate-block deletion, no-clobber, injectable guard, admit_knowledge, wrappers); memory-view rebuild + `list_current_knowledge` filter/ordering; text scanners (`find_probable_injection`, `find_high_entropy_spans`, `correction_score`, `AFFIRMATION_RE`); config module (§3); fsutil move with cli re-export shim left for lane 5.

### Lane 2 — safeguards & gate demolition
**Files:** `src/ocbrain/safeguards.py` (new), `src/ocbrain/events.py`, `src/ocbrain/excerpt.py`, `src/ocbrain/maintenance.py`, `src/ocbrain/loops.py`, `src/ocbrain/mcp.py`, `src/ocbrain/proposals.py` (delete), `tests/test_safeguards.py`, `tests/test_mcp.py`, `tests/test_loops.py`.
**Brief:** §5.5 quarantine + tripwire registry; §5.1 items 4-8 (approval-packet deletion, auto_decide_compilations, mcp allow_writes removal, loops gate flip); heal_conflicts signal emission + supersede-pair callback; excerpt belt-and-suspenders; audit `dream.py`/`teacher.py`/`retrieve.py` for gate couplings (report, don't own).

### Lane 3 — autolabel, judge, review, promotion
**Files:** `src/ocbrain/autolabel.py` (new), `src/ocbrain/judge.py` (new), `src/ocbrain/review.py` (new), `src/ocbrain/promote.py` (new), `tests/test_autolabel.py`, `tests/test_judge.py`, `tests/test_review.py`, `tests/test_promote.py`.
**Brief:** §5.2-5.4 signal taxonomy/miners/fold/judge; §6 post-turn review (consumes `dataset.transcripts` DTOs per spec — develop against fixture Sessions until lane 4 lands); §5.7 promotion/decay. Imports from lane 1 (text, config, db) and dataset DTO shapes from this spec.

### Lane 4 — dataset mining
**Files:** `src/ocbrain/dataset/__init__.py`, `src/ocbrain/dataset/transcripts.py`, `src/ocbrain/dataset/mine_sft.py`, `src/ocbrain/dataset/mine_dpo.py`, `src/ocbrain/dataset/mine_persona.py`, `src/ocbrain/dataset/quality.py`, `tests/test_transcripts.py`, `tests/test_mine_sft.py`, `tests/test_mine_dpo.py`, `tests/test_mine_persona.py`, `tests/test_dataset_quality.py`.
**Brief:** §7.1-7.4 — three verified-format parsers, envelope classifier, SFT/DPO/persona miners, quality scrub, provenance + composed scope, `dataset_sources` incremental ledger. Build fixtures from the verified line formats in this spec, not from live transcripts.

### Lane 5 — pipeline, export, CLI, ops (integration lane, lands last)
**Files:** `src/ocbrain/autopilot.py` (new), `src/ocbrain/dataset/export.py`, `src/ocbrain/dataset/stats.py`, `src/ocbrain/cli.py`, `ops/com.jonathangu.ocbrain.autopilot.plist` (new), `pyproject.toml`, `tests/test_autopilot.py`, `tests/test_dataset_export.py`, `tests/test_cli_v2.py`.
**Brief:** §4 stage machine (lock, snapshot, budgets, autopilot_runs); §7.5-7.6 deterministic export + manifest + egress audit + stats; §8 CLI wiring incl. propose deletion and fsutil re-export shim; §9 launchd; version bump.

### Integration order

1. **Lane 1 merges first** (everything imports it). Gate: migration tests green against a COPY of the live DB; snapshot taken.
2. **Lanes 2, 3, 4 build in parallel** against lane-1 main + this spec's interfaces; merge in any order once green (no shared files; lane 3's review↔lane 4's transcripts contract is the frozen DTO shapes in §7.1).
3. **Lane 5 merges last**, wiring CLI/autopilot over all of it; runs the full suite + a dry-run `ocbrain autopilot` against a DB copy, then first live run manually before loading the launchd job.
4. Per shared-tree ops doctrine: each lane in its own worktree, no `git add -A`, no pushes while another integration is in flight.

---

## 12. Open questions (non-blocking; defaults chosen)

1. **Judge egress**: hosted OpenAI judging of workspace-scope knowledge bodies (redacted, scope-gated, audited, $0.50/day cap) — confirm Jonathan accepts, or set `judge.enabled=false`/point at a local model. Default: enabled but inert until the API key env is provided.
2. **Origin backfill**: run the one-time `UPDATE knowledge SET origin='backfill' WHERE origin IS NULL`, or leave NULL (treated as non-human either way). Default: leave NULL.
3. **Persona identity defaults**: confirm telegram `sender_id <TELEGRAM_SENDER_ID>` / username `<TELEGRAM_USERNAME>` and git author strings before first persona export.
4. **Content-duplicate consolidation**: ~4.3% duplicate content in knowledge/evidence — v0.2 handles it at the dataset layer only; a knowledge-consolidation (merge) pass is deferred to v0.3.
5. **Status-enum rebuild**: a proper `status='quarantined'` value needs the `_migrate_retrieval_uses` rebuild pattern — deferred to v0.3; v0.2 uses the additive `quarantine_reason` column.
6. **New brain_events kinds** (e.g. `autopilot_run`, `quarantine_applied`): deferred to v0.3 with the same rebuild pattern; v0.2 reuses existing kinds + `autopilot_runs`.
