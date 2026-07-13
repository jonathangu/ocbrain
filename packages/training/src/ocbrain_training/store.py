"""Companion-owned SQLite store for training artifacts and provenance."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_TRAINING_DB = Path(
    os.environ.get("OCBRAIN_TRAINING_DB", "~/.ocbrain/training.sqlite")
).expanduser()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS companion_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_runtime TEXT,
  source_uri TEXT,
  content_hash TEXT NOT NULL,
  claim TEXT NOT NULL,
  artifact_uri TEXT,
  artifact_hash TEXT,
  verifier_status TEXT DEFAULT 'unknown',
  loop_tags TEXT,
  project TEXT,
  privacy_scope TEXT DEFAULT 'workspace',
  occurred_at TEXT,
  ingested_at TEXT NOT NULL,
  UNIQUE(source_uri, content_hash)
);

CREATE TABLE IF NOT EXISTS dataset_examples (
  id TEXT PRIMARY KEY,
  dataset TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  dedup_key TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  source_uri TEXT,
  source_span TEXT,
  evidence_ids TEXT NOT NULL,
  privacy_scope TEXT NOT NULL DEFAULT 'workspace',
  quality_label TEXT NOT NULL,
  quality_confidence REAL,
  quality_reasons TEXT,
  n_turns INTEGER,
  n_chars INTEGER,
  example_json TEXT NOT NULL,
  session_id TEXT,
  occurred_at TEXT,
  grade_score REAL,
  grade_json TEXT,
  grade_model TEXT,
  grade_prompt_version TEXT,
  graded_at TEXT,
  train_class TEXT,
  train_class_reason TEXT,
  train_classified_at TEXT,
  train_selected INTEGER NOT NULL DEFAULT 0,
  train_selection_rank INTEGER,
  train_selection_reason TEXT,
  train_selected_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(dataset, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_dsx_dataset_label
  ON dataset_examples(dataset, quality_label);
CREATE INDEX IF NOT EXISTS idx_dsx_dedup ON dataset_examples(dataset, dedup_key);
CREATE INDEX IF NOT EXISTS idx_dsx_occurred ON dataset_examples(dataset, occurred_at);
CREATE INDEX IF NOT EXISTS idx_dsx_grade ON dataset_examples(dataset, grade_score);
CREATE INDEX IF NOT EXISTS idx_dsx_train_class
  ON dataset_examples(dataset, train_class, grade_score);
CREATE INDEX IF NOT EXISTS idx_dsx_train_selected
  ON dataset_examples(dataset, train_selected, train_class, grade_score);

CREATE TABLE IF NOT EXISTS dataset_sources (
  source_uri TEXT NOT NULL,
  dataset TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  mined_at TEXT NOT NULL,
  examples_emitted INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'mined',
  detail TEXT,
  PRIMARY KEY (source_uri, dataset)
);

CREATE TABLE IF NOT EXISTS dataset_grade_runs (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  endpoint TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  item_count INTEGER NOT NULL DEFAULT 0,
  error_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  request_hash TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS idx_dataset_grade_runs_ts ON dataset_grade_runs(ts);

CREATE TABLE IF NOT EXISTS dataset_calibrations (
  eval_id TEXT PRIMARY KEY,
  winner TEXT NOT NULL,
  reason TEXT,
  ideal_response TEXT,
  ideal_response_source TEXT,
  preference_strength TEXT,
  labeled_by TEXT,
  labeled_at TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS egress_audits (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  target TEXT NOT NULL,
  context_json TEXT NOT NULL,
  query TEXT,
  included_json TEXT NOT NULL,
  rejected_json TEXT NOT NULL,
  payload_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_exports (
  id TEXT PRIMARY KEY,
  ts TEXT NOT NULL,
  dataset TEXT NOT NULL,
  path TEXT NOT NULL,
  min_scope TEXT NOT NULL,
  min_label TEXT NOT NULL,
  min_grade REAL,
  format TEXT NOT NULL,
  example_count INTEGER NOT NULL,
  excluded_count INTEGER NOT NULL,
  bytes INTEGER NOT NULL,
  payload_hash TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  egress_audit_id TEXT
);
"""


def connect_training(path: Path | str = DEFAULT_TRAINING_DB) -> sqlite3.Connection:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    init_training_store(conn)
    return conn


def init_training_store(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR REPLACE INTO companion_meta(key, value) VALUES ('schema', 'ocbrain-training-v1')"
    )
    conn.commit()


__all__ = ["DEFAULT_TRAINING_DB", "connect_training", "init_training_store"]
