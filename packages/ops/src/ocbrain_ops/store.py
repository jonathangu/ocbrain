"""Companion-owned operational ledger for manual ops and watchdog state."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_OPS_DB = Path(os.environ.get("OCBRAIN_OPS_DB", "~/.ocbrain/ops.sqlite")).expanduser()

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS companion_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signal_events (
  id TEXT PRIMARY KEY, kind TEXT NOT NULL, polarity TEXT NOT NULL,
  weight REAL NOT NULL, source TEXT NOT NULL, source_ref TEXT NOT NULL,
  session_key TEXT, knowledge_id TEXT, evidence_id TEXT, details TEXT,
  occurred_at TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS harvest_watermarks (
  domain TEXT NOT NULL, stream TEXT NOT NULL, watermark TEXT NOT NULL,
  updated_at TEXT NOT NULL, PRIMARY KEY (domain, stream)
);
CREATE TABLE IF NOT EXISTS judge_runs (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
  item_count INTEGER NOT NULL DEFAULT 0, request_hash TEXT, prompt_tokens INTEGER,
  completion_tokens INTEGER, cost_usd REAL NOT NULL DEFAULT 0,
  response_json TEXT, egress_audit_id TEXT
);
CREATE TABLE IF NOT EXISTS embed_runs (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, items INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0, status TEXT NOT NULL, error TEXT
);
CREATE TABLE IF NOT EXISTS autopilot_runs (
  id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL DEFAULT 'running', stages_json TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS loop_liveness (
  loop_id TEXT NOT NULL, run_id TEXT, last_heartbeat_at TEXT,
  last_ledger_write_at TEXT, expected_interval_seconds INTEGER,
  deadman_due_at TEXT, PRIMARY KEY (loop_id, run_id)
);
CREATE TABLE IF NOT EXISTS family_scores (
  loop_id TEXT NOT NULL, family TEXT NOT NULL, attempts INTEGER, kept INTEGER,
  reverted INTEGER, approach_failures INTEGER, verifier_pass_rate REAL,
  mean_primary_delta REAL, recency TEXT, state TEXT, refreshed_at TEXT,
  PRIMARY KEY (loop_id, family)
);
CREATE TABLE IF NOT EXISTS egress_audits (
  id TEXT PRIMARY KEY, ts TEXT NOT NULL, target TEXT NOT NULL,
  context_json TEXT NOT NULL, query TEXT, included_json TEXT NOT NULL,
  rejected_json TEXT NOT NULL, payload_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stall_pages (
  fingerprint TEXT PRIMARY KEY, stall_class TEXT NOT NULL, unit_id TEXT NOT NULL,
  terminal_signature TEXT, first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
  paged_at TEXT, retired_at TEXT, retire_reason TEXT,
  run_count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS watchdog_findings (
  id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, stall_class TEXT NOT NULL,
  unit_id TEXT NOT NULL, terminal_signature TEXT, snippet TEXT NOT NULL,
  artifact_path TEXT, occurred_at TEXT, observed_at TEXT NOT NULL,
  UNIQUE(fingerprint)
);
"""


def connect_ops(path: Path | str = DEFAULT_OPS_DB) -> sqlite3.Connection:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    init_ops_store(conn)
    return conn


def init_ops_store(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR REPLACE INTO companion_meta(key, value) VALUES ('schema', 'ocbrain-ops-v1')"
    )
    conn.commit()


__all__ = ["DEFAULT_OPS_DB", "connect_ops", "init_ops_store"]
