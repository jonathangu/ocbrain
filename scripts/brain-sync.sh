#!/bin/bash
# brain-sync.sh — incremental local-activity harvest into the OCBrain v1 core.
#
# Fingerprint-gated: unchanged transcript files are skipped, so this is cheap
# to run every few minutes. Safe under concurrency: SQLite WAL + busy timeout,
# and ocbrain evidence ids are stable/deduped.
set -uo pipefail

# The shared core per ~/.ocbrain/install-receipt.md — the same DB Codex,
# Cursor, Claude Desktop, and Hermes use via MCP. The repo data/ocbrain.sqlite
# is dev scratch only; do not harvest into it.
REPO="$HOME/Developer/ocbrain"
PY="$REPO/.venv/bin/python"
DB="$HOME/.ocbrain/ocbrain.sqlite"
export OCBRAIN_CONFIG="$REPO/data/ocbrain.config.json"

echo "== $(date -u +%FT%TZ) brain-sync start =="

# Single-instance: a 4GB cold harvest can outlive the 15-min launchd interval.
exec 9>"$HOME/.ocbrain/brain-sync.lock"
flock -n 9 || { echo "another brain-sync is running; exiting"; exit 0; }

# 1. Hermes transcripts: state.db -> JSONL export (content-compared writes).
"$PY" "$REPO/scripts/export-hermes-transcripts.py"

# 2. Agent runtime history (Codex, Claude Code, Hermes) -> v1 core.
"$PY" -m ocbrain.cli --db "$DB" import-history \
  "$HOME/.codex/sessions" \
  "$HOME/.codex/archived_sessions" \
  "$HOME/.claude/projects" \
  "$HOME/.hermes/sessions" \
  --project coframe --privacy-scope private --batch-size 25

# 3. Agent memory/instruction files.
"$PY" -m ocbrain.cli --db "$DB" import-memory \
  "$HOME/.claude/CLAUDE.md" \
  "$HOME/.codex/AGENTS.md" \
  "$HOME/.hermes/SOUL.md" \
  "$HOME/.hermes/memories" \
  --project coframe --privacy-scope private

# 4. Reconcile core projections.
"$PY" -m ocbrain.cli --db "$DB" sync

echo "== $(date -u +%FT%TZ) brain-sync done =="
