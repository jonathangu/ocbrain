#!/bin/bash
# brain-sync.sh — incremental local-activity harvest into the OCBrain v1 core.
#
# Fingerprint-gated: unchanged transcript files are skipped, so this is cheap
# to run every few minutes. Safe under concurrency: SQLite WAL + busy timeout,
# and ocbrain evidence ids are stable/deduped.
set -uo pipefail

REPO="$HOME/Developer/ocbrain"
PY="$REPO/.venv/bin/python"
DB="$REPO/data/ocbrain.sqlite"
export OCBRAIN_CONFIG="$REPO/data/ocbrain.config.json"

echo "== $(date -u +%FT%TZ) brain-sync start =="

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
