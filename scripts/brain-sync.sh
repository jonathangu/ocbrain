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

# Hard ceiling for the whole harvest. A cold multi-GB import can take a while,
# but a stuck run must never block the launchd schedule indefinitely (launchd
# will not fire a new instance while a previous one is still alive).
HARVEST_BUDGET_SECONDS="${OCBRAIN_SYNC_BUDGET_SECONDS:-2700}"

echo "== $(date -u +%FT%TZ) brain-sync start =="

# Single-instance, portable: macOS has no flock(1). A mkdir lock is atomic on
# POSIX, and we recover from stale locks left by killed runs via PID liveness.
LOCKDIR="$HOME/.ocbrain/brain-sync.lock.d"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  holder="$(cat "$LOCKDIR/pid" 2>/dev/null || echo '')"
  if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
    echo "another brain-sync (pid $holder) is running; exiting"
    exit 0
  fi
  echo "stale lock from pid ${holder:-unknown}; reclaiming"
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR" 2>/dev/null || { echo "lock contention; exiting"; exit 0; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT

# Run one command under a hard time budget; partial batches stay committed and
# the next run resumes via the fingerprint/dedup gates.
run_with_budget() {
  local budget="$1"; shift
  "$@" &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    waited=$((waited + 15))
    if (( waited >= budget )); then
      echo "budget ${budget}s exceeded for: $* — killing (partial work stays committed)"
      kill "$pid" 2>/dev/null
      sleep 5
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

event_count() {
  sqlite3 "$DB" "SELECT COUNT(*) FROM brain_events;" 2>/dev/null || echo "?"
}

before="$(event_count)"
echo "brain_events before: $before"

# 1. Hermes transcripts: state.db -> JSONL export (content-compared writes).
"$PY" "$REPO/scripts/export-hermes-transcripts.py"

# 2. Cursor chats: state.vscdb -> JSONL export (content-compared writes).
"$PY" "$REPO/scripts/export-cursor-chats.py"

# 3. Agent runtime history (Codex, Claude Code, Hermes, Cursor) -> v1 core.
run_with_budget "$HARVEST_BUDGET_SECONDS" \
  "$PY" -m ocbrain.cli --db "$DB" import-history \
  "$HOME/.codex/sessions" \
  "$HOME/.codex/archived_sessions" \
  "$HOME/.claude/projects" \
  "$HOME/.hermes/sessions" \
  "$HOME/.ocbrain/exports/cursor" \
  --project coframe --privacy-scope private --batch-size 25

# 4. Agent memory/instruction files.
"$PY" -m ocbrain.cli --db "$DB" import-memory \
  "$HOME/.claude/CLAUDE.md" \
  "$HOME/.codex/AGENTS.md" \
  "$HOME/.hermes/SOUL.md" \
  "$HOME/.hermes/memories" \
  --project coframe --privacy-scope private

# 5. Reconcile core projections.
"$PY" -m ocbrain.cli --db "$DB" sync

after="$(event_count)"
echo "brain_events after: $after"
echo "== $(date -u +%FT%TZ) brain-sync done =="
