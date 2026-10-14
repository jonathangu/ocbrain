#!/bin/bash
# brain-sync.sh — incremental local-activity harvest into the OCBrain v1 core.
#
# Fingerprint-gated: unchanged transcript files are skipped, so this is cheap
# to run every few minutes. Safe under concurrency: SQLite WAL + busy timeout,
# and ocbrain evidence ids are stable/deduped.
set -uo pipefail

# Resolve the installed checkout and the same active core used by MCP clients.
# Explicit environment values win; otherwise use the checkout-local active DB
# pointer and only fall back to ~/.ocbrain for installations without one.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${OCBRAIN_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PY="${OCBRAIN_PYTHON:-$REPO/.venv/bin/python}"
ACTIVE_DB_FILE="${OCBRAIN_ACTIVE_DB_FILE:-$REPO/data/active-core.path}"
DB="${OCBRAIN_DB:-}"
if [[ -z "$DB" && -r "$ACTIVE_DB_FILE" ]]; then
  IFS= read -r DB < "$ACTIVE_DB_FILE"
fi
DB="${DB:-$HOME/.ocbrain/ocbrain.sqlite}"
if [[ "$DB" != /* ]]; then
  echo "OCBrain DB path must be absolute: $DB" >&2
  exit 2
fi
# Do NOT pin the config into the checkout. ocbrain resolves
# ~/.ocbrain/ocbrain.config.json first, which survives a `git clean -xfd`, a
# fresh clone, and a worktree switch; forcing $REPO/data here would reintroduce
# the failure where operator settings vanish with the working tree and the loop
# keeps exiting 0 while promoting nothing. Set OCBRAIN_CONFIG yourself to
# override.

# Claude Code seats live in ~/.claude and in every ~/.claude-*; test each
# candidate so an unmatched glob never reaches import-history as a literal.
HISTORY_DIRS=(
  "$HOME/.codex/sessions"
  "$HOME/.codex/archived_sessions"
)
for claude_projects in "$HOME/.claude/projects" "$HOME"/.claude-*/projects; do
  [[ -d "$claude_projects" ]] && HISTORY_DIRS+=("$claude_projects")
done
HISTORY_DIRS+=(
  "$HOME/.hermes/sessions"
  "$HOME/.ocbrain/exports/cursor"
)

# Agent memory/instruction files. The two ~/.hermes entries are the retired
# single-gateway home; the fleet now keeps one home per agent profile, each with
# its own SOUL.md and memories/, and none of those were ever imported.
MEMORY_PATHS=(
  "$HOME/.claude/CLAUDE.md"
  "$HOME/.codex/AGENTS.md"
  "$HOME/.hermes/SOUL.md"
  "$HOME/.hermes/memories"
)
# Test each candidate rather than passing a glob. import-memory drops a missing
# directory silently, but an unmatched glob keeps its literal '*' through path
# selection and is only rejected later, as an unreadable-path skip on every run.
# A profile need not have both, and an empty memories/ is normal (togepicoframe
# has one today); import-memory sweeps *.md, so the .lock files beside them and
# the credentials in the profile root are never candidates.
for profile_dir in "$HOME/.hermes/profiles"/*/; do
  profile_dir="${profile_dir%/}"
  [[ -f "$profile_dir/SOUL.md" ]] && MEMORY_PATHS+=("$profile_dir/SOUL.md")
  [[ -d "$profile_dir/memories" ]] && MEMORY_PATHS+=("$profile_dir/memories")
done
if [[ -n "${OCBRAIN_SYNC_EXTRA_MEMORY_PATHS:-}" ]]; then
  IFS=: read -r -a extra_memory <<< "$OCBRAIN_SYNC_EXTRA_MEMORY_PATHS"
  for extra_path in "${extra_memory[@]}"; do
    [[ -n "$extra_path" && -e "$extra_path" ]] && MEMORY_PATHS+=("$extra_path")
  done
fi

if [[ "${1:-}" == "--print-plan" ]]; then
  for entry in "${HISTORY_DIRS[@]}"; do printf 'history:%s\0' "$entry"; done
  for entry in "${MEMORY_PATHS[@]}"; do printf 'memory:%s\0' "$entry"; done
  exit 0
fi

SYNC_PROJECT="${OCBRAIN_SYNC_PROJECT:-workspace}"
SYNC_PRIVACY_SCOPE="${OCBRAIN_SYNC_PRIVACY_SCOPE:-workspace}"
SYNC_BATCH_SIZE="${OCBRAIN_SYNC_BATCH_SIZE:-25}"

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

# Yield CPU to interactive/agent work; the harvest is background maintenance.
renice -n 10 -p $$ >/dev/null 2>&1 || true

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
  "${HISTORY_DIRS[@]}" \
  --project "$SYNC_PROJECT" --privacy-scope "$SYNC_PRIVACY_SCOPE" \
  --batch-size "$SYNC_BATCH_SIZE" --evidence-only

# 4. Agent memory/instruction files (paths are collected above).
echo "import-memory paths: ${#MEMORY_PATHS[@]}"

"$PY" -m ocbrain.cli --db "$DB" import-memory \
  "${MEMORY_PATHS[@]}" \
  --project "$SYNC_PROJECT" --privacy-scope "$SYNC_PRIVACY_SCOPE" --evidence-only

if [[ -n "${OCBRAIN_RECEIPT_ROOTS:-}" ]]; then
  run_with_budget "$HARVEST_BUDGET_SECONDS" \
    "$PY" "$REPO/scripts/receipt-closeouts.py" --db "$DB" \
    --roots "$OCBRAIN_RECEIPT_ROOTS" \
    --project "${OCBRAIN_RECEIPT_PROJECT:-workspace}" \
    --since-days "${OCBRAIN_RECEIPT_SINCE_DAYS:-7}" \
    || echo "receipt-closeouts failed; continuing"
fi

# 5. Reconcile core projections. `sync` refuses past its --max-events bound
# rather than doing partial work, and a cold or long-delayed harvest can enqueue
# far more than the 1000 default, so raise the ceiling for the scheduled path.
"$PY" -m ocbrain.cli --db "$DB" sync --max-events "${OCBRAIN_SYNC_MAX_EVENTS:-200000}"

after="$(event_count)"
echo "brain_events after: $after"
echo "== $(date -u +%FT%TZ) brain-sync done =="
