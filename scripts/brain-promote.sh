#!/bin/bash
# brain-promote.sh — turn harvested evidence into servable knowledge, and retire
# knowledge that has stopped earning its place.
#
# brain-sync.sh only records evidence; nothing it does makes a fact retrievable.
# Without a promotion pass the corpus freezes at whatever was last curated by
# hand, and without a retirement pass it only ever grows. This script is the
# other half of the loop:
#
#   1. curate   — compile eligible evidence into wiki facts (hosted model call)
#   2. hygiene  — retire expired and redundant beliefs
#   3. wiki     — rematerialize the wiki, which is also what deletes orphan pages
#   4. lint     — post-condition check on the materialized tree
#   5. vectors  — rebuild the dense sidecar so new facts are semantically findable
#   6. procmine — mine the trajectory corpus (opt-in: OCBRAIN_PROCMINE=1)
#
# NOT installed by default. OCBrain ships no scheduler; an operator opts in by
# loading a launchd agent (see docs/SCHEDULED_MAINTENANCE.md). Step 1 is the only
# one that leaves the machine by default, and it is digest-gated: an unchanged
# corpus makes no API call, so a quiet cycle is free.
set -uo pipefail

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

# Which project scopes get curated is the operator's config (curator.projects),
# not a pin in this script. A single pinned project freezes the wiki the moment
# work moves to a second scope, and the evidence keeps arriving regardless.
# OCBRAIN_PROMOTE_PROJECT still curates exactly one scope, for a one-off run.
PROMOTE_PROJECT="${OCBRAIN_PROMOTE_PROJECT:-}"
if [[ -n "$PROMOTE_PROJECT" ]]; then
  CURATE_SCOPE_ARGS=(--project "$PROMOTE_PROJECT")
else
  CURATE_SCOPE_ARGS=(--projects-from-config)
fi
PROMOTE_PROVIDER="${OCBRAIN_PROMOTE_PROVIDER:-anthropic}"
PROMOTE_MAX_BELIEFS="${OCBRAIN_PROMOTE_MAX_BELIEFS:-24}"
# Moonshot's prompt and completion share the model's context window, while the
# Anthropic and OpenAI defaults can safely leave more room for completion.
case "$PROMOTE_PROVIDER" in
  moonshot) PROMOTE_DEFAULT_MAX_TOKENS=8000 ;;
  *) PROMOTE_DEFAULT_MAX_TOKENS=16000 ;;
esac
PROMOTE_MAX_TOKENS="${OCBRAIN_PROMOTE_MAX_TOKENS:-$PROMOTE_DEFAULT_MAX_TOKENS}"
WIKI_DIR="${OCBRAIN_WIKI_DIR:-$(dirname -- "$DB")/wiki}"
BUDGET_SECONDS="${OCBRAIN_PROMOTE_BUDGET_SECONDS:-1800}"

CURATE_ARGS=(
  "$PY" "$REPO/scripts/wiki-curator.py"
  --db "$DB"
  --provider "$PROMOTE_PROVIDER"
  "${CURATE_SCOPE_ARGS[@]}"
  --wiki-dir "$WIKI_DIR"
  --max-beliefs "$PROMOTE_MAX_BELIEFS"
  --max-tokens "$PROMOTE_MAX_TOKENS"
  --apply
)
[[ -n "${OCBRAIN_PROMOTE_BASE_URL:-}" ]] && CURATE_ARGS+=(--base-url "$OCBRAIN_PROMOTE_BASE_URL")
[[ -n "${OCBRAIN_PROMOTE_MODEL:-}" ]] && CURATE_ARGS+=(--model "$OCBRAIN_PROMOTE_MODEL")
[[ -n "${OCBRAIN_PROMOTE_API_KEY_ENV:-}" ]] && CURATE_ARGS+=(--api-key-env "$OCBRAIN_PROMOTE_API_KEY_ENV")
[[ -n "${OCBRAIN_PROMOTE_ENV_FILE:-}" ]] && CURATE_ARGS+=(--env-file "$OCBRAIN_PROMOTE_ENV_FILE")

MANIFEST_APPLY_ARGS=(curated-apply --actor "${OCBRAIN_PROMOTE_MANIFEST_ACTOR:-brain-promote}")
[[ "${OCBRAIN_PROMOTE_MANIFEST_ALLOW_HOSTED:-}" == "1" ]] && MANIFEST_APPLY_ARGS+=(--allow-hosted-egress)

# Side-effect-free probe for deterministic tests of the real shell expansion.
if [[ "${1:-}" == "--print-curate-argv" ]]; then
  printf '%s\0' "${CURATE_ARGS[@]}"
  exit 0
fi
if [[ "${1:-}" == "--print-manifest-apply-argv" ]]; then
  printf '%s\0' "${MANIFEST_APPLY_ARGS[@]}"
  exit 0
fi

# Which hygiene classes may apply unattended. Both are unambiguous: `expired`
# acts on an explicit supersession or a passed valid_until, `redundant` only on
# same-scope restatements above the token threshold, keeping the newest.
# `redundant` was once missing here, which is why duplicate wiki facts
# accumulated to a quarter of the serving corpus while the loop reported clean.
HYGIENE_CLASSES="${OCBRAIN_HYGIENE_CLASSES:---class expired --class redundant}"
# Report-only by default. Set OCBRAIN_HYGIENE_APPLY=1 to let it retire beliefs;
# every retraction is soft and undoable with `ocbrain hygiene --restore`.
HYGIENE_APPLY="${OCBRAIN_HYGIENE_APPLY:-0}"

# The deslop steps that used to sit here are gone. Across 155 consecutive
# hourly runs the belief sweep reported `actionable: 0, repairs: [], judged:
# false` every single time: the mechanical rules do fire, but as the curator's
# write-time gate, before a bad claim is ever stored. Re-running them over an
# already-gated corpus found nothing, by construction.

# Snapshot before mutating. Once per UTC day rather than hourly: the core is
# ~150MB and this job runs every hour. Uses the SQLite online-backup API, so it
# is safe against the live WAL. Rotation only ever touches the auto- family,
# leaving hand-made pre-* backups alone.
PROMOTE_BACKUP="${OCBRAIN_PROMOTE_BACKUP:-1}"
BACKUP_DIR="${OCBRAIN_BACKUP_DIR:-$HOME/.ocbrain/backups}"
BACKUP_KEEP="${OCBRAIN_BACKUP_KEEP:-7}"

echo "== $(date -u +%FT%TZ) brain-promote start =="

LOCKDIR="$HOME/.ocbrain/brain-promote.lock.d"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  holder="$(cat "$LOCKDIR/pid" 2>/dev/null || echo '')"
  if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
    echo "another brain-promote (pid $holder) is running; exiting"
    exit 0
  fi
  echo "stale lock from pid ${holder:-unknown}; reclaiming"
  rm -rf "$LOCKDIR"
  mkdir "$LOCKDIR" 2>/dev/null || { echo "lock contention; exiting"; exit 0; }
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT

run_with_budget() {
  local budget="$1"; shift
  "$@" &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= budget )); then
      echo "budget ${budget}s exceeded for: $* — killing"
      kill "$pid" 2>/dev/null
      sleep 5
      kill -9 "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
  done
  wait "$pid"
}

renice -n 10 -p $$ >/dev/null 2>&1 || true

serving_count() {
  sqlite3 "$DB" \
    "SELECT COUNT(*) FROM current_beliefs WHERE serve=1 AND status='current';" \
    2>/dev/null || echo "?"
}

before="$(serving_count)"
echo "serving beliefs before: $before"

# 0. Daily snapshot, before anything below can retire or rewrite a belief.
if [[ "$PROMOTE_BACKUP" == "1" ]]; then
  snap="$BACKUP_DIR/ocbrain-auto-$(date -u +%Y%m%d).sqlite"
  if [[ -f "$snap" ]]; then
    echo "backup for today already present: $snap"
  else
    mkdir -p "$BACKUP_DIR"
    if "$PY" -c 'import sys; from ocbrain.fsutil import snapshot_sqlite; snapshot_sqlite(sys.argv[1], sys.argv[2])' "$DB" "$snap"; then
      echo "backup written: $snap"
      # Keep the newest $BACKUP_KEEP auto snapshots; never touch pre-*/manual ones.
      ls -1t "$BACKUP_DIR"/ocbrain-auto-*.sqlite 2>/dev/null \
        | tail -n "+$((BACKUP_KEEP + 1))" \
        | while read -r old; do rm -f -- "$old" && echo "rotated out: $old"; done
    else
      echo "backup failed; refusing to run mutating steps this cycle"
      exit 1
    fi
  fi
fi

# 1. Curate. Digest-gated per project, so a scope whose evidence has not changed
# makes no API call and a fully quiet cycle is free.
run_with_budget "$BUDGET_SECONDS" \
  "${CURATE_ARGS[@]}" \
  || echo "curate step failed or was capped; continuing with retirement"

if [[ -n "${OCBRAIN_PROMOTE_MANIFEST_CMDS:-}" ]]; then
  IFS=: read -r -a manifest_cmds <<< "$OCBRAIN_PROMOTE_MANIFEST_CMDS"
  for manifest_cmd in "${manifest_cmds[@]}"; do
    [[ -n "$manifest_cmd" ]] || continue
    if ! manifest_path="$("$manifest_cmd" | tail -n 1)" || [[ -z "$manifest_path" ]]; then
      echo "manifest command failed: $manifest_cmd; continuing"
      continue
    fi
    "$PY" -m ocbrain.cli --db "$DB" "${MANIFEST_APPLY_ARGS[@]}" "$manifest_path" \
      || echo "curated-apply failed for $manifest_path; continuing"
  done
fi

# 2. Retire. Reported either way; only applied when explicitly enabled.
hygiene_args=(--db "$DB" hygiene $HYGIENE_CLASSES)
if [[ "$HYGIENE_APPLY" == "1" ]]; then
  hygiene_args+=(--apply)
fi
"$PY" -m ocbrain.cli "${hygiene_args[@]}" \
  || echo "hygiene step failed; continuing"

# 3. Rematerialize the wiki. A full rebuild + atomic swap is what removes pages
# for beliefs retired above; retirements outside a curate run leave orphans until
# this happens.
"$PY" - "$DB" "$WIKI_DIR" <<'PYEOF' || echo "wiki rematerialize failed; continuing"
import sys
from pathlib import Path

from ocbrain.db import connect
from ocbrain.wiki import materialize_wiki

db_path, wiki_dir = Path(sys.argv[1]), Path(sys.argv[2])
conn = connect(db_path)
try:
    count = materialize_wiki(conn, wiki_dir, run={"action": "scheduled-rematerialize"})
    print(f"wiki pages: {count}")
finally:
    conn.close()
PYEOF

# 4. Post-condition check on the tree we just wrote. Non-zero means findings, so
# surface them in the log rather than swallowing the exit code.
"$PY" "$REPO/scripts/wiki-lint.py" "$WIKI_DIR" --db "$DB" \
  || echo "wiki-lint reported findings (see above)"

# 5. Rebuild the dense sidecar so newly promoted facts are semantically findable.
# Retrieval degrades to lexical-only against a stale sidecar, so a promote that
# skipped this would leave new knowledge half-reachable.
"$PY" -m ocbrain.cli --db "$DB" vector-build \
  || echo "vector-build failed (is the local embedder running?); continuing"

# 6. Mine procedural memory. Opt-in twice over, and off by default.
#
# OCBRAIN_PROCMINE=1 turns the stage on at all; OCBRAIN_PROCMINE_APPLY=1 is what
# lets the mint write beliefs. Two switches rather than one because the two
# risks are different: extraction reads every transcript on the machine, and
# minting adds rows to the serving corpus. An operator may reasonably want the
# atlas refreshed without the second.
#
# Extraction is incremental — unchanged source files replay from cached
# segments under $PROCMINE_DIR/cache — so a quiet cycle costs seconds rather
# than a full re-walk of the corpus.
PROCMINE="${OCBRAIN_PROCMINE:-0}"
PROCMINE_APPLY="${OCBRAIN_PROCMINE_APPLY:-0}"
PROCMINE_DIR="${OCBRAIN_PROCMINE_DIR:-$HOME/.ocbrain/procmine}"
PROCMINE_BUDGET_SECONDS="${OCBRAIN_PROCMINE_BUDGET_SECONDS:-1800}"
if [[ "$PROCMINE" == "1" ]]; then
  mkdir -p "$PROCMINE_DIR"
  procmine_traces="$PROCMINE_DIR/traces.jsonl"
  if run_with_budget "$PROCMINE_BUDGET_SECONDS" \
      env PYTHONPATH="$REPO/src:$REPO/scripts" "$PY" -m procmine extract \
        --out "$procmine_traces" --state-dir "$PROCMINE_DIR"; then
    # Atlas refresh. Also rewrites $PROCMINE_DIR/episodes.json, the standing
    # per-episode step-sequence artifact, so the sequences the miner computes
    # stop being recomputed and thrown away on every run.
    run_with_budget "$PROCMINE_BUDGET_SECONDS" \
      env PYTHONPATH="$REPO/src:$REPO/scripts" "$PY" -m procmine atlas \
        --traces "$procmine_traces" \
        --brain-db "$DB" \
        --report "$PROCMINE_DIR/PROCEDURE-ATLAS.md" \
        --json "$PROCMINE_DIR/procedures.json" \
        --dump-episodes "$PROCMINE_DIR/episodes.json" \
      || echo "procmine atlas failed; continuing"

    mint_args=(--traces "$procmine_traces" --brain-db "$DB")
    if [[ "$PROCMINE_APPLY" == "1" ]]; then
      mint_args+=(--apply)
    fi
    env PYTHONPATH="$REPO/src:$REPO/scripts" "$PY" -m procmine mint "${mint_args[@]}" \
      || echo "procmine mint failed; continuing"
  else
    echo "procmine extract failed or was capped; skipping atlas and mint"
  fi
else
  echo "procmine stage disabled (set OCBRAIN_PROCMINE=1 to enable)"
fi

after="$(serving_count)"
echo "serving beliefs after: $after"
echo "== $(date -u +%FT%TZ) brain-promote done =="
