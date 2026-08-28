#!/usr/bin/env bash
# Mutation proof for the closeout-discipline gates added on fix/closeout-discipline.
#
# Every gate here is mutated so it SHOULD fail, the named test is run, and the
# run must fail. Then the mutation is reverted and the test must pass again.
# A gate whose failing input is unreachable is the defect class this exists for.
#
# Two documented traps are handled explicitly:
#   (a) a size-preserving mutation restored inside the same second can run the
#       OTHER version's .pyc and reverse the verdict both ways -- so every
#       __pycache__ is removed before every run and PYTHONDONTWRITEBYTECODE=1
#       is set for all of them;
#   (b) the expected result of a probe is never printed before the probe
#       returns -- this script reports what happened, it does not assert it.
#
# Usage: ops/mutation-proof-closeout-discipline.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${OCBRAIN_PYTHON:-$ROOT/../../../.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src"

clear_bytecode() {
  find "$ROOT/src" "$ROOT/tests" "$ROOT/scripts" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
}

run_test() {
  clear_bytecode
  "$PY" -m pytest -q -p no:cacheprovider "$1" >/tmp/mutproof.$$ 2>&1
  local rc=$?
  tail -3 /tmp/mutproof.$$ | tr '\n' ' '
  rm -f /tmp/mutproof.$$
  return $rc
}

# mutate FILE FROM TO TEST LABEL
mutate() {
  local file="$1" from="$2" to="$3" test="$4" label="$5"
  local backup
  backup="$(mktemp)"
  cp "$file" "$backup"
  "$PY" - "$file" "$from" "$to" <<'EOF'
import pathlib, sys
path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = path.read_text()
if text.count(old) != 1:
    print(f"MUTATION ANCHOR NOT UNIQUE ({text.count(old)}) in {path}", file=sys.stderr)
    raise SystemExit(2)
path.write_text(text.replace(old, new))
EOF
  if [ $? -ne 0 ]; then
    printf '%-58s ANCHOR MISSING\n' "$label"
    cp "$backup" "$file"; rm -f "$backup"; return 1
  fi
  local out
  out="$(run_test "$test")"
  local mutated_rc=$?
  cp "$backup" "$file"; rm -f "$backup"
  # pytest exits 4 when a named test id does not resolve and 5 when it
  # collected nothing at all. Either way a renamed test would otherwise read as
  # `mutated=FAILS`, which is this harness scoring a missing instrument as a
  # passing proof -- the exact failure it exists to catch elsewhere. Verified by
  # pointing one entry at a name that does not exist and seeing this line.
  if [ $mutated_rc -eq 4 ] || [ $mutated_rc -eq 5 ]; then
    printf '%-58s TEST NOT FOUND (%s)\n' "$label" "$test"
    return 1
  fi
  local restored_out
  restored_out="$(run_test "$test")"
  local restored_rc=$?
  printf '%-58s mutated=%s restored=%s\n' "$label" \
    "$([ $mutated_rc -ne 0 ] && echo FAILS || echo PASSES)" \
    "$([ $restored_rc -eq 0 ] && echo PASSES || echo FAILS)"
  printf '    mutated : %s\n    restored: %s\n' "$out" "$restored_out"
}

C=src/ocbrain/closeout.py
T=tests/test_closeout_discipline.py

echo "== defect 1: session shape gate =="
mutate "$C" \
  'RUNTIME_SESSION_SHAPES = frozenset({"runtime_uuid", "runtime_hex"})' \
  'RUNTIME_SESSION_SHAPES = frozenset({"runtime_uuid", "runtime_hex", "slug", "date_like", "filesystem_path", "contains_space"})' \
  "$T::test_a_hand_written_session_id_is_refused_and_the_error_says_where_to_get_one" \
  "shape gate admits every shape"

mutate "$C" \
  '        if policy == "enforce":
            raise ValueError(_session_id_error(claimed, shape))' \
  '        if policy == "never":
            raise ValueError(_session_id_error(claimed, shape))' \
  "$T::test_a_hand_written_session_id_is_refused_and_the_error_says_where_to_get_one" \
  "enforce branch made unreachable"

mutate "$C" \
  '    if hint is not None and is_runtime_session_id(hint):' \
  '    if False and hint is not None and is_runtime_session_id(hint):' \
  "$T::test_the_harness_attested_hint_outranks_the_model_and_the_disagreement_is_kept" \
  "harness hint no longer outranks the model"

mutate "$C" \
  '    elif observed is not None and observed.server_connection_id:' \
  '    elif False and observed is not None and observed.server_connection_id:' \
  "$T::test_omitting_the_session_is_legal_and_the_server_fills_it_from_its_own_connection" \
  "server-connection fallback removed"

echo
echo "== defect 2: runtime family =="
mutate "$C" \
  '            if segments & tokens:' \
  '            if any(token in folded for token in tokens):' \
  "$T::test_a_normaliser_matching_substrings_invents_data" \
  "segment matching reverted to substring"

mutate "$C" \
  '        if mapped in RUNTIME_FAMILIES:' \
  '        if mapped is not None:' \
  "$T::test_an_operator_alias_can_name_an_install_specific_label" \
  "alias may invent a family"

echo
echo "== defect 3: unresolved gate =="
mutate "$C" \
  '    return status not in CLEAN_SUCCESS_STATUSES or verification_status == "failed"' \
  '    return status not in CLEAN_SUCCESS_STATUSES' \
  "$T::test_the_unresolved_gate_catches_282_of_the_1239_live_closeouts" \
  "verifier trigger dropped (status only)"

mutate "$C" \
  '        problems.append(_unresolved_error(status, verification_status))' \
  '        pass' \
  "$T::test_a_completed_closeout_with_a_failed_verifier_must_say_what_failed" \
  "unresolved refusal removed"

mutate "$C" \
  '        raise ValueError("\n\n".join(problems))' \
  '        raise ValueError(problems[0])' \
  "$T::test_both_gates_report_together_so_one_retry_fixes_both" \
  "only the first refusal is reported"

mutate "$C" \
  'CLEAN_SUCCESS_STATUSES = {"completed"}' \
  'CLEAN_SUCCESS_STATUSES = {"completed", "partial", "blocked", "failed", "cancelled"}' \
  "$T::test_every_non_completion_status_owes_an_explanation" \
  "every status counted as a clean success"

echo
echo "== config fail-open =="
mutate "$C" \
  '    if settings.session_id_policy not in SESSION_ID_POLICIES:
        return replace(settings, session_id_policy=default.session_id_policy)' \
  '    pass' \
  "$T::test_a_misspelled_policy_falls_back_instead_of_taking_the_write_path_down" \
  "typo policy takes the write path down"

echo
echo "== migration =="
mutate src/ocbrain/core_v1.py \
  '    ("task_closeouts", "session_id_source", "TEXT"),' \
  '' \
  "$T::test_an_existing_core_gains_the_columns_before_the_first_closeout_lands" \
  "session_id_source column not migrated"

mutate src/ocbrain/db.py \
  '    for column, decl in _V7_TASK_CLOSEOUT_COLUMNS:
        _ensure_column(conn, "task_closeouts", column, decl)' \
  '    pass' \
  "$T::test_the_legacy_initializer_also_migrates_an_existing_database" \
  "legacy db.py migration skipped"

echo
echo "== remediation: the reader that makes the unresolved gate honest =="
B=src/ocbrain/briefing.py

mutate "$B" \
  '                "unresolved": receipt.get("unresolved"),' \
  '                "unresolved": None,' \
  "$T::test_the_ledger_serves_the_unresolved_sentence_the_gate_charges_for" \
  "ledger row projection drops unresolved"

mutate "$B" \
  '                "unresolved": row["unresolved"],' \
  '                "unresolved": None,' \
  "$T::test_the_ledger_serves_the_unresolved_sentence_the_gate_charges_for" \
  "failed_attempts drops unresolved"

mutate "$B" \
  '        "latest_unresolved": latest["unresolved"],' \
  '        "latest_unresolved": None,' \
  "$T::test_the_ledger_serves_the_unresolved_sentence_the_gate_charges_for" \
  "ledger entry drops latest_unresolved"

mutate "$B" \
  '        because = entry["latest_unresolved"] or entry["latest_summary"]' \
  '        because = entry["latest_summary"]' \
  "$T::test_the_briefings_failed_line_carries_what_did_not_work_not_just_the_summary" \
  "briefing FAILED line reverts to the summary"

mutate "$B" \
  '        because = entry["latest_unresolved"] or entry["latest_summary"]' \
  '        because = entry["latest_unresolved"]' \
  "$T::test_a_failed_attempt_with_no_unresolved_still_reports_its_summary" \
  "briefing FAILED line loses the pre-gate fallback"

echo
echo "== remediation: the harness-attested hint is the highest-trust door =="
mutate "$C" \
  '    return classify_session_id(value) in RUNTIME_SESSION_SHAPES' \
  '    return classify_session_id(value) != "absent"' \
  "$T::test_a_junk_harness_hint_never_wins_the_identity_column" \
  "hint shape check accepts anything non-empty (reviewer G1)"

mutate "$C" \
  '    return classify_session_id(value) in RUNTIME_SESSION_SHAPES' \
  '    return classify_session_id(value) != "absent"' \
  "$T::test_is_runtime_session_id_admits_exactly_the_two_machine_shapes" \
  "the predicate itself, asserted directly"

mutate "$C" \
  '    return classify_session_id(value) in RUNTIME_SESSION_SHAPES' \
  '    return classify_session_id(value) != "absent"' \
  "$T::test_a_junk_hint_does_not_rescue_a_caller_who_omitted_everything" \
  "junk hint fills an otherwise-empty column"

echo
echo "== remediation: three normalisers, reconciled =="
E=scripts/procmine/episodes.py

mutate "$C" \
  '        exact = RUNTIME_FAMILY_EXACT.get(folded)
        if exact is not None:
            return exact' \
  '        pass' \
  "$T::test_this_repos_own_runtime_is_not_unknown_to_its_own_normaliser" \
  "this repo's own runtime is unknown again"

mutate "$C" \
  '    "ocbrain-runtime-call": "mcp",' \
  '    "ocbrain": "mcp",' \
  "$T::test_this_repos_own_runtime_is_not_unknown_to_its_own_normaliser" \
  "exact spelling widened to a token (66 rows misplaced)"

mutate "$C" \
  '        fold_runtime_label(k): str(v).strip() for k, v in (aliases or {}).items()' \
  '        str(k).strip().lower(): str(v).strip() for k, v in (aliases or {}).items()' \
  "$T::test_an_alias_key_with_a_space_is_reachable" \
  "alias keys lowercased but not folded"

mutate "$E" \
  '    shared = runtime_family(text)
    if shared != "unknown":
        return _SHARED_FAMILY[shared]' \
  '    pass' \
  "$T::test_the_write_time_enum_and_the_mining_taxonomy_never_contradict_each_other" \
  "mining taxonomy stops asking the shared folder"

mutate "$E" \
  '    (re.compile(r"telegram|kanban|gateway|hermeswork", re.I), "hermes"),' \
  '    (re.compile(r"hermes|codex|cursor|claude", re.I), "hermes"),' \
  "$T::test_the_mining_taxonomy_no_longer_matches_family_tokens_as_substrings" \
  "family tokens substring-matched in the miner again"

echo
echo "== remediation: the miner reads the authoritative column =="
mutate "$E" \
  '            if stored_session_source in _SERVER_OBSERVED_SOURCES
            or (stored_session_source is None and session_hint)' \
  '            if session_hint' \
  "$T::test_the_miner_reads_the_authoritative_source_column_not_its_own_guess" \
  "session_source back to the hint heuristic"

mutate "$E" \
  '_SERVER_OBSERVED_SOURCES = frozenset({"harness_attested", "server_connection"})' \
  '_SERVER_OBSERVED_SOURCES = frozenset({"harness_attested"})' \
  "$T::test_the_miner_reads_the_authoritative_source_column_not_its_own_guess" \
  "server-minted conn: id called model_reported"

echo
echo "== remediation: the sibling column on retrieval_uses =="
V=src/ocbrain/core_v1.py
D=src/ocbrain/db.py

mutate "$V" \
  '    identity = resolve_session_identity(session_id, prov, policy="quarantine")' \
  '    identity = resolve_session_identity(session_id, prov, policy="off")' \
  "$T::test_the_retrieval_receipt_gets_the_same_identity_discipline" \
  "v1 retrieval stores the slug again"

mutate "$V" \
  '    identity = resolve_session_identity(session_id, prov, policy="quarantine")' \
  '    identity = resolve_session_identity(session_id, prov, policy="enforce")' \
  "$T::test_a_retrieval_is_never_refused_for_its_session_label" \
  "a read refused for its session label"

mutate "$D" \
  '    identity = resolve_session_identity(session_id, observed, policy="quarantine")' \
  '    identity = resolve_session_identity(session_id, observed, policy="off")' \
  "$T::test_the_legacy_retrieval_writer_is_gated_too" \
  "legacy retrieval writer left ungated (the sibling)"

mutate "$V" \
  '    ("retrieval_uses", "session_id_source", "TEXT"),' \
  '' \
  "$T::test_an_existing_core_gains_the_retrieval_column_before_the_first_read" \
  "retrieval column not migrated on an existing core"

mutate "$D" \
  '    for column, decl in _V7_RETRIEVAL_USE_COLUMNS:
        _ensure_column(conn, "retrieval_uses", column, decl)' \
  '    pass' \
  "$T::test_the_legacy_initializer_also_adds_the_retrieval_column" \
  "legacy db.py retrieval migration skipped"
