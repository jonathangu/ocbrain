from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from ocbrain.db import now_iso
from ocbrain.events import canonical_json
from ocbrain.ids import content_hash, stable_id
from ocbrain.provenance import EMPTY_PROVENANCE, Provenance
from ocbrain.scope import ScopeContext

CLOSEOUT_SCHEMA_VERSION = "ocbrain.closeout.v1"
ACTION_SCHEMA_VERSION = "ocbrain.action.v1"
OUTCOME_SCHEMA_VERSION = "ocbrain.outcome.v1"
CLOSEOUT_STATUSES = {"completed", "partial", "blocked", "failed", "cancelled"}
DECISION_IMPACTS = {"none", "informed", "changed", "prevented_error", "unknown"}
VERIFIER_STATUSES = {"passed", "failed", "unknown", "not_required"}

# A closeout is a clean success only when the agent claims completion AND no
# verifier it filed says otherwise. Everything else -- every non-completed
# status, and a `completed` carrying a failed verifier -- has to name what did
# not work. See ``_requires_unresolved``.
CLEAN_SUCCESS_STATUSES = {"completed"}

# Wrapper syntax clients paste in front of an otherwise-fine reference, so that
# `ocbrain:COFASC-292` and `COFASC-292` land in the same chain. Matched
# case-insensitively because the wrapper is punctuation, not identity.
TASK_REF_WRAPPER_PREFIXES: tuple[str, ...] = ("ocbrain:", "task:")
# Long enough for every task_ref in a real 1,148-row corpus (longest: 164
# chars), short enough that a pasted document cannot become an index key.
MAX_TASK_REF_NORM = 256
_TASK_REF_WHITESPACE = re.compile(r"\s+")


def normalize_task_ref(task_ref: Any) -> str:
    """Fold a free-text ``task_ref`` into the key two closeouts chain on.

    Trims, collapses internal whitespace, strips the wrapper prefixes above, and
    bounds the length. **Case is preserved.** This column carries Linear ids
    (`COFASC-292`) and raw UUIDs, and ``scope.py`` gives the reasoning for the
    same decision about task and session ids: they are machine-minted,
    high-cardinality, and often case-significant, so folding them risks
    collapsing two distinct references into one. Only the spellings a human
    varies by accident are folded.

    Idempotent: ``normalize_task_ref(normalize_task_ref(x)) == normalize_task_ref(x)``.
    A value that is nothing but wrapper prefixes folds back to its trimmed self
    rather than to the empty string, so an odd input cannot chain itself onto
    every other odd input.

    The raw ``task_ref`` column keeps the verbatim value forever; this is a
    derived key stored beside it, never a replacement.
    """
    collapsed = _TASK_REF_WHITESPACE.sub(" ", str(task_ref or "")).strip()
    stripped = collapsed
    peeled = True
    while peeled:
        peeled = False
        for prefix in TASK_REF_WRAPPER_PREFIXES:
            if stripped[: len(prefix)].lower() == prefix:
                stripped = stripped[len(prefix) :].strip()
                peeled = True
    return (stripped or collapsed)[:MAX_TASK_REF_NORM]


# --------------------------------------------------------------------------- #
# Session identity
# --------------------------------------------------------------------------- #

# The two shapes a runtime actually mints. Claude Code writes a hyphenated UUID
# that is byte-identical to its transcript filename; Codex writes a UUIDv7 in the
# same shape; a bare 32- or 40-char hex digest is the other machine-minted form
# seen in the wild (Hermes runtime ids). Everything else in the live corpus is a
# human typing something they thought was descriptive.
_UUID_SESSION = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_HEX_SESSION = re.compile(r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40})$")
_DATE_SESSION = re.compile(r"^\d{4}-?\d{2}-?\d{2}")
_PATH_SESSION = re.compile(r"[/\\]|^~")

# Shapes that ARE a runtime id. Membership, not a regex, so a caller can ask.
RUNTIME_SESSION_SHAPES = frozenset({"runtime_uuid", "runtime_hex"})

# Where the id in ``task_closeouts.session_id`` came from, in descending trust.
# Recorded beside the value so nobody has to guess later which rows are joinable.
SESSION_ID_SOURCES = frozenset(
    {"harness_attested", "agent_reported", "server_connection", "none"}
)
# Server-minted fallback ids wear a prefix. A connection id is NOT a transcript
# id and must never be mistaken for one by a later join; the prefix makes that
# structural rather than a convention somebody has to remember.
SERVER_CONNECTION_SESSION_PREFIX = "conn:"

# What happens to a session id that is not runtime-shaped.
#   ``enforce``     refuse the closeout, naming the shape and where to get one.
#   ``quarantine``  keep the claim in the receipt, keep it out of the identity
#                   column, and fall through to the server-observed id.
#   ``off``         the pre-2026-08-28 behaviour: store whatever was sent.
SESSION_ID_POLICIES = frozenset({"enforce", "quarantine", "off"})

# Synthetic but shape-valid, so public error output teaches the contract without
# publishing a caller's real session identifier.
_EXAMPLE_SESSION_ID = "018f27db-3a4c-7b19-92ef-123456789abc"


def classify_session_id(value: Any) -> str:
    """Name the shape of a claimed session id.

    Returns ``absent`` for nothing at all, one of :data:`RUNTIME_SESSION_SHAPES`
    for a machine-minted id, and otherwise the specific way it is wrong --
    ``filesystem_path``, ``contains_space``, ``date_like``, ``slug`` -- so the
    refusal can say what the caller actually sent.

    Measured over the 1,239 closeouts in a read-only backup of the live core
    taken 2026-08-28 12:30 PDT: 211 ``runtime_uuid``, 431 ``absent``, 296
    ``date_like``, 239 ``slug``, 35 ``contains_space``, 27 ``filesystem_path``.
    Every one of the 94 closeouts that joins a Claude Code transcript is
    ``runtime_uuid``; the other 1,145 join nothing, which is what makes this a
    shape question rather than a taste question.
    """
    if not isinstance(value, str) or not value.strip():
        return "absent"
    text = value.strip()
    if _UUID_SESSION.match(text):
        return "runtime_uuid"
    if _HEX_SESSION.match(text):
        return "runtime_hex"
    if _PATH_SESSION.search(text):
        return "filesystem_path"
    if any(char.isspace() for char in text):
        return "contains_space"
    if _DATE_SESSION.match(text):
        return "date_like"
    return "slug"


def is_runtime_session_id(value: Any) -> bool:
    """Whether ``value`` is a machine-minted session id rather than a label."""
    return classify_session_id(value) in RUNTIME_SESSION_SHAPES


def _session_id_error(claim: str, shape: str) -> str:
    return (
        f"context.session must be the runtime's own session id, not {claim!r} ({shape}). "
        f"A valid one is a UUID like {_EXAMPLE_SESSION_ID}, or a bare 32/40-character "
        "hex id. Claude Code exports it as $CLAUDE_CODE_SESSION_ID; any other client "
        "can export $OCBRAIN_SESSION_ID and the server will read it from its own "
        "environment. If this runtime has no session id, omit context.session "
        "entirely and the server records its own connection id instead. Do not "
        "invent one: of the 597 hand-written session ids in this core, zero join "
        "a transcript."
    )


def resolve_session_identity(
    claim: Any,
    observed: Provenance,
    *,
    policy: str = "enforce",
) -> dict[str, Any]:
    """Decide what goes in the identity column, and say where it came from.

    Precedence is descending trust, which is the opposite of descending
    convenience:

    1. ``client_session_hint`` -- read by the server from its own process
       environment, so no model can type it in.
    2. the caller's ``context.session``, but only if it is runtime-shaped.
    3. the server's own connection id, prefixed ``conn:``.
    4. nothing.

    A caller that supplies a non-runtime-shaped id is refused under the default
    ``enforce`` policy. The gate is always satisfiable: omitting the field is
    legal and lands on rule 3 or 4, so no client is ever unable to file a
    closeout. What it is not is *silently* satisfiable -- the error is the only
    channel that has ever reached the agent, and six weeks of prose guidance
    moved the UUID rate from 15.1% (July) to 19.6% (August).

    The caller's claim is never destroyed. It stays verbatim in
    ``context.session`` inside the receipt, and is echoed as ``session_id_claim``
    whenever it differs from what was stored.
    """
    if policy not in SESSION_ID_POLICIES:
        raise ValueError(
            f"session_id_policy must be one of: {', '.join(sorted(SESSION_ID_POLICIES))}"
        )
    claimed = claim.strip() if isinstance(claim, str) and claim.strip() else None
    shape = classify_session_id(claimed)
    if policy == "off":
        return {
            "session_id": claimed,
            "session_id_source": "agent_reported" if claimed else "none",
            "session_id_shape": shape,
        }
    if claimed is not None and shape not in RUNTIME_SESSION_SHAPES:
        if policy == "enforce":
            raise ValueError(_session_id_error(claimed, shape))
        claimed = None  # quarantine: recorded below, never stored as identity

    hint = observed.client_session_hint if observed else None
    hint = hint.strip() if isinstance(hint, str) and hint.strip() else None
    resolved: dict[str, Any] = {"session_id_shape": shape}
    if hint is not None and is_runtime_session_id(hint):
        resolved["session_id"] = hint
        resolved["session_id_source"] = "harness_attested"
    elif claimed is not None:
        resolved["session_id"] = claimed
        resolved["session_id_source"] = "agent_reported"
    elif observed is not None and observed.server_connection_id:
        resolved["session_id"] = (
            f"{SERVER_CONNECTION_SESSION_PREFIX}{observed.server_connection_id}"
        )
        resolved["session_id_source"] = "server_connection"
    else:
        resolved["session_id"] = None
        resolved["session_id_source"] = "none"
    stored = resolved["session_id"]
    original = claim.strip() if isinstance(claim, str) and claim.strip() else None
    if original is not None and original != stored:
        # Two runtime-shaped ids that disagree is the Claude Code subagent case:
        # the harness hint is inherited from the parent, the model may know its
        # own. Both are kept; the server-observed one is what the column serves.
        resolved["session_id_claim"] = original
        if shape in RUNTIME_SESSION_SHAPES:
            resolved["session_id_conflict"] = True
    return resolved


# --------------------------------------------------------------------------- #
# Runtime family
# --------------------------------------------------------------------------- #

# ``context.runtime`` was free text and arrived as 160 distinct spellings across
# 1,239 live closeouts -- five of "local mac", four of "codex desktop", and
# values like "local macOS + analytics ClickHouse" that describe an
# environment rather than a client. Nothing can be grouped by that column.
#
# These are the client families a closeout can be grouped by. `unknown` is a
# real member, not a failure: "local", "desktop" and "macOS" name the machine,
# and inventing a client for them would be guessing.
RUNTIME_FAMILIES = ("claude-code", "codex", "cursor", "hermes", "mcp", "cli", "unknown")

# Ordered rules, matched against whole segments of the folded spelling. Order is
# precedence and matters: "local Codex desktop and Hermes gateway" names two,
# and a stable answer beats an accurate-sounding one. Every token here appears in
# the live corpus listing; none was invented.
#
# Segments, not substrings. A substring match on "cli" classified
# "local macOS + analytics ClickHouse" (13 live rows) as the CLI family,
# which is how a normaliser quietly invents data.
RUNTIME_FAMILY_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("claude-code", frozenset({"claude"})),
    ("codex", frozenset({"codex"})),
    ("cursor", frozenset({"cursor"})),
    ("hermes", frozenset({"hermes"})),
    ("cli", frozenset({"launchd", "cron", "cli", "dagster"})),
    ("mcp", frozenset({"mcp"})),
)
# Whole folded spellings whose family is known but whose tokens cannot safely be
# added to the rules above. `ocbrain-runtime-call` is set by
# ``ocbrain.runtime_call`` -- this repo's own one-shot MCP path, 2 live rows --
# and the sibling mapper in scripts/procmine has placed it since it was written.
# It is matched exactly rather than by a token because `ocbrain` also appears in
# `local-agent-mode-ocbrain`, the Claude Code client key on 66 live rows, and
# folding those into `mcp` would be the invent-a-family failure the segment rule
# exists to prevent. Anything install-specific goes in ``runtime_aliases``, not
# here; the bar for this table is a spelling this repository itself emits.
RUNTIME_FAMILY_EXACT: dict[str, str] = {
    "ocbrain-runtime-call": "mcp",
}

# Everything a real spelling uses to join two words: whitespace, punctuation,
# and the path/profile separators in values like `hermes@example` and
# `~/hermes/example`.
_RUNTIME_SEPARATORS = re.compile(r"[^0-9a-z]+")


def fold_runtime_label(value: Any) -> str:
    """Lowercase one runtime spelling and reduce every separator to ``-``."""
    return _RUNTIME_SEPARATORS.sub("-", str(value or "").strip().lower()).strip("-")


def runtime_family(
    *candidates: Any, aliases: dict[str, str] | None = None
) -> str:
    """Map runtime spellings onto one of :data:`RUNTIME_FAMILIES`.

    Candidates are tried in order and the first that resolves wins, so callers
    pass the server-observed ``client_runtime_key`` before the model's claim:
    what the process saw outranks what the model typed. A candidate that names
    only an environment ("local", "macOS") resolves to nothing and falls through
    to the next one.

    ``aliases`` is the operator's table for install-specific labels, mapping a
    folded spelling to a family. It ships empty and lives in config for the same
    reason ``scopes.aliases`` does: a real fleet's profile names are operator
    data, and this repo is public. Its keys are folded exactly the way a
    candidate is, so an entry written `{"claude code desktop": ...}` reaches the
    candidate `claude-code-desktop`; a key that could never match would be a
    trap with a config file in front of it.

    This is the one runtime folder in the repo that write paths use.
    ``procmine.episodes.normalize_runtime`` asks it first and only falls through
    to its own install-specific rules when this abstains, so the two can differ
    by abstention but never by contradiction --
    ``tests/test_closeout_discipline.py`` asserts that over the live census.

    Pure and history-independent, so the same function classifies a row written
    today and a row written in July. ``task_closeouts`` is append-only and
    historical spellings can never be rewritten in place; this is what keeps
    them analysable.
    """
    table = {
        fold_runtime_label(k): str(v).strip() for k, v in (aliases or {}).items()
    }
    for candidate in candidates:
        folded = fold_runtime_label(candidate)
        if not folded:
            continue
        mapped = table.get(folded)
        if mapped in RUNTIME_FAMILIES:
            return mapped
        exact = RUNTIME_FAMILY_EXACT.get(folded)
        if exact is not None:
            return exact
        segments = set(folded.split("-"))
        for family, tokens in RUNTIME_FAMILY_RULES:
            if segments & tokens:
                return family
    return "unknown"


def record_closeout(
    conn: sqlite3.Connection,
    *,
    task_ref: str,
    status: str,
    summary: str,
    context: ScopeContext | None = None,
    retrieval_use_ids: list[str] | None = None,
    decision_impact: str = "unknown",
    decision_note: str | None = None,
    artifact_refs: list[dict[str, Any]] | None = None,
    verifier_refs: list[dict[str, Any]] | None = None,
    actions: list[dict[str, Any]] | None = None,
    outcomes: list[dict[str, Any]] | None = None,
    awaiting: str | None = None,
    unresolved: str | None = None,
    runtime_detail: str | None = None,
    actor: str = "agent",
    parent_closeout_id: str | None = None,
    provenance: Provenance | None = None,
) -> dict[str, Any]:
    """Append a generic execution outcome receipt without promoting knowledge.

    ``provenance`` is what the server observed about the connection that sent
    the closeout; ``context.session`` and ``context.runtime`` remain what the
    model said. Both are recorded, separately, because only one of them can be
    trusted and the receipt should say which. The observed fields join into the
    hashed provenance block, so two byte-identical closeouts written from two
    different connections are two distinct receipts rather than a UNIQUE
    collision on ``content_hash``.

    ``parent_closeout_id`` names the closeout this one continues. It is
    validated against ``task_closeouts.id``, and an unresolved value is recorded
    in the receipt as a claim with ``chain.parent_unresolved`` set rather than
    refused: a closeout must never fail over a bad parent, for the same reason
    an unknown ``retrieval_use_id`` no longer voids the whole receipt. Only a
    resolved parent reaches the column, so the pointer is never dangling.

    ``unresolved`` names what did not work. It is required whenever the receipt
    is not a clean success -- see :func:`_requires_unresolved` -- because
    ``brain.ledger``'s only job is stopping the next session repeating this
    afternoon, and it cannot do that from a status word alone.

    ``runtime_detail`` is where the environment goes: "analytics ClickHouse",
    "launchd", "zone-a". It exists so that detail stops being crammed into
    ``context.runtime``, which is meant to name the client and nothing else.
    """
    task_ref = _required_text(task_ref, "task_ref")
    summary = _required_text(summary, "summary")
    actor = _required_text(actor, "actor")
    settings = _closeout_settings()
    if status not in CLOSEOUT_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(CLOSEOUT_STATUSES))}")
    if decision_impact not in DECISION_IMPACTS:
        raise ValueError(
            f"decision_impact must be one of: {', '.join(sorted(DECISION_IMPACTS))}"
        )
    if status == "blocked" and not (awaiting and awaiting.strip()):
        raise ValueError("blocked closeouts require awaiting")
    retrieval_ids, unmatched_retrieval_ids = _partition_retrieval_ids(
        conn, _dedupe_text(retrieval_use_ids or [])
    )
    artifacts = [_normalize_artifact_ref(value) for value in artifact_refs or []]
    verifiers = [_normalize_verifier_ref(value) for value in verifier_refs or []]
    normalized_actions = [_normalize_action(value) for value in actions or []]
    normalized_outcomes = [_normalize_outcome(value) for value in outcomes or []]
    verification_status = _verification_status(verifiers)
    unresolved_text = _optional_text(unresolved)
    detail = _optional_text(runtime_detail)
    resolved = context or ScopeContext()
    closed_at = now_iso()
    task_ref_norm = normalize_task_ref(task_ref)
    parent_claim = parent_closeout_id.strip() if isinstance(parent_closeout_id, str) else None
    resolved_parent = _resolve_parent(conn, parent_claim)
    chain: dict[str, Any] = {
        "parent_closeout_id": parent_claim or None,
        # One indexed read on (task_ref_norm, closed_at). Historical rows carry
        # a NULL norm and are deliberately not rewritten, so a chain begins at
        # the first closeout written by a server that has this column.
        "previous_in_chain": _previous_in_chain(conn, task_ref_norm),
    }
    if parent_claim and resolved_parent is None:
        chain["parent_unresolved"] = True
    observed = provenance or EMPTY_PROVENANCE
    # Both write-time gates are evaluated before either is raised, so a caller
    # with two problems learns both in one round trip. Telling an unattended
    # agent about one refusal at a time costs it two retries for one closeout.
    problems: list[str] = []
    identity: dict[str, Any] = {}
    try:
        identity = resolve_session_identity(
            resolved.session, observed, policy=settings.session_id_policy
        )
    except ValueError as exc:
        problems.append(str(exc))
    if (
        settings.require_unresolved
        and unresolved_text is None
        and _requires_unresolved(status, verification_status)
    ):
        problems.append(_unresolved_error(status, verification_status))
    if problems:
        raise ValueError("\n\n".join(problems))
    family = runtime_family(
        observed.client_runtime_key,
        resolved.runtime,
        aliases=settings.runtime_aliases,
    )
    provenance_block: dict[str, Any] = {
        "source": "agent_reported",
        "actor": actor,
        # Verbatim, as it always was. ``runtime_family`` beside it is the
        # groupable form; neither replaces the other.
        "runtime": resolved.runtime or "mcp",
        "runtime_family": family,
        "runtime_detail": detail,
        # Unchanged meaning: what the model claimed. Every historical receipt
        # reads this way and a silently re-pointed field is worse than a new one.
        "session_id": resolved.session,
        # What actually went in the identity column, and on whose word.
        "session_identity": identity,
        "reported_at": closed_at,
        # Named so nobody has to read this file to know which half is a claim.
        "server_observed": observed.to_dict(),
    }
    base_receipt: dict[str, Any] = {
        "schema_version": CLOSEOUT_SCHEMA_VERSION,
        "closed_at": closed_at,
        "task_ref": task_ref,
        "status": status,
        "summary": summary,
        "decision": {
            "impact": decision_impact,
            "note": decision_note.strip() if decision_note and decision_note.strip() else None,
        },
        "retrieval_use_ids": retrieval_ids,
        "unmatched_retrieval_use_ids": unmatched_retrieval_ids,
        "artifact_refs": artifacts,
        "verifier_refs": verifiers,
        "actions": normalized_actions,
        "outcomes": normalized_outcomes,
        "verification_status": verification_status,
        "awaiting": awaiting.strip() if awaiting and awaiting.strip() else None,
        "unresolved": unresolved_text,
        "task_ref_norm": task_ref_norm,
        "chain": chain,
        "context": resolved.to_dict(),
        "provenance": provenance_block,
    }
    digest = content_hash(canonical_json(base_receipt))
    closeout_id = stable_id("close", task_ref, closed_at, digest)
    receipt = {"id": closeout_id, "content_hash": digest, **base_receipt}
    conn.execute(
        """
        INSERT INTO task_closeouts (
          id, schema_version, closed_at, task_ref, status, summary,
          decision_impact, decision_note, awaiting, runtime, session_id,
          context_json, artifact_refs_json, verifier_refs_json, provenance_json,
          receipt_json, content_hash,
          server_connection_id, client_session_hint, client_runtime_key,
          parent_closeout_id, task_ref_norm,
          session_id_source, runtime_family, unresolved
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            closeout_id,
            CLOSEOUT_SCHEMA_VERSION,
            closed_at,
            task_ref,
            status,
            summary,
            decision_impact,
            base_receipt["decision"]["note"],
            base_receipt["awaiting"],
            provenance_block["runtime"],
            identity["session_id"],
            canonical_json(base_receipt["context"]),
            canonical_json(artifacts),
            canonical_json(verifiers),
            canonical_json(provenance_block),
            canonical_json(receipt),
            digest,
            observed.server_connection_id,
            observed.client_session_hint,
            observed.client_runtime_key,
            resolved_parent,
            task_ref_norm,
            identity["session_id_source"],
            family,
            unresolved_text,
        ),
    )
    for retrieval_use_id in retrieval_ids:
        conn.execute(
            "INSERT INTO task_closeout_retrievals (closeout_id, retrieval_use_id) "
            "VALUES (?, ?)",
            (closeout_id, retrieval_use_id),
        )
        conn.execute(
            "UPDATE retrieval_uses SET affected_decision = ? WHERE id = ?",
            (_affected_decision(decision_impact), retrieval_use_id),
        )
    return receipt


def get_closeout(conn: sqlite3.Connection, closeout_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT receipt_json FROM task_closeouts WHERE id = ?",
        (closeout_id,),
    ).fetchone()
    return json.loads(row["receipt_json"]) if row is not None else None


def _resolve_parent(conn: sqlite3.Connection, parent_closeout_id: str | None) -> str | None:
    """Return the parent id only if a closeout by that id exists.

    An unresolved claim is kept in the receipt and out of the column, mirroring
    how ``_partition_retrieval_ids`` treats an unknown retrieval id: the claim
    is evidence, a dangling pointer is not.
    """
    if not parent_closeout_id:
        return None
    row = conn.execute(
        "SELECT id FROM task_closeouts WHERE id=?", (parent_closeout_id,)
    ).fetchone()
    return str(row["id"]) if row is not None else None


def _previous_in_chain(conn: sqlite3.Connection, task_ref_norm: str) -> str | None:
    """The most recent closeout already filed against the same normalized ref.

    This is what gives an agent chain continuity without having to remember an
    id across sessions: it never has to pass ``parent_closeout_id`` to find out
    what the last run on this task concluded. Ordered by ``closed_at`` with the
    id as a tiebreaker, so two closeouts written inside the same clock tick
    still resolve deterministically.
    """
    if not task_ref_norm:
        return None
    row = conn.execute(
        "SELECT id FROM task_closeouts WHERE task_ref_norm=? "
        "ORDER BY closed_at DESC, id DESC LIMIT 1",
        (task_ref_norm,),
    ).fetchone()
    return str(row["id"]) if row is not None else None


def _partition_retrieval_ids(
    conn: sqlite3.Connection, retrieval_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Split linked retrieval ids into (known, unknown), refusing neither.

    This used to raise on any unknown id, which voided the entire receipt: an
    agent holding one mangled id — a live fleet retried `ocbret_…` three times
    in one evening — cannot repair its own context, so the retry fails
    identically and the closeout is simply lost. A receipt with one unlinked id
    recorded as unmatched is strictly more evidence than no receipt. Unknown ids
    are carried in the receipt verbatim, and never inserted into
    ``task_closeout_retrievals``, so no join is ever fabricated.
    """
    if not retrieval_ids:
        return [], []
    placeholders = ",".join("?" for _ in retrieval_ids)
    found = {
        str(row["id"])
        for row in conn.execute(
            f"SELECT id FROM retrieval_uses WHERE id IN ({placeholders})",  # noqa: S608
            retrieval_ids,
        )
    }
    known = [value for value in retrieval_ids if value in found]
    unknown = [value for value in retrieval_ids if value not in found]
    return known, unknown


def _normalize_artifact_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("artifact_refs entries must be objects")
    uri = _required_text(value.get("uri"), "artifact_refs[].uri")
    result: dict[str, Any] = {"uri": uri}
    for key in ("kind", "sha256", "label"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"artifact_refs[].{key}")
    return result


def _normalize_verifier_ref(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("verifier_refs entries must be objects")
    uri = _required_text(value.get("uri"), "verifier_refs[].uri")
    status = str(value.get("status") or "unknown")
    if status not in VERIFIER_STATUSES:
        raise ValueError(f"verifier status must be one of: {', '.join(sorted(VERIFIER_STATUSES))}")
    result: dict[str, Any] = {"uri": uri, "status": status}
    for key in ("kind", "sha256", "detail"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"verifier_refs[].{key}")
    return result


def _normalize_action(value: Any) -> dict[str, Any]:
    """Preserve a portable action envelope without pretending it is a reward."""
    if not isinstance(value, dict):
        raise ValueError("actions entries must be objects")
    target = _json_object(value.get("target"), "actions[].target", required=True)
    result: dict[str, Any] = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "mechanism": _required_text(value.get("mechanism"), "actions[].mechanism"),
        "semantic_role": _required_text(
            value.get("semantic_role"), "actions[].semantic_role"
        ),
        "target": target,
    }
    for key in ("action_id", "occurred_at"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"actions[].{key}")
    for key in ("context_before", "policy", "cost", "provenance"):
        item = value.get(key)
        if item is not None:
            result[key] = _json_object(item, f"actions[].{key}", required=False)
    features = value.get("features")
    if features is not None:
        normalized_features = _json_object(
            features, "actions[].features", required=False
        )
        if normalized_features:
            result["features"] = normalized_features
    if "features" in result:
        result["feature_schema"] = _required_text(
            value.get("feature_schema"), "actions[].feature_schema"
        )
    elif value.get("feature_schema") is not None:
        raise ValueError("actions[].feature_schema requires non-empty actions[].features")
    return result


def _normalize_outcome(value: Any) -> dict[str, Any]:
    """Keep outcome components and local meaning instead of one scalar reward."""
    if not isinstance(value, dict):
        raise ValueError("outcomes entries must be objects")
    if "value" not in value:
        raise ValueError("outcomes[].value is required")
    result: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "metric": _required_text(value.get("metric"), "outcomes[].metric"),
        "value": _json_value(value["value"], "outcomes[].value"),
        "role": _required_text(value.get("role") or "primary", "outcomes[].role"),
        "interpretation": _required_text(
            value.get("interpretation"), "outcomes[].interpretation"
        ),
    }
    for key in ("unit", "observed_at"):
        item = value.get(key)
        if item is not None:
            result[key] = _required_text(item, f"outcomes[].{key}")
    for key in (
        "observation_window",
        "baseline",
        "counterfactual",
        "attribution",
        "uncertainty",
    ):
        item = value.get(key)
        if item is not None:
            result[key] = _json_value(item, f"outcomes[].{key}")
    features = value.get("features")
    if features is not None:
        normalized_features = _json_object(
            features, "outcomes[].features", required=False
        )
        if normalized_features:
            result["features"] = normalized_features
    if "features" in result:
        result["feature_schema"] = _required_text(
            value.get("feature_schema"), "outcomes[].feature_schema"
        )
    elif value.get("feature_schema") is not None:
        raise ValueError("outcomes[].feature_schema requires non-empty outcomes[].features")
    return result


def _json_object(value: Any, name: str, *, required: bool) -> dict[str, Any]:
    if not isinstance(value, dict) or (required and not value):
        suffix = "a non-empty object" if required else "an object"
        raise ValueError(f"{name} must be {suffix}")
    return _json_value(value, name)


def _json_value(value: Any, name: str) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON") from exc
    return json.loads(encoded)


def _requires_unresolved(status: str, verification_status: str) -> bool:
    """Whether this receipt carries evidence that something did not work.

    Two independent triggers, and either alone is enough:

    * the agent's own status is not a completion, or
    * a verifier the agent filed says ``failed``.

    The second is what makes this more than a restatement of ``status``. On the
    live core, 95 closeouts claim ``completed`` while carrying a failed
    verifier -- 88 of them alongside passing ones -- and none of them has a field
    saying which check failed or why it did not stop the claim.

    Deliberately NOT a status override. Seven of the twelve closeouts whose
    verifiers *all* failed are read-only audits where the FAIL verdict is the
    deliverable ("Read-only re-review found remaining blockers; verdict FAIL"),
    so deriving ``failed`` from the evidence would relabel successful work.
    The caller keeps the verdict and owes an explanation.
    """
    return status not in CLEAN_SUCCESS_STATUSES or verification_status == "failed"


def _unresolved_error(status: str, verification_status: str) -> str:
    if verification_status == "failed" and status in CLEAN_SUCCESS_STATUSES:
        because = (
            f"status is {status!r} but a verifier_ref reports 'failed'"
        )
    else:
        because = f"status is {status!r}, which is not a completion"
    return (
        f"unresolved is required: {because}. State what did not work and is still "
        "not working, in the caller's own words -- the failing check, the thing "
        "that was not tried, the question left open. brain.ledger reads this to "
        "stop the next session repeating the attempt, and it cannot do that from "
        "a status word. If nothing is outstanding, the closeout is 'completed' "
        "with no failed verifier and this field is not asked for."
    )


def _closeout_settings() -> Any:
    """Resolve the ``closeout`` config section, failing open to shipped defaults.

    Mirrors ``scope._scope_settings``: this sits in front of every closeout
    write, and a malformed config file must not be the reason an agent cannot
    file a receipt. A misspelled ``session_id_policy`` falls back to the shipped
    default for that field alone, rather than raising on every closeout after
    it -- a typo in a config file must not take the write path down.
    """
    from dataclasses import replace

    from ocbrain.config import CloseoutConfig

    default = CloseoutConfig()
    try:
        from ocbrain.config import load_config

        settings = load_config().closeout
    except Exception:  # noqa: BLE001 - config problems must not block a closeout
        return default
    if settings.session_id_policy not in SESSION_ID_POLICIES:
        return replace(settings, session_id_policy=default.session_id_policy)
    return settings


def _verification_status(verifiers: list[dict[str, Any]]) -> str:
    if any(value["status"] == "failed" for value in verifiers):
        return "failed"
    if verifiers and all(value["status"] == "passed" for value in verifiers):
        return "verified"
    return "agent_reported"


def _affected_decision(decision_impact: str) -> int | None:
    if decision_impact in {"informed", "changed", "prevented_error"}:
        return 1
    if decision_impact == "none":
        return 0
    return None


def _optional_text(value: Any) -> str | None:
    """Trim to a non-empty string, or ``None``. Blank is not a statement."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _dedupe_text(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = _required_text(value, "retrieval_use_ids[]")
        if text not in result:
            result.append(text)
    return result
