"""Closeouts, label grades, and the join back to traces.

Three problems have to be solved honestly here.

**The runtime field is free text.** 1,103 closeouts carry ~170 distinct
``runtime`` strings ("codex-desktop", "Codex desktop local Mac", "local macOS +
readonlyprod ClickHouse"). :func:`normalize_runtime` folds them into a handful of
families; anything it cannot place stays ``unknown`` rather than being guessed
into a bucket.

**"completed" is not an outcome.** 933 of 1,103 closeouts say completed and 98%
carry at least one verifier ref, so neither field discriminates on its own. The
grade ladder in :func:`grade_closeout` therefore turns on whether a receipt is
*checkable today* — a verifier or artifact URI that still resolves on disk — not
on what the agent asserted.

**``session_id`` is mostly not a session id.** Only 122 of 1,103 are UUID-shaped;
the rest are human slugs like ``2026-07-20-cro-oracle-images``. The join runs in
tiers and every tier's yield is reported separately, because the shortfall is a
finding about the closeout contract, not noise to smooth over.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ocbrain.closeout import runtime_family
from procmine.runtimes import canonical_runtime

BRAIN_DB = Path(os.path.expanduser("~/.ocbrain/ocbrain.sqlite"))

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

# Verifier kinds a machine could have produced. Used only to separate a claimed
# machine check from a prose assertion; it is strictly weaker than a resolvable
# path, and the grade ladder ranks it that way.
MACHINE_VERIFIER_KINDS = {
    "pytest", "ruff", "mypy", "lint", "test", "tests", "typecheck", "ci",
    "command", "git", "compileall", "build", "smoke", "mcp_smoke", "unit",
    "integration", "e2e", "coverage", "typescript",
}

GRADE_ORDER = [
    "blocked-or-failed",
    "partial",
    "self-reported-completed",
    "artifact-linked",
    "verifier-claimed",
    "verifier-receipted",
]
GRADE_RANK = {name: index for index, name in enumerate(GRADE_ORDER)}

# This taxonomy's name for each family ``ocbrain.closeout.runtime_family``
# returns. Two names differ and the rest are identical: `mcp-direct` and `mcp`
# are the same answer, and so are `host-batch` and `cli`. Declared rather than
# assumed, because it is what lets the two mappers be compared instead of merely
# looked at.
# Values of `task_closeouts.session_id_source` that mean the server, not the
# model, put the value in the column. `harness_attested` is read from the MCP
# child's own environment and `server_connection` is minted by the server; the
# other two (`agent_reported`, `none`) are the model's word or nothing.
_SERVER_OBSERVED_SOURCES = frozenset({"harness_attested", "server_connection"})

_SHARED_FAMILY: dict[str, str] = {
    "claude-code": "claude-code",
    "codex": "codex",
    "cursor": "cursor",
    "hermes": "hermes",
    "mcp": "mcp-direct",
    "cli": "host-batch",
}

# What is left after the shipped families moved to the shared folder: the
# install- and environment-specific tokens, which are this file's actual job.
# A closeout's `runtime` is free text from a fleet whose profile names live on
# one operator's machine, and those cannot go in a public repository's write
# path. `re.search` is the intended semantics *here* -- `hermes-runtimes` is a
# path fragment and an install-specific profile-hash prefix -- but it is no longer
# applied to any family token, because a substring match on "cli" reads
# "ClickHouse" as a CLI. That trap is why the shared folder matches segments.
_RUNTIME_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"telegram|kanban|gateway|hermeswork", re.I), "hermes"),
    (re.compile(r"gpt-5\.6-sol", re.I), "hermes"),
    (re.compile(r"^f15a38ee|hermes-runtimes", re.I), "hermes"),
    (re.compile(r"asa[12]|trino|gcloud", re.I), "host-batch"),
    (re.compile(r"local|mac|darwin|desktop|workspace|worktree", re.I), "unattributed-local"),
]


def normalize_runtime(raw: str | None) -> str:
    """Fold one free-text runtime spelling into a mining family.

    Asks ``ocbrain.closeout.runtime_family`` first -- the same folder the
    closeout write path uses -- and falls through to the install-specific rules
    above only when it abstains. Three normalisers for one question is how two
    of them drift; this one is now a superset of the shipped one rather than a
    rival to it, so the two can differ by abstention but never by contradiction.
    ``tests/test_closeout_discipline.py`` asserts exactly that over the live
    runtime census.
    """
    if not raw:
        return "unknown"
    text = raw.strip()
    if not text:
        return "unknown"
    shared = runtime_family(text)
    if shared != "unknown":
        return _SHARED_FAMILY[shared]
    for pattern, family in _RUNTIME_RULES:
        if pattern.search(text):
            return family
    return "unknown"


def _json_list(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _uri_resolves(uri: str | None) -> bool:
    if not isinstance(uri, str) or not uri:
        return False
    if not uri.startswith(("/", "~")):
        return False
    return os.path.exists(os.path.expanduser(uri))


@dataclass(slots=True)
class Episode:
    """A closeout, graded, with whatever trace we could attach to it."""

    closeout_id: str
    closed_at: str
    task_ref: str
    status: str
    summary: str
    runtime_raw: str | None
    runtime: str
    runtime_slug: str | None
    session_id: str | None
    session_hint: str | None
    project: str | None
    repo: str | None
    grade: str
    grade_rank: int
    n_verifiers: int
    n_verifiers_failed: int
    n_artifacts: int
    resolvable_verifier_uris: int
    resolvable_artifact_uris: int
    session_source: str = "model_reported"
    trace_id: str | None = None
    trace_runtime: str | None = None
    join_tier: str = "unjoined"
    segmented: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = {
            key: getattr(self, key)
            for key in (
                "closeout_id", "closed_at", "task_ref", "status", "summary",
                "runtime_raw", "runtime", "runtime_slug", "session_id",
                "session_hint", "session_source", "project", "repo",
                "grade", "grade_rank", "n_verifiers", "n_verifiers_failed",
                "n_artifacts", "resolvable_verifier_uris",
                "resolvable_artifact_uris", "trace_id", "trace_runtime",
                "join_tier", "segmented",
            )
        }
        data["n_events"] = len(self.events)
        return data


def grade_closeout(
    status: str,
    verifier_refs: list[dict[str, Any]],
    artifact_refs: list[dict[str, Any]],
) -> tuple[str, int, int]:
    """Return ``(grade, resolvable_verifier_uris, resolvable_artifact_uris)``.

    The ladder, strongest first:

    ``verifier-receipted``
        a *passed* verifier whose URI still resolves on this filesystem.
    ``verifier-claimed``
        a passed verifier of a machine-producible kind, but nothing to check.
    ``artifact-linked``
        no usable verifier, but an artifact path that still resolves.
    ``self-reported-completed``
        status completed, and nothing above holds.
    ``partial`` / ``blocked-or-failed``
        taken straight from status.
    """
    resolvable_verifiers = sum(
        1
        for ref in verifier_refs
        if ref.get("status") == "passed" and _uri_resolves(ref.get("uri"))
    )
    resolvable_artifacts = sum(1 for ref in artifact_refs if _uri_resolves(ref.get("uri")))

    if status in {"blocked", "failed", "cancelled"}:
        return "blocked-or-failed", resolvable_verifiers, resolvable_artifacts
    if status == "partial":
        return "partial", resolvable_verifiers, resolvable_artifacts

    if resolvable_verifiers:
        return "verifier-receipted", resolvable_verifiers, resolvable_artifacts
    machine_claimed = any(
        ref.get("status") == "passed"
        and str(ref.get("kind") or "").lower() in MACHINE_VERIFIER_KINDS
        for ref in verifier_refs
    )
    if machine_claimed:
        return "verifier-claimed", resolvable_verifiers, resolvable_artifacts
    if resolvable_artifacts:
        return "artifact-linked", resolvable_verifiers, resolvable_artifacts
    return "self-reported-completed", resolvable_verifiers, resolvable_artifacts


def load_episodes(db_path: Path | None = None) -> list[Episode]:
    path = db_path or BRAIN_DB
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        # `client_session_hint` only exists on cores written after
        # server-captured provenance landed. Probe rather than assume: this
        # miner is pointed at old snapshots as often as at the live core.
        columns = {row[1] for row in conn.execute("pragma table_info(task_closeouts)")}
        hint_column = "client_session_hint" if "client_session_hint" in columns else "NULL"
        runtime_key_column = "client_runtime_key" if "client_runtime_key" in columns else "NULL"
        # Written by ocbrain.closeout from 2026-08-28: who filled `session_id`.
        # Authoritative where present, and NULL on every earlier row, which is
        # why the hint heuristic below is kept rather than replaced.
        source_column = "session_id_source" if "session_id_source" in columns else "NULL"
        rows = conn.execute(
            "select id, closed_at, task_ref, status, summary, runtime, session_id, "
            "context_json, artifact_refs_json, verifier_refs_json, provenance_json, "
            f"{hint_column}, {runtime_key_column}, {source_column} "
            "from task_closeouts order by closed_at"
        ).fetchall()
    finally:
        conn.close()

    episodes: list[Episode] = []
    for (
        closeout_id, closed_at, task_ref, status, summary, runtime_raw, session_id,
        context_json, artifact_json, verifier_json, provenance_json,
        session_hint, runtime_key, stored_session_source,
    ) in rows:
        verifier_refs = _json_list(verifier_json)
        artifact_refs = _json_list(artifact_json)
        grade, resolvable_verifiers, resolvable_artifacts = grade_closeout(
            status, verifier_refs, artifact_refs
        )
        try:
            context = json.loads(context_json) if context_json else {}
        except Exception:
            context = {}
        try:
            provenance = json.loads(provenance_json) if provenance_json else {}
        except Exception:
            provenance = {}
        # The server-observed hint outranks anything the model typed: it came
        # from the MCP child's own environment, and on Claude Code it is
        # byte-identical to the transcript filename this episode wants to join
        # to. It is still only harness-attested -- see ocbrain.provenance -- so
        # the model-supplied value is kept beside it, not overwritten.
        reported_session = session_id or provenance.get("session_id") or context.get("session")
        effective_session = session_hint or reported_session
        # Where the write path recorded who filled the column, believe it. A
        # `conn:<32hex>` id is minted by the server and carries no hint, so the
        # hint heuristic alone called every one of those `model_reported` --
        # precisely backwards. The heuristic stays as the fallback for the rows
        # written before the column existed, where it is all there is.
        session_source = (
            "server_observed"
            if stored_session_source in _SERVER_OBSERVED_SOURCES
            or (stored_session_source is None and session_hint)
            else "model_reported"
        )
        effective_runtime = runtime_raw or provenance.get("runtime") or context.get("runtime")
        episodes.append(
            Episode(
                closeout_id=closeout_id,
                closed_at=closed_at,
                task_ref=task_ref,
                status=status,
                summary=summary,
                runtime_raw=effective_runtime,
                runtime=normalize_runtime(runtime_key or effective_runtime),
                runtime_slug=canonical_runtime(runtime_key or effective_runtime),
                session_id=effective_session or None,
                session_hint=session_hint or None,
                session_source=session_source,
                project=context.get("project") if isinstance(context, dict) else None,
                repo=context.get("repo") if isinstance(context, dict) else None,
                grade=grade,
                grade_rank=GRADE_RANK[grade],
                n_verifiers=len(verifier_refs),
                n_verifiers_failed=sum(1 for r in verifier_refs if r.get("status") == "failed"),
                n_artifacts=len(artifact_refs),
                resolvable_verifier_uris=resolvable_verifiers,
                resolvable_artifact_uris=resolvable_artifacts,
            )
        )
    return episodes


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


_TRACE_FAMILY = {
    "claude-code": "claude-code",
    "claude-code-subagent": "claude-code",
    "codex": "codex",
    "hermes-legacy": "hermes",
}


_WORD = re.compile(r"[a-z0-9]{4,}")
_WEAK_CONTEXT_TOKENS = {
    "users", "home", "documents", "coframe", "work", "repo",
    "local", "macos", "codex", "hermes", "profiles", "sessions", "main", "root",
    # See the note in dag._STOPWORDS: derived, not hardcoded.
    os.path.basename(os.path.expanduser("~")).lower(),
}


def _context_tokens(*values: str | None) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        if value:
            tokens.update(_WORD.findall(value.lower()))
    return tokens - _WEAK_CONTEXT_TOKENS


def _disambiguate(episode: Episode, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Break a tie between concurrently-open sessions using where they ran.

    Jonathan runs several sessions at once, so "a closeout landed while this
    session was open" is frequently true of three sessions. When the closeout
    names a repo or project and exactly one of the candidates was working in a
    directory (or, for hermes, a profile) that mentions it, that is a real
    identification rather than a coin flip. Anything less decisive stays
    ambiguous.
    """
    wanted = _context_tokens(episode.repo, episode.project, episode.task_ref)
    if not wanted:
        return None
    scored: list[tuple[int, dict[str, Any]]] = []
    for candidate in candidates:
        have = _context_tokens(
            candidate.get("cwd_tail"), candidate.get("cwd"), candidate.get("runtime")
        )
        scored.append((len(wanted & have), candidate))
    scored.sort(key=lambda item: -item[0])
    best = scored[0][0]
    if best == 0:
        return None
    if len(scored) > 1 and scored[1][0] == best:
        return None
    return scored[0][1]


def _trace_family(runtime: str) -> str:
    if runtime.startswith("hermes:"):
        return "hermes"
    return _TRACE_FAMILY.get(runtime, runtime)


def join_episodes(
    episodes: list[Episode],
    traces: list[dict[str, Any]],
    *,
    temporal_grace_minutes: int = 45,
) -> dict[str, int]:
    """Attach traces to episodes in tiers, mutating ``episodes`` in place.

    Tiers, strongest first:

    ``exact``
        ``session_id`` equals a trace id.
    ``uuid``
        a UUID embedded in ``session_id`` equals a trace id (codex slugs
        sometimes wrap the rollout uuid in extra text).
    ``temporal``
        same runtime family, and ``closed_at`` falls inside exactly one trace
        window extended by ``temporal_grace_minutes``.
    ``temporal-ambiguous``
        the same test, but more than one session was open. The nearest match is
        still attached so the yield table is honest, but the tier is separate
        because such an episode is not evidence about any one session and
        :func:`mining_set` drops it.

    Returns per-tier counts. Nothing falls back silently: an episode with no
    match keeps ``join_tier == "unjoined"``.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for trace in traces:
        by_id.setdefault(trace["trace_id"], trace)

    windows: dict[str, list[tuple[datetime, datetime, dict[str, Any]]]] = {}
    for trace in traces:
        start = _parse_time(trace.get("started_at"))
        end = _parse_time(trace.get("ended_at")) or start
        if start is None or end is None:
            continue
        windows.setdefault(_trace_family(trace["runtime"]), []).append((start, end, trace))
    for entries in windows.values():
        entries.sort(key=lambda item: item[0])

    counts = {
        "exact": 0, "uuid": 0, "temporal": 0, "temporal-context": 0,
        "temporal-ambiguous": 0, "unjoined": 0,
    }
    grace = timedelta(minutes=temporal_grace_minutes)

    for episode in episodes:
        trace = None
        tier = "unjoined"
        session = episode.session_id or ""
        if session and session in by_id:
            trace = by_id[session]
            tier = "exact"
        else:
            match = _UUID.search(session)
            if match and match.group(0).lower() in by_id:
                trace = by_id[match.group(0).lower()]
                tier = "uuid"
        if trace is None:
            closed = _parse_time(episode.closed_at)
            family = episode.runtime
            candidates = windows.get(family, []) if closed else []
            hits = [
                entry
                for entry in candidates
                if entry[0] - grace <= closed <= entry[1] + grace
            ]
            if hits:
                hits.sort(key=lambda entry: abs((closed - entry[1]).total_seconds()))
                if len(hits) == 1:
                    trace, tier = hits[0][2], "temporal"
                else:
                    resolved = _disambiguate(episode, [entry[2] for entry in hits])
                    if resolved is not None:
                        trace, tier = resolved, "temporal-context"
                    else:
                        trace, tier = hits[0][2], "temporal-ambiguous"
        if trace is None:
            counts["unjoined"] += 1
            continue
        episode.trace_id = trace["trace_id"]
        episode.trace_runtime = trace["runtime"]
        episode.join_tier = tier
        episode.events = trace["events"]
        counts[tier] += 1
    return counts


MINING_TIERS = ("exact", "uuid", "temporal", "temporal-context")

# The `brain.closeout` call is logged by the runtime a moment *after* the row it
# creates is timestamped, so a naive boundary hands every segment the previous
# episode's closing call as its first step. Shifting both bounds by the same
# slack keeps the segments disjoint and puts each closeout call at the end of the
# episode it closes, which is where it belongs.
_CLOSEOUT_SLACK = timedelta(seconds=120)

# A segment of one or two calls cannot describe a procedure; it is usually just
# the closeout call itself. Mining it would only manufacture support.
MIN_EPISODE_EVENTS = 3


def mining_set(episodes: list[Episode]) -> tuple[list[Episode], dict[str, int]]:
    """The subset of joined episodes that can carry an independence claim.

    Two filters, both necessary and both costly in volume:

    *Ambiguous temporal joins are dropped.* If three codex sessions were open
    when a closeout landed, that closeout is not evidence about any one of them.

    *Shared traces are segmented, not duplicated.* A long session files many
    closeouts — the codex heartbeat loop is the extreme case, 60 closeouts
    against one rollout. Attaching the whole rollout to each of them would make
    one session look like 60 independent confirmations, and every support count
    downstream would be inflated by its length. Since both events and closeouts
    carry timestamps, each closeout instead keeps only the events between the
    previous closeout on that trace and its own ``closed_at``: disjoint slices,
    and the natural reading of what a closeout is a receipt *for*. A closeout
    whose slice is empty is dropped rather than credited with someone else's work.

    Returns ``(episodes, counts)`` where ``counts`` explains the attrition.
    """
    eligible = [
        episode
        for episode in episodes
        if episode.join_tier in MINING_TIERS and episode.events
    ]
    by_trace: dict[str, list[Episode]] = {}
    for episode in eligible:
        by_trace.setdefault(str(episode.trace_id), []).append(episode)

    kept: list[Episode] = []
    segmented = 0
    emptied = 0
    for group in by_trace.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        segmented += len(group)
        group.sort(key=lambda episode: episode.closed_at)
        lower: datetime | None = None
        for episode in group:
            upper = _parse_time(episode.closed_at)
            bound = upper + _CLOSEOUT_SLACK if upper else None
            slice_events = [
                event
                for event in episode.events
                if _within(_parse_time(event.get("at")), lower, bound)
            ]
            lower = bound or lower
            if not slice_events:
                emptied += 1
                continue
            episode.events = slice_events
            episode.segmented = True
            kept.append(episode)

    too_short = sum(1 for episode in kept if len(episode.events) < MIN_EPISODE_EVENTS)
    kept = [episode for episode in kept if len(episode.events) >= MIN_EPISODE_EVENTS]
    kept.sort(key=lambda episode: episode.closed_at)
    return kept, {
        "joined": sum(1 for e in episodes if e.join_tier != "unjoined"),
        "dropped_ambiguous": sum(
            1 for e in episodes if e.join_tier.startswith("temporal-ambiguous")
        ),
        "eligible": len(eligible),
        "shared_a_trace": segmented,
        "dropped_empty_segment": emptied,
        "dropped_too_short": too_short,
        "mining_episodes": len(kept),
    }


def _within(when: datetime | None, lower: datetime | None, upper: datetime | None) -> bool:
    if when is None:
        return False
    if lower is not None and when <= lower:
        return False
    return not (upper is not None and when > upper)
