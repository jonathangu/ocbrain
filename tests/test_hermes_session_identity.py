"""Hermes native session ids: verified admission to the closeout identity column.

A Hermes gateway mints ``YYYYMMDD_HHMMSS_<6 or 8 hex>`` in a per-profile SQLite
registry, one MCP child multiplexes every session of that profile, and the old
two-shape gate classified those ids ``date_like`` and refused them. The fix has
two halves that the tests below keep separate: the *shape* admits a
``runtime_hermes`` id, and the *storage* of one requires the originating
profile's own registry -- read only, rooted at a validated profile name -- to
hold it. A registry fixture is built here for every case; no live Hermes
database or transcript payload is read.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ocbrain.closeout import (
    HERMES_PROFILES_ROOT_ENV,
    classify_session_id,
    hermes_profile_from_runtime_key,
    hermes_registry_lookup,
    is_runtime_session_id,
    record_closeout,
    resolve_session_identity,
)
from ocbrain.core_v1 import init_core_v1
from ocbrain.db import connect
from ocbrain.provenance import Provenance
from ocbrain.scope import ScopeContext

# The real-format parent and child that motivated the fix: a Telegram session
# and a native child it spawned, both registry rows of one profile, plus a cron
# run id exactly as the Hermes scheduler mints it
# (cron_<12-hex job id>_<YYYYMMDD>_<HHMMSS>). Ids are metadata, not transcript
# content; the registry fixture below mirrors the live schema's ``sessions.id``
# key (PRAGMA-confirmed 2026-09-17) rather than a fabricated column.
PARENT_SESSION = "20260915_175454_9776060d"
CHILD_SESSION = "20260915_180902_4f19c2ae"
CRON_SESSION = "cron_0b59d3345a70_20260825_143534"
# The api-server platform derives a stable conversation id per Open WebUI-style
# chat: ``api-`` plus the first 16 lowercase hex chars of a sha256 digest
# (gateway/platforms/api_server.py ``_derive_chat_session_id``), keyed on the
# same ``sessions.id`` registry. This is the real planner row that motivated
# the admission (see the live test below).
API_SESSION = "api-77806f5ba8ae308d"
# The api-server platform also mints ``api_<10-digit epoch>_<8 lowercase hex>``
# for a native peer POST /api/sessions row (and its fork) —
# f"api_{int(time.time())}_{uuid4().hex[:8]}", gateway/platforms/api_server.py
# 2817/2977. This is the real thinkercoframe planner row that motivated the
# admission.
API_PEER_SESSION = "api_1789673743_bed6bf3c"
PROFILE = "squirtle"

# The live registry this machine actually serves, used only for read-only
# existence checks; the fixture above carries the behavioural cases.
LIVE_REGISTRY_DB = Path.home() / ".hermes" / "profiles" / "squirtlecoframe" / "state.db"


def _registry(tmp_path, session_ids, profile=PROFILE):
    root = tmp_path / "profiles"
    db = root / profile / "state.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    # Column names and the ``id`` key mirror the live schema so the lookup is
    # exercised against the shape it will actually read.
    conn.execute(
        "CREATE TABLE sessions ("
        "id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL NOT NULL, "
        "ended_at REAL, message_count INTEGER DEFAULT 0, parent_session_id TEXT, "
        "archived INTEGER DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, 'telegram', 1.0)",
        [(session_id,) for session_id in session_ids],
    )
    conn.commit()
    conn.close()
    return root


def _hermes_provenance(**kwargs):
    return Provenance(
        server_connection_id="cafe" * 8, client_runtime_key=f"hermes:{PROFILE}", **kwargs
    )


def _core(tmp_path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def test_the_real_format_hermes_ids_are_runtime_shaped():
    """The regression this pass exists for: the old gate called these date_like.

    Every id below is byte-for-byte the shape a Hermes gateway mints -- parent
    with an 8-hex tail, child with the same, and a scheduled run's
    ``cron_<12-hex>_<date>_<time>`` -- and the old two-shape gate refused them
    at closeout, so a Hermes gateway could never join its own receipts to its
    own sessions or its cron runs.
    """
    for value in (PARENT_SESSION, CHILD_SESSION, "20260801_075651_56e87d", CRON_SESSION):
        assert is_runtime_session_id(value), value
    assert classify_session_id(PARENT_SESSION) == "runtime_hermes"
    assert classify_session_id(CRON_SESSION) == "runtime_hermes_cron"


@pytest.mark.parametrize(
    "value",
    [
        "2026-07-22",
        "2026-07-21 release checklist",
        "20260801_075651",
        "20260801_075651_56e87d0b-and-then-prose",
        "20261301_075651_56e87d0b",
        "20260801_256551_56e87d0b",
        "20260801_075651_56e87d0",
        "cron_g59d3345a70z_20260825_143534",
        "cron_0b59d3345a7_20260825_143534",
        "cron_0b59d3345a70_20261325_143534",
    ],
)
def test_near_misses_of_the_hermes_shape_stay_refused_shapes(value):
    """Bounded admission: bare dates, impossible calendar fields and trailing
    prose are not runtime ids, and a 7-hex tail is not a Hermes tail either."""
    assert not is_runtime_session_id(value)
    assert classify_session_id(value) in {
        "date_like",
        "contains_space",
        "slug",
    }


def test_a_verified_hermes_claim_is_stored_with_its_registry_label(tmp_path, monkeypatch):
    root = _registry(tmp_path, [PARENT_SESSION, CHILD_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    resolved = resolve_session_identity(PARENT_SESSION, _hermes_provenance())
    assert resolved["session_id"] == PARENT_SESSION
    assert resolved["session_id_source"] == "agent_reported"
    assert resolved["session_id_shape"] == "runtime_hermes"
    assert resolved["session_id_registry"] == f"hermes_registry:{PROFILE}"


def test_a_verified_hermes_child_lands_in_the_closeout_identity_column(tmp_path, monkeypatch):
    """End to end through the write path the old gate refused."""
    root = _registry(tmp_path, [PARENT_SESSION, CHILD_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="HERMES-1",
        status="completed",
        summary="Child session filed its own receipt under its own id.",
        context=ScopeContext(project="coframe", runtime="hermes", session=CHILD_SESSION),
        provenance=_hermes_provenance(),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == CHILD_SESSION
    assert row["session_id_source"] == "agent_reported"
    assert receipt["provenance"]["session_identity"]["session_id_registry"] == (
        f"hermes_registry:{PROFILE}"
    )


def test_two_children_in_one_connection_keep_separate_identities(tmp_path, monkeypatch):
    """One MCP child multiplexes every session; per-call claims, verified per
    call, must not collapse onto each other or onto the connection."""
    root = _registry(tmp_path, [PARENT_SESSION, CHILD_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    conn = _core(tmp_path)
    parent = record_closeout(
        conn,
        task_ref="HERMES-PARENT",
        status="completed",
        summary="Parent receipt.",
        context=ScopeContext(project="coframe", runtime="hermes", session=PARENT_SESSION),
        provenance=_hermes_provenance(),
    )
    child = record_closeout(
        conn,
        task_ref="HERMES-CHILD",
        status="completed",
        summary="Child receipt.",
        context=ScopeContext(project="coframe", runtime="hermes", session=CHILD_SESSION),
        provenance=_hermes_provenance(),
    )
    conn.commit()
    stored = {
        row[0]
        for row in conn.execute(
            "SELECT session_id FROM task_closeouts WHERE id IN (?, ?)",
            (parent["id"], child["id"]),
        )
    }
    assert stored == {PARENT_SESSION, CHILD_SESSION}
    assert parent["provenance"]["session_identity"]["session_id"] != (
        child["provenance"]["session_identity"]["session_id"]
    )


def test_a_cron_id_the_registry_holds_is_stored(tmp_path, monkeypatch):
    """A scheduled run's ``cron_<12-hex>_<date>_<time>`` id gates exactly like a
    native one: shape plus a registry row in the bound profile."""
    root = _registry(tmp_path, [CRON_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="HERMES-CRON",
        status="completed",
        summary="Scheduled run filed under its own cron id.",
        context=ScopeContext(project="coframe", runtime="hermes", session=CRON_SESSION),
        provenance=_hermes_provenance(),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == CRON_SESSION
    assert row["session_id_source"] == "agent_reported"
    assert receipt["provenance"]["session_identity"]["session_id_registry"] == (
        f"hermes_registry:{PROFILE}"
    )


def test_a_cron_id_the_registry_does_not_hold_is_refused(tmp_path, monkeypatch):
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(CRON_SESSION, _hermes_provenance())
    assert "does not hold this id" in str(excinfo.value)


def test_the_real_format_api_conversation_id_is_runtime_shaped():
    """Only the mint's exact form -- ``api-`` plus 16 lowercase hex -- admits."""
    assert classify_session_id(API_SESSION) == "runtime_hermes_api"
    assert is_runtime_session_id(API_SESSION)
    for value in (
        "api-77806f5ba8ae308",
        "api-77806f5ba8ae308d0",
        "api-77806F5BA8AE308D",
        "api_77806f5ba8ae308d",
        "77806f5ba8ae308d",
    ):
        assert not is_runtime_session_id(value), value
        assert classify_session_id(value) == "slug", value


def test_a_verified_api_conversation_id_is_stored_end_to_end(tmp_path, monkeypatch):
    """An api-server conversation id gates exactly like native and cron: shape
    plus a registry row in the profile bound to the connection."""
    root = _registry(tmp_path, [PARENT_SESSION, API_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="HERMES-API",
        status="completed",
        summary="Planner API conversation filed under its own api- id.",
        context=ScopeContext(project="coframe", runtime="hermes", session=API_SESSION),
        provenance=_hermes_provenance(),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == API_SESSION
    assert row["session_id_source"] == "agent_reported"
    assert receipt["provenance"]["session_identity"]["session_id_registry"] == (
        f"hermes_registry:{PROFILE}"
    )


def test_an_api_conversation_id_absent_from_the_registry_is_refused(tmp_path, monkeypatch):
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(API_SESSION, _hermes_provenance())
    assert "does not hold this id" in str(excinfo.value)


def test_an_api_conversation_id_of_another_profile_is_refused(tmp_path, monkeypatch):
    """The id is real, but of a profile this connection is not bound to: the
    bound profile's registry reads fine and simply does not hold it."""
    _registry(tmp_path, [PARENT_SESSION])
    _registry(tmp_path, [API_SESSION], profile="bulbasaur")
    monkeypatch.setenv(
        HERMES_PROFILES_ROOT_ENV, str(tmp_path / "profiles")
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(API_SESSION, _hermes_provenance())
    assert "does not hold this id" in str(excinfo.value)


def test_the_real_format_api_peer_session_id_is_runtime_shaped():
    """Only the mint's exact form -- ``api_`` plus a 10-digit epoch and 8
    lowercase hex -- admits; lookalikes stay slugs, and the hyphenated
    conversation shape stays its own shape."""
    assert classify_session_id(API_PEER_SESSION) == "runtime_hermes_api_peer"
    assert is_runtime_session_id(API_PEER_SESSION)
    for value in (
        "api_178967374_bed6bf3c",
        "api_1789673743_bed6bf3c0",
        "api_1789673743_BED6BF3C",
        "api-1789673743_bed6bf3c",
        "api_1789673743-bed6bf3c",
    ):
        assert not is_runtime_session_id(value), value
        assert classify_session_id(value) == "slug", value
    # A bare epoch+hex pair without the api_ prefix stays date_like, exactly as
    # it was before this shape existed.
    assert classify_session_id("1789673743_bed6bf3c") == "date_like"
    assert not is_runtime_session_id("1789673743_bed6bf3c")


def test_a_verified_api_peer_session_id_is_stored_end_to_end(tmp_path, monkeypatch):
    """A peer POST session id gates exactly like the derived conversation shape:
    registry row in the profile bound to the connection, never shape alone."""
    root = _registry(tmp_path, [PARENT_SESSION, API_PEER_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="HERMES-API-PEER",
        status="completed",
        summary="Planner peer POST session filed under its own api_ id.",
        context=ScopeContext(project="coframe", runtime="hermes", session=API_PEER_SESSION),
        provenance=_hermes_provenance(),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["session_id"] == API_PEER_SESSION
    assert row["session_id_source"] == "agent_reported"
    assert receipt["provenance"]["session_identity"]["session_id_registry"] == (
        f"hermes_registry:{PROFILE}"
    )


def test_an_api_peer_session_id_absent_from_the_registry_is_refused(tmp_path, monkeypatch):
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(API_PEER_SESSION, _hermes_provenance())
    assert "does not hold this id" in str(excinfo.value)


def test_an_api_peer_session_id_of_another_profile_is_refused(tmp_path, monkeypatch):
    """The id is a real row, but of a profile this connection is not bound to."""
    _registry(tmp_path, [PARENT_SESSION])
    _registry(tmp_path, [API_PEER_SESSION], profile="bulbasaur")
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(tmp_path / "profiles"))
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(API_PEER_SESSION, _hermes_provenance())
    assert "does not hold this id" in str(excinfo.value)


def test_an_api_shaped_hint_still_never_overrides_a_child_claim(tmp_path, monkeypatch):
    """The api shape rides the same registry path as native and cron, so a
    process-level env hint naming the parent conversation is dropped the same
    way and the per-call child claim wins untouched."""
    root = _registry(tmp_path, [PARENT_SESSION, API_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    resolved = resolve_session_identity(
        PARENT_SESSION,
        Provenance(
            server_connection_id="cafe" * 8,
            client_session_hint=API_SESSION,
            client_runtime_key=f"hermes:{PROFILE}",
        ),
    )
    assert resolved["session_id"] == PARENT_SESSION
    assert resolved["session_id_registry"] == f"hermes_registry:{PROFILE}"
    assert "session_id_claim" not in resolved


def test_a_uuid_claim_is_unchanged_by_the_api_shape_gate(tmp_path, monkeypatch):
    """A UUID claim on a Hermes-bound connection never consults the registry."""
    uuid_claim = "5e2dc531-45b2-483b-bdfd-14b6d9767b4b"
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(tmp_path / "profiles-empty"))
    resolved = resolve_session_identity(uuid_claim, _hermes_provenance())
    assert resolved["session_id"] == uuid_claim
    assert resolved["session_id_source"] == "agent_reported"
    assert resolved["session_id_shape"] == "runtime_uuid"
    assert "session_id_registry" not in resolved


def test_a_claim_the_registry_does_not_hold_is_refused_with_the_reason(tmp_path, monkeypatch):
    """Wrong profile, stale id, typo: registry readable, id absent."""
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(CHILD_SESSION, _hermes_provenance())
    message = str(excinfo.value)
    assert "does not hold this id" in message
    assert "OCBRAIN_SESSION_ID" in message
    assert "omit context.session" in message
    assert repr(CHILD_SESSION) in message


def test_a_connection_without_a_bound_profile_cannot_verify_anything(tmp_path, monkeypatch):
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    for observed in (
        Provenance(server_connection_id="cafe" * 8),
        Provenance(server_connection_id="cafe" * 8, client_runtime_key="hermes"),
        Provenance(server_connection_id="cafe" * 8, client_runtime_key="claude-code"),
    ):
        with pytest.raises(ValueError) as excinfo:
            resolve_session_identity(PARENT_SESSION, observed)
        assert "no Hermes profile is bound" in str(excinfo.value)


def test_an_unreadable_registry_is_could_not_verify_not_no_such_session(tmp_path, monkeypatch):
    monkeypatch.setenv(
        HERMES_PROFILES_ROOT_ENV, str(tmp_path / "profiles-that-do-not-exist")
    )
    with pytest.raises(ValueError) as excinfo:
        resolve_session_identity(PARENT_SESSION, _hermes_provenance())
    assert "could not be read" in str(excinfo.value)


def test_quarantine_keeps_an_unverified_hermes_claim_out_of_the_column(tmp_path, monkeypatch):
    monkeypatch.setenv(
        HERMES_PROFILES_ROOT_ENV, str(tmp_path / "profiles-that-do-not-exist")
    )
    resolved = resolve_session_identity(
        PARENT_SESSION, _hermes_provenance(), policy="quarantine"
    )
    assert resolved["session_id"] == f"conn:{'cafe' * 8}"
    assert resolved["session_id_source"] == "server_connection"
    assert resolved["session_id_claim"] == PARENT_SESSION


def test_a_hermes_shaped_hint_never_wins_even_when_the_registry_holds_it(
    tmp_path, monkeypatch
):
    """One Hermes MCP child serves every session of a profile, so a process-level
    env hint names at best the parent that spawned the child. Even a hint the
    registry holds is dropped: with no per-call claim there is no parent stamp,
    and the connection id serves instead."""
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    resolved = resolve_session_identity(
        None,
        Provenance(
            server_connection_id="cafe" * 8,
            client_session_hint=PARENT_SESSION,
            client_runtime_key=f"hermes:{PROFILE}",
        ),
    )
    assert resolved["session_id"] == f"conn:{'cafe' * 8}"
    assert resolved["session_id_source"] == "server_connection"


def test_a_verified_child_claim_beats_a_registry_held_parent_hint(tmp_path, monkeypatch):
    """The multiplexed-child case the review flagged: the env carries the parent's
    id while the call claims the child's. Both are registry rows of the profile,
    but only the per-call claim is this call's session, so the child wins and the
    hint leaves no trace."""
    root = _registry(tmp_path, [PARENT_SESSION, CHILD_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    resolved = resolve_session_identity(
        CHILD_SESSION,
        Provenance(
            server_connection_id="cafe" * 8,
            client_session_hint=PARENT_SESSION,
            client_runtime_key=f"hermes:{PROFILE}",
        ),
    )
    assert resolved["session_id"] == CHILD_SESSION
    assert resolved["session_id_source"] == "agent_reported"
    assert resolved["session_id_registry"] == f"hermes_registry:{PROFILE}"
    assert "session_id_claim" not in resolved


def test_a_uuid_hint_still_wins_for_non_hermes_runtimes():
    """Only the Hermes shapes are dropped; a harness-attested UUID hint keeps its
    pre-gate meaning when no claim is made."""
    uuid_hint = "5e2dc531-45b2-483b-bdfd-14b6d9767b4b"
    resolved = resolve_session_identity(
        None, Provenance(server_connection_id="cafe" * 8, client_session_hint=uuid_hint)
    )
    assert resolved["session_id"] == uuid_hint
    assert resolved["session_id_source"] == "harness_attested"


@pytest.mark.parametrize(
    "runtime_key",
    [
        "hermes:../escape",
        "hermes:.hidden",
        "hermes:a/b",
        "hermes:..\\escape",
        "claude-code",
        "hermes",
    ],
)
def test_profile_selection_cannot_traverse_or_bind_a_foreign_key(runtime_key):
    """The profile is one validated path segment cut from the server-observed
    runtime key; anything that could leave the profiles root binds to nothing."""
    assert hermes_profile_from_runtime_key(runtime_key) is None


def test_the_registry_lookup_reads_nothing_outside_its_root(tmp_path):
    """Even a hand-called lookup cannot be walked out of the profiles root."""
    assert hermes_registry_lookup(PARENT_SESSION, "../escape", profiles_root=tmp_path) is None
    assert hermes_registry_lookup(PARENT_SESSION, ".hidden", profiles_root=tmp_path) is None


def test_the_registry_lookup_never_writes(tmp_path):
    """mode=ro: a writable fixture database stays byte-identical after a check."""
    root = _registry(tmp_path, [PARENT_SESSION])
    db = root / PROFILE / "state.db"
    before = db.read_bytes()
    assert hermes_registry_lookup(PARENT_SESSION, PROFILE, profiles_root=root) is True
    assert hermes_registry_lookup(CHILD_SESSION, PROFILE, profiles_root=root) is False
    assert db.read_bytes() == before


def test_a_foreign_registry_schema_could_not_be_consulted(tmp_path):
    """A database without the sessions table is not evidence of absence."""
    root = tmp_path / "profiles"
    db = root / PROFILE / "state.db"
    db.parent.mkdir(parents=True)
    sqlite3.connect(db).close()
    assert hermes_registry_lookup(PARENT_SESSION, PROFILE, profiles_root=root) is None


@pytest.mark.skipif(
    not LIVE_REGISTRY_DB.is_file(), reason="live Hermes registry not present on this machine"
)
def test_the_live_registry_holds_the_real_format_ids_read_only():
    """Independent of any fixture: the registry this machine actually serves
    holds the real-format parent id and a real cron row, and does not hold a
    shape-valid id it never issued. Existence checks only, read-only."""
    assert hermes_registry_lookup(PARENT_SESSION, "squirtlecoframe") is True
    assert hermes_registry_lookup(CRON_SESSION, "squirtlecoframe") is True
    assert hermes_registry_lookup("20990101_000000_000000aa", "squirtlecoframe") is False


LIVE_API_REGISTRY_DB = (
    Path.home() / ".hermes" / "profiles" / "thinkercoframe" / "state.db"
)


@pytest.mark.skipif(
    not LIVE_API_REGISTRY_DB.is_file(),
    reason="live thinkercoframe registry not present on this machine",
)
def test_the_live_thinkercoframe_registry_holds_the_real_api_conversation_id():
    """The motivating row: a planner API conversation the old gate called a slug."""
    assert hermes_registry_lookup(API_SESSION, "thinkercoframe") is True
    assert hermes_registry_lookup("api-" + "0" * 16, "thinkercoframe") is False
    assert hermes_registry_lookup(API_PEER_SESSION, "thinkercoframe") is True
    assert hermes_registry_lookup("api_" + "0" * 10 + "_" + "0" * 8, "thinkercoframe") is False
