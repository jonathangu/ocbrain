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

import pytest

from ocbrain.closeout import (
    HERMES_PROFILES_ROOT_ENV,
    HERMES_SESSION_SHAPE,
    RUNTIME_SESSION_SHAPES,
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
# and a native child it spawned, both registry rows of one profile. An id is
# metadata, not transcript content.
PARENT_SESSION = "20260915_175454_9776060d"
CHILD_SESSION = "20260915_180902_4f19c2ae"
PROFILE = "squirtle"


def _registry(tmp_path, session_ids, profile=PROFILE):
    root = tmp_path / "profiles"
    db = root / profile / "state.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, message_count INTEGER)")
    conn.executemany(
        "INSERT INTO sessions VALUES (?, 3)", [(session_id,) for session_id in session_ids]
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
    with an 8-hex tail, child with the same -- and the old two-shape gate
    refused them at closeout, so a Hermes gateway could never join its own
    receipts to its own sessions.
    """
    for value in (PARENT_SESSION, CHILD_SESSION, "20260801_075651_56e87d"):
        assert classify_session_id(value) == HERMES_SESSION_SHAPE, value
        assert is_runtime_session_id(value)
        assert HERMES_SESSION_SHAPE in RUNTIME_SESSION_SHAPES


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
    assert resolved["session_id_shape"] == HERMES_SESSION_SHAPE
    assert resolved["session_id_verified"] == f"hermes_registry:{PROFILE}"


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
    assert receipt["provenance"]["session_identity"]["session_id_verified"] == (
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


def test_a_hermes_shaped_hint_never_wins_without_verification(tmp_path, monkeypatch):
    """A per-process environment value is at best a parent stamp: one Hermes MCP
    child serves every session of a profile, so an env id copied to every child
    would give them all one transcript. Verified, a hint may win; unverified, it
    loses to the connection."""
    root = _registry(tmp_path, [PARENT_SESSION])
    monkeypatch.setenv(HERMES_PROFILES_ROOT_ENV, str(root))
    verified = resolve_session_identity(
        None,
        Provenance(
            server_connection_id="cafe" * 8,
            client_session_hint=PARENT_SESSION,
            client_runtime_key=f"hermes:{PROFILE}",
        ),
    )
    assert verified["session_id"] == PARENT_SESSION
    assert verified["session_id_source"] == "harness_attested"
    unverified = resolve_session_identity(
        None,
        Provenance(
            server_connection_id="cafe" * 8,
            client_session_hint=CHILD_SESSION,
            client_runtime_key=f"hermes:{PROFILE}",
        ),
    )
    assert unverified["session_id"] == f"conn:{'cafe' * 8}"
    assert unverified["session_id_source"] == "server_connection"


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
