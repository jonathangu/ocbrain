"""Server-captured caller identity, and the boundary it must not cross.

The whole value of this feature is that the three identities stay apart. A test
suite that only checked "a session id got stored" would pass just as happily
with the model's free text copied into the authoritative column, which is the
failure it exists to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

from ocbrain import closeout as closeout_module
from ocbrain.closeout import record_closeout
from ocbrain.core_v1 import init_core_v1, record_core_v1_retrieval
from ocbrain.db import connect
from ocbrain.mcp import handle_request
from ocbrain.provenance import Provenance, connection_provenance
from ocbrain.scope import ScopeContext

CLAUDE_ENV = {
    "CLAUDE_CODE_SESSION_ID": "3ebe3a24-6162-4af2-a4ee-4e8c1de121f7",
    "AI_AGENT": "claude-code_2-1-237_harness",
}


def _core(tmp_path: Path):
    conn = connect(tmp_path / "core.sqlite")
    init_core_v1(conn)
    return conn


def test_capture_mints_a_connection_id_and_reads_the_harness_environment() -> None:
    prov = Provenance.capture(client_name="claude-code", env=dict(CLAUDE_ENV))
    assert len(prov.server_connection_id) == 32
    assert prov.client_session_hint == CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"]
    assert prov.client_session_hint_source == "env:CLAUDE_CODE_SESSION_ID"
    assert prov.client_runtime_key == "claude-code_2-1-237_harness"
    assert prov.client_runtime_key_source == "env:AI_AGENT"
    payload = prov.to_dict()
    # The trust level is data, not a docstring: a consumer must be able to see
    # that this value was attested by the harness rather than verified here.
    assert payload["client_session_hint_trust"] == "harness_attested"
    assert payload["server_connection_id_trust"] == "server_minted"


def test_an_operator_set_client_key_outranks_the_harness_and_the_handshake() -> None:
    prov = Provenance.capture(
        client_name="cursor",
        env={"OCBRAIN_CLIENT": "hermes:charmander", "AI_AGENT": "claude-code_2-1-237_harness"},
    )
    assert prov.client_runtime_key == "hermes:charmander"
    assert prov.client_runtime_key_source == "env:OCBRAIN_CLIENT"
    # The handshake name is still recorded, just not promoted over the operator's.
    assert prov.client_name == "cursor"


def test_the_handshake_name_is_the_last_resort_runtime_key() -> None:
    prov = Provenance.capture(client_name="claude-code", env={})
    assert prov.client_runtime_key == "claude-code"
    assert prov.client_runtime_key_source == "client_info.name"
    assert prov.client_session_hint is None


def test_a_bare_environment_degrades_to_the_connection_id_alone() -> None:
    """Codex is unverified and Cursor gains nothing; neither may break."""
    prov = Provenance.capture(client_name=None, env={})
    assert prov.server_connection_id
    assert prov.client_session_hint is None
    assert prov.client_runtime_key is None
    assert prov.to_dict().keys() == {"server_connection_id", "server_connection_id_trust"}


def test_one_connection_keeps_one_id_and_the_next_connection_gets_another() -> None:
    first: dict[str, object] = {"client_name": "claude-code"}
    second: dict[str, object] = {"client_name": "claude-code"}
    a1 = connection_provenance(first)
    a2 = connection_provenance(first)
    b1 = connection_provenance(second)
    assert a1.server_connection_id == a2.server_connection_id
    assert a1.server_connection_id != b1.server_connection_id


def test_a_caller_that_is_not_a_connection_gets_no_id_rather_than_a_fresh_one() -> None:
    """A "connection id" that changes between two calls would be worse than none."""
    assert connection_provenance(None).server_connection_id is None


def test_the_session_hint_cannot_come_from_the_model(tmp_path: Path) -> None:
    """The point of the hint is that no model can type it.

    ``context.session`` is model-supplied and must never reach
    ``client_session_hint``; if it did, the honest/attested split would be
    decorative. That is what this test is for, and it is unchanged.

    What did change: a model-typed slug no longer reaches the identity column
    either. It is quarantined -- kept in ``provenance_json`` as the claim it is,
    while the column carries the server's own connection id -- because 967 of
    the 1,115 session ids on the live core were hand-written and none of them
    joined a transcript.
    """
    conn = _core(tmp_path)
    retrieval_id = record_core_v1_retrieval(
        conn,
        query="probe",
        context={"project": "bountiful"},
        items=[],
        runtime="codex",
        task_ref=None,
        session_id="model-typed-this",
        provenance=Provenance.capture(client_name="codex", env={}),
    )
    conn.commit()
    row = conn.execute(
        "SELECT session_id, session_id_source, client_session_hint, provenance_json "
        "FROM retrieval_uses WHERE id=?",
        (retrieval_id,),
    ).fetchone()
    assert row["session_id"].startswith("conn:")
    assert row["session_id_source"] == "server_connection"
    assert row["client_session_hint"] is None
    identity = json.loads(row["provenance_json"])["session_identity"]
    assert identity["session_id_claim"] == "model-typed-this"


def test_closeout_records_the_observed_identity_beside_the_claimed_one(tmp_path: Path) -> None:
    conn = _core(tmp_path)
    receipt = record_closeout(
        conn,
        task_ref="t",
        status="completed",
        summary="Threaded server-observed provenance into both write paths.",
        context=ScopeContext(project="bountiful", runtime="Claude Code, sort of"),
        provenance=Provenance.capture(client_name="claude-code", env=dict(CLAUDE_ENV)),
    )
    conn.commit()
    row = conn.execute(
        "SELECT runtime, runtime_family, session_id, session_id_source, "
        "server_connection_id, client_session_hint, "
        "client_runtime_key, provenance_json FROM task_closeouts WHERE id=?",
        (receipt["id"],),
    ).fetchone()
    assert row["runtime"] == "Claude Code, sort of"
    assert row["runtime_family"] == "claude-code"
    # No session was claimed, so the harness-attested hint fills the column and
    # says so. Before, this row would have carried whatever the model typed.
    assert row["session_id"] == CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"]
    assert row["session_id_source"] == "harness_attested"
    assert row["client_session_hint"] == CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"]
    assert row["client_runtime_key"] == CLAUDE_ENV["AI_AGENT"]
    assert len(row["server_connection_id"]) == 32
    observed = json.loads(row["provenance_json"])["server_observed"]
    assert observed["client_session_hint_trust"] == "harness_attested"
    # The receipt the caller got back says the same thing.
    assert receipt["provenance"]["server_observed"]["client_session_hint"] == (
        CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"]
    )


def test_two_connections_writing_the_same_closeout_are_two_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The UNIQUE content_hash used to collapse them into one insert failure.

    ``closed_at`` is frozen so the connection is the *only* thing that differs;
    otherwise the sub-second clock would separate the digests on its own and
    the test would pass even with provenance dropped from the receipt.
    """
    conn = _core(tmp_path)
    monkeypatch.setattr(closeout_module, "now_iso", lambda: "2026-08-25T00:00:00+00:00")
    digests = set()
    for index in range(2):
        receipt = record_closeout(
            conn,
            task_ref="t",
            status="completed",
            summary="Two agents finished the identical task in parallel lanes.",
            context=ScopeContext(project="bountiful"),
            provenance=Provenance.capture(
                client_name="claude-code",
                env={},
                connection_id=f"connection-{index}",
            ),
        )
        digests.add(receipt["content_hash"])
    conn.commit()
    assert len(digests) == 2


def test_tools_call_carries_the_connection_identity_all_the_way_to_the_row(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The dispatch thread is the actual work; pin it end to end.

    ``tools/call`` did not receive ``session_state`` at all before this change,
    so nothing the server observed at ``initialize`` could reach a write.
    """
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"])
    monkeypatch.setenv("AI_AGENT", CLAUDE_ENV["AI_AGENT"])
    monkeypatch.delenv("OCBRAIN_CLIENT", raising=False)
    monkeypatch.delenv("OCBRAIN_SESSION_ID", raising=False)
    conn = _core(tmp_path)
    session_state: dict[str, object] = {}

    handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"clientInfo": {"name": "claude-code"}},
        },
        session_state=session_state,
    )
    handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "brain.context",
                "arguments": {"query": "anything", "context": {"project": "bountiful"}},
            },
        },
        allow_writes=True,
        session_state=session_state,
    )
    row = conn.execute(
        "SELECT server_connection_id, client_session_hint, client_runtime_key "
        "FROM retrieval_uses ORDER BY served_at DESC LIMIT 1"
    ).fetchone()
    assert row["client_session_hint"] == CLAUDE_ENV["CLAUDE_CODE_SESSION_ID"]
    assert row["client_runtime_key"] == CLAUDE_ENV["AI_AGENT"]
    assert row["server_connection_id"] == session_state["provenance"].server_connection_id


def test_a_call_before_initialize_still_gets_a_connection_id(tmp_path: Path) -> None:
    """Not every client sends initialize first; the id must not depend on it."""
    conn = _core(tmp_path)
    session_state: dict[str, object] = {}
    handle_request(
        conn,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "brain.context", "arguments": {"query": "anything"}},
        },
        allow_writes=True,
        session_state=session_state,
    )
    row = conn.execute(
        "SELECT server_connection_id FROM retrieval_uses ORDER BY served_at DESC LIMIT 1"
    ).fetchone()
    assert row["server_connection_id"]
