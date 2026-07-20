from __future__ import annotations

import json
from pathlib import Path

from ocbrain.db import connect, init_db
from ocbrain.fsutil import history_runtime
from ocbrain_training.dataset.transcripts import (
    INJECTED_PREFIXES,
    classify_user_text,
    is_conversation_transcript,
    iter_unmined_transcripts,
    parse_claude_session,
    parse_codex_session,
    parse_openclaw_session,
    parse_transcript,
    record_source,
    strip_timestamp_prefix,
)

AUTHOR_IDS = ["1000000001", "persona_user"]


def _write_jsonl(path: Path, objs: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(o) for o in objs) + "\n", encoding="utf-8")
    return path


def _openclaw_lines() -> list[dict]:
    return [
        {
            "type": "session",
            "id": "sess1",
            "cwd": "/x",
            "version": "1",
            "timestamp": "2026-05-20T10:00:00Z",
        },
        {
            "type": "message",
            "message": {
                "role": "user",
                "content": "[Wed 2026-05-20 10:03 PDT] please help me with the deploy",
                "timestamp": "2026-05-20T10:03:00Z",
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "text": "SECRET_COT should never surface"},
                    {"type": "text", "text": "Here is the deploy plan you asked for."},
                    {"type": "toolCall", "name": "bash", "input": {}},
                ],
                "timestamp": "2026-05-20T10:03:10Z",
            },
        },
        {
            "type": "message",
            "message": {"role": "toolResult", "toolName": "bash", "content": "X" * 900},
        },
    ]


def test_parse_openclaw_session(tmp_path: Path):
    path = _write_jsonl(tmp_path / "sess1.jsonl", _openclaw_lines())
    session = parse_openclaw_session(path, author_ids=AUTHOR_IDS, tool_result_truncate=500)
    assert session.source_kind == "openclaw_session"
    assert session.session_id == "sess1"
    roles = [t.role for t in session.turns]
    assert roles == ["user", "assistant", "tool"]
    # thinking never enters text
    assert all("SECRET_COT" not in t.text for t in session.turns)
    # timestamp prefix stripped from the user turn
    assert session.turns[0].text.startswith("please help")
    # tool call counted, tool result truncated to 500 chars
    assert session.turns[1].n_tool_calls == 1
    assert len(session.turns[2].text) == 500


def test_structured_tool_error_flags_survive_without_error_text(tmp_path: Path):
    openclaw = _openclaw_lines()
    openclaw[-1]["message"]["content"] = "command stopped cleanly"
    openclaw[-1]["message"]["isError"] = True
    oc_path = _write_jsonl(tmp_path / "oc-error.jsonl", openclaw)
    assert parse_openclaw_session(oc_path).turns[-1].tool_error is True

    claude = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "command stopped", "is_error": True}
                ],
            },
        }
    ]
    claude_path = _write_jsonl(tmp_path / "claude-error.jsonl", claude)
    assert parse_claude_session(claude_path).turns[-1].tool_error is True


def test_openclaw_agent_from_path(tmp_path: Path):
    root = tmp_path / "agents" / "planner" / "sessions"
    root.mkdir(parents=True)
    path = _write_jsonl(root / "sess1.jsonl", _openclaw_lines())
    session = parse_openclaw_session(path, author_ids=AUTHOR_IDS)
    assert session.agent == "planner"


def test_parse_claude_session(tmp_path: Path):
    lines = [
        {"type": "queue-operation", "operation": "noop"},  # junk, skipped
        {
            "type": "user",
            "sessionId": "cs1",
            "timestamp": "2026-07-07T11:01:00Z",
            "message": {"role": "user", "content": "[Tue 2026-07-07 11:01 PDT] do the thing"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-07-07T11:01:05Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Done, I edited the file for you."},
                    {"type": "tool_use", "name": "Edit", "input": {}},
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-07-07T11:01:06Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "Error: boom exploded"}],
            },
        },
    ]
    path = _write_jsonl(tmp_path / "cs1.jsonl", lines)
    session = parse_claude_session(path, author_ids=AUTHOR_IDS)
    assert session.source_kind == "claude_session"
    assert session.session_id == "cs1"
    roles = [t.role for t in session.turns]
    assert roles == ["user", "assistant", "tool"]
    assert session.turns[1].n_tool_calls == 1
    assert session.turns[2].tool_error is True


def test_parse_codex_session(tmp_path: Path):
    lines = [
        {
            "type": "session_meta",
            "payload": {"id": "cx1", "cwd": "/y"},
            "timestamp": "2026-05-14T06:06:44Z",
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "developer system instructions"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "please compute the sum"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "content": None,
                "summary": [],
                "encrypted_content": "SECRET_COT",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "shell",
                "arguments": "{}",
                "call_id": "c1",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "output": "AssertionError: nope",
                "call_id": "c1",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "the result is 42, all set"}],
            },
        },
    ]
    path = _write_jsonl(tmp_path / "rollout-2026-05-14-cx1.jsonl", lines)
    session = parse_codex_session(path, author_ids=AUTHOR_IDS)
    assert session.source_kind == "codex_session"
    assert session.session_id == "cx1"
    # developer -> injected user turn; reasoning dropped; function_call -> assistant tool call
    assert session.turns[0].role == "user" and session.turns[0].kind == "injected"
    assert session.turns[1].role == "user" and session.turns[1].kind == "bare"
    assert session.turns[1].text == "please compute the sum"
    assert all("SECRET_COT" not in t.text for t in session.turns)
    tool_turns = [t for t in session.turns if t.role == "tool"]
    assert tool_turns and tool_turns[0].tool_error is True
    assistant = [t for t in session.turns if t.role == "assistant"]
    assert sum(t.n_tool_calls for t in assistant) == 1


def test_sidecar_and_junk_predicate(tmp_path: Path):
    assert is_conversation_transcript(tmp_path / "real.jsonl")
    assert not is_conversation_transcript(tmp_path / "x.trajectory.jsonl")
    assert not is_conversation_transcript(tmp_path / "y.trajectory-path.json")
    assert not is_conversation_transcript(tmp_path / "z.jsonl.codex-app-server.json")
    assert not is_conversation_transcript(tmp_path / "sessions.json")
    assert not is_conversation_transcript(tmp_path / "session_index.jsonl")
    assert not is_conversation_transcript(tmp_path / "history.jsonl")
    assert not is_conversation_transcript(tmp_path / "codex-home" / ".tmp" / "a.jsonl")
    assert not is_conversation_transcript(tmp_path / "plugin-config.json")


def test_parse_migrated_chatgpt_codex_rollout(tmp_path: Path):
    """Current ChatGPT desktop rollouts keep Codex response items plus new
    bookkeeping records. Tool pairs must survive; reasoning/world state must not.
    """
    lines = [
        {
            "type": "session_meta",
            "timestamp": "2026-07-09T20:00:00Z",
            "payload": {"id": "migrated-1", "cwd": "/workspace"},
        },
        {"type": "world_state", "timestamp": "2026-07-09T20:00:01Z", "payload": {}},
        {"type": "turn_context", "timestamp": "2026-07-09T20:00:02Z", "payload": {}},
        {
            "type": "event_msg",
            "timestamp": "2026-07-09T20:00:03Z",
            "payload": {"type": "user_message", "message": "duplicate bookkeeping"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:03Z",
            "payload": {
                "type": "agent_message",
                "author": "worker",
                "recipient": "root",
                "content": [
                    {"type": "input_text", "text": "bounded inter-agent context"},
                    {"type": "encrypted_content", "encrypted_content": "AGENT_SECRET"},
                ],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:03Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "inspect the repo"}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:04Z",
            "payload": {"type": "reasoning", "encrypted_content": "NEVER_TRAIN"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:05Z",
            "payload": {
                "type": "custom_tool_call",
                "name": "exec_command",
                "call_id": "c1",
                "input": "{}",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:06Z",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": "repository status is clean",
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-09T20:00:07Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Inspection complete."}],
            },
        },
    ]
    path = _write_jsonl(tmp_path / "rollout-migrated.jsonl", lines)
    session = parse_codex_session(path)
    assert session.session_id == "migrated-1"
    assert [turn.role for turn in session.turns] == [
        "user",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.turns[0].kind == "injected"
    assert session.turns[0].text == "bounded inter-agent context"
    assert session.turns[1].kind == "bare"
    assert session.turns[2].n_tool_calls == 1
    assert all("NEVER_TRAIN" not in turn.text for turn in session.turns)
    assert all("AGENT_SECRET" not in turn.text for turn in session.turns)
    assert all("duplicate bookkeeping" not in turn.text for turn in session.turns)


def test_embedded_openclaw_harness_runtime_attribution(tmp_path: Path):
    codex = tmp_path / ".openclaw" / "agents" / "work" / "agent" / "codex-home"
    codex_path = codex / "sessions" / "rollout-current.jsonl"
    assert history_runtime(codex_path) == "codex"

    claude_path = (
        tmp_path
        / ".openclaw"
        / "agents"
        / "work"
        / "agent"
        / "claude-home"
        / "projects"
        / "current.jsonl"
    )
    assert history_runtime(claude_path) == "claude"


def test_hermes_runtime_attribution(tmp_path: Path):
    export = (
        tmp_path
        / ".hermes"
        / "sessions"
        / "export"
        / "hermes-20260719_101552_e5038d20.jsonl"
    )
    assert history_runtime(export) == "hermes"


def test_cursor_runtime_attribution(tmp_path: Path):
    export = (
        tmp_path
        / ".ocbrain"
        / "exports"
        / "cursor"
        / "cursor-0d254e880906ec61816dd14b697edff1.jsonl"
    )
    assert history_runtime(export) == "cursor"

    dot_home = tmp_path / ".cursor" / "chats" / "current.jsonl"
    assert history_runtime(dot_home) == "cursor"

    dump = tmp_path / ".hermes" / "sessions" / "request_dump_x.json"
    assert history_runtime(dump) == "hermes"


def test_telegram_envelope_author_verification():
    envelope = {"message_id": "1", "sender_id": "1000000001"}
    text = (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(envelope)
        + "\n```\nship the release please"
    )
    cls = classify_user_text(text, author_ids=AUTHOR_IDS, agent="main", direct_agents=["main"])
    assert cls.kind == "telegram_envelope"
    assert cls.sender_verified is True
    assert cls.authored_by == "1000000001"
    assert cls.text == "ship the release please"


def test_production_runtime_wrappers_are_injected_case_insensitively():
    wrappers = (
        "System (untrusted): completed background command 884422",
        "<goal_context>automated continuation</goal_context>",
        "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>\nprivate machine context",
        "Pre-compaction memory flush. Store durable memories now.",
        "Read HEARTBEAT.md if it exists (workspace context).",
        "[cron:daily] perform the scheduled check",
        "[Cron:daily] perform the scheduled check",
        "Warning: apply_patch was requested through exec_command.",
        "You have 8192 weighted tokens left",
        "A new session was started via /new or /reset. Say hello.",
    )
    for wrapper in wrappers:
        cls = classify_user_text(wrapper, author_ids=AUTHOR_IDS)
        assert cls.kind == "injected", wrapper


def test_sender_envelope_author_verification_and_context_removal():
    envelope = {
        "label": "Persona Tester (1000000001)",
        "id": "1000000001",
        "name": "Persona Tester",
        "username": "persona_user",
    }
    text = (
        "System (untrusted): completed background command 884422\n\n"
        "Sender (untrusted metadata):\n```json\n"
        + json.dumps(envelope)
        + "\n```\nMake sure the release is actually verified before we ship it."
    )
    cls = classify_user_text(text, author_ids=AUTHOR_IDS)
    assert cls.kind == "telegram_envelope"
    assert cls.sender_verified is True
    assert cls.authored_by == "1000000001"
    assert cls.text == "Make sure the release is actually verified before we ship it."
    assert all(token not in cls.text for token in ("884422", "1000000001", "persona_user", "```"))


def test_telegram_envelope_removes_replied_message_context_for_both_variants():
    reply = {
        "sender_label": "agent",
        "body": "quoted context with account 998877 and @context_user",
    }
    variants = (
        (
            "Conversation info (untrusted metadata):",
            {"message_id": "10", "sender_id": "1000000001"},
        ),
        (
            "Sender (untrusted metadata):",
            {"label": "Persona Tester (1000000001)", "id": "1000000001"},
        ),
    )
    for marker, envelope in variants:
        text = (
            f"{marker}\n```json\n{json.dumps(envelope)}\n```\n\n"
            "Replied message (untrusted, for context):\n```json\n"
            f"{json.dumps(reply)}\n```\n\nHard cutover. Burn the bridges."
        )
        cls = classify_user_text(text, author_ids=AUTHOR_IDS)
        assert cls.sender_verified is True
        assert cls.authored_by == "1000000001"
        assert cls.text == "Hard cutover. Burn the bridges."
        residue = ("998877", "context_user", "sender_label", "```")
        assert all(token not in cls.text for token in residue)


def test_telegram_envelope_unverified_sender():
    envelope = {"message_id": "2", "sender_id": "999999"}
    text = (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(envelope)
        + "\n```\nhello from a stranger"
    )
    cls = classify_user_text(text, author_ids=AUTHOR_IDS)
    assert cls.kind == "telegram_envelope"
    assert cls.sender_verified is False
    assert cls.authored_by is None


def test_username_verification_from_config():
    # Author verification reads the username from config, never a hardcoded value.
    envelope = {"message_id": "3", "username": "persona_user"}
    text = (
        "Conversation info (untrusted metadata):\n```json\n"
        + json.dumps(envelope)
        + "\n```\nrun it"
    )
    cls = classify_user_text(text, author_ids=["1000000001", "persona_user"])
    assert cls.sender_verified is True
    assert cls.authored_by == "persona_user"
    # With a different configured username the same envelope does NOT verify.
    cls2 = classify_user_text(text, author_ids=["1000000001", "someoneelse"])
    assert cls2.sender_verified is False


def test_injected_prefix_after_timestamp_strip():
    ts, remainder = strip_timestamp_prefix("[Wed 2026-05-20 10:03 PDT] [Subagent Context] work")
    assert ts == "[Wed 2026-05-20 10:03 PDT]"
    assert remainder.startswith("[Subagent Context]")
    cls = classify_user_text(
        "[Sat 2026-04-11 02:10 PDT] [Subagent Task]: do lane D2", author_ids=AUTHOR_IDS
    )
    assert cls.kind == "injected"
    # every declared prefix classifies as injected
    for prefix in INJECTED_PREFIXES:
        assert classify_user_text(prefix + " tail", author_ids=AUTHOR_IDS).kind == "injected"


def test_bare_and_media_classification():
    bare = classify_user_text("just a normal question about the code", author_ids=AUTHOR_IDS)
    assert bare.kind == "bare"
    media = classify_user_text("[image #4]", author_ids=AUTHOR_IDS)
    assert media.kind == "media"


def test_parse_transcript_dispatch(tmp_path: Path):
    oc = _write_jsonl(tmp_path / "oc.jsonl", _openclaw_lines())
    session = parse_transcript(oc, author_ids=AUTHOR_IDS)
    assert session is not None and session.source_kind == "openclaw_session"
    # junk returns None
    junk = tmp_path / "bad.trajectory.jsonl"
    junk.write_text("{}\n", encoding="utf-8")
    assert parse_transcript(junk, author_ids=AUTHOR_IDS) is None


def test_fingerprint_remine_on_append(tmp_path: Path):
    conn = connect(tmp_path / "db.sqlite")
    init_db(conn)
    path = _write_jsonl(tmp_path / "sess1.jsonl", _openclaw_lines())
    unmined = list(iter_unmined_transcripts(conn, [tmp_path], "sft"))
    assert [p for p, _ in unmined] == [path]
    fingerprint = unmined[0][1]
    record_source(conn, str(path), "sft", fingerprint, 1)
    # unchanged file is not re-yielded
    assert list(iter_unmined_transcripts(conn, [tmp_path], "sft")) == []
    # appending a line changes the fingerprint (append-only) -> re-yielded
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"type": "message", "message": {"role": "user", "content": "one more thing"}}
            )
            + "\n"
        )
    reyielded = list(iter_unmined_transcripts(conn, [tmp_path], "sft"))
    assert [p for p, _ in reyielded] == [path]
