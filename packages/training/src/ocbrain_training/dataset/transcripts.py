"""Transcript parsing — the SOLE transcript authority for ocbrain v0.2 (spec §7.1).

Three verified on-disk formats are normalized to :class:`Session` / :class:`Turn`
DTOs that the whole dataset factory (and lane-3's ``review.py``) consumes:

* **openclaw** session files ``.../agents/<agent>/sessions/<id>.jsonl`` — a
  ``{"type":"session"}`` header line followed by ``{"type":"message"}`` lines
  whose ``message`` carries ``role`` (``user``/``assistant``/``toolResult``) and
  ``content`` (a plain string or a list of ``text``/``toolCall``/``toolResult``
  blocks).
* **claude-code** project JSONL ``~/.claude/projects/<slug>/<id>.jsonl`` — lines
  of ``{"type":"user"|"assistant", "message": {role, content}}`` (Anthropic
  block shapes ``text``/``tool_use``/``tool_result``/``thinking``) interleaved
  with non-conversation bookkeeping lines that are skipped.
* **codex** rollouts ``rollout-*.jsonl`` — ``{"type":"response_item",
  "payload": {...}}`` lines whose payload is a ``message`` (roles
  ``user``/``assistant``/``developer``, content blocks ``input_text`` /
  ``output_text``), an inter-agent ``agent_message`` (injected context), a
  ``reasoning`` block (dropped), or a tool call/output.

Normalization rules (spec §7.1): compatible consecutive same-role turns collapse;
``thinking``/``reasoning`` blocks NEVER enter text (other models' CoT is not
Jonathan-agent signal); tool results become ``role='tool'`` turns truncated to
``cfg.tool_result_truncate`` chars with an ``ERROR_RESULT_RE`` error flag; user
turns with different authors or classifications stay separate and are classified
(telegram-envelope / injected / media / bare) with config-driven author verification.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from ocbrain.fsutil import file_fingerprint, history_runtime
from ocbrain.write_batch import DatasetWriteBatch

# --- normalization constants -------------------------------------------------

# Leading local-time stamp openclaw/claude prepend to human turns, e.g.
# "[Wed 2026-05-20 10:03 PDT] ..." or "[Sat 2026-04-11 02:10:30 PDT] ...".
_TS_PREFIX_RE = re.compile(
    r"^\[\w{3}\s+\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?\s+[A-Za-z]{2,5}\]\s*"
)

# Prefixes (after timestamp strip) that mark a machine-injected user turn: sub-
# agent context, boot/heartbeat checks, cron ticks, compaction flushes, and the
# harness reminder envelopes. ~70% of main-agent "user" turns are these.
INJECTED_PREFIXES: tuple[str, ...] = (
    "[Subagent Context]",
    "[Subagent Task]",
    "[System]",
    "[System Reminder]",
    "<system-reminder>",
    "[Heartbeat",
    "[Heartbeat Check]",
    "[Boot]",
    "[Boot Check]",
    "[Bootstrap]",
    "[Cron]",
    "[Cron Tick]",
    "[Scheduled Task]",
    "[Compaction]",
    "[Compaction Flush]",
    "[Context Compaction]",
    "[Auto]",
    "[Automated]",
    "[Reminder]",
    "Caveat: The messages below",
    "This session is being continued from a previous",
    "[Continued Session]",
    "[Resumed Session]",
    # OpenClaw runtime envelopes observed in the production transcript corpus.
    "System (untrusted):",
    "<goal_context",
    "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>",
    "<<<END_OPENCLAW_INTERNAL_CONTEXT>>>",
    "<task-notification",
    "<environment_context",
    "Pre-compaction memory flush.",
    "Read HEARTBEAT.md if it exists",
    "You are running a boot check",
    "[cron:",
    "Warning: apply_patch was requested",
    "Warning: The maximum number of unified exec processes",
    "A scheduled reminder has been triggered",
    "A new session was started via /new or /reset.",
)

_INJECTED_RUNTIME_RE = re.compile(
    r"(?is)^\s*(?:"
    r"warning:\s*you have \d+ weighted tokens left\b|"
    r"you have \d+ weighted tokens left\b|"
    r"system:\s*\[[^\]]+\]\s*reminder\b"
    r")"
)

# A tool result that signals failure (spec §7.1 ``tool_errors``).
ERROR_RESULT_RE = re.compile(
    r"(?i)(?:^|\b)(?:error|traceback \(most recent call last\)|"
    r"exception|assertionerror|command failed|failed with|"
    r"exit code [1-9]|non-zero exit|fatal:|permission denied|"
    r"no such file|✕|✗)\b"
)

# A media-only user turn (attachment placeholder, no authored text).
_MEDIA_RE = re.compile(
    r"^(?:\[(?:image|images?\s*#?\d*|photo|voice message|audio|video|file|"
    r"document|attachment|sticker|gif)\s*[^\]]*\]|<media[^>]*>)\s*$",
    re.IGNORECASE,
)

_AUTHOR_ENVELOPE_MARKERS = (
    "Conversation info (untrusted metadata):",
    "Sender (untrusted metadata):",
)
_REPLY_ENVELOPE_MARKER = "Replied message (untrusted, for context):"
_ENVELOPE_MARKERS = (*_AUTHOR_ENVELOPE_MARKERS, _REPLY_ENVELOPE_MARKER)
_ENVELOPE_BLOCK_RE = re.compile(
    rf"(?P<marker>{'|'.join(re.escape(marker) for marker in _ENVELOPE_MARKERS)})"
    r"\s*```json\s*(?P<payload>.*?)\s*```",
    re.DOTALL,
)

# Sidecar / junk file suffixes and names excluded from the transcript corpus.
_SIDECAR_SUFFIXES = (
    ".trajectory.jsonl",
    ".trajectory-path.json",
    ".codex-app-server.json",
    ".jsonl.codex-app-server.json",
)
_SIDECAR_NAMES = (
    "sessions.json",
    # ChatGPT desktop / Codex migration bookkeeping. These are indexes or
    # prompt-history ledgers, not paired conversation rollouts.
    "session_index.jsonl",
    "history.jsonl",
)

_THINKING_BLOCK_TYPES = {"thinking", "reasoning", "redacted_thinking"}


# --- DTOs --------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One normalized conversational turn."""

    role: str  # 'user' | 'assistant' | 'tool' | 'system'
    text: str
    kind: str = ""  # user turns: telegram_envelope|injected|media|bare
    ts: str | None = None
    tool_name: str | None = None
    tool_error: bool = False
    n_tool_calls: int = 0  # assistant turns: collapsed tool-call count
    authored_by: str | None = None
    sender_verified: bool = False

    @property
    def injected(self) -> bool:
        return self.kind == "injected"


@dataclass(frozen=True)
class Session:
    """A normalized transcript."""

    session_id: str
    source_kind: str  # 'openclaw_session' | 'claude_session' | 'codex_session'
    source_uri: str
    runtime: str
    agent: str
    turns: tuple[Turn, ...] = ()
    cwd: str | None = None
    occurred_at: str | None = None


@dataclass(frozen=True)
class UserClass:
    """Result of classifying a user turn's text."""

    kind: str  # 'telegram_envelope' | 'injected' | 'media' | 'bare'
    text: str = ""  # cleaned message text (envelope removed)
    authored_by: str | None = None
    sender_verified: bool = False


# --- classification ----------------------------------------------------------


def strip_timestamp_prefix(text: str) -> tuple[str | None, str]:
    """Split a leading ``[Day YYYY-MM-DD HH:MM TZ]`` stamp from ``text``."""
    match = _TS_PREFIX_RE.match(text)
    if not match:
        return None, text
    return match.group(0).strip(), text[match.end() :]


def parse_telegram_envelope(text: str) -> tuple[dict | None, str]:
    """Extract untrusted sender metadata and the authored message after it.

    Returns ``(envelope_dict_or_None, message_text)``. When no envelope marker is
    present the input is returned unchanged with a ``None`` envelope. OpenClaw
    has emitted both ``Conversation info`` and ``Sender`` author envelopes; the
    latter may be followed by a quoted/replied-message context envelope. Only
    author-envelope fields are returned, and none of those envelope blocks (or
    any machine context before them) is admitted into the authored message.
    """
    matches = list(_ENVELOPE_BLOCK_RE.finditer(text))
    first_author = next(
        (i for i, match in enumerate(matches) if match.group("marker") in _AUTHOR_ENVELOPE_MARKERS),
        None,
    )
    if first_author is None:
        return None, text

    envelope: dict | None = None
    cursor = matches[first_author].start()
    for match in matches[first_author:]:
        # Sender/reply blocks form one leading envelope. Stop before anything
        # authored rather than stripping marker-like text later in the message.
        if text[cursor : match.start()].strip():
            break
        cursor = match.end()
        if match.group("marker") not in _AUTHOR_ENVELOPE_MARKERS:
            continue
        try:
            loaded = json.loads(match.group("payload"))
            if isinstance(loaded, dict):
                if envelope is None:
                    envelope = {}
                envelope.update(loaded)
        except (ValueError, TypeError):
            continue

    message = text[cursor:].strip()
    return envelope, message


def classify_user_text(
    text: str,
    *,
    author_ids: Sequence[str] = (),
    agent: str | None = None,
    direct_agents: Sequence[str] = (),
    founder_ids: Sequence[str] = (),
) -> UserClass:
    """Classify a user turn (spec §7.1 ``classify_user_text``).

    ``author_ids`` are the config-driven telegram identities (sender_id /
    username) that verify a turn as persona (Jonathan) authored — NEVER hardcoded
    here. ``founder_ids`` are additional feedback-author identities (e.g. a
    co-founder) that get ``authored_by`` stamped for attribution/weighting but are
    NOT persona-verified: their turns must never enter the persona/voice stream.
    """
    ts, body = strip_timestamp_prefix(text or "")
    stripped = body.strip()

    # Telegram envelope: parse metadata, verify authorship against config ids.
    if any(marker in stripped for marker in _AUTHOR_ENVELOPE_MARKERS):
        envelope, message = parse_telegram_envelope(stripped)
        authored_by: str | None = None
        verified = False
        if envelope is not None:
            ids = {str(a) for a in author_ids}
            fids = {str(a) for a in founder_ids}
            sender = str(envelope.get("sender_id") or envelope.get("id") or "").strip()
            username = str(envelope.get("username") or "").strip()
            for candidate in (sender, username):
                if not candidate:
                    continue
                if candidate in ids:
                    # Persona author: verified, admissible as voice.
                    authored_by = candidate
                    verified = True
                    break
                if candidate in fids:
                    # Founder feedback author (non-persona): attributed but never
                    # verified — records who spoke without admitting persona voice.
                    authored_by = candidate
                    break
        # A media-only envelope message stays classified as media residue.
        if message and _MEDIA_RE.match(message):
            return UserClass("media", text=message)
        return UserClass(
            "telegram_envelope",
            text=message,
            authored_by=authored_by,
            sender_verified=verified,
        )

    folded = stripped.casefold()
    if any(folded.startswith(prefix.casefold()) for prefix in INJECTED_PREFIXES) or (
        _INJECTED_RUNTIME_RE.match(stripped)
    ):
        return UserClass("injected", text=stripped)

    if not stripped or _MEDIA_RE.match(stripped):
        return UserClass("media", text=stripped)

    # Bare human text: authored (unverified) only when the session's own agent is
    # a configured direct-driver agent (e.g. 'main').
    verified = False
    authored_by = None
    if agent and agent in set(direct_agents):
        authored_by = None  # bare turns are never *verified*; persona flags them
    return UserClass("bare", text=stripped, authored_by=authored_by, sender_verified=verified)


# --- block extraction helpers ------------------------------------------------


def _text_from_blocks(content: object, text_keys: Sequence[str] = ("text",)) -> str:
    """Concatenate text blocks, dropping thinking/reasoning blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str):
                parts.append(block)
            continue
        btype = block.get("type")
        if btype in _THINKING_BLOCK_TYPES:
            continue
        for key in text_keys:
            value = block.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return "\n".join(p for p in parts if p)


def _tool_result_text(content: object, truncate: int) -> str:
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content") or ""
                if isinstance(value, str):
                    chunks.append(value)
            elif isinstance(block, str):
                chunks.append(block)
        text = "\n".join(c for c in chunks if c)
    else:
        text = str(content or "")
    return text[:truncate]


def _explicit_tool_error(*values: object) -> bool:
    """Read structured tool error flags without interpreting arbitrary text."""
    for value in values:
        if isinstance(value, list):
            if _explicit_tool_error(*value):
                return True
            continue
        if not isinstance(value, dict):
            continue
        if value.get("isError") is True or value.get("is_error") is True:
            return True
        status = value.get("status")
        if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
            return True
    return False


def _distinct_authors(a: Turn, b: Turn) -> bool:
    """True when two turns carry different identified authors (multi-user group).

    Collapsing user turns from different senders (e.g. a founder and the operator
    speaking back-to-back in a telegram group) would erase the per-sender
    attribution that founder weighting and persona isolation depend on, so such
    turns are kept separate.
    """
    return bool(a.authored_by and b.authored_by and a.authored_by != b.authored_by)


def _distinct_user_kinds(a: Turn, b: Turn) -> bool:
    """Keep injected/media/human boundaries from inheriting each other's class."""
    return a.role == "user" and b.role == "user" and a.kind != b.kind


def _collapse(turns: list[Turn]) -> tuple[Turn, ...]:
    """Collapse consecutive same-role turns (spec §7.1).

    Same-role turns merge, EXCEPT consecutive user turns authored by two distinct
    identified senders, which stay separate so multi-user telegram groups keep
    Patrick-vs-Jonathan-vs-agent attribution intact.
    """
    out: list[Turn] = []
    for turn in turns:
        if (
            out
            and out[-1].role == turn.role
            and not _distinct_authors(out[-1], turn)
            and not _distinct_user_kinds(out[-1], turn)
        ):
            prev = out[-1]
            merged_text = "\n".join(t for t in (prev.text, turn.text) if t)
            out[-1] = replace(
                prev,
                text=merged_text,
                n_tool_calls=prev.n_tool_calls + turn.n_tool_calls,
                tool_error=prev.tool_error or turn.tool_error,
                # keep the first turn's classification/authorship/timestamp
                kind=prev.kind or turn.kind,
                tool_name=prev.tool_name or turn.tool_name,
                authored_by=prev.authored_by or turn.authored_by,
                sender_verified=prev.sender_verified or turn.sender_verified,
            )
        else:
            out.append(turn)
    return tuple(out)


def _agent_from_path(path: Path, runtime: str) -> str:
    parts = path.parts
    if "agents" in parts:
        idx = parts.index("agents")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return runtime


# --- parsers -----------------------------------------------------------------


def _iter_json_lines(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def parse_openclaw_session(
    path: Path,
    *,
    author_ids: Sequence[str] = (),
    direct_agents: Sequence[str] = (),
    tool_result_truncate: int = 500,
    founder_ids: Sequence[str] = (),
) -> Session:
    runtime = "openclaw"
    agent = _agent_from_path(path, runtime)
    session_id = path.stem
    cwd: str | None = None
    occurred_at: str | None = None
    raw: list[Turn] = []
    for obj in _iter_json_lines(path):
        otype = obj.get("type")
        if otype == "session":
            session_id = str(obj.get("id") or session_id)
            cwd = obj.get("cwd")
            occurred_at = obj.get("timestamp") or occurred_at
            continue
        if otype != "message":
            continue
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        ts = message.get("timestamp") or obj.get("timestamp")
        occurred_at = occurred_at or ts
        content = message.get("content")
        if role == "user":
            text = _text_from_blocks(content)
            cls = classify_user_text(
                text,
                author_ids=author_ids,
                agent=agent,
                direct_agents=direct_agents,
                founder_ids=founder_ids,
            )
            raw.append(
                Turn(
                    role="user",
                    text=cls.text,
                    kind=cls.kind,
                    ts=ts,
                    authored_by=cls.authored_by,
                    sender_verified=cls.sender_verified,
                )
            )
        elif role == "assistant":
            text = _text_from_blocks(content)
            n_calls = _count_blocks(content, {"toolCall", "tool_use"})
            raw.append(Turn(role="assistant", text=text, ts=ts, n_tool_calls=n_calls))
        elif role in ("toolResult", "tool"):
            result = _tool_result_text(
                message.get("content") if content is None else content, tool_result_truncate
            )
            raw.append(
                Turn(
                    role="tool",
                    text=result,
                    ts=ts,
                    tool_name=message.get("toolName") or message.get("name"),
                    tool_error=(
                        _explicit_tool_error(message, content)
                        or bool(ERROR_RESULT_RE.search(result))
                    ),
                )
            )
    return Session(
        session_id=session_id,
        source_kind="openclaw_session",
        source_uri=str(path),
        runtime=runtime,
        agent=agent,
        turns=_collapse(raw),
        cwd=cwd,
        occurred_at=occurred_at,
    )


def _count_blocks(content: object, types: set[str]) -> int:
    if not isinstance(content, list):
        return 0
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") in types)


def parse_claude_session(
    path: Path,
    *,
    author_ids: Sequence[str] = (),
    direct_agents: Sequence[str] = (),
    tool_result_truncate: int = 500,
    founder_ids: Sequence[str] = (),
) -> Session:
    runtime = "claude"
    agent = _agent_from_path(path, runtime)
    session_id = path.stem
    cwd: str | None = None
    occurred_at: str | None = None
    raw: list[Turn] = []
    for obj in _iter_json_lines(path):
        otype = obj.get("type")
        if otype not in ("user", "assistant"):
            continue
        session_id = str(obj.get("sessionId") or session_id)
        cwd = obj.get("cwd") or cwd
        ts = obj.get("timestamp")
        occurred_at = occurred_at or ts
        message = obj.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if otype == "user":
            # A user line carrying tool_result blocks is a tool turn, not human text.
            if _count_blocks(content, {"tool_result"}) and not _text_from_blocks(content):
                result = _tool_result_text(content, tool_result_truncate)
                raw.append(
                    Turn(
                        role="tool",
                        text=result,
                        ts=ts,
                        tool_error=(
                            _explicit_tool_error(message, content)
                            or bool(ERROR_RESULT_RE.search(result))
                        ),
                    )
                )
                continue
            text = _text_from_blocks(content)
            cls = classify_user_text(
                text,
                author_ids=author_ids,
                agent=agent,
                direct_agents=direct_agents,
                founder_ids=founder_ids,
            )
            raw.append(
                Turn(
                    role="user",
                    text=cls.text,
                    kind=cls.kind,
                    ts=ts,
                    authored_by=cls.authored_by,
                    sender_verified=cls.sender_verified,
                )
            )
        else:
            text = _text_from_blocks(content)
            n_calls = _count_blocks(content, {"tool_use"})
            raw.append(Turn(role="assistant", text=text, ts=ts, n_tool_calls=n_calls))
    return Session(
        session_id=session_id,
        source_kind="claude_session",
        source_uri=str(path),
        runtime=runtime,
        agent=agent,
        turns=_collapse(raw),
        cwd=cwd,
        occurred_at=occurred_at,
    )


_CODEX_TOOL_CALL_TYPES = {"function_call", "custom_tool_call", "local_shell_call"}
_CODEX_TOOL_OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
}


def parse_codex_session(
    path: Path,
    *,
    author_ids: Sequence[str] = (),
    direct_agents: Sequence[str] = (),
    tool_result_truncate: int = 500,
    founder_ids: Sequence[str] = (),
) -> Session:
    runtime = "codex"
    agent = _agent_from_path(path, runtime)
    session_id = path.stem
    cwd: str | None = None
    occurred_at: str | None = None
    raw: list[Turn] = []
    for obj in _iter_json_lines(path):
        otype = obj.get("type")
        ts = obj.get("timestamp")
        occurred_at = occurred_at or ts
        if otype == "session_meta":
            payload = obj.get("payload")
            if isinstance(payload, dict):
                session_id = str(payload.get("id") or session_id)
                cwd = payload.get("cwd") or cwd
            continue
        if otype != "response_item":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if ptype == "reasoning":
            continue  # other-model CoT never enters text
        if ptype == "message":
            role = payload.get("role")
            text = _text_from_blocks(payload.get("content"), text_keys=("text",))
            if role == "assistant":
                raw.append(Turn(role="assistant", text=text, ts=ts))
            elif role == "developer":
                # Developer instructions are injected context, never human text.
                raw.append(Turn(role="user", text=text, kind="injected", ts=ts))
            else:  # user
                cls = classify_user_text(
                    text,
                    author_ids=author_ids,
                    agent=agent,
                    direct_agents=direct_agents,
                    founder_ids=founder_ids,
                )
                raw.append(
                    Turn(
                        role="user",
                        text=cls.text,
                        kind=cls.kind,
                        ts=ts,
                        authored_by=cls.authored_by,
                        sender_verified=cls.sender_verified,
                    )
                )
        elif ptype == "agent_message":
            # Current Codex/ChatGPT rollouts persist cross-agent messages as
            # response items. Preserve the plaintext context but never treat an
            # agent's words as operator-authored/persona training signal.
            text = _text_from_blocks(payload.get("content"), text_keys=("text",))
            if text:
                raw.append(Turn(role="user", text=text, kind="injected", ts=ts))
        elif ptype in _CODEX_TOOL_CALL_TYPES:
            raw.append(
                Turn(
                    role="assistant",
                    text="",
                    ts=ts,
                    n_tool_calls=1,
                    tool_name=payload.get("name"),
                )
            )
        elif ptype in _CODEX_TOOL_OUTPUT_TYPES:
            result = _tool_result_text(payload.get("output"), tool_result_truncate)
            raw.append(
                Turn(
                    role="tool",
                    text=result,
                    ts=ts,
                    tool_error=(
                        _explicit_tool_error(payload, payload.get("output"))
                        or bool(ERROR_RESULT_RE.search(result))
                    ),
                )
            )
    return Session(
        session_id=session_id,
        source_kind="codex_session",
        source_uri=str(path),
        runtime=runtime,
        agent=agent,
        turns=_collapse(raw),
        cwd=cwd,
        occurred_at=occurred_at,
    )


# --- corpus predicate + dispatch ---------------------------------------------


def is_conversation_transcript(path: Path) -> bool:
    """True for a real conversation transcript; False for sidecars/junk (spec §7.1)."""
    name = path.name
    lowered = name.lower()
    if not lowered.endswith(".jsonl"):
        return False
    if lowered in _SIDECAR_NAMES:
        return False
    for suffix in _SIDECAR_SUFFIXES:
        if lowered.endswith(suffix):
            return False
    parts = [p.lower() for p in path.parts]
    if "codex-home" in parts and ".tmp" in parts:
        return False
    if any(part == ".tmp" for part in parts):
        return False
    return True


def detect_format(path: Path, first_obj: dict | None) -> str:
    """Return 'openclaw'|'claude'|'codex' for a transcript path/first line."""
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]
    if name.startswith("rollout-") or ".codex" in parts:
        return "codex"
    if first_obj is not None:
        otype = first_obj.get("type")
        if otype == "session_meta":
            return "codex"
        if otype in ("queue-operation", "summary"):
            return "claude"
        if otype == "session" and "version" in first_obj:
            return "openclaw"
    if ".claude" in parts or "projects" in parts:
        return "claude"
    if ".openclaw" in parts or "agents" in parts:
        return "openclaw"
    return {"codex": "codex", "claude": "claude", "openclaw": "openclaw"}.get(
        history_runtime(path), "openclaw"
    )


def parse_transcript(
    path: Path,
    *,
    author_ids: Sequence[str] = (),
    direct_agents: Sequence[str] = (),
    tool_result_truncate: int = 500,
    founder_ids: Sequence[str] = (),
) -> Session | None:
    """Parse any supported transcript into a :class:`Session` (None for junk)."""
    path = Path(path)
    if not is_conversation_transcript(path):
        return None
    first_obj: dict | None = None
    for obj in _iter_json_lines(path):
        first_obj = obj
        break
    fmt = detect_format(path, first_obj)
    kwargs = {
        "author_ids": author_ids,
        "direct_agents": direct_agents,
        "tool_result_truncate": tool_result_truncate,
        "founder_ids": founder_ids,
    }
    if fmt == "codex":
        return parse_codex_session(path, **kwargs)
    if fmt == "claude":
        return parse_claude_session(path, **kwargs)
    return parse_openclaw_session(path, **kwargs)


# --- incremental discovery ---------------------------------------------------


def iter_transcript_files(roots: Iterable[str | Path]) -> Iterator[Path]:
    """Yield every conversation-transcript path under ``roots`` (sorted, deduped)."""
    seen: set[str] = set()
    for root in roots:
        base = Path(root).expanduser()
        if base.is_file():
            candidates: Iterable[Path] = [base]
        elif base.is_dir():
            candidates = sorted(base.rglob("*.jsonl"))
        else:
            continue
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if is_conversation_transcript(candidate):
                yield candidate


def resolve_transcript_evidence(
    conn: sqlite3.Connection,
    session: Session,
    *,
    write_batch: DatasetWriteBatch | None = None,
) -> tuple[str, str]:
    """Return ``(evidence_id, privacy_scope)`` for a session's transcript.

    Reuses the history-import evidence row (same ``source_uri``) when the harvest
    stage already created it; otherwise upserts a minimal transcript evidence row
    so every mined example carries a real provenance id (spec §7.3, ≥1 evidence).
    The evidence row's ``privacy_scope`` becomes the example's composed scope.
    """
    from ocbrain.db import upsert_evidence  # local import avoids a cycle at import time

    row = conn.execute(
        """
        SELECT id, privacy_scope FROM evidence
        WHERE source_uri = ? ORDER BY ingested_at DESC LIMIT 1
        """,
        (session.source_uri,),
    ).fetchone()
    if row is not None:
        return row["id"], row["privacy_scope"] or "workspace"
    path = Path(session.source_uri)
    try:
        digest = file_fingerprint(path)
    except OSError:
        digest = session.session_id
    if write_batch is not None:
        write_batch.ensure()
    evidence_id = upsert_evidence(
        conn,
        source_type=f"{session.runtime}_history_file",
        source_runtime=session.runtime,
        source_uri=session.source_uri,
        content_hash=digest,
        claim=f"{session.source_kind} transcript {session.session_id}",
        privacy_scope="workspace",
        occurred_at=session.occurred_at,
    )
    if write_batch is not None:
        write_batch.operation()
        # The caller parses and scores transcript examples next; never carry
        # this evidence transaction into that CPU-heavy work.
        write_batch.flush()
    return evidence_id, "workspace"


def _mined_fingerprints(conn: sqlite3.Connection, dataset: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT source_uri, fingerprint FROM dataset_sources WHERE dataset = ?",
        (dataset,),
    ).fetchall()
    return {row["source_uri"]: row["fingerprint"] for row in rows}


def record_source(
    conn: sqlite3.Connection,
    source_uri: str,
    dataset: str,
    fingerprint: str,
    emitted: int,
    *,
    status: str = "mined",
    detail: str | None = None,
) -> None:
    """Upsert the ``dataset_sources`` fingerprint ledger for one transcript file."""
    from ocbrain.db import now_iso

    conn.execute(
        """
        INSERT INTO dataset_sources (
          source_uri, dataset, fingerprint, mined_at, examples_emitted, status, detail
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_uri, dataset) DO UPDATE SET
          fingerprint = excluded.fingerprint,
          mined_at = excluded.mined_at,
          examples_emitted = excluded.examples_emitted,
          status = excluded.status,
          detail = excluded.detail
        """,
        (source_uri, dataset, fingerprint, now_iso(), emitted, status, detail),
    )


def iter_unmined_transcripts(
    conn: sqlite3.Connection,
    roots: Iterable[str | Path],
    dataset: str,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(path, fingerprint)`` for transcripts new or changed since last mine.

    Session files are append-only, so a changed ``file_fingerprint`` (path + size
    + mtime_ns) re-parses the file; ``UNIQUE(dataset, content_hash)`` dedups
    previously-emitted examples (spec §4.3).
    """
    mined = _mined_fingerprints(conn, dataset)
    for path in iter_transcript_files(roots):
        try:
            fingerprint = file_fingerprint(path)
        except OSError:
            continue
        if mined.get(str(path)) == fingerprint:
            continue
        yield path, fingerprint
