"""Persona / Jonathan-voice mining (spec §7.2 Persona).

Three sources of Jonathan's own words as assistant TARGETS:

1. **Telegram** — envelope-verified user turns (his messages) become the
   assistant target; the user side is the preceding assistant (agent) turn
   ≤4000 chars; openers (no preceding turn) are skipped. Bare unverified texts
   are admitted with ``sender_verified=false`` and −0.2 confidence unless
   ``--verified-only``.
2. **Git commits** — ``git log`` for configured authors, EXCLUDING agent-authored
   commits (``Co-Authored-By: Claude`` / ``🤖 Generated with``); prompt is
   ``git show --stat`` (≤2000 chars), target is subject+body. Each commit upserts
   a ``git_commit`` evidence row so persona examples carry real provenance.
3. **Authored docs** — OFF by default (``persona_authored_globs`` empty; the
   memory files are mostly agent-written and would poison the voice).

Style-consistency filtering keeps pasted code / commands / bare links out of the
voice set.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ocbrain.config import OcbrainConfig, load_config
from ocbrain.dataset.quality import store_example
from ocbrain.dataset.transcripts import Session, iter_transcript_files, parse_transcript
from ocbrain.ids import content_hash as _content_hash
from ocbrain.text import redact_secrets

_VERIFIED_CONF = 0.85
_UNVERIFIED_PENALTY = 0.2
_USER_SIDE_MAX = 4000
_STAT_MAX = 2000

_AGENT_COMMIT_MARKERS = ("Co-Authored-By: Claude", "🤖 Generated with", "Co-authored-by: Claude")

# Style rejects: slash-commands, bare URLs/paths, or code-fence pastes are not
# representative of Jonathan's prose voice.
_SLASH_CMD_RE = re.compile(r"^\s*/[a-zA-Z]")
_BARE_URL_RE = re.compile(r"^\s*https?://\S+\s*$")
_BARE_PATH_RE = re.compile(r"^\s*(?:/|~/|\./)?[\w./-]+\.\w{1,5}\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```")


def is_style_admissible(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _SLASH_CMD_RE.match(stripped):
        return False
    if _BARE_URL_RE.match(stripped):
        return False
    if _BARE_PATH_RE.match(stripped) and " " not in stripped:
        return False
    if _CODE_FENCE_RE.match(stripped):
        return False
    # Mostly-symbol pastes (little natural-language content) are rejected.
    letters = sum(ch.isalpha() or ch.isspace() for ch in stripped)
    if letters / max(1, len(stripped)) < 0.5:
        return False
    return True


def telegram_examples(
    session: Session,
    cfg: OcbrainConfig | None = None,
    *,
    verified_only: bool = False,
) -> list[dict[str, Any]]:
    """Yield persona example dicts (not yet stored) from Jonathan's turns."""
    cfg = cfg or load_config()
    system = cfg.dataset.persona_system_prompt
    out: list[dict[str, Any]] = []
    turns = session.turns
    for i, turn in enumerate(turns):
        if turn.role != "user":
            continue
        if turn.kind == "injected" or turn.kind == "media":
            continue
        verified = turn.sender_verified
        if not verified:
            if verified_only:
                continue
            # bare admitted only when the agent is a configured direct driver
            if session.agent not in set(cfg.dataset.persona_direct_agents):
                continue
        target = redact_secrets(turn.text.strip())
        if not is_style_admissible(target):
            continue
        # user side = preceding assistant turn; openers (none) are skipped.
        prev_assistant = None
        for j in range(i - 1, -1, -1):
            if turns[j].role == "assistant" and turns[j].text.strip():
                prev_assistant = turns[j].text.strip()[:_USER_SIDE_MAX]
                break
        if prev_assistant is None:
            continue
        confidence = _VERIFIED_CONF if verified else _VERIFIED_CONF - _UNVERIFIED_PENALTY
        out.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prev_assistant},
                    {"role": "assistant", "content": target},
                ],
                "target_text": target,
                "confidence": confidence,
                "sender_verified": verified,
                "occurred_at": turn.ts or session.occurred_at,
                "reasons": ["verified"] if verified else ["bare_unverified"],
            }
        )
    return out


def discover_git_repos(base: str | Path) -> list[Path]:
    base_path = Path(base).expanduser()
    if not base_path.is_dir():
        return []
    return sorted(p.parent for p in base_path.glob("*/.git"))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def _is_agent_commit(body: str) -> bool:
    return any(marker in body for marker in _AGENT_COMMIT_MARKERS)


def commit_examples(
    conn: sqlite3.Connection,
    repo: str | Path,
    cfg: OcbrainConfig | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Mine Jonathan's non-agent commits from one repo (upserts git_commit evidence)."""
    cfg = cfg or load_config()
    repo_path = Path(repo).expanduser()
    if not (repo_path / ".git").exists():
        return []
    from ocbrain.db import upsert_evidence

    author_args: list[str] = []
    for author in cfg.dataset.persona_git_authors:
        author_args.extend(["--author", author])
    sep = "\x1e"
    unit = "\x1f"
    fmt = unit.join(["%H", "%an", "%ae", "%s", "%b"]) + sep
    log_args = ["log", "--no-merges", f"--format={fmt}"]
    if author_args:
        log_args[1:1] = author_args
    if limit:
        log_args.append(f"-n{limit * 4}")  # over-fetch; agent commits get filtered
    raw = _git(repo_path, *log_args)
    out: list[dict[str, Any]] = []
    for record in raw.split(sep):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(unit)
        if len(parts) < 4:
            continue
        sha, _an, _ae, subject = parts[0], parts[1], parts[2], parts[3]
        body = parts[4] if len(parts) > 4 else ""
        if _is_agent_commit(body):
            continue
        message = subject if not body.strip() else f"{subject}\n\n{body.strip()}"
        message = redact_secrets(message.strip())
        stat_raw = _git(repo_path, "show", "--stat", "--format=", sha).strip()
        stat = redact_secrets(stat_raw)[:_STAT_MAX]
        if not stat:
            continue
        prompt = f"Write a commit message for these changes:\n{stat}"
        source_uri = f"git://{repo_path.name}#{sha}"
        evidence_id = upsert_evidence(
            conn,
            source_type="git_commit",
            source_runtime="git",
            source_uri=source_uri,
            content_hash=sha,
            claim=f"commit {sha[:12]}: {subject[:200]}",
            privacy_scope="workspace",
        )
        out.append(
            {
                "messages": [
                    {"role": "system", "content": cfg.dataset.persona_system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": message},
                ],
                "target_text": message,
                "confidence": _VERIFIED_CONF,
                "sender_verified": True,
                "evidence_ids": [evidence_id],
                "source_kind": "git_commit",
                "source_uri": source_uri,
                "occurred_at": None,
                "reasons": ["git_commit"],
            }
        )
        if limit and len(out) >= limit:
            break
    return out


def doc_examples(
    conn: sqlite3.Connection, cfg: OcbrainConfig | None = None
) -> list[dict[str, Any]]:
    """Authored-doc persona targets — OFF unless ``persona_authored_globs`` set."""
    cfg = cfg or load_config()
    from ocbrain.db import upsert_evidence

    out: list[dict[str, Any]] = []
    for pattern in cfg.dataset.persona_authored_globs:
        for path in sorted(Path().glob(pattern)):
            if not path.is_file():
                continue
            text = redact_secrets(path.read_text(encoding="utf-8", errors="replace").strip())
            if not is_style_admissible(text):
                continue
            source_uri = str(path)
            evidence_id = upsert_evidence(
                conn,
                source_type="authored_doc",
                source_runtime="openclaw",
                source_uri=source_uri,
                content_hash=_content_hash(text),
                claim=f"authored doc {path.name}",
                privacy_scope="workspace",
            )
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": cfg.dataset.persona_system_prompt},
                        {"role": "user", "content": f"Write the document '{path.stem}'."},
                        {"role": "assistant", "content": text},
                    ],
                    "target_text": text,
                    "confidence": _VERIFIED_CONF,
                    "sender_verified": True,
                    "evidence_ids": [evidence_id],
                    "source_kind": "authored_doc",
                    "source_uri": source_uri,
                    "occurred_at": None,
                    "reasons": ["authored_doc"],
                }
            )
    return out


def mine_persona(
    conn: sqlite3.Connection,
    *,
    cfg: OcbrainConfig | None = None,
    sessions: Iterable[Session] | None = None,
    roots: Iterable[str] | None = None,
    repos: Iterable[str | Path] | None = None,
    verified_only: bool = False,
    limit: int | None = None,
    time_budget_seconds: float | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    from ocbrain.dataset.transcripts import record_source, resolve_transcript_evidence

    started = time.monotonic()
    stored = 0
    excluded = 0
    examined = 0

    def _store(candidate: dict[str, Any], *, evidence_ids: list[str], scope: str,
               source_kind: str, source_uri: str | None, session_id: str | None) -> None:
        nonlocal stored, excluded, examined
        examined += 1
        result = store_example(
            conn,
            dataset="persona",
            source_kind=source_kind,
            source_uri=source_uri,
            source_span=None,
            evidence_ids=evidence_ids,
            privacy_scope=scope,
            body={"messages": candidate["messages"]},
            metadata={
                "sender_verified": candidate["sender_verified"],
                "session_id": session_id,
                "source_kind": source_kind,
            },
            target_text=candidate["target_text"],
            base_label="good",
            base_confidence=candidate["confidence"],
            base_reasons=candidate.get("reasons", []),
            n_turns=len(candidate["messages"]),
            session_id=session_id,
            occurred_at=candidate.get("occurred_at"),
        )
        if result is None:
            return
        if result["quality_label"] == "excluded":
            excluded += 1
        else:
            stored += 1

    # Telegram (from provided sessions or discovered transcripts).
    def _emit_session(session: Session) -> None:
        evidence_id, scope = resolve_transcript_evidence(conn, session)
        for candidate in telegram_examples(session, cfg, verified_only=verified_only):
            _store(
                candidate,
                evidence_ids=[evidence_id],
                scope=scope,
                source_kind=session.source_kind,
                source_uri=session.source_uri,
                session_id=session.session_id,
            )

    if sessions is not None:
        for session in sessions:
            _emit_session(session)
    elif roots is not None:
        for path in iter_transcript_files(roots):
            if time_budget_seconds is not None and time.monotonic() - started > time_budget_seconds:
                break
            session = parse_transcript(
                path,
                author_ids=cfg.dataset.persona_author_ids,
                direct_agents=cfg.dataset.persona_direct_agents,
                tool_result_truncate=cfg.dataset.tool_result_truncate,
            )
            if session is None:
                continue
            before = examined
            _emit_session(session)
            record_source(conn, str(path), "persona", _fingerprint(path), examined - before)

    # Git commits.
    repo_list: list[Path] = []
    if repos is not None:
        repo_list = [Path(r).expanduser() for r in repos]
    elif cfg.dataset.persona_git_repos:
        repo_list = [Path(r).expanduser() for r in cfg.dataset.persona_git_repos]
    else:
        repo_list = discover_git_repos("~/.openclaw/workspace")
    for repo in repo_list:
        for candidate in commit_examples(conn, repo, cfg, limit=limit):
            _store(
                candidate,
                evidence_ids=candidate["evidence_ids"],
                scope="workspace",
                source_kind="git_commit",
                source_uri=candidate["source_uri"],
                session_id=None,
            )

    # Authored docs (off by default).
    for candidate in doc_examples(conn, cfg):
        _store(
            candidate,
            evidence_ids=candidate["evidence_ids"],
            scope="workspace",
            source_kind="authored_doc",
            source_uri=candidate["source_uri"],
            session_id=None,
        )

    return {
        "ok": True,
        "dataset": "persona",
        "examined": examined,
        "stored": stored,
        "excluded": excluded,
    }


def _fingerprint(path: Path) -> str:
    from ocbrain.fsutil import file_fingerprint

    try:
        return file_fingerprint(path)
    except OSError:
        return ""
