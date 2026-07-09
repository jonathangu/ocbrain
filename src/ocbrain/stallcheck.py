"""ocbrain stall checker — a passive, always-on progress watchdog.

Jonathan's agents sometimes end a work turn *waiting* ("standing by",
"waiting on the monitor") and then never move again because the notification
they expect never arrives. Nobody notices until he pings to follow up. This
module is that follow-up, automated: every 15 minutes it sweeps the agent
exhaust for the signature of a parked-and-forgotten turn, records the finding
into the brain's deadman engine, and sends Jonathan ONE Telegram digest of any
*new* stalls (deduplicated so a persistent stall pages exactly once).

Three readers feed it:

  READER A (filesystem) — the money reader. Scans subagent workflow dirs for
    ``agent-*.jsonl`` transcripts whose LAST record is an assistant ``end_turn``
    whose text matches a passive-wait lexicon, and task ``.output`` files that
    are zero-byte or opened (``start:``) but never closed (``exit:``). A
    workflow whose journal shows *recent* result activity, or whose files were
    *recently* appended, is alive and is never flagged.

  READER B (sqlite, read-only) — runner ``task_runs`` that are ``lost`` or have
    been ``running``/``pending`` with no event past the stale threshold.

  READER C (sqlite, read-only) — ``channel_ingress_events`` that failed with
    ``handler-timeout`` inside the lookback window (dropped inbound work).

Every finding upserts a ``loop_liveness`` row + a ``loop_tripwire`` evidence
row into the brain DB, so the existing liveness sweep and the weekly review see
it. The checker also writes its OWN heartbeat row (``loop_id='stallcheck'``) so
the weekly review notices if the watchman itself dies.

SECURITY: the Telegram bot token is read from ``openclaw.json`` at send time and
is NEVER printed, logged, or stored. The committed config defaults are empty —
without a local ``stall_pager`` config (chat id + openclaw.json path) the pager
is inert and the module only scans + records.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.db import connect, upsert_evidence
from ocbrain.ids import content_hash, stable_id

# --- Brain connection hardening ------------------------------------------------
# The live brain DB has heavy concurrent writers (autopilot, mcp). Every brain
# connection this module opens must wait rather than fail-fast on a lock.
BRAIN_BUSY_TIMEOUT_MS = 5000

# --- Config defaults -----------------------------------------------------------
# Defaults are intentionally operational-but-inert: the scan roots point at the
# real exhaust, but the *pager* stays silent until a local (gitignored) config
# supplies a chat id + openclaw.json path. Committed code carries no chat ids and
# no token paths.
DEFAULT_STALE_MINUTES = 20
DEFAULT_TERMINAL_BACKLOG_HOURS = 48
DEFAULT_INGRESS_WINDOW_HOURS = 720
DEFAULT_SELF_INTERVAL_SECONDS = 900
# Telegram rejects a sendMessage body over 4096 chars with HTTP 400, so the
# digest is bounded two ways: at most this many stalls per run (the rest ride the
# next 15-min cycle), and a hard character budget below the API ceiling.
DEFAULT_MAX_PAGES_PER_RUN = 8
MAX_MESSAGE_CHARS = 3900

DEFAULT_WORKFLOW_GLOBS: tuple[str, ...] = (
    "/Users/guclaw/.claude/projects/*/*/subagents/workflows/*/",
)
DEFAULT_TASK_OUTPUT_GLOBS: tuple[str, ...] = (
    "/private/tmp/claude-501/*/*/tasks/",
)
DEFAULT_RUNNER_DB = "/Users/guclaw/.openclaw/state/openclaw.sqlite"

DEFAULT_PASSIVE_WAIT_LEXICON: tuple[str, ...] = (
    "waiting on the monitor",
    "standing by",
    "i'll hold here",
    "no further action is useful until",
    "let the monitor notify me",
    "waiting for the monitor",
)

# Cap the byte count read for the start:/exit: text probe so a giant transcript
# masquerading as a .output file never blows up memory. Zero-byte detection is a
# stat, unaffected by this cap.
_OUTPUT_PROBE_MAX_BYTES = 512 * 1024


@dataclass(frozen=True)
class StallCheckConfig:
    workflow_globs: tuple[str, ...] = DEFAULT_WORKFLOW_GLOBS
    task_output_globs: tuple[str, ...] = DEFAULT_TASK_OUTPUT_GLOBS
    runner_db: str = DEFAULT_RUNNER_DB
    passive_wait_lexicon: tuple[str, ...] = DEFAULT_PASSIVE_WAIT_LEXICON
    stale_threshold_minutes: int = DEFAULT_STALE_MINUTES
    terminal_backlog_hours: int = DEFAULT_TERMINAL_BACKLOG_HOURS
    ingress_window_hours: int = DEFAULT_INGRESS_WINDOW_HOURS
    self_interval_seconds: int = DEFAULT_SELF_INTERVAL_SECONDS
    max_pages_per_run: int = DEFAULT_MAX_PAGES_PER_RUN
    # Spec default: a zero-byte .output is a stall. In practice this system's
    # .output files are also used as background-command stdout spools, many of
    # which are legitimately empty, so an operator can turn the zero-byte clause
    # off in local config while keeping the (precise) start-without-exit clause.
    flag_zero_byte_output: bool = True
    # Pager: empty/None on committed defaults -> inert.
    pager_account: str = "default"
    pager_chat_id: str | None = None
    pager_openclaw_json: str | None = None

    @property
    def stale_threshold_seconds(self) -> int:
        return self.stale_threshold_minutes * 60

    @property
    def terminal_backlog_seconds(self) -> int:
        return self.terminal_backlog_hours * 3600


def load_config(config_path: Path | str | None = None) -> StallCheckConfig:
    """Layer the ``stall_check`` / ``stall_pager`` sections of the ocbrain JSON
    config over the hard defaults. A missing file yields pure defaults (inert
    pager)."""
    path = Path(
        config_path
        or os.environ.get("OCBRAIN_CONFIG", "data/ocbrain.config.json")
    ).expanduser()
    raw: dict[str, Any] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            raw = {}
    sc = raw.get("stall_check", {}) if isinstance(raw.get("stall_check"), dict) else {}
    pg = raw.get("stall_pager", {}) if isinstance(raw.get("stall_pager"), dict) else {}

    def _tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        val = sc.get(key)
        if isinstance(val, list) and val:
            return tuple(str(x) for x in val)
        return default

    return StallCheckConfig(
        workflow_globs=_tuple("workflow_globs", DEFAULT_WORKFLOW_GLOBS),
        task_output_globs=_tuple("task_output_globs", DEFAULT_TASK_OUTPUT_GLOBS),
        runner_db=str(sc.get("runner_db", DEFAULT_RUNNER_DB)),
        passive_wait_lexicon=_tuple("passive_wait_lexicon", DEFAULT_PASSIVE_WAIT_LEXICON),
        stale_threshold_minutes=int(sc.get("stale_threshold_minutes", DEFAULT_STALE_MINUTES)),
        terminal_backlog_hours=int(
            sc.get("terminal_backlog_hours", DEFAULT_TERMINAL_BACKLOG_HOURS)
        ),
        ingress_window_hours=int(sc.get("ingress_window_hours", DEFAULT_INGRESS_WINDOW_HOURS)),
        self_interval_seconds=int(sc.get("self_interval_seconds", DEFAULT_SELF_INTERVAL_SECONDS)),
        max_pages_per_run=int(sc.get("max_pages_per_run", DEFAULT_MAX_PAGES_PER_RUN)),
        flag_zero_byte_output=bool(sc.get("flag_zero_byte_output", True)),
        pager_account=str(pg.get("account", "default")),
        pager_chat_id=(str(pg["chat_id"]) if pg.get("chat_id") is not None else None),
        pager_openclaw_json=(
            str(pg["openclaw_json"]) if pg.get("openclaw_json") is not None else None
        ),
    )


# --- Findings ------------------------------------------------------------------
@dataclass(frozen=True)
class Finding:
    stall_class: str
    unit_id: str
    terminal_signature: str
    snippet: str
    artifact_path: str
    age_seconds: float
    occurred_at: str

    @property
    def fingerprint(self) -> str:
        # Stable across runs for the same stall; changes when the terminal
        # signature changes (a genuinely new stall on the same unit).
        return stable_id(
            "stall", self.stall_class, self.unit_id, self.terminal_signature
        )

    @property
    def loop_id(self) -> str:
        return f"stall/{self.stall_class}"

    @property
    def suggested_action(self) -> str:
        return (
            f"point a fresh agent at {self.artifact_path} with: "
            "review the diff, finish from last step"
        )


def _snip(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat[:limit]


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# --- Reader A: filesystem ------------------------------------------------------
def _last_jsonl_record(path: Path) -> dict[str, Any] | None:
    """Return the last non-blank JSON record of a .jsonl file, or None."""
    last: dict[str, Any] | None = None
    try:
        with path.open("r", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    last = obj
    except OSError:
        return None
    return last


def _assistant_end_turn_text(record: dict[str, Any]) -> str | None:
    """If ``record`` is an assistant ``end_turn`` message, return its joined text."""
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    if message.get("role") != "assistant" or message.get("stop_reason") != "end_turn":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return None


def _journal_recent_result(journal: Path, cutoff_epoch: float) -> bool:
    """True if the workflow journal recorded a ``result`` row after ``cutoff``.

    A workflow that is still emitting results is alive; never flag it. We use the
    file mtime as the activity clock (the journal is append-only, so a result
    newer than the cutoff means the file itself was touched after the cutoff).
    """
    if not journal.is_file():
        return False
    if _mtime(journal) < cutoff_epoch:
        return False
    try:
        with journal.open("r", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and obj.get("type") == "result":
                    return True
    except OSError:
        return False
    return False


def _dir_recently_active(directory: Path, cutoff_epoch: float) -> bool:
    """True if any file in ``directory`` was modified after ``cutoff`` — the
    workflow is still being appended to, so nothing in it is stalled."""
    try:
        for child in directory.iterdir():
            if child.is_file() and _mtime(child) >= cutoff_epoch:
                return True
    except OSError:
        return False
    return False


def scan_workflow_stalls(cfg: StallCheckConfig, now: datetime) -> list[Finding]:
    now_epoch = now.timestamp()
    cutoff = now_epoch - cfg.stale_threshold_seconds
    lexicon = [kw.lower() for kw in cfg.passive_wait_lexicon]
    findings: list[Finding] = []
    seen_dirs: dict[Path, bool] = {}
    for pattern in cfg.workflow_globs:
        for match in sorted(Path("/").glob(pattern.lstrip("/"))):
            if not match.is_dir():
                continue
            for agent_file in sorted(match.glob("agent-*.jsonl")):
                mtime = _mtime(agent_file)
                if mtime == 0.0 or mtime >= cutoff:
                    continue  # fresh / still being appended
                record = _last_jsonl_record(agent_file)
                if record is None:
                    continue
                text = _assistant_end_turn_text(record)
                if not text:
                    continue
                low = text.lower()
                matched = next((kw for kw in lexicon if kw in low), None)
                if matched is None:
                    continue
                # False-positive guards, evaluated once per workflow dir.
                if match not in seen_dirs:
                    seen_dirs[match] = _dir_recently_active(
                        match, cutoff
                    ) or _journal_recent_result(match / "journal.jsonl", cutoff)
                if seen_dirs[match]:
                    continue
                findings.append(
                    Finding(
                        stall_class="workflow_passive_wait",
                        unit_id=agent_file.stem.removeprefix("agent-") or agent_file.stem,
                        terminal_signature=f"passive_wait:{matched}",
                        snippet=_snip(text),
                        artifact_path=str(agent_file),
                        age_seconds=now_epoch - mtime,
                        occurred_at=datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
                    )
                )
    return findings


def scan_task_output_stalls(cfg: StallCheckConfig, now: datetime) -> list[Finding]:
    now_epoch = now.timestamp()
    cutoff = now_epoch - cfg.stale_threshold_seconds
    findings: list[Finding] = []
    for pattern in cfg.task_output_globs:
        for tasks_dir in sorted(Path("/").glob(pattern.lstrip("/"))):
            if not tasks_dir.is_dir():
                continue
            for output in sorted(tasks_dir.glob("*.output")):
                mtime = _mtime(output)
                if mtime == 0.0 or mtime >= cutoff:
                    continue
                try:
                    size = output.stat().st_size
                except OSError:
                    continue
                signature: str | None = None
                snippet = ""
                if size == 0:
                    if not cfg.flag_zero_byte_output:
                        continue
                    signature = "zero_byte"
                    snippet = "task output is empty (never wrote a byte)"
                else:
                    has_start, has_exit = _probe_start_exit(output)
                    if has_start and not has_exit:
                        signature = "start_no_exit"
                        snippet = "task started but never recorded an exit line"
                if signature is None:
                    continue
                findings.append(
                    Finding(
                        stall_class="task_output_stall",
                        unit_id=output.stem,
                        terminal_signature=signature,
                        snippet=snippet,
                        artifact_path=str(output),
                        age_seconds=now_epoch - mtime,
                        occurred_at=datetime.fromtimestamp(mtime, tz=UTC).isoformat(),
                    )
                )
    return findings


def _probe_start_exit(path: Path) -> tuple[bool, bool]:
    has_start = False
    has_exit = False
    try:
        with path.open("r", errors="replace") as handle:
            read = 0
            for line in handle:
                read += len(line)
                stripped = line.lstrip()
                if stripped.startswith("start:"):
                    has_start = True
                elif stripped.startswith("exit:"):
                    has_exit = True
                if read >= _OUTPUT_PROBE_MAX_BYTES:
                    break
    except OSError:
        return (False, False)
    return (has_start, has_exit)


# --- Readers B + C: runner sqlite (read-only) ----------------------------------
def open_runner_ro(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def scan_runner_task_runs(
    runner: sqlite3.Connection, cfg: StallCheckConfig, now: datetime
) -> list[Finding]:
    """READER B — lost, or running/pending past the stale threshold."""
    now_ms = int(now.timestamp() * 1000)
    stale_ms = now_ms - cfg.stale_threshold_seconds * 1000
    findings: list[Finding] = []
    # Uses idx_task_runs_status (status is the leading, selective predicate).
    rows = runner.execute(
        """
        SELECT task_id, status, label, task, last_event_at, error, progress_summary
        FROM task_runs
        WHERE status = 'lost'
           OR (status IN ('running', 'pending') AND last_event_at < ?)
        ORDER BY last_event_at ASC
        """,
        (stale_ms,),
    ).fetchall()
    for row in rows:
        last_event = row["last_event_at"] or now_ms
        age = max(0.0, (now_ms - last_event) / 1000.0)
        note = (row["progress_summary"] or row["error"] or row["label"] or row["task"] or "")
        findings.append(
            Finding(
                stall_class="task_run_lost" if row["status"] == "lost" else "task_run_stale",
                unit_id=row["task_id"],
                terminal_signature=f"status:{row['status']}",
                snippet=_snip(str(note)),
                artifact_path=f"runner:task_runs/{row['task_id']}",
                age_seconds=age,
                occurred_at=datetime.fromtimestamp(last_event / 1000.0, tz=UTC).isoformat(),
            )
        )
    return findings


def scan_ingress_timeouts(
    runner: sqlite3.Connection, cfg: StallCheckConfig, now: datetime
) -> list[Finding]:
    """READER C — inbound events that failed with a handler-timeout in-window."""
    now_ms = int(now.timestamp() * 1000)
    window_ms = now_ms - cfg.ingress_window_hours * 3600 * 1000
    findings: list[Finding] = []
    rows = runner.execute(
        """
        SELECT queue_name, event_id, channel_id, account_id, failed_reason,
               failed_at, last_error
        FROM channel_ingress_events
        WHERE status = 'failed'
          AND failed_reason = 'handler-timeout'
          AND failed_at >= ?
        ORDER BY failed_at ASC
        """,
        (window_ms,),
    ).fetchall()
    for row in rows:
        failed_at = row["failed_at"] or now_ms
        age = max(0.0, (now_ms - failed_at) / 1000.0)
        note = row["last_error"] or f"{row['queue_name']} handler timed out"
        findings.append(
            Finding(
                stall_class="ingress_handler_timeout",
                unit_id=f"{row['queue_name']}/{row['event_id']}",
                terminal_signature="handler-timeout",
                snippet=_snip(str(note)),
                artifact_path=f"runner:channel_ingress_events/{row['event_id']}",
                age_seconds=age,
                occurred_at=datetime.fromtimestamp(failed_at / 1000.0, tz=UTC).isoformat(),
            )
        )
    return findings


# --- Brain writes: deadman engine + pager ledger -------------------------------
def open_brain(path: Path | str) -> sqlite3.Connection:
    conn = connect(Path(path))
    conn.execute(f"PRAGMA busy_timeout={BRAIN_BUSY_TIMEOUT_MS}")
    return conn


STALL_PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS stall_pages (
  fingerprint TEXT PRIMARY KEY,
  stall_class TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  terminal_signature TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  paged_at TEXT,
  run_count INTEGER NOT NULL DEFAULT 1
);
"""

_FIRSTRUN_MARKER = "__firstrun__"


def ensure_stall_pages(conn: sqlite3.Connection) -> None:
    conn.executescript(STALL_PAGES_SCHEMA)


def is_first_run(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM stall_pages WHERE fingerprint = ? LIMIT 1", (_FIRSTRUN_MARKER,)
    ).fetchone()
    return row is None


def mark_first_run_done(conn: sqlite3.Connection, now: datetime) -> None:
    stamp = now.isoformat()
    conn.execute(
        """
        INSERT INTO stall_pages (
          fingerprint, stall_class, unit_id, terminal_signature,
          first_seen_at, last_seen_at, paged_at, run_count
        ) VALUES (?, 'marker', 'firstrun', 'firstrun', ?, ?, ?, 1)
        ON CONFLICT(fingerprint) DO UPDATE SET last_seen_at = excluded.last_seen_at
        """,
        (_FIRSTRUN_MARKER, stamp, stamp, stamp),
    )


def already_paged(conn: sqlite3.Connection, fingerprint: str) -> bool:
    """True only if this stall was ACTUALLY delivered (paged_at set). A finding
    that was recorded but whose page never sent (inert pager, HTTP error, or held
    over a per-run cap) stays eligible so the next run retries it."""
    row = conn.execute(
        "SELECT 1 FROM stall_pages WHERE fingerprint = ? AND paged_at IS NOT NULL LIMIT 1",
        (fingerprint,),
    ).fetchone()
    return row is not None


def record_page(
    conn: sqlite3.Connection, finding: Finding, now: datetime, *, paged: bool
) -> None:
    stamp = now.isoformat()
    conn.execute(
        """
        INSERT INTO stall_pages (
          fingerprint, stall_class, unit_id, terminal_signature,
          first_seen_at, last_seen_at, paged_at, run_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(fingerprint) DO UPDATE SET
          last_seen_at = excluded.last_seen_at,
          paged_at = COALESCE(stall_pages.paged_at, excluded.paged_at),
          run_count = stall_pages.run_count + 1
        """,
        (
            finding.fingerprint,
            finding.stall_class,
            finding.unit_id,
            finding.terminal_signature,
            stamp,
            stamp,
            stamp if paged else None,
        ),
    )


def feed_deadman(conn: sqlite3.Connection, finding: Finding, now: datetime) -> str:
    """Upsert a loop_liveness row (deadman already due) + a loop_tripwire evidence
    row for one finding. Idempotent: one evidence row per stall fingerprint."""
    stamp = now.isoformat()
    conn.execute(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(loop_id, run_id) DO UPDATE SET
          last_heartbeat_at = excluded.last_heartbeat_at,
          last_ledger_write_at = excluded.last_ledger_write_at,
          expected_interval_seconds = excluded.expected_interval_seconds,
          deadman_due_at = excluded.deadman_due_at
        """,
        (
            finding.loop_id,
            finding.unit_id,
            finding.occurred_at,
            finding.occurred_at,
            None,
            stamp,  # deadman already due — the stall is, by definition, overdue
        ),
    )
    body = json.dumps(
        {
            "stall_class": finding.stall_class,
            "unit_id": finding.unit_id,
            "terminal_signature": finding.terminal_signature,
        },
        sort_keys=True,
    )
    return upsert_evidence(
        conn,
        source_type="loop_tripwire",
        source_runtime="ocbrain-stallcheck",
        source_uri=(
            f"ocbrain://stall/{finding.stall_class}/{finding.unit_id}"
            f"/{finding.terminal_signature}"
        ),
        content_hash=content_hash(body),
        claim=f"Stall detected: {finding.stall_class} {finding.unit_id} — {finding.snippet}",
        artifact_uri=finding.artifact_path,
        verifier_status="not_required",
        loop_tags={
            "loop_id": finding.loop_id,
            "run_id": finding.unit_id,
            "tripwire": f"stall_{finding.stall_class}",
        },
        privacy_scope="workspace",
        occurred_at=stamp,
    )


def upsert_self_heartbeat(conn: sqlite3.Connection, cfg: StallCheckConfig, now: datetime) -> None:
    """Record the watchman's own liveness so the weekly review notices if it dies."""
    stamp = now.isoformat()
    interval = cfg.self_interval_seconds
    due = datetime.fromtimestamp(now.timestamp() + 2 * interval, tz=UTC).isoformat()
    conn.execute(
        """
        INSERT INTO loop_liveness (
          loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
          expected_interval_seconds, deadman_due_at
        ) VALUES ('stallcheck', 'heartbeat', ?, ?, ?, ?)
        ON CONFLICT(loop_id, run_id) DO UPDATE SET
          last_heartbeat_at = excluded.last_heartbeat_at,
          last_ledger_write_at = excluded.last_ledger_write_at,
          expected_interval_seconds = excluded.expected_interval_seconds,
          deadman_due_at = excluded.deadman_due_at
        """,
        (stamp, stamp, interval, due),
    )


# --- Pager (Telegram, stdlib only) ---------------------------------------------
def read_bot_token(openclaw_json: str, account: str) -> str | None:
    """Read the Telegram bot token from openclaw.json at send time.

    The token is returned to the immediate caller (``send_telegram``) and used
    once to build the request URL. It is NEVER printed, logged, or stored.
    """
    try:
        data = json.loads(Path(openclaw_json).expanduser().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    account_cfg = (
        data.get("channels", {})
        .get("telegram", {})
        .get("accounts", {})
        .get(account, {})
    )
    token = account_cfg.get("botToken")
    return token or None


def _finding_block(f: Finding) -> str:
    age_h = f.age_seconds / 3600.0
    lines = [f"• {f.stall_class} — {f.unit_id} (idle {age_h:.1f}h)"]
    if f.snippet:
        lines.append(f'  "{_snip(f.snippet)}"')
    lines.append(f"  {f.artifact_path}")
    lines.append("  → fresh agent: review the diff, finish from last step")
    return "\n".join(lines)


def build_digest_message(
    findings: list[Finding],
    *,
    prefix: str = "",
    backlog: bool = False,
    max_chars: int = MAX_MESSAGE_CHARS,
) -> tuple[str, int]:
    """Render a bounded digest. Returns ``(text, n_included)`` — blocks are added
    until the character budget is hit, and any remainder is summarized in a
    footer so the message never exceeds Telegram's 4096-char limit."""
    total = len(findings)
    header = f"{total} stalled agent(s)"
    if backlog:
        header += " (backlog on first run)"
    header = f"{prefix}{header}:" if prefix else f"{header}:"
    body: list[str] = [header]
    included = 0
    for f in findings:
        block = _finding_block(f)
        remaining = total - included - 1
        footer = f"\n… +{remaining} more (next digest)" if remaining > 0 else ""
        candidate = "\n".join(body + ["", block]) + footer
        if len(candidate) > max_chars and included > 0:
            body.append(f"\n… +{total - included} more (next digest)")
            break
        body.append("")
        body.append(block)
        included += 1
    return "\n".join(body), included


def send_telegram(cfg: StallCheckConfig, text: str) -> int | None:
    """Send ONE Telegram DM. Returns the HTTP status, or None if the pager is
    inert (no chat id / openclaw.json configured) or the token is unavailable.

    The token never leaves this function's local scope and is never logged.
    """
    if not cfg.pager_chat_id or not cfg.pager_openclaw_json:
        return None
    token = read_bot_token(cfg.pager_openclaw_json, cfg.pager_account)
    if not token:
        return None
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode(
        {"chat_id": cfg.pager_chat_id, "text": text, "disable_web_page_preview": "true"}
    ).encode()
    request = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


# --- Orchestration -------------------------------------------------------------
@dataclass
class RunReport:
    findings: list[Finding] = field(default_factory=list)
    new_findings: list[Finding] = field(default_factory=list)
    paged: list[Finding] = field(default_factory=list)
    first_run: bool = False
    page_status: int | None = None
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        by_class: dict[str, int] = {}
        for f in self.findings:
            by_class[f.stall_class] = by_class.get(f.stall_class, 0) + 1
        return by_class


def collect_findings(
    cfg: StallCheckConfig,
    now: datetime,
    *,
    runner: sqlite3.Connection | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(scan_workflow_stalls(cfg, now))
    findings.extend(scan_task_output_stalls(cfg, now))
    if runner is not None:
        findings.extend(scan_runner_task_runs(runner, cfg, now))
        findings.extend(scan_ingress_timeouts(runner, cfg, now))
    # De-dup within one run by fingerprint (a unit can only be found once).
    unique: dict[str, Finding] = {}
    for f in findings:
        unique.setdefault(f.fingerprint, f)
    return list(unique.values())


def run(
    cfg: StallCheckConfig,
    brain: sqlite3.Connection,
    *,
    runner: sqlite3.Connection | None = None,
    now: datetime | None = None,
    message_prefix: str = "",
    send: bool = True,
) -> RunReport:
    """Execute one full stall-check pass: scan, feed the deadman engine, self-
    heartbeat, then page a single deduplicated digest of new findings."""
    now = now or datetime.now(UTC)
    ensure_stall_pages(brain)
    first_run = is_first_run(brain)
    report = RunReport(first_run=first_run)

    findings = collect_findings(cfg, now, runner=runner)
    report.findings = findings

    # Feed the deadman engine for EVERY finding (idempotent), every run.
    for finding in findings:
        report.evidence_ids.append(feed_deadman(brain, finding, now))

    backlog_cutoff = cfg.terminal_backlog_seconds
    new_findings = [f for f in findings if not already_paged(brain, f.fingerprint)]
    report.new_findings = new_findings

    if first_run:
        # One-time backlog digest: page everything new, including >48h terminals.
        candidates = list(new_findings)
    else:
        # Steady state: page only new findings that are not already terminal-and-old.
        candidates = [f for f in new_findings if f.age_seconds <= backlog_cutoff]
    candidates.sort(key=lambda f: f.age_seconds, reverse=True)
    # Bound each run to one digest of at most max_pages_per_run stalls; the rest
    # remain un-paged (paged_at NULL) and ride the next cycle.
    to_page = candidates[: max(0, cfg.max_pages_per_run)]
    report.paged = to_page

    # Send the digest, then figure out which fingerprints were actually delivered.
    delivered_fps: set[str] = set()
    pager_configured = bool(cfg.pager_chat_id and cfg.pager_openclaw_json)
    send_attempted = False
    if to_page and send:
        message, included = build_digest_message(
            to_page, prefix=message_prefix, backlog=first_run
        )
        send_attempted = pager_configured
        report.page_status = send_telegram(cfg, message)
        delivered = report.page_status is not None and 200 <= report.page_status < 300
        if delivered:
            delivered_fps = {f.fingerprint for f in to_page[:included]}

    # Ledger every new finding so run_count/last_seen advance; stamp paged_at ONLY
    # on genuine delivery, so an undelivered stall stays eligible for the next run.
    for finding in new_findings:
        record_page(brain, finding, now, paged=finding.fingerprint in delivered_fps)

    upsert_self_heartbeat(brain, cfg, now)
    # Retire the first-run backlog window only once the backlog has actually been
    # delivered (or there was nothing to deliver / the pager is inert). A failed
    # send leaves first_run set so the whole backlog — including >48h items — retries.
    send_failed = bool(to_page) and send_attempted and not delivered_fps
    if not send_failed:
        mark_first_run_done(brain, now)
    brain.commit()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocbrain.stallcheck", description=__doc__)
    parser.add_argument("--config", default=None, help="path to ocbrain config JSON")
    parser.add_argument(
        "--brain-db",
        default=os.environ.get("OCBRAIN_DB"),
        help="path to the ocbrain brain sqlite (defaults to $OCBRAIN_DB)",
    )
    parser.add_argument(
        "--message-prefix",
        default="",
        help="prefix prepended to the Telegram digest (e.g. a test marker)",
    )
    parser.add_argument(
        "--no-send", action="store_true", help="scan + record but do not page"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report findings without writing to the brain or paging",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    now = datetime.now(UTC)

    if args.dry_run:
        runner = None
        try:
            runner = open_runner_ro(cfg.runner_db)
        except sqlite3.Error:
            runner = None
        try:
            findings = collect_findings(cfg, now, runner=runner)
        finally:
            if runner is not None:
                runner.close()
        by_class: dict[str, int] = {}
        for f in findings:
            by_class[f.stall_class] = by_class.get(f.stall_class, 0) + 1
        print(f"[stallcheck] dry-run: {len(findings)} finding(s) {by_class}")
        for f in sorted(findings, key=lambda x: x.age_seconds, reverse=True):
            print(
                f"  {f.stall_class} {f.unit_id} idle={f.age_seconds / 3600:.1f}h "
                f"sig={f.terminal_signature} :: {f.snippet[:80]}"
            )
        return 0

    brain_path = args.brain_db
    if not brain_path:
        print("[stallcheck] no brain DB (set $OCBRAIN_DB or --brain-db)", file=sys.stderr)
        return 2
    brain = open_brain(brain_path)
    runner = None
    try:
        try:
            runner = open_runner_ro(cfg.runner_db)
        except sqlite3.Error:
            runner = None
        report = run(
            cfg,
            brain,
            runner=runner,
            now=now,
            message_prefix=args.message_prefix,
            send=not args.no_send,
        )
    finally:
        if runner is not None:
            runner.close()
        brain.close()

    status = "-" if report.page_status is None else str(report.page_status)
    print(
        f"[stallcheck] {now.isoformat()} findings={len(report.findings)} "
        f"new={len(report.new_findings)} paged={len(report.paged)} "
        f"first_run={report.first_run} page_status={status} counts={report.counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
