"""OCBrain's optional, explicitly invoked progress watchdog.

Jonathan's agents sometimes end a work turn *waiting* ("standing by",
"waiting on the monitor") and then never move again because the notification
they expect never arrives. Nobody notices until he pings to follow up. This
module is that follow-up. On explicit invocation it sweeps the agent exhaust
for the signature of a parked-and-forgotten turn, records the finding in the
companion-owned ops ledger, and can send Jonathan ONE Telegram digest of any
*new* stalls (deduplicated so a persistent stall pages exactly once). The
package does not install or enable a scheduler.

Four readers feed it:

  READER A (filesystem) — the money reader. Scans subagent workflow dirs for
    ``agent-*.jsonl`` transcripts whose LAST record is an assistant ``end_turn``
    and flags a parked-and-forgotten turn via two independent signals:
      (1) PRIMARY, lexicon-free — a monitor / background-task ``tool_use`` appears
          in the trailing N records with no later ``tool_result`` / user event,
          i.e. the turn closed while a monitor was still pending. This catches
          every passive-wait paraphrase with zero text matching.
      (2) SECONDARY, regex lexicon — the end_turn text matches a case-insensitive
          passive-wait regex family (generalizer + high-precision seeds).
    It also flags task ``.output`` files that are zero-byte or opened (``start:``)
    but never closed (``exit:``). A workflow whose journal shows *recent* result
    activity, or whose files were *recently* appended, is alive and never flagged.

  READER B (sqlite, read-only) — runner ``task_runs`` that are ``lost`` or have
    been ``running``/``pending`` with no event past the stale threshold.

  READER C (sqlite, read-only) — ``channel_ingress_events`` that failed with
    ``handler-timeout`` inside the lookback window (dropped inbound work).

  READER D (optional read-only legacy/core sqlite) — overdue ``loop_liveness``
    deadmen and archived autopilot run failures, when those tables exist.

Every finding is written only to the companion-owned ops ledger.  An optional
v1 core path is opened read-only to inspect old deadman state; the watchdog never
writes knowledge or evidence into the brain.

SECURITY: the Telegram bot token is read from ``openclaw.json`` at send time and
is NEVER printed, logged, or stored. The committed config defaults are empty —
without a local ``stall_pager`` config (chat id + openclaw.json path) the pager
is inert and the module only scans + records.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ocbrain.ids import content_hash, stable_id

from ocbrain_ops.store import DEFAULT_OPS_DB, connect_ops

# --- Brain connection hardening ------------------------------------------------
# The companion ledger may have another explicitly invoked writer. Wait rather
# than fail fast on a short-lived SQLite lock.
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
# next invocation), and a hard character budget below the API ceiling.
DEFAULT_MAX_PAGES_PER_RUN = 8
MAX_MESSAGE_CHARS = 3900

_HOME = Path.home()
DEFAULT_WORKFLOW_GLOBS: tuple[str, ...] = (
    str(_HOME / ".claude/projects/*/*/subagents/workflows/*/"),
)
DEFAULT_TASK_OUTPUT_GLOBS: tuple[str, ...] = (f"/private/tmp/claude-{os.getuid()}/*/*/tasks/",)
DEFAULT_RUNNER_DB = str(_HOME / ".openclaw/state/openclaw.sqlite")

# Secondary signal. Case-insensitive REGEX families (not exact substrings), so
# paraphrases like "waiting for the clean live run to land via the monitor" match
# without an exact seed. The leading family pattern is the generalizer; the
# remaining entries preserve the original high-precision seeds. Any operator-
# supplied ``passive_wait_lexicon`` entries are likewise treated as regex.
DEFAULT_PASSIVE_WAIT_LEXICON: tuple[str, ...] = (
    # <wait/hold/standing-by verb> … within 80 chars … <arrival/callback noun>
    r"\b(wait|waiting|waits|hold|holding|holds|stand(?:ing)? by|park(?:ed|ing)?"
    r"|block(?:ed|ing)?|pause[ds]?)\b.{0,80}\b(monitor|notif(?:y|ication|ies)"
    r"|callback|ping|land(?:s|ed|ing)?|complete[sd]?|completion|arriv(?:e|es|al)"
    r"|resolve[sd]?|finish(?:es|ed)?|come[s]? back|return[s]?)\b",
    r"\bwaiting on the monitor\b",
    r"\bstanding by\b",
    r"\bi'?ll hold here\b",
    r"\bno further action is useful until\b",
    r"\blet the monitor notify me\b",
    r"\bwaiting for the monitor\b",
)

# Primary signal. Tool-use names (or a truthy ``run_in_background`` input) that
# denote a background/monitor call whose result the turn is waiting on. Matched
# case-insensitively against the tool_use ``name``.
DEFAULT_MONITOR_TOOL_PATTERN = r"monitor|background|spawn_task|remote_?trigger|push_?notification"

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
    # Primary (lexicon-free) signal: flag a turn that closed while a monitor /
    # background-task tool_use was still pending (no later tool_result / user
    # event). Config-gated, default on. ``monitor_tool_pattern`` is the regex
    # that identifies a monitor-ish tool_use name; ``trailing_record_window`` is
    # how many tail records of the transcript the scan inspects.
    pending_monitor_signal: bool = True
    monitor_tool_pattern: str = DEFAULT_MONITOR_TOOL_PATTERN
    trailing_record_window: int = 12
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
    autopilot_failure_window_hours: int = 48
    autopilot_running_stale_minutes: int = 300
    judge_failure_streak: int = 2
    daily_canary_enabled: bool = False
    daily_canary_hour_utc: int = 18

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
        config_path or os.environ.get("OCBRAIN_CONFIG", "data/ocbrain.config.json")
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
        pending_monitor_signal=bool(sc.get("pending_monitor_signal", True)),
        monitor_tool_pattern=str(sc.get("monitor_tool_pattern", DEFAULT_MONITOR_TOOL_PATTERN)),
        trailing_record_window=int(sc.get("trailing_record_window", 12)),
        stale_threshold_minutes=int(sc.get("stale_threshold_minutes", DEFAULT_STALE_MINUTES)),
        terminal_backlog_hours=int(
            sc.get("terminal_backlog_hours", DEFAULT_TERMINAL_BACKLOG_HOURS)
        ),
        ingress_window_hours=int(sc.get("ingress_window_hours", DEFAULT_INGRESS_WINDOW_HOURS)),
        self_interval_seconds=int(sc.get("self_interval_seconds", DEFAULT_SELF_INTERVAL_SECONDS)),
        max_pages_per_run=int(sc.get("max_pages_per_run", DEFAULT_MAX_PAGES_PER_RUN)),
        flag_zero_byte_output=bool(sc.get("flag_zero_byte_output", True)),
        autopilot_failure_window_hours=int(sc.get("autopilot_failure_window_hours", 48)),
        autopilot_running_stale_minutes=int(sc.get("autopilot_running_stale_minutes", 300)),
        judge_failure_streak=int(sc.get("judge_failure_streak", 2)),
        daily_canary_enabled=bool(sc.get("daily_canary_enabled", False)),
        daily_canary_hour_utc=int(sc.get("daily_canary_hour_utc", 18)),
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
        return stable_id("stall", self.stall_class, self.unit_id, self.terminal_signature)

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
def _tail_jsonl_records(path: Path, n: int) -> list[dict[str, Any]]:
    """Return the last ``n`` non-blank JSON dict records of a .jsonl file."""
    records: list[dict[str, Any]] = []
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
                    records.append(obj)
                    if n > 0 and len(records) > n:
                        records.pop(0)
    except OSError:
        return []
    return records


def _content_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    if not isinstance(message, dict):
        return []
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _is_user_record(record: dict[str, Any]) -> bool:
    if record.get("type") == "user":
        return True
    message = record.get("message")
    return isinstance(message, dict) and message.get("role") == "user"


def _has_tool_result(record: dict[str, Any]) -> bool:
    return any(block.get("type") == "tool_result" for block in _content_blocks(record))


def _is_assistant_end_turn(record: dict[str, Any]) -> bool:
    message = record.get("message")
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("stop_reason") == "end_turn"
    )


def _tool_use_is_monitor(block: dict[str, Any], monitor_re: re.Pattern[str]) -> bool:
    if block.get("type") != "tool_use":
        return False
    if monitor_re.search(str(block.get("name", ""))):
        return True
    inp = block.get("input")
    return isinstance(inp, dict) and bool(inp.get("run_in_background"))


def detect_pending_monitor(
    records: list[dict[str, Any]], monitor_re: re.Pattern[str]
) -> str | None:
    """LEXICON-FREE primary signal. Return the name of a monitor / background-task
    ``tool_use`` in ``records`` whose result never lands — i.e. no later record is
    a user event or carries a ``tool_result`` — else ``None``. Catches every
    passive-wait paraphrase with zero text matching."""
    total = len(records)
    for i, record in enumerate(records):
        for block in _content_blocks(record):
            if not _tool_use_is_monitor(block, monitor_re):
                continue
            resolved_later = any(
                _is_user_record(records[j]) or _has_tool_result(records[j])
                for j in range(i + 1, total)
            )
            if not resolved_later:
                return str(block.get("name") or "monitor")
    return None


def _compile_lexicon(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def match_passive_wait(text: str, compiled: list[re.Pattern[str]]) -> str | None:
    """SECONDARY signal. Return the matched (normalized) span if any passive-wait
    regex family matches ``text``, else ``None``."""
    flat = " ".join(text.split())
    for pat in compiled:
        found = pat.search(flat)
        if found:
            return _snip(found.group(0), 80).lower()
    return None


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
    compiled = _compile_lexicon(cfg.passive_wait_lexicon)
    monitor_re = re.compile(cfg.monitor_tool_pattern, re.IGNORECASE)
    window = cfg.trailing_record_window
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
                records = _tail_jsonl_records(agent_file, window)
                # Both signals require the turn to have CLOSED on an assistant
                # end_turn (a parked-and-forgotten turn), not a still-running one.
                if not records or not _is_assistant_end_turn(records[-1]):
                    continue
                text = _assistant_end_turn_text(records[-1]) or ""

                signature: str | None = None
                # PRIMARY (lexicon-free): turn closed with a monitor pending.
                if cfg.pending_monitor_signal:
                    monitor = detect_pending_monitor(records, monitor_re)
                    if monitor:
                        signature = f"pending_monitor:{monitor}"
                # SECONDARY (regex lexicon): passive-wait phrasing in the text.
                if signature is None:
                    matched = match_passive_wait(text, compiled)
                    if matched is not None:
                        signature = f"passive_wait:{matched}"
                if signature is None:
                    continue

                snippet = _snip(text) or "turn closed while a background monitor was pending"
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
                        terminal_signature=signature,
                        snippet=snippet,
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
        note = row["progress_summary"] or row["error"] or row["label"] or row["task"] or ""
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


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def scan_brain_deadmans(brain: sqlite3.Connection, now: datetime) -> list[Finding]:
    """READER D — overdue producer heartbeats in the shared brain ledger.

    Rows created *by* stall findings use a ``stall/`` loop id and are excluded,
    otherwise the observer would recursively report its own reports. A stable
    state hash changes only after the producer makes progress, so one stuck
    run pages once while a later, genuinely new stall on the same profile can
    page again.
    """
    findings: list[Finding] = []
    rows = brain.execute(
        """
        SELECT loop_id, run_id, last_heartbeat_at, last_ledger_write_at,
               deadman_due_at
        FROM loop_liveness
        WHERE deadman_due_at IS NOT NULL
          AND loop_id NOT LIKE 'stall/%'
        ORDER BY deadman_due_at, loop_id, run_id
        """
    ).fetchall()
    for row in rows:
        due_at = _parse_iso(row["deadman_due_at"])
        if due_at is None or due_at > now:
            continue
        heartbeat = _parse_iso(row["last_heartbeat_at"])
        ledger_write = _parse_iso(row["last_ledger_write_at"])
        heartbeat_stale = heartbeat is None or heartbeat <= due_at
        ledger_stale = ledger_write is None or ledger_write <= due_at
        if not heartbeat_stale and not ledger_stale:
            continue
        loop_id = str(row["loop_id"])
        run_id = str(row["run_id"] or "unknown-run")
        state = json.dumps(
            {
                "loop_id": loop_id,
                "run_id": run_id,
                "heartbeat": row["last_heartbeat_at"],
                "ledger": row["last_ledger_write_at"],
            },
            sort_keys=True,
        )
        stale_parts = []
        if heartbeat_stale:
            stale_parts.append("heartbeat")
        if ledger_stale:
            stale_parts.append("ledger")
        findings.append(
            Finding(
                stall_class="loop_deadman",
                unit_id=f"{loop_id}/{run_id}",
                terminal_signature=f"deadman:{content_hash(state)[:16]}",
                snippet=(
                    f"{loop_id} {run_id} crossed its deadman with "
                    f"{' and '.join(stale_parts)} progress stale"
                ),
                artifact_path=f"brain:loop_liveness/{loop_id}/{run_id}",
                age_seconds=max(0.0, (now - due_at).total_seconds()),
                occurred_at=due_at.isoformat(),
            )
        )
    return findings


def _autopilot_stage_errors(stages_json: str | None) -> list[tuple[str, str]]:
    """Return leaf stage/sub-stage errors from one run ledger payload."""
    if not stages_json:
        return []
    try:
        payload = json.loads(stages_json)
    except (TypeError, json.JSONDecodeError):
        return [("ledger", "malformed stages_json")]
    found: list[tuple[str, str]] = []

    def walk(value: Any, path: str) -> None:
        if not isinstance(value, dict):
            return
        error = value.get("error")
        if error:
            found.append((path or "run", str(error)))
        for key in ("stages",):
            child = value.get(key)
            if isinstance(child, dict):
                for name, item in child.items():
                    walk(item, f"{path}.{name}" if path else str(name))

    for name, item in payload.items() if isinstance(payload, dict) else []:
        walk(item, str(name))
    return found


def scan_autopilot_failures(
    brain: sqlite3.Connection,
    cfg: StallCheckConfig,
    now: datetime,
) -> list[Finding]:
    """Page completed partial/error runs, stale running rows, and judge streaks."""
    findings: list[Finding] = []
    cutoff = now.timestamp() - max(1, cfg.autopilot_failure_window_hours) * 3600
    rows = brain.execute(
        """
        SELECT id, started_at, finished_at, status, stages_json, error
        FROM autopilot_runs
        ORDER BY started_at DESC
        LIMIT 200
        """
    ).fetchall()
    recent = []
    for row in rows:
        started = _parse_iso(row["started_at"])
        if started is None or started.timestamp() < cutoff:
            continue
        recent.append(row)
        status = str(row["status"] or "unknown")
        finished = _parse_iso(row["finished_at"])
        errors = _autopilot_stage_errors(row["stages_json"])
        if status in {"partial", "error", "failed"}:
            detail = "; ".join(f"{path}: {error}" for path, error in errors[:3])
            detail = detail or str(row["error"] or "run did not complete cleanly")
            occurred = finished or started
            findings.append(
                Finding(
                    stall_class="autopilot_failed",
                    unit_id=str(row["id"]),
                    terminal_signature=f"{status}:{content_hash(detail)[:16]}",
                    snippet=f"scheduled run ended {status}: {detail}",
                    artifact_path=f"brain:autopilot_runs/{row['id']}",
                    age_seconds=max(0.0, (now - occurred).total_seconds()),
                    occurred_at=occurred.isoformat(),
                )
            )
        elif status == "running":
            stale_after = max(1, cfg.autopilot_running_stale_minutes) * 60
            age = max(0.0, (now - started).total_seconds())
            if age >= stale_after:
                findings.append(
                    Finding(
                        stall_class="autopilot_stuck_running",
                        unit_id=str(row["id"]),
                        terminal_signature=f"running:{row['started_at']}",
                        snippet=(
                            "scheduled run remains running beyond its allowed observation window"
                        ),
                        artifact_path=f"brain:autopilot_runs/{row['id']}",
                        age_seconds=age,
                        occurred_at=started.isoformat(),
                    )
                )

    streak_target = max(2, cfg.judge_failure_streak)
    completed = [row for row in recent if row["status"] != "running"]
    judge_failures: list[tuple[sqlite3.Row, str]] = []
    for row in completed:
        matches = [
            error
            for path, error in _autopilot_stage_errors(row["stages_json"])
            if path.endswith(".judge")
        ]
        if not matches:
            break
        judge_failures.append((row, matches[0]))
        if len(judge_failures) >= streak_target:
            latest, latest_error = judge_failures[0]
            findings.append(
                Finding(
                    stall_class="judge_failure_streak",
                    unit_id="autopilot/judge",
                    terminal_signature=f"{latest['id']}:{content_hash(latest_error)[:16]}",
                    snippet=(
                        f"judge failed in {len(judge_failures)} consecutive completed "
                        f"runs: {latest_error}"
                    ),
                    artifact_path=f"brain:autopilot_runs/{latest['id']}",
                    age_seconds=max(
                        0.0,
                        (now - (_parse_iso(latest["finished_at"]) or now)).total_seconds(),
                    ),
                    occurred_at=(_parse_iso(latest["finished_at"]) or now).isoformat(),
                )
            )
            break
    return findings


# --- Brain writes: deadman engine + pager ledger -------------------------------
def open_brain(path: Path | str) -> sqlite3.Connection:
    """Compatibility name for opening the companion-owned ops ledger."""
    conn = connect_ops(Path(path))
    conn.execute(f"PRAGMA busy_timeout={BRAIN_BUSY_TIMEOUT_MS}")
    return conn


def open_brain_ro(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# The ops ledger may have another explicit writer. busy_timeout makes a single
# statement wait out a lock, but if
# the competing writer holds its transaction longer than one busy_timeout
# window, that statement still raises OperationalError. Mirrors mcp.py's
# _call_tool_with_lock_retry: bound-retry on 'database is locked', rolling
# back the (aborted, never-partially-applied) transaction between attempts.
# Every brain write in this module is an idempotent upsert (ON CONFLICT DO
# UPDATE), so re-running a call that aborted mid-write is safe.
WRITE_LOCK_RETRIES = 3
WRITE_LOCK_BACKOFF_SECONDS = 0.25


def _write_with_lock_retry(conn: sqlite3.Connection, fn, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs), bound-retrying on 'database is locked'."""
    for attempt in range(WRITE_LOCK_RETRIES):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == WRITE_LOCK_RETRIES - 1:
                raise
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            time.sleep(WRITE_LOCK_BACKOFF_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


STALL_PAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS loop_liveness (
  loop_id TEXT NOT NULL,
  run_id TEXT,
  last_heartbeat_at TEXT,
  last_ledger_write_at TEXT,
  expected_interval_seconds INTEGER,
  deadman_due_at TEXT,
  PRIMARY KEY (loop_id, run_id)
);
CREATE TABLE IF NOT EXISTS stall_pages (
  fingerprint TEXT PRIMARY KEY,
  stall_class TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  terminal_signature TEXT,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  paged_at TEXT,
  retired_at TEXT,
  retire_reason TEXT,
  run_count INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS watchdog_findings (
  id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL,
  stall_class TEXT NOT NULL,
  unit_id TEXT NOT NULL,
  terminal_signature TEXT,
  snippet TEXT NOT NULL,
  artifact_path TEXT,
  occurred_at TEXT,
  observed_at TEXT NOT NULL,
  UNIQUE(fingerprint)
);
"""

_FIRSTRUN_MARKER = "__firstrun__"


def ensure_stall_pages(conn: sqlite3.Connection) -> None:
    conn.executescript(STALL_PAGES_SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(stall_pages)")}
    if "retired_at" not in columns:
        conn.execute("ALTER TABLE stall_pages ADD COLUMN retired_at TEXT")
    if "retire_reason" not in columns:
        conn.execute("ALTER TABLE stall_pages ADD COLUMN retire_reason TEXT")


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


def record_canary_delivery(conn: sqlite3.Connection, key: str, now: datetime) -> None:
    stamp = now.isoformat()
    conn.execute(
        """
        INSERT INTO stall_pages (
          fingerprint, stall_class, unit_id, terminal_signature,
          first_seen_at, last_seen_at, paged_at, run_count
        ) VALUES (?, 'pager_canary', ?, ?, ?, ?, ?, 1)
        ON CONFLICT(fingerprint) DO UPDATE SET
          last_seen_at=excluded.last_seen_at,
          paged_at=excluded.paged_at,
          run_count=stall_pages.run_count + 1
        """,
        (key, now.date().isoformat(), key, stamp, stamp, stamp),
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


def already_handled(conn: sqlite3.Connection, fingerprint: str) -> bool:
    """True when a finding was delivered or deliberately retired.

    Old (> backlog window) findings discovered after bootstrap are not paged,
    but they must not remain "new" forever and rewrite their ledger on every
    invocation. Failed/disabled delivery still has both fields NULL and remains
    eligible for retry.
    """
    row = conn.execute(
        """
        SELECT 1 FROM stall_pages
        WHERE fingerprint = ?
          AND (paged_at IS NOT NULL OR retired_at IS NOT NULL)
        LIMIT 1
        """,
        (fingerprint,),
    ).fetchone()
    return row is not None


def record_page(
    conn: sqlite3.Connection,
    finding: Finding,
    now: datetime,
    *,
    paged: bool,
    retire_reason: str | None = None,
) -> None:
    stamp = now.isoformat()
    conn.execute(
        """
        INSERT INTO stall_pages (
          fingerprint, stall_class, unit_id, terminal_signature,
          first_seen_at, last_seen_at, paged_at, retired_at, retire_reason, run_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(fingerprint) DO UPDATE SET
          last_seen_at = excluded.last_seen_at,
          paged_at = COALESCE(stall_pages.paged_at, excluded.paged_at),
          retired_at = COALESCE(stall_pages.retired_at, excluded.retired_at),
          retire_reason = COALESCE(stall_pages.retire_reason, excluded.retire_reason),
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
            stamp if retire_reason else None,
            retire_reason,
        ),
    )


def feed_deadman(conn: sqlite3.Connection, finding: Finding, now: datetime) -> str:
    """Upsert a deadman plus companion-local watchdog finding."""
    if (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='watchdog_findings'"
        ).fetchone()
        is None
    ):
        ensure_stall_pages(conn)
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
    finding_id = stable_id("watch", finding.fingerprint)
    conn.execute(
        """
        INSERT INTO watchdog_findings (
          id, fingerprint, stall_class, unit_id, terminal_signature,
          snippet, artifact_path, occurred_at, observed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
          snippet = excluded.snippet,
          artifact_path = excluded.artifact_path,
          occurred_at = excluded.occurred_at,
          observed_at = excluded.observed_at
        """,
        (
            finding_id,
            finding.fingerprint,
            finding.stall_class,
            finding.unit_id,
            finding.terminal_signature,
            finding.snippet,
            finding.artifact_path,
            finding.occurred_at,
            stamp,
        ),
    )
    return finding_id


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
    account_cfg = data.get("channels", {}).get("telegram", {}).get("accounts", {}).get(account, {})
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
    retired: list[Finding] = field(default_factory=list)
    first_run: bool = False
    page_status: int | None = None
    canary_status: int | None = None
    finding_ids: list[str] = field(default_factory=list)

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
    brain: sqlite3.Connection | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(scan_workflow_stalls(cfg, now))
    findings.extend(scan_task_output_stalls(cfg, now))
    if runner is not None:
        findings.extend(scan_runner_task_runs(runner, cfg, now))
        findings.extend(scan_ingress_timeouts(runner, cfg, now))
    if brain is not None:
        findings.extend(scan_brain_deadmans(brain, now))
        findings.extend(scan_autopilot_failures(brain, cfg, now))
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
    source_brain: sqlite3.Connection | None = None,
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

    findings = collect_findings(
        cfg,
        now,
        runner=runner,
        brain=source_brain if source_brain is not None else brain,
    )
    report.findings = findings

    # Feed the deadman engine for EVERY finding (idempotent), every run.
    for finding in findings:
        report.finding_ids.append(_write_with_lock_retry(brain, feed_deadman, brain, finding, now))
    # Companion findings are durable before an optional Telegram request, and the
    # pager's network latency never owns SQLite's writer slot.
    _write_with_lock_retry(brain, brain.commit)

    backlog_cutoff = cfg.terminal_backlog_seconds
    new_findings = [f for f in findings if not already_handled(brain, f.fingerprint)]
    report.new_findings = new_findings

    if first_run:
        # One-time backlog digest: page everything new, including >48h terminals.
        candidates = list(new_findings)
    else:
        # Steady state: page only new findings that are not already terminal-and-old.
        candidates = [f for f in new_findings if f.age_seconds <= backlog_cutoff]
        report.retired = [f for f in new_findings if f.age_seconds > backlog_cutoff]
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
        message, included = build_digest_message(to_page, prefix=message_prefix, backlog=first_run)
        send_attempted = pager_configured
        report.page_status = send_telegram(cfg, message)
        delivered = report.page_status is not None and 200 <= report.page_status < 300
        if delivered:
            delivered_fps = {f.fingerprint for f in to_page[:included]}

    canary_key = f"__canary__:{now.date().isoformat()}"
    canary_due = (
        cfg.daily_canary_enabled
        and now.hour >= min(max(cfg.daily_canary_hour_utc, 0), 23)
        and not already_paged(brain, canary_key)
    )
    if canary_due and send:
        digest_proved_delivery = bool(delivered_fps)
        if digest_proved_delivery:
            report.canary_status = report.page_status
        elif not to_page:
            report.canary_status = send_telegram(
                cfg,
                f"OCBrain pager canary — delivery path healthy for {now.date().isoformat()} UTC.",
            )
        canary_delivered = report.canary_status is not None and 200 <= report.canary_status < 300
        if canary_delivered:
            _write_with_lock_retry(brain, record_canary_delivery, brain, canary_key, now)

    # Ledger every new finding so run_count/last_seen advance; stamp paged_at ONLY
    # on genuine delivery, so an undelivered stall stays eligible for the next run.
    retired_fps = {finding.fingerprint for finding in report.retired}
    for finding in new_findings:
        _write_with_lock_retry(
            brain,
            record_page,
            brain,
            finding,
            now,
            paged=finding.fingerprint in delivered_fps,
            retire_reason=(
                "outside_backlog_window" if finding.fingerprint in retired_fps else None
            ),
        )

    _write_with_lock_retry(brain, upsert_self_heartbeat, brain, cfg, now)
    # Retire the first-run backlog window only once the backlog has actually been
    # delivered (or there was nothing to deliver / the pager is inert). A failed
    # send leaves first_run set so the whole backlog — including >48h items — retries.
    send_failed = bool(to_page) and send_attempted and not delivered_fps
    if not send_failed:
        _write_with_lock_retry(brain, mark_first_run_done, brain, now)
    _write_with_lock_retry(brain, brain.commit)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocbrain-watchdog", description=__doc__)
    parser.add_argument("--config", default=None, help="path to ocbrain config JSON")
    parser.add_argument(
        "--ops-db",
        type=Path,
        default=DEFAULT_OPS_DB,
        help="companion-owned watchdog ledger (default: ~/.ocbrain/ops.sqlite)",
    )
    parser.add_argument(
        "--core-db",
        type=Path,
        help="optional v1 core to inspect read-only",
    )
    parser.add_argument("--brain-db", dest="core_db", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--message-prefix",
        default="",
        help="prefix prepended to the Telegram digest (e.g. a test marker)",
    )
    parser.add_argument("--no-send", action="store_true", help="scan + record but do not page")
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
        brain = None
        try:
            runner = open_runner_ro(cfg.runner_db)
        except sqlite3.Error:
            runner = None
        if args.core_db:
            try:
                brain = open_brain_ro(args.core_db)
            except sqlite3.Error:
                brain = None
        try:
            findings = collect_findings(cfg, now, runner=runner, brain=brain)
        finally:
            if runner is not None:
                runner.close()
            if brain is not None:
                brain.close()
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

    brain = open_brain(args.ops_db)
    source_brain = None
    runner = None
    report = None
    lock_skip = False
    try:
        try:
            runner = open_runner_ro(cfg.runner_db)
        except sqlite3.Error:
            runner = None
        if args.core_db:
            try:
                source_brain = open_brain_ro(args.core_db)
            except sqlite3.Error:
                source_brain = None
        try:
            report = run(
                cfg,
                brain,
                runner=runner,
                source_brain=source_brain,
                now=now,
                message_prefix=args.message_prefix,
                send=not args.no_send,
            )
        except sqlite3.OperationalError as exc:
            # _write_with_lock_retry already bound-retried every brain write.
            # If the brain is STILL busy after that budget (a competing
            # writer — e.g. autopilot's multi-minute review/tripwires stage —
            # held the lock the whole window), don't crash the watchman: every
            # write here is an idempotent upsert over freshly re-scanned
            # findings, so skipping this cycle loses nothing — the next
            # later invocation re-feeds it. A clean lock-budget skip keeps the
            # explicitly invoked process from being reported as a false crash.
            if "database is locked" not in str(exc).lower():
                raise
            lock_skip = True
    finally:
        if runner is not None:
            runner.close()
        if source_brain is not None:
            source_brain.close()
        brain.close()

    if lock_skip:
        print(
            f"[stallcheck] {now.isoformat()} SKIPPED: brain database busy after "
            f"bound retries, will retry next cycle"
        )
        return 0

    status = "-" if report.page_status is None else str(report.page_status)
    print(
        f"[stallcheck] {now.isoformat()} findings={len(report.findings)} "
        f"new={len(report.new_findings)} paged={len(report.paged)} "
        f"retired={len(report.retired)} "
        f"first_run={report.first_run} page_status={status} counts={report.counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
