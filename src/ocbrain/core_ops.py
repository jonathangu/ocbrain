"""Explicit, local-only operations for the OCBrain core ledger.

This module intentionally has no imports from autopilot, dataset, judge,
embedding, teacher, or trainer code.  ``sync_core`` is a bounded one-shot
projection reconciliation; it cannot schedule follow-up work or make a hosted
call, regardless of configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ocbrain.core_v1 import (
    CORE_V1_TABLES,
    init_core_v1,
    is_core_v1,
    project_core_v1,
)
from ocbrain.db import connect
from ocbrain.events import rebuild_projection
from ocbrain.text import redact_secrets

CORE_TABLES: tuple[str, ...] = (
    "evidence",
    "knowledge",
    "knowledge_evidence",
    "retrieval_uses",
    "family_scores",
    "brain_events",
    "current_beliefs",
    "egress_audits",
    "signal_events",
    "harvest_watermarks",
    "projection_cursor",
    "search_index",
)

COMPANION_TABLES: tuple[str, ...] = (
    "loop_liveness",
    "judge_runs",
    "autopilot_runs",
    "dataset_examples",
    "dataset_sources",
    "dataset_grade_runs",
    "dataset_calibrations",
    "dataset_exports",
    "embed_runs",
)

FORBIDDEN_SYNC_STAGES: tuple[str, ...] = (
    "autopilot",
    "judge",
    "embed",
    "teacher",
    "dataset",
    "trainer",
    "network",
    "scheduler",
)


def _read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def logical_table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return counts for public logical tables, excluding FTS shadow tables."""
    if is_core_v1(conn):
        return {
            table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in CORE_V1_TABLES
            if _table_exists(conn, table) and table != "search_index"
        }
    return {
        table: int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table in (*CORE_TABLES, *COMPANION_TABLES)
        if _table_exists(conn, table)
    }


def _one_check(conn: sqlite3.Connection, pragma: str) -> str:
    row = conn.execute(pragma).fetchone()
    return str(row[0]) if row is not None else "missing"


def database_status(path: Path) -> dict[str, Any]:
    """Inspect an existing database without initializing or migrating it."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {
            "status": "missing",
            "path": str(resolved),
            "exists": False,
            "healthy": False,
        }
    try:
        conn = _read_only(resolved)
        try:
            quick_check = _one_check(conn, "PRAGMA quick_check(1)")
            foreign_key_rows = conn.execute("PRAGMA foreign_key_check").fetchmany(20)
            counts = logical_table_counts(conn)
            journal_mode = _one_check(conn, "PRAGMA journal_mode")
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            v1 = is_core_v1(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {
            "status": "error",
            "path": str(resolved),
            "exists": True,
            "healthy": False,
            "error": str(exc),
        }
    wal = Path(f"{resolved}-wal")
    shm = Path(f"{resolved}-shm")
    healthy = quick_check == "ok" and not foreign_key_rows
    return {
        "status": "ok" if healthy else "degraded",
        "path": str(resolved),
        "exists": True,
        "healthy": healthy,
        "integrity": quick_check,
        "foreign_key_violations": len(foreign_key_rows),
        "journal_mode": journal_mode,
        "user_version": user_version,
        "bytes": resolved.stat().st_size,
        "wal_bytes": wal.stat().st_size if wal.exists() else 0,
        "shm_bytes": shm.stat().st_size if shm.exists() else 0,
        "counts": counts,
        "core_schema": "ocbrain.core.v1" if v1 else "legacy",
        "core_rows": sum(counts.get(name, 0) for name in (CORE_V1_TABLES if v1 else CORE_TABLES)),
        "companion_rows": (0 if v1 else sum(counts.get(name, 0) for name in COMPANION_TABLES)),
    }


def sync_core(
    path: Path,
    *,
    max_events: int = 1_000,
    time_budget_seconds: float = 10.0,
) -> dict[str, Any]:
    """Boundedly reconcile the event projection, with no other work dispatch.

    The event count is checked before any projection write.  SQLite's progress
    handler enforces the wall-clock deadline for database work, and a savepoint
    makes an interrupted projection atomic.
    """
    if max_events < 0:
        raise ValueError("max_events must be non-negative")
    if time_budget_seconds <= 0:
        raise ValueError("time_budget_seconds must be positive")

    resolved = path.expanduser().resolve()
    created = not resolved.exists()
    conn = connect(resolved)
    try:
        if created:
            init_core_v1(conn)
        elif not _table_exists(conn, "brain_events") or not _table_exists(
            conn, "projection_cursor"
        ):
            raise ValueError(
                "existing database is not an initialized OCBrain ledger; run `ocbrain init` first"
            )

        max_rowid = int(
            conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM brain_events").fetchone()[0]
        )
        cursor_row = conn.execute(
            "SELECT last_event_rowid FROM projection_cursor WHERE id = 1"
        ).fetchone()
        cursor = (
            int(cursor_row["last_event_rowid"])
            if cursor_row is not None and cursor_row["last_event_rowid"] is not None
            else None
        )
        full_rebuild = cursor is None or cursor > max_rowid
        if full_rebuild:
            pending = int(conn.execute("SELECT COUNT(*) FROM brain_events").fetchone()[0])
        else:
            pending = int(
                conn.execute(
                    "SELECT COUNT(*) FROM brain_events WHERE rowid > ?", (cursor,)
                ).fetchone()[0]
            )

        policy = {
            "mode": "explicit_one_shot",
            "scheduled": False,
            "network_allowed": False,
            "hosted_calls": 0,
            "stages": ["event_projection", "sqlite_quick_check", "foreign_key_check"],
            "forbidden_stages": list(FORBIDDEN_SYNC_STAGES),
        }
        if pending > max_events:
            return {
                "action": "sync",
                "status": "bounded_refusal",
                "changed": False,
                "reason": "pending event count exceeds max_events",
                "pending_events": pending,
                "max_events": max_events,
                "projection_cursor": cursor,
                "event_head_rowid": max_rowid,
                "policy": policy,
            }

        deadline = time.monotonic() + time_budget_seconds

        def timed_out() -> int:
            return int(time.monotonic() >= deadline)

        conn.execute("SAVEPOINT core_sync")
        conn.set_progress_handler(timed_out, 1_000)
        try:
            if is_core_v1(conn):
                project_core_v1(conn, full=full_rebuild)
            else:
                rebuild_projection(conn, full=full_rebuild)
            quick_check = _one_check(conn, "PRAGMA quick_check(1)")
            foreign_key_rows = conn.execute("PRAGMA foreign_key_check").fetchmany(20)
            conn.execute("RELEASE SAVEPOINT core_sync")
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.set_progress_handler(None, 0)
            conn.execute("ROLLBACK TO SAVEPOINT core_sync")
            conn.execute("RELEASE SAVEPOINT core_sync")
            conn.rollback()
            if "interrupted" not in str(exc).lower():
                raise
            return {
                "action": "sync",
                "status": "time_budget_exceeded",
                "changed": False,
                "pending_events": pending,
                "max_events": max_events,
                "time_budget_seconds": time_budget_seconds,
                "policy": policy,
            }
        finally:
            conn.set_progress_handler(None, 0)

        after = conn.execute(
            "SELECT last_event_rowid FROM projection_cursor WHERE id = 1"
        ).fetchone()
        after_cursor = after["last_event_rowid"] if after is not None else None
        healthy = quick_check == "ok" and not foreign_key_rows
        return {
            "action": "sync",
            "status": "ok" if healthy else "degraded",
            "changed": pending > 0 or full_rebuild,
            "created": created,
            "processed_events": pending,
            "projection_cursor_before": cursor,
            "projection_cursor_after": after_cursor,
            "event_head_rowid": max_rowid,
            "integrity": quick_check,
            "foreign_key_violations": len(foreign_key_rows),
            "elapsed_seconds": round(time_budget_seconds - max(deadline - time.monotonic(), 0), 6),
            "policy": policy,
        }
    finally:
        conn.close()


def _default_mcp_command(smoke_db: Path) -> tuple[list[str], dict[str, str], str]:
    env = dict(os.environ)
    env["OCBRAIN_DB"] = str(smoke_db)
    env.setdefault("PYTHONUNBUFFERED", "1")
    root = Path(__file__).resolve().parents[2]
    launcher = root / "scripts" / "ocbrain-mcp"
    if launcher.is_file() and os.access(launcher, os.X_OK):
        return [str(launcher)], env, str(launcher)
    return (
        [sys.executable, "-m", "ocbrain.cli", "--db", str(smoke_db), "mcp"],
        env,
        f"{sys.executable} -m ocbrain.cli",
    )


def stdio_mcp_smoke(
    *,
    timeout_seconds: float = 8.0,
    launcher: Path | None = None,
) -> dict[str, Any]:
    """Start a real child MCP server and complete initialize/ping/tools-list."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    with tempfile.TemporaryDirectory(prefix="ocbrain-doctor-") as tmp:
        smoke_db = Path(tmp) / "smoke.sqlite"
        if launcher is None:
            command, env, transport = _default_mcp_command(smoke_db)
        else:
            resolved = launcher.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"MCP launcher not found: {resolved}")
            command = [str(resolved)]
            env = dict(os.environ)
            env["OCBRAIN_DB"] = str(smoke_db)
            env.setdefault("PYTHONUNBUFFERED", "1")
            transport = str(resolved)

        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}},
        ]
        payload = "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in requests)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                input=payload,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "timeout",
                "healthy": False,
                "transport": transport,
                "timeout_seconds": timeout_seconds,
                "stderr": redact_secrets((exc.stderr or "")[-2_000:]),
            }

        responses: dict[int, dict[str, Any]] = {}
        parse_errors: list[str] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                parse_errors.append(line[:200])
                continue
            if isinstance(response, dict) and isinstance(response.get("id"), int):
                responses[response["id"]] = response
        initialize = responses.get(1, {}).get("result", {})
        ping = responses.get(2, {}).get("result")
        tools_result = responses.get(3, {}).get("result", {})
        tools = tools_result.get("tools", []) if isinstance(tools_result, dict) else []
        tool_names = {
            item.get("name") for item in tools if isinstance(item, dict) and item.get("name")
        }
        required = {"brain.search", "brain.digest", "brain.feedback"}
        missing = sorted(required - tool_names)
        healthy = (
            completed.returncode == 0
            and initialize.get("serverInfo", {}).get("name") == "ocbrain"
            and ping == {}
            and not missing
            and not parse_errors
        )
        return {
            "status": "ok" if healthy else "failed",
            "healthy": healthy,
            "transport": transport,
            "protocol": "json-rpc-newline-stdio",
            "protocol_version": initialize.get("protocolVersion"),
            "server": initialize.get("serverInfo"),
            "tool_count": len(tool_names),
            "missing_required_tools": missing,
            "response_ids": sorted(responses),
            "returncode": completed.returncode,
            "parse_errors": parse_errors,
            "stderr": redact_secrets(completed.stderr[-2_000:]),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }


def _run_client_check(name: str, command: list[str], timeout_seconds: float) -> dict[str, Any]:
    binary = shutil.which(command[0])
    if binary is None:
        return {"name": name, "status": "missing", "healthy": False, "command": command}
    env = dict(os.environ)
    env.setdefault("NO_COLOR", "1")
    try:
        completed = subprocess.run(
            [binary, *command[1:]],
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "timeout",
            "healthy": False,
            "command": command,
            "stderr": redact_secrets((exc.stderr or "")[-2_000:]),
        }
    return {
        "name": name,
        "status": "ok" if completed.returncode == 0 else "failed",
        "healthy": completed.returncode == 0,
        "command": command,
        "returncode": completed.returncode,
        "stdout": redact_secrets(completed.stdout[-4_000:]),
        "stderr": redact_secrets(completed.stderr[-2_000:]),
    }


def runtime_client_checks(*, timeout_seconds: float = 12.0) -> list[dict[str, Any]]:
    """Run the documented, read-only registration/probe commands."""
    commands = (
        ("codex", ["codex", "mcp", "get", "ocbrain"]),
        ("claude", ["claude", "mcp", "list"]),
        ("openclaw-doctor", ["openclaw", "mcp", "doctor", "ocbrain"]),
        ("openclaw-probe", ["openclaw", "mcp", "probe", "ocbrain"]),
    )
    return [_run_client_check(name, command, timeout_seconds) for name, command in commands]


def local_control_file_security() -> dict[str, Any]:
    """Check optional local control files without reading or exposing contents."""
    candidates = {
        "active_db_pointer": Path(
            os.environ.get("OCBRAIN_ACTIVE_DB_FILE", "data/active-core.path")
        ).expanduser(),
        "config": Path(os.environ.get("OCBRAIN_CONFIG", "data/ocbrain.config.json")).expanduser(),
    }
    files: dict[str, dict[str, Any]] = {}
    healthy = True
    for name, path in candidates.items():
        if not path.exists():
            files[name] = {"status": "missing_optional", "secure": True}
            continue
        mode = path.stat().st_mode & 0o777
        secure = path.is_file() and mode & 0o077 == 0
        healthy = healthy and secure
        files[name] = {
            "status": "owner_only" if secure else "permissions_too_open",
            "secure": secure,
            "mode": f"{mode:04o}",
        }
    return {"healthy": healthy, "files": files}


def doctor(
    path: Path,
    *,
    timeout_seconds: float = 8.0,
    launcher: Path | None = None,
    check_clients: bool = False,
) -> dict[str, Any]:
    db = database_status(path)
    mcp = stdio_mcp_smoke(timeout_seconds=timeout_seconds, launcher=launcher)
    clients = runtime_client_checks(timeout_seconds=timeout_seconds) if check_clients else []
    local_files = local_control_file_security()
    healthy = (
        bool(db.get("healthy")) and bool(mcp.get("healthy")) and bool(local_files.get("healthy"))
    )
    if check_clients:
        healthy = healthy and all(item["healthy"] for item in clients)
    return {
        "action": "runtime-check" if check_clients else "doctor",
        "status": "ok" if healthy else "failed",
        "healthy": healthy,
        "database": db,
        "mcp_stdio": mcp,
        "clients": clients,
        "local_control_files": local_files,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_target(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() or Path(f"{resolved}-wal").exists() or Path(f"{resolved}-shm").exists():
        raise FileExistsError(f"{label} must be a fresh path: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _online_copy(source: Path, destination: Path) -> None:
    source_conn = _read_only(source)
    destination_conn = sqlite3.connect(destination)
    try:
        source_conn.backup(destination_conn, pages=2_048, sleep=0.01)
    finally:
        destination_conn.close()
        source_conn.close()
    os.chmod(destination, 0o600)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def backup_database(
    source: Path,
    destination: Path,
    *,
    manifest: Path | None = None,
) -> dict[str, Any]:
    """Create and verify a transaction-consistent SQLite backup at a fresh path."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source database not found: {source}")
    destination = _fresh_target(destination, label="backup destination")
    if source == destination:
        raise ValueError("backup destination must differ from source")
    manifest = (
        manifest.expanduser().resolve()
        if manifest is not None
        else Path(f"{destination}.manifest.json")
    )
    _fresh_target(manifest, label="backup manifest")
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        _online_copy(source, temp)
        report = database_status(temp)
        if not report.get("healthy"):
            raise RuntimeError(f"backup verification failed: {report}")
        payload = {
            "format": "ocbrain-backup-v1",
            "source": str(source),
            "backup": str(destination),
            "sha256": sha256_file(temp),
            "bytes": temp.stat().st_size,
            "integrity": report["integrity"],
            "foreign_key_violations": report["foreign_key_violations"],
            "counts": report["counts"],
        }
        os.replace(temp, destination)
        _atomic_json(manifest, payload)
        return payload | {"manifest": str(manifest)}
    finally:
        temp.unlink(missing_ok=True)


def restore_database(
    backup: Path,
    destination: Path,
    *,
    manifest: Path | None = None,
) -> dict[str, Any]:
    """Verify a backup and restore it to a fresh path; never replace a live DB."""
    backup = backup.expanduser().resolve()
    if not backup.is_file():
        raise FileNotFoundError(f"backup database not found: {backup}")
    destination = _fresh_target(destination, label="restore destination")
    if backup == destination:
        raise ValueError("restore destination must differ from backup")
    manifest_path = manifest.expanduser().resolve() if manifest else Path(f"{backup}.manifest.json")
    expected_sha: str | None = None
    if manifest_path.is_file():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest_payload.get("sha256")
        actual_sha = sha256_file(backup)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError("backup SHA-256 does not match its manifest")
    elif manifest is not None:
        raise FileNotFoundError(f"backup manifest not found: {manifest_path}")

    source_report = database_status(backup)
    if not source_report.get("healthy"):
        raise RuntimeError(f"backup database failed verification: {source_report}")
    temp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        _online_copy(backup, temp)
        restored_report = database_status(temp)
        if not restored_report.get("healthy"):
            raise RuntimeError(f"restored database failed verification: {restored_report}")
        if restored_report["counts"] != source_report["counts"]:
            raise RuntimeError("restored logical table counts do not match backup")
        os.replace(temp, destination)
        return {
            "format": "ocbrain-restore-v1",
            "backup": str(backup),
            "destination": str(destination),
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
            "manifest_sha256_verified": expected_sha is not None,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "integrity": restored_report["integrity"],
            "foreign_key_violations": restored_report["foreign_key_violations"],
            "counts": restored_report["counts"],
            "live_database_replaced": False,
        }
    finally:
        temp.unlink(missing_ok=True)


__all__ = [
    "COMPANION_TABLES",
    "CORE_TABLES",
    "FORBIDDEN_SYNC_STAGES",
    "backup_database",
    "database_status",
    "doctor",
    "logical_table_counts",
    "restore_database",
    "runtime_client_checks",
    "sha256_file",
    "stdio_mcp_smoke",
    "sync_core",
]
