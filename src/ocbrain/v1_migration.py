"""Archive-first streaming migration into the event-authoritative v1 core.

The source database is never modified.  One consistent online archive snapshot
becomes the sole migration input.  Core, training, and operations databases are
built at fresh temporary paths, verified, and only then atomically published.
Activation is always a separate operator action.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ocbrain.core_ops import (
    _atomic_json,
    _fresh_target,
    _online_copy,
    _read_only,
    _table_exists,
    sha256_file,
)
from ocbrain.core_v1 import (
    CORE_V1_FTS_TABLES,
    CORE_V1_TABLES,
    assert_core_v1_inventory,
    canonical_json,
    conservative_legacy_scope,
    init_core_v1,
    project_core_v1,
    rebuild_core_v1_search,
    set_core_v1_search_triggers,
    sha256_text,
    verify_event_chain,
)
from ocbrain.ids import stable_id

MIGRATION_FORMAT = "ocbrain-core-v1-migration"
IMPORT_SCHEMA_VERSION = "ocbrain.legacy-import.v1"
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_PROGRESS_INTERVAL = 10_000

TRAINING_TABLES: tuple[str, ...] = (
    "dataset_examples",
    "dataset_sources",
    "dataset_grade_runs",
    "dataset_calibrations",
    "dataset_exports",
)

OPS_TABLES: tuple[str, ...] = (
    "signal_events",
    "harvest_watermarks",
    "judge_runs",
    "embed_runs",
    "autopilot_runs",
    "loop_liveness",
    "family_scores",
    "stall_pages",
)

CORE_AUDIT_TABLES: tuple[str, ...] = (
    "egress_audits",
    "context_source_handles",
    "context_source_handle_issues",
    "task_closeouts",
    "task_closeout_retrievals",
)

SEMANTIC_SOURCE_TABLES: tuple[str, ...] = (
    "brain_events",
    "evidence",
    "knowledge",
    "knowledge_evidence",
    "signal_events",
    "retrieval_uses",
)


def _quote(name: str) -> str:
    if not name or not all(character.isalnum() or character == "_" for character in name):
        raise ValueError(f"unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def _count(conn: sqlite3.Connection, table: str) -> int:
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0])


def _canonical_value(value: Any, *, preserve_bytes: bool = True) -> Any:
    if isinstance(value, bytes):
        if preserve_bytes:
            return {"$bytes": base64.b64encode(value).decode("ascii")}
        return {
            "$bytes_sha256": hashlib.sha256(value).hexdigest(),
            "$bytes_length": len(value),
        }
    if isinstance(value, float):
        return {"$float": value.hex()}
    return value


def _safe_row(row: sqlite3.Row, *, preserve_bytes: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        # Semantic event payloads retain JSON numbers as numbers. Exact
        # float/byte encodings belong only in the separate row digest.
        result[str(key)] = (
            _canonical_value(value, preserve_bytes=preserve_bytes)
            if isinstance(value, bytes)
            else value
        )
    return result


def _row_sha256(row: sqlite3.Row) -> str:
    exact = {
        str(key): _canonical_value(row[key], preserve_bytes=True) for key in row.keys()
    }
    return sha256_text(canonical_json(exact))


def table_sha256(
    conn: sqlite3.Connection,
    table: str,
    *,
    order_columns: Sequence[str] | None = None,
) -> str:
    """Hash declared columns and logical rows in stable primary-key order."""
    if not _table_exists(conn, table):
        return hashlib.sha256(b"missing-table\n").hexdigest()
    quoted = _quote(table)
    info = list(conn.execute(f"PRAGMA table_info({quoted})"))
    columns = [str(row["name"]) for row in info]
    primary_key = [
        str(row["name"])
        for row in sorted(info, key=lambda item: int(item["pk"]))
        if row["pk"]
    ]
    stable_order = list(order_columns) if order_columns is not None else primary_key
    order = ", ".join(_quote(name) for name in stable_order) if stable_order else "rowid"
    names = ", ".join(_quote(name) for name in columns)
    digest = hashlib.sha256()
    digest.update(canonical_json(columns).encode())
    digest.update(b"\n")
    cursor = conn.execute(
        f"SELECT {names} FROM {quoted} ORDER BY {order}"  # noqa: S608 - quoted local schema
    )
    while rows := cursor.fetchmany(1_000):
        for row in rows:
            digest.update(
                canonical_json(
                    [_canonical_value(row[index]) for index in range(len(row))]
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def event_prefix_sha256(conn: sqlite3.Connection, *, through_seq: int | None = None) -> str:
    """Hash the legacy event prefix including its explicit sequence/rowid."""
    sequence = "event_seq" if _column_exists(conn, "brain_events", "event_seq") else "rowid"
    where = ""
    params: tuple[Any, ...] = ()
    if through_seq is not None:
        where = f" WHERE {sequence} <= ?"
        params = (through_seq,)
    digest = hashlib.sha256()
    columns = (
        "id",
        "ts",
        "kind",
        "writer",
        "session_id",
        "body_json",
        "body_hash",
        "prev_hash",
        "event_hash",
    )
    rows = conn.execute(
        f"SELECT {sequence} AS event_seq, {', '.join(columns)} "  # noqa: S608
        f"FROM brain_events{where} ORDER BY {sequence}",
        params,
    )
    for row in rows:
        digest.update(
            canonical_json(
                [row["event_seq"], *[row[column] for column in columns]]
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({_quote(table)})"))


def _derived_companion_paths(core: Path) -> tuple[Path, Path]:
    stem = core.stem
    return (
        core.with_name(f"{stem}-training.sqlite"),
        core.with_name(f"{stem}-ops.sqlite"),
    )


def _resolve_outputs(
    core: Path,
    training: Path | None,
    ops: Path | None,
) -> tuple[Path, Path, Path]:
    derived_training, derived_ops = _derived_companion_paths(core)
    return core, training or derived_training, ops or derived_ops


def _path_blockers(
    source: Path,
    outputs: dict[str, Path],
) -> list[str]:
    blockers: list[str] = []
    source = source.expanduser().resolve()
    resolved = {name: path.expanduser().resolve() for name, path in outputs.items()}
    all_paths = [source, *resolved.values()]
    if len(set(all_paths)) != len(all_paths):
        blockers.append("source and every output path must differ")
    if not source.is_file():
        blockers.append(f"source database not found: {source}")
    for label, path in resolved.items():
        if path.exists() or Path(f"{path}-wal").exists() or Path(f"{path}-shm").exists():
            blockers.append(f"{label} output path is not fresh: {path}")
    return blockers


def migration_plan(
    source: Path,
    core: Path,
    archive: Path,
    manifest: Path,
    training: Path | None = None,
    ops: Path | None = None,
) -> dict[str, Any]:
    """Return a read-only plan; it never initializes or creates an output."""
    source = source.expanduser().resolve()
    core, training_path, ops_path = _resolve_outputs(core, training, ops)
    outputs = {
        "core": core.expanduser().resolve(),
        "training": training_path.expanduser().resolve(),
        "ops": ops_path.expanduser().resolve(),
        "archive": archive.expanduser().resolve(),
        "manifest": manifest.expanduser().resolve(),
    }
    blockers = _path_blockers(source, outputs)
    semantic_counts: dict[str, int] = {}
    training_counts: dict[str, int] = {}
    ops_counts: dict[str, int] = {}
    if source.is_file():
        conn = _read_only(source)
        try:
            semantic_counts = {table: _count(conn, table) for table in SEMANTIC_SOURCE_TABLES}
            training_counts = {table: _count(conn, table) for table in TRAINING_TABLES}
            ops_counts = {table: _count(conn, table) for table in OPS_TABLES}
        finally:
            conn.close()
    return {
        "format": MIGRATION_FORMAT,
        "action": "plan",
        "ready": not blockers,
        "source": str(source),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "v1_core_tables": list(CORE_V1_TABLES),
        "training_tables": list(TRAINING_TABLES),
        "ops_tables": list(OPS_TABLES),
        "source_semantic_counts": semantic_counts,
        "source_training_counts": training_counts,
        "source_ops_counts": ops_counts,
        # Compatibility keys retained for the v0.4.1 CLI while it migrates to
        # the explicit core/training/ops vocabulary.
        "source_core_counts": semantic_counts,
        "source_archive_only_counts": training_counts | ops_counts,
        "blockers": blockers,
        "safety": {
            "source_opened_read_only": True,
            "archive_first": True,
            "in_place_drop": False,
            "in_place_vacuum": False,
            "automatic_activation": False,
            "outputs_must_be_fresh": True,
            "hosted_calls": 0,
            "network_calls": 0,
            "schedulers_started": 0,
        },
        "compatibility_note": (
            "The v1 core contains no relational knowledge/evidence or companion tables. "
            "The immutable archive preserves every historical table."
        ),
    }


class _MigrationAppender:
    """Fast deterministic event appender with a cached chain head."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        row = conn.execute(
            "SELECT event_hash FROM brain_events ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        self.previous = str(row[0]) if row else None

    def append(
        self,
        kind: str,
        body: dict[str, Any],
        *,
        ts: str,
        writer: str = "ocbrain-v1-migration",
    ) -> str:
        body_json = canonical_json(body)
        body_hash = sha256_text(body_json)
        event_hash = sha256_text(
            canonical_json(
                {
                    "ts": ts,
                    "kind": kind,
                    "writer": writer,
                    "session_id": None,
                    "body_hash": body_hash,
                    "prev_hash": self.previous,
                }
            )
        )
        event_id = stable_id("evt", kind, event_hash)
        self.conn.execute(
            """
            INSERT INTO brain_events(
              id, ts, kind, writer, session_id, body_json, body_hash, prev_hash, event_hash
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (event_id, ts, kind, writer, body_json, body_hash, self.previous, event_hash),
        )
        self.previous = event_hash
        return event_id


def _timestamp(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return str(value)
    return "1970-01-01T00:00:00+00:00"


def _copy_event_prefix(archive: Path, core: Path) -> dict[str, Any]:
    source = _read_only(archive)
    try:
        if not _table_exists(source, "brain_events"):
            source_chain = {"verified": True, "events": 0, "last_rowid": 0, "last_event_hash": None}
            source_digest = event_prefix_sha256_empty()
            max_seq = 0
        else:
            source_chain = verify_event_chain(source)
            if not source_chain["verified"]:
                raise RuntimeError(f"legacy event prefix is corrupt: {source_chain}")
            source_digest = event_prefix_sha256(source)
            max_seq = int(
                source.execute("SELECT COALESCE(MAX(rowid),0) FROM brain_events").fetchone()[0]
            )
    finally:
        source.close()

    conn = sqlite3.connect(core)
    conn.row_factory = sqlite3.Row
    try:
        init_core_v1(conn)
        conn.execute("ATTACH DATABASE ? AS archived", (str(archive),))
        if max_seq:
            conn.execute(
                """
                INSERT INTO brain_events(
                  event_seq, id, ts, kind, writer, session_id, body_json,
                  body_hash, prev_hash, event_hash
                )
                SELECT rowid, id, ts, kind, writer, session_id, body_json,
                       body_hash, prev_hash, event_hash
                FROM archived.brain_events
                ORDER BY rowid
                """
            )
        conn.commit()
        conn.execute("DETACH DATABASE archived")
        copied_chain = verify_event_chain(conn, through_rowid=max_seq)
        copied_digest = event_prefix_sha256(conn, through_seq=max_seq)
        if copied_chain != source_chain or copied_digest != source_digest:
            raise RuntimeError(
                "legacy event prefix mismatch: "
                f"source_chain={source_chain}; copied_chain={copied_chain}; "
                f"source_sha={source_digest}; copied_sha={copied_digest}"
            )
        set_core_v1_search_triggers(conn, enabled=False)
        project_core_v1(conn, full=True)
        conn.commit()
    finally:
        conn.close()
    os.chmod(core, 0o600)
    return {
        "count": source_chain["events"],
        "max_event_seq": max_seq,
        "sha256": source_digest,
        "last_event_hash": source_chain.get("last_event_hash"),
        "verified": True,
    }


def event_prefix_sha256_empty() -> str:
    return hashlib.sha256(b"").hexdigest()


def _event_evidence_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["evidence_id"]): str(row["content_hash"])
        for row in conn.execute("SELECT evidence_id, content_hash FROM evidence_objects")
    }


def _legacy_row_body(row: dict[str, Any]) -> str:
    if row.get("type") == "value":
        value: Any = row.get("value_text")
        if row.get("value_bool") is not None:
            value = str(bool(row["value_bool"]))
        elif row.get("value_numeric") is not None:
            value = str(row["value_numeric"])
        return f"{row.get('subject') or ''} {row.get('predicate') or ''} {value or ''}".strip()
    return " ".join(str(value or "") for value in (row.get("title"), row.get("body_uri"))).strip()


def _link_groups(conn: sqlite3.Connection) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    if not _table_exists(conn, "knowledge_evidence"):
        return
    current: str | None = None
    links: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT knowledge_id, evidence_id, relation, created_at "
        "FROM knowledge_evidence ORDER BY knowledge_id, created_at, evidence_id, relation"
    ):
        knowledge_id = str(row["knowledge_id"])
        if current is not None and knowledge_id != current:
            yield current, links
            links = []
        current = knowledge_id
        links.append(dict(row))
    if current is not None:
        yield current, links


def _append_import_events(
    archive: Path,
    core: Path,
    *,
    import_batch_id: str,
    batch_size: int,
    progress_interval: int,
    progress: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    source = _read_only(archive)
    destination = sqlite3.connect(core)
    destination.row_factory = sqlite3.Row
    destination.execute("PRAGMA foreign_keys=ON")
    counts = {
        "legacy_evidence_imported": 0,
        "legacy_knowledge_imported": 0,
        "legacy_signal_imported": 0,
        "retrieval_snapshot_imported": 0,
    }
    try:
        appender = _MigrationAppender(destination)
        existing_evidence = _event_evidence_hashes(destination)
        evidence_map: dict[str, str] = {}
        since_commit = 0
        total = 0

        if _table_exists(source, "evidence"):
            for row in source.execute("SELECT * FROM evidence ORDER BY id"):
                data = _safe_row(row)
                source_id = str(data["id"])
                claim_hash = sha256_text(str(data.get("claim") or ""))
                if source_id not in existing_evidence or existing_evidence[source_id] == claim_hash:
                    canonical_id = source_id
                else:
                    canonical_id = f"legacy:{source_id}:{claim_hash[:12]}"
                evidence_map[source_id] = canonical_id
                body = {
                    "schema_version": IMPORT_SCHEMA_VERSION,
                    "import_batch_id": import_batch_id,
                    "subject": {"kind": "evidence", "id": canonical_id},
                    "legacy_evidence_id": source_id,
                    "canonical_evidence_id": canonical_id,
                    "legacy_row_sha256": _row_sha256(row),
                    "row": data,
                    "scope": conservative_legacy_scope(data).to_dict(),
                    "collision": canonical_id != source_id,
                }
                appender.append(
                    "legacy_evidence_imported",
                    body,
                    ts=_timestamp(data, "ingested_at", "occurred_at"),
                )
                counts["legacy_evidence_imported"] += 1
                total, since_commit = _advance_progress(
                    destination,
                    total,
                    since_commit,
                    batch_size,
                    progress_interval,
                    progress,
                    "evidence",
                )

        link_iter = iter(_link_groups(source))
        pending_links = next(link_iter, None)
        if _table_exists(source, "knowledge"):
            for row in source.execute("SELECT * FROM knowledge ORDER BY id"):
                raw = dict(row)
                data = _safe_row(row)
                knowledge_id = str(data["id"])
                while pending_links is not None and pending_links[0] < knowledge_id:
                    pending_links = next(link_iter, None)
                links = (
                    pending_links[1]
                    if pending_links and pending_links[0] == knowledge_id
                    else []
                )
                if pending_links and pending_links[0] == knowledge_id:
                    pending_links = next(link_iter, None)
                canonical_links = [
                    {
                        "evidence_id": evidence_map.get(
                            str(link["evidence_id"]), str(link["evidence_id"])
                        ),
                        "legacy_evidence_id": str(link["evidence_id"]),
                        "relation": link.get("relation") or "supports",
                        "created_at": link.get("created_at"),
                    }
                    for link in links
                ]
                embedding = raw.get("embedding")
                if isinstance(embedding, bytes):
                    data["embedding"] = {
                        "$bytes_sha256": hashlib.sha256(embedding).hexdigest(),
                        "$bytes_length": len(embedding),
                    }
                canonical_id = f"legacy:{knowledge_id}"
                body = {
                    "schema_version": IMPORT_SCHEMA_VERSION,
                    "import_batch_id": import_batch_id,
                    "subject": {"kind": "belief", "id": canonical_id},
                    "legacy_knowledge_id": knowledge_id,
                    "canonical_belief_id": canonical_id,
                    "legacy_row_sha256": _row_sha256(row),
                    "embedding_sha256": (
                        hashlib.sha256(embedding).hexdigest()
                        if isinstance(embedding, bytes)
                        else None
                    ),
                    "body": _legacy_row_body(data),
                    "row": data,
                    "evidence_links": canonical_links,
                    "scope": conservative_legacy_scope(data).to_dict(),
                }
                appender.append(
                    "legacy_knowledge_imported",
                    body,
                    ts=_timestamp(data, "updated_at", "created_at"),
                )
                counts["legacy_knowledge_imported"] += 1
                total, since_commit = _advance_progress(
                    destination,
                    total,
                    since_commit,
                    batch_size,
                    progress_interval,
                    progress,
                    "knowledge",
                )

        if _table_exists(source, "signal_events"):
            for row in source.execute("SELECT * FROM signal_events ORDER BY created_at, id"):
                data = _safe_row(row)
                appender.append(
                    "legacy_signal_imported",
                    {
                        "schema_version": IMPORT_SCHEMA_VERSION,
                        "import_batch_id": import_batch_id,
                        "subject": {
                            "kind": "signal",
                            "id": str(data.get("id") or ""),
                        },
                        "legacy_row_sha256": _row_sha256(row),
                        "row": data,
                    },
                    ts=_timestamp(data, "created_at", "occurred_at"),
                )
                counts["legacy_signal_imported"] += 1
                total, since_commit = _advance_progress(
                    destination,
                    total,
                    since_commit,
                    batch_size,
                    progress_interval,
                    progress,
                    "signals",
                )

        if _table_exists(source, "retrieval_uses"):
            for row in source.execute("SELECT * FROM retrieval_uses ORDER BY served_at, id"):
                data = _safe_row(row)
                appender.append(
                    "retrieval_snapshot_imported",
                    {
                        "schema_version": IMPORT_SCHEMA_VERSION,
                        "import_batch_id": import_batch_id,
                        "subject": {"kind": "retrieval", "id": str(data["id"])},
                        "legacy_row_sha256": _row_sha256(row),
                        "row": data,
                    },
                    ts=_timestamp(data, "served_at"),
                )
                counts["retrieval_snapshot_imported"] += 1
                total, since_commit = _advance_progress(
                    destination,
                    total,
                    since_commit,
                    batch_size,
                    progress_interval,
                    progress,
                    "retrievals",
                )
        destination.commit()
        if progress:
            progress({"stage": "imports", "rows": total, "status": "complete"})

        # Search triggers were suspended before the prefix projection.  The
        # incremental fold populates keyed search_documents only; FTS is rebuilt once.
        projection = project_core_v1(destination)
        rebuild_core_v1_search(destination)
        set_core_v1_search_triggers(destination, enabled=True)
        destination.commit()
        chain = verify_event_chain(destination)
        if not chain["verified"]:
            raise RuntimeError(f"v1 event chain failed after imports: {chain}")
        return {
            "counts": counts,
            "total_import_events": sum(counts.values()),
            "projection": projection,
            "chain": chain,
        }
    finally:
        destination.close()
        source.close()


def _advance_progress(
    conn: sqlite3.Connection,
    total: int,
    since_commit: int,
    batch_size: int,
    progress_interval: int,
    progress: Callable[[dict[str, Any]], None] | None,
    stage: str,
) -> tuple[int, int]:
    total += 1
    since_commit += 1
    if since_commit >= batch_size:
        conn.commit()
        since_commit = 0
    if progress and total % progress_interval == 0:
        progress({"stage": stage, "rows": total, "status": "running"})
    return total, since_commit


def _copy_intersection_tables(archive: Path, core: Path) -> dict[str, int]:
    conn = sqlite3.connect(core)
    conn.row_factory = sqlite3.Row
    copied: dict[str, int] = {}
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("ATTACH DATABASE ? AS archived", (str(archive),))
        for table in CORE_AUDIT_TABLES:
            if not _table_exists(conn, table):
                raise RuntimeError(f"v1 core missing audit table: {table}")
            source_exists = conn.execute(
                "SELECT 1 FROM archived.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if source_exists is None:
                copied[table] = 0
                continue
            target_columns = [
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({_quote(table)})")
            ]
            source_columns = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA archived.table_info({_quote(table)})")
            }
            columns = [name for name in target_columns if name in source_columns]
            names = ", ".join(_quote(name) for name in columns)
            conn.execute(
                f"INSERT INTO {_quote(table)} ({names}) "  # noqa: S608 - allow-listed table
                f"SELECT {names} FROM archived.{_quote(table)} ORDER BY rowid"
            )
            copied[table] = int(conn.execute("SELECT changes()").fetchone()[0])
        conn.commit()
        conn.execute("DETACH DATABASE archived")
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"v1 core has {len(violations)} foreign-key violations")
    finally:
        conn.close()
    return copied


def _build_companion_extract(
    archive: Path,
    output: Path,
    *,
    kind: str,
    tables: Sequence[str],
) -> dict[str, Any]:
    conn = sqlite3.connect(output)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("CREATE TABLE schema_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO schema_meta VALUES (?, ?)",
            (("companion_kind", kind), ("source", "ocbrain-v0-archive")),
        )
        conn.execute("ATTACH DATABASE ? AS archived", (str(archive),))
        for table in tables:
            exists = conn.execute(
                "SELECT 1 FROM archived.sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                continue
            conn.execute(
                f"CREATE TABLE {_quote(table)} AS "  # noqa: S608 - fixed table allow-list
                f"SELECT * FROM archived.{_quote(table)} WHERE 0"
            )
            conn.execute(
                f"INSERT INTO {_quote(table)} SELECT * "  # noqa: S608
                f"FROM archived.{_quote(table)} ORDER BY rowid"
            )
        conn.commit()
        conn.execute("DETACH DATABASE archived")
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError(f"{kind} companion integrity failed: {integrity}")
    finally:
        conn.close()
    os.chmod(output, 0o600)
    source = _read_only(archive)
    destination = _read_only(output)
    try:
        verification: dict[str, dict[str, Any]] = {}
        for table in tables:
            primary_key = [
                str(row["name"])
                for row in sorted(
                    source.execute(f"PRAGMA table_info({_quote(table)})"),
                    key=lambda item: int(item["pk"]),
                )
                if row["pk"]
            ] if _table_exists(source, table) else []
            verification[table] = {
                "source_count": _count(source, table),
                "extract_count": _count(destination, table),
                "source_sha256": table_sha256(
                    source, table, order_columns=primary_key or None
                ),
                "extract_sha256": table_sha256(
                    destination, table, order_columns=primary_key or None
                ),
            }
    finally:
        destination.close()
        source.close()
    failed = [
        table
        for table, item in verification.items()
        if item["source_count"] != item["extract_count"]
        or item["source_sha256"] != item["extract_sha256"]
    ]
    if failed:
        raise RuntimeError(f"{kind} companion verification failed: {failed}")
    return {"kind": kind, "tables": verification, "verified": True}


def _logical_source_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _source_catalog(archive: Path) -> dict[str, Any]:
    conn = _read_only(archive)
    try:
        tables: dict[str, dict[str, Any]] = {}
        for table in _logical_source_tables(conn):
            if table.startswith("search_index_"):
                # FTS shadow pages are implementation-derived and can be very
                # large BLOBs. The byte-identical archive file hash preserves
                # them; enumerate/count them explicitly without a second full
                # content pass through the 1.9 GB source.
                tables[table] = {
                    "count": _count(conn, table),
                    "sha256": None,
                    "preservation": "byte_exact_archive_only_derived",
                }
            else:
                tables[table] = {
                    "count": _count(conn, table),
                    "sha256": table_sha256(conn, table),
                    "preservation": "archive_and_logical_hash",
                }
        objects = [
            {
                "type": str(row["type"]),
                "name": str(row["name"]),
                "table": row["tbl_name"],
                "sql_sha256": sha256_text(str(row["sql"] or "")),
            }
            for row in conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            )
        ]
        return {"tables": tables, "schema_objects": objects}
    finally:
        conn.close()


def _verify_core(core: Path, prefix: dict[str, Any], imports: dict[str, Any]) -> dict[str, Any]:
    conn = _read_only(core)
    try:
        assert_core_v1_inventory(conn)
        chain = verify_event_chain(conn)
        if not chain["verified"]:
            raise RuntimeError(f"v1 event chain verification failed: {chain}")
        prefix_sha = event_prefix_sha256(conn, through_seq=prefix["max_event_seq"])
        prefix_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM brain_events WHERE event_seq <= ?",
                (prefix["max_event_seq"],),
            ).fetchone()[0]
        )
        if prefix_sha != prefix["sha256"] or prefix_count != prefix["count"]:
            raise RuntimeError("published v1 core does not preserve the exact legacy event prefix")
        integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
        if integrity != "ok" or foreign_keys:
            raise RuntimeError(
                f"v1 core database checks failed: integrity={integrity}; fk={len(foreign_keys)}"
            )
        counts = {
            table: _count(conn, table)
            for table in CORE_V1_TABLES
            if table not in CORE_V1_FTS_TABLES
        }
        expected_events = prefix["count"] + imports["total_import_events"]
        if counts["brain_events"] != expected_events:
            raise RuntimeError(
                f"event coverage mismatch: expected={expected_events}; got={counts['brain_events']}"
            )
        forbidden = sorted(
            set(TRAINING_TABLES)
            | set(OPS_TABLES)
            | {"evidence", "knowledge", "knowledge_evidence", "memory"}
        )
        leaked = [table for table in forbidden if _table_exists(conn, table)]
        if leaked:
            raise RuntimeError(f"legacy/companion tables leaked into v1 core: {leaked}")
        return {
            "verified": True,
            "integrity": integrity,
            "foreign_key_violations": 0,
            "event_chain": chain,
            "legacy_event_prefix": {
                "count": prefix_count,
                "sha256": prefix_sha,
                "verified": True,
            },
            "counts": counts,
            "exact_table_inventory": sorted(
                set(CORE_V1_TABLES) | set(CORE_V1_FTS_TABLES)
            ),
        }
    finally:
        conn.close()


def migrate_core_v1(
    source: Path,
    core: Path,
    archive: Path,
    manifest: Path,
    training: Path | None = None,
    ops: Path | None = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build and verify archive/core/training/ops artifacts without activation."""
    if batch_size <= 0 or progress_interval <= 0:
        raise ValueError("batch_size and progress_interval must be positive")
    plan = migration_plan(source, core, archive, manifest, training, ops)
    if plan["blockers"]:
        raise ValueError("; ".join(plan["blockers"]))
    source = source.expanduser().resolve()
    core_path = _fresh_target(Path(plan["outputs"]["core"]), label="core output")
    training_path = _fresh_target(
        Path(plan["outputs"]["training"]), label="training output"
    )
    ops_path = _fresh_target(Path(plan["outputs"]["ops"]), label="ops output")
    archive_path = _fresh_target(Path(plan["outputs"]["archive"]), label="archive output")
    manifest_path = _fresh_target(
        Path(plan["outputs"]["manifest"]), label="migration manifest"
    )
    token = uuid.uuid4().hex
    temporary = {
        "archive": archive_path.with_name(f".{archive_path.name}.{token}.tmp"),
        "core": core_path.with_name(f".{core_path.name}.{token}.tmp"),
        "training": training_path.with_name(f".{training_path.name}.{token}.tmp"),
        "ops": ops_path.with_name(f".{ops_path.name}.{token}.tmp"),
    }
    published: list[Path] = []
    try:
        if progress:
            progress({"stage": "archive", "rows": 0, "status": "running"})
        _online_copy(source, temporary["archive"])
        archive_sha = sha256_file(temporary["archive"])
        import_batch_id = stable_id("mig", MIGRATION_FORMAT, archive_sha)
        source_catalog = _source_catalog(temporary["archive"])

        prefix = _copy_event_prefix(temporary["archive"], temporary["core"])
        imports = _append_import_events(
            temporary["archive"],
            temporary["core"],
            import_batch_id=import_batch_id,
            batch_size=batch_size,
            progress_interval=progress_interval,
            progress=progress,
        )
        audit_copies = _copy_intersection_tables(temporary["archive"], temporary["core"])
        core_verification = _verify_core(temporary["core"], prefix, imports)

        training_report = _build_companion_extract(
            temporary["archive"],
            temporary["training"],
            kind="training",
            tables=TRAINING_TABLES,
        )
        ops_report = _build_companion_extract(
            temporary["archive"],
            temporary["ops"],
            kind="ops",
            tables=OPS_TABLES,
        )

        core_sha = sha256_file(temporary["core"])
        training_sha = sha256_file(temporary["training"])
        ops_sha = sha256_file(temporary["ops"])
        payload = {
            "format": MIGRATION_FORMAT,
            "action": "migrate",
            "status": "verified",
            "import_batch_id": import_batch_id,
            "source": {
                "path": str(source),
                "bytes_at_finish": source.stat().st_size,
                "modified_ns_at_finish": source.stat().st_mtime_ns,
            },
            "archive": {
                "path": str(archive_path),
                "bytes": temporary["archive"].stat().st_size,
                "sha256": archive_sha,
                "catalog": source_catalog,
            },
            "core": {
                "path": str(core_path),
                "bytes": temporary["core"].stat().st_size,
                "sha256": core_sha,
                "verification": core_verification,
                "imports": imports,
                "audit_copies": audit_copies,
            },
            "training": {
                "path": str(training_path),
                "bytes": temporary["training"].stat().st_size,
                "sha256": training_sha,
                **training_report,
            },
            "ops": {
                "path": str(ops_path),
                "bytes": temporary["ops"].stat().st_size,
                "sha256": ops_sha,
                **ops_report,
            },
            "safety": plan["safety"]
            | {
                "live_database_replaced": False,
                "live_database_repointed": False,
            },
            "activation": (
                "Manual only: verify all three runtime clients against the fresh core, "
                "then explicitly repoint OCBRAIN_DB. This command never activates it."
            ),
        }

        for name, final in (
            ("archive", archive_path),
            ("core", core_path),
            ("training", training_path),
            ("ops", ops_path),
        ):
            os.replace(temporary[name], final)
            os.chmod(final, 0o600)
            published.append(final)
        _atomic_json(manifest_path, payload)
        os.chmod(manifest_path, 0o600)
        published.append(manifest_path)
        if progress:
            progress({"stage": "publish", "rows": 5, "status": "complete"})
        return payload | {"manifest": str(manifest_path)}
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
            Path(f"{path}-wal").unlink(missing_ok=True)
            Path(f"{path}-shm").unlink(missing_ok=True)


__all__ = [
    "CORE_AUDIT_TABLES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PROGRESS_INTERVAL",
    "IMPORT_SCHEMA_VERSION",
    "MIGRATION_FORMAT",
    "OPS_TABLES",
    "SEMANTIC_SOURCE_TABLES",
    "TRAINING_TABLES",
    "event_prefix_sha256",
    "migrate_core_v1",
    "migration_plan",
    "table_sha256",
]
