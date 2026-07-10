"""Bounded SQLite write transactions for dataset mining.

Dataset parsing is intentionally incremental, but the original miners left the
first implicit write transaction open until the whole SFT/DPO/persona stage
finished.  On a large corpus that turned minutes of read/parse work into one
database-writer window.  ``DatasetWriteBatch`` makes the boundary explicit and
records the time spent waiting for and holding SQLite's writer lock.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any


class DatasetWriteBatch:
    """Own short ``BEGIN IMMEDIATE`` transactions and report lock telemetry."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        max_operations: int = 50,
        max_seconds: float = 2.0,
    ) -> None:
        if max_operations < 1:
            raise ValueError("max_operations must be positive")
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.conn = conn
        self.max_operations = max_operations
        self.max_seconds = max_seconds
        self._opened_at: float | None = None
        self._pending = 0
        self.operations = 0
        self.batches_committed = 0
        self.lock_wait_seconds = 0.0
        self.max_lock_wait_seconds = 0.0
        self.writer_seconds = 0.0
        self.max_writer_seconds = 0.0

    def ensure(self) -> None:
        """Acquire the writer lock immediately before a mutating unit of work."""
        if self.conn.in_transaction:
            # A caller may enter with a pre-existing transaction. We cannot know
            # when its lock began, so start the observable window here.
            if self._opened_at is None:
                self._opened_at = time.monotonic()
            return
        wait_started = time.monotonic()
        self.conn.execute("BEGIN IMMEDIATE")
        acquired = time.monotonic()
        waited = acquired - wait_started
        self.lock_wait_seconds += waited
        self.max_lock_wait_seconds = max(self.max_lock_wait_seconds, waited)
        self._opened_at = acquired

    def operation(self) -> None:
        """Record one completed mutating unit and commit at either bound."""
        if not self.conn.in_transaction:
            raise RuntimeError("dataset write operation completed outside a transaction")
        self._pending += 1
        self.operations += 1
        elapsed = time.monotonic() - (self._opened_at or time.monotonic())
        if self._pending >= self.max_operations or elapsed >= self.max_seconds:
            self.flush()

    def flush(self) -> None:
        """Commit the current batch, if any, and close its measured lock window."""
        if not self.conn.in_transaction:
            self._opened_at = None
            self._pending = 0
            return
        opened = self._opened_at or time.monotonic()
        self.conn.commit()
        held = time.monotonic() - opened
        self.writer_seconds += held
        self.max_writer_seconds = max(self.max_writer_seconds, held)
        self.batches_committed += 1
        self._opened_at = None
        self._pending = 0

    def rollback(self) -> None:
        if self.conn.in_transaction:
            self.conn.rollback()
        self._opened_at = None
        self._pending = 0

    def metrics(self) -> dict[str, Any]:
        """Return JSON-safe, explicitly named writer-lock measurements."""
        return {
            "batch_max_operations": self.max_operations,
            "batch_max_seconds": self.max_seconds,
            "operations": self.operations,
            "batches_committed": self.batches_committed,
            "lock_wait_seconds": round(self.lock_wait_seconds, 6),
            "max_lock_wait_seconds": round(self.max_lock_wait_seconds, 6),
            "writer_lock_seconds": round(self.writer_seconds, 6),
            "max_writer_lock_seconds": round(self.max_writer_seconds, 6),
        }
