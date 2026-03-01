"""
app/database/db.py – SQLite persistence layer.

Features
--------
* Auto-creates the ``traffic_logs`` table on first connection.
* ``BatchWriter`` accumulates inserts and flushes every ``BATCH_SIZE``
  records, avoiding per-frame writes.
* Thread-local connections so every thread gets its own SQLite handle.
* ``fetch_recent(n)``  – last *n* rows (for /vehicles).
* ``fetch_all()``      – every row (for /download-csv).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List

from app.config import DB_PATH

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 10

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS traffic_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    vehicle_type  TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    density       TEXT    NOT NULL
);
"""

# ── Thread-local connection ────────────────────────────────────────────────────
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not getattr(_local, "con", None):
        _local.con = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.con.row_factory = sqlite3.Row
        _local.con.execute(_CREATE_TABLE)
        _local.con.commit()
        logger.debug("SQLite connection opened [thread=%s]",
                     threading.current_thread().name)
    return _local.con


# ── BatchWriter ────────────────────────────────────────────────────────────────

class BatchWriter:
    """Thread-safe accumulator that flushes to SQLite on every BATCH_SIZE rows."""

    def __init__(self) -> None:
        self._buf:  List[tuple] = []
        self._lock = threading.Lock()

    def add(
        self,
        timestamp:    datetime,
        vehicle_type: str,
        direction:    str,
        density:      str,
    ) -> None:
        row = (timestamp.isoformat(timespec="seconds"), vehicle_type, direction, density)
        with self._lock:
            self._buf.append(row)
            if len(self._buf) >= BATCH_SIZE:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        _conn().executemany(
            "INSERT INTO traffic_logs (timestamp, vehicle_type, direction, density) "
            "VALUES (?, ?, ?, ?)",
            self._buf,
        )
        _conn().commit()
        logger.debug("DB flush: %d rows written.", len(self._buf))
        self._buf.clear()


# ── Module-level singleton ─────────────────────────────────────────────────────
_writer = BatchWriter()


def log_crossing(
    timestamp:    datetime,
    vehicle_type: str,
    direction:    str,
    density:      str,
) -> None:
    """Queue a line-crossing event for batch insertion."""
    _writer.add(timestamp, vehicle_type, direction, density)


def flush() -> None:
    """Force-write any remaining buffered rows (call on shutdown)."""
    _writer.flush()


# ── Queries ────────────────────────────────────────────────────────────────────

def fetch_recent(n: int = 50) -> List[Dict[str, Any]]:
    """Return the *n* most recent log rows, newest first."""
    rows = _conn().execute(
        "SELECT id, timestamp, vehicle_type, direction, density "
        "FROM traffic_logs ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_all() -> List[Dict[str, Any]]:
    """Return all rows ordered oldest→newest (for CSV export)."""
    rows = _conn().execute(
        "SELECT id, timestamp, vehicle_type, direction, density "
        "FROM traffic_logs ORDER BY id ASC"
    ).fetchall()
    return [dict(r) for r in rows]
