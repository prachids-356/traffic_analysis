"""
db.py – SQLite persistence layer for traffic logs.

Features
--------
* Auto-creates the ``traffic_logs`` table on first use.
* Accumulates inserts in a local batch; flushes every BATCH_SIZE records.
* fetch_recent()  → last 50 rows as list-of-dicts (for the API).
* fetch_all()     → all rows as list-of-dicts (for CSV export).
"""

from __future__ import annotations

import sqlite3
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any

from config import DB_PATH

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 10

# ── Schema ─────────────────────────────────────────────────────────────────────
_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS traffic_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    vehicle_type  TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    density       TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    alert_type    TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    description   TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS traffic_summary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    total_count   INTEGER,
    health_score  INTEGER,
    risk_score    INTEGER,
    peak_hour     TEXT
);
CREATE TABLE IF NOT EXISTS detected_plates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    track_id      INTEGER NOT NULL,
    vehicle_type  TEXT    NOT NULL,
    plate_text    TEXT    NOT NULL,
    confidence    REAL    NOT NULL
);
"""


# ── Connection helper (one connection per thread) ──────────────────────────────
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "con") or _local.con is None:
        _local.con = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.con.row_factory = sqlite3.Row
        _local.con.executescript(_CREATE_TABLES)
        _local.con.commit()
        logger.debug("SQLite connection opened for thread %s", threading.current_thread().name)
    return _local.con


# ── Batch writer ───────────────────────────────────────────────────────────────
class BatchWriter:
    """
    Accumulate crossing events locally; flush to SQLite every BATCH_SIZE rows.

    Usage
    -----
    writer = BatchWriter()
    writer.add(timestamp, vehicle_type, direction, density)  # call per crossing
    writer.flush()    # call on shutdown to persist remaining rows
    """

    def __init__(self) -> None:
        self._buffer: List[tuple] = []
        self._lock = threading.Lock()

    def add(
        self,
        timestamp: datetime,
        vehicle_type: str,
        direction: str,
        density: str,
    ) -> None:
        row = (timestamp.isoformat(timespec="seconds"), vehicle_type, direction, density)
        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) >= BATCH_SIZE:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        con = _conn()
        con.executemany(
            "INSERT INTO traffic_logs (timestamp, vehicle_type, direction, density) "
            "VALUES (?, ?, ?, ?)",
            self._buffer,
        )
        con.commit()
        logger.debug("Flushed %d rows to DB.", len(self._buffer))
        self._buffer.clear()


# ── Module-level singleton writer (used by traffic_service) ────────────────────
_writer = BatchWriter()


def log_crossing(
    timestamp: datetime,
    vehicle_type: str,
    direction: str,
    density: str,
) -> None:
    """Convenience wrapper around the module-level BatchWriter."""
    _writer.add(timestamp, vehicle_type, direction, density)


def flush() -> None:
    """Force-flush any buffered rows (call on shutdown)."""
    _writer.flush()


# ── Query helpers ──────────────────────────────────────────────────────────────

def _rows_to_dicts(rows) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows]


def fetch_recent(n: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent *n* traffic log rows, newest first."""
    con = _conn()
    rows = con.execute(
        "SELECT id, timestamp, vehicle_type, direction, density "
        "FROM traffic_logs ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return _rows_to_dicts(rows)


def fetch_all() -> List[Dict[str, Any]]:
    """Return every row in the table, oldest first (for CSV export)."""
    con = _conn()
    rows = con.execute(
        "SELECT id, timestamp, vehicle_type, direction, density "
        "FROM traffic_logs ORDER BY id ASC"
    ).fetchall()
    return _rows_to_dicts(rows)
# ── Alert helpers ──────────────────────────────────────────────────────────────

def log_alert(
    alert_type: str,
    severity: str,
    description: str,
) -> None:
    """Log an alert to the database."""
    ts = datetime.now().isoformat(timespec="seconds")
    con = _conn()
    con.execute(
        "INSERT INTO alerts (timestamp, alert_type, severity, description) "
        "VALUES (?, ?, ?, ?)",
        (ts, alert_type, severity, description),
    )
    con.commit()


def fetch_alerts(n: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent *n* alerts."""
    con = _conn()
    rows = con.execute(
        "SELECT id, timestamp, alert_type, severity, description "
        "FROM alerts ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return _rows_to_dicts(rows)


def log_summary(
    total_count: int,
    health_score: int,
    risk_score: int,
    peak_hour: str,
) -> None:
    """Log a periodic summary snapshot."""
    ts = datetime.now().isoformat(timespec="seconds")
    con = _conn()
    con.execute(
        "INSERT INTO traffic_summary (timestamp, total_count, health_score, risk_score, peak_hour) "
        "VALUES (?, ?, ?, ?, ?)",
        (ts, total_count, health_score, risk_score, peak_hour),
    )
    con.commit()


def log_plate(
    track_id: int,
    vehicle_type: str,
    plate_text: str,
    confidence: float,
) -> None:
    """Log a recognized license plate to the database."""
    ts = datetime.now().isoformat(timespec="seconds")
    con = _conn()
    con.execute(
        "INSERT INTO detected_plates (timestamp, track_id, vehicle_type, plate_text, confidence) "
        "VALUES (?, ?, ?, ?, ?)",
        (ts, track_id, vehicle_type, plate_text, confidence),
    )
    con.commit()


def fetch_plates(n: int = 20) -> List[Dict[str, Any]]:
    """Return the most recent *n* recognized plates."""
    con = _conn()
    rows = con.execute(
        "SELECT id, timestamp, track_id, vehicle_type, plate_text, confidence "
        "FROM detected_plates ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return _rows_to_dicts(rows)
