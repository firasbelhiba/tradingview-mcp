"""
Alert Store — the server's only persistent state.

Every other tool here is a pure read: ask a question, get an answer, keep
nothing. Alerts are different by nature — an alert that cannot be remembered
between calls is not an alert — so this module owns the one piece of state
the server keeps, and keeps it in one place rather than scattering files
around the package.

SQLite from the standard library: no new dependency, and it handles the
concurrent access a threaded MCP server produces, which a JSON file does not.
A connection is opened per call and closed again; alert traffic is a handful
of rows, so pooling would add failure modes for no gain.

Storage location, in order:
  TRADINGVIEW_MCP_ALERTS_DB   explicit path to the database file
  TRADINGVIEW_MCP_DATA_DIR    directory to put ``alerts.sqlite3`` in
  ~/.tradingview-mcp/alerts.sqlite3
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

_SCHEMA_VERSION = 1
_init_lock = Lock()
_initialised: set = set()

ARMED = "armed"
TRIGGERED = "triggered"


def db_path() -> Path:
    explicit = os.environ.get("TRADINGVIEW_MCP_ALERTS_DB")
    if explicit:
        return Path(explicit).expanduser()
    data_dir = os.environ.get("TRADINGVIEW_MCP_DATA_DIR")
    base = Path(data_dir).expanduser() if data_dir else Path.home() / ".tradingview-mcp"
    return base / "alerts.sqlite3"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL lets a check run while another call writes; without it a concurrent
    # writer makes readers fail with "database is locked".
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    _ensure_schema(conn, str(path))
    return conn


def _ensure_schema(conn: sqlite3.Connection, key: str) -> None:
    if key in _initialised:
        return
    with _init_lock:
        if key in _initialised:
            return
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    exchange        TEXT    NOT NULL,
                    metric          TEXT    NOT NULL,
                    operator        TEXT    NOT NULL,
                    threshold       REAL    NOT NULL,
                    timeframe       TEXT    NOT NULL,
                    note            TEXT,
                    repeat_after_fire INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT    NOT NULL,
                    triggered_at    TEXT,
                    triggered_value REAL,
                    last_checked_at TEXT,
                    last_value      REAL,
                    check_error     TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_alerts_state ON alerts(triggered_at)"
            )
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        _initialised.add(key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["state"] = TRIGGERED if d.get("triggered_at") else ARMED
    d["repeat_after_fire"] = bool(d.get("repeat_after_fire"))
    return d


def add_alert(symbol: str, exchange: str, metric: str, operator: str,
              threshold: float, timeframe: str, note: Optional[str] = None,
              repeat_after_fire: bool = False) -> Dict[str, Any]:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO alerts
              (symbol, exchange, metric, operator, threshold, timeframe, note,
               repeat_after_fire, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol, exchange, metric, operator, float(threshold), timeframe,
             note, 1 if repeat_after_fire else 0, _now()),
        )
        alert_id = cur.lastrowid
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return _row_to_dict(row)


def list_alerts(state: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM alerts"
    params: tuple = ()
    if state == ARMED:
        sql += " WHERE triggered_at IS NULL"
    elif state == TRIGGERED:
        sql += " WHERE triggered_at IS NOT NULL"
    sql += " ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_alert(alert_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return _row_to_dict(row) if row else None


def delete_alert(alert_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
    return cur.rowcount > 0


def delete_all(state: Optional[str] = None) -> int:
    sql = "DELETE FROM alerts"
    if state == ARMED:
        sql += " WHERE triggered_at IS NULL"
    elif state == TRIGGERED:
        sql += " WHERE triggered_at IS NOT NULL"
    with _connect() as conn:
        cur = conn.execute(sql)
    return cur.rowcount


def mark_triggered(alert_id: int, value: float) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE alerts SET triggered_at = ?, triggered_value = ?, "
            "last_checked_at = ?, last_value = ?, check_error = NULL WHERE id = ?",
            (_now(), float(value), _now(), float(value), alert_id),
        )


def record_check(alert_id: int, value: Optional[float],
                 error: Optional[str] = None) -> None:
    """Record that an alert was evaluated without firing.

    Kept so a caller can tell "checked, condition not met" from "never
    checked" or "the feed was down" — otherwise a silent data outage looks
    exactly like a quiet market.
    """
    with _connect() as conn:
        conn.execute(
            "UPDATE alerts SET last_checked_at = ?, last_value = ?, check_error = ? "
            "WHERE id = ?",
            (_now(), None if value is None else float(value), error, alert_id),
        )


def rearm(alert_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE alerts SET triggered_at = NULL, triggered_value = NULL "
            "WHERE id = ?",
            (alert_id,),
        )
    return cur.rowcount > 0
