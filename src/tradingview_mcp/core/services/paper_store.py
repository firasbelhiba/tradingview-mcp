"""
Paper Store — persistence for the simulated wallet.

NOTHING HERE TOUCHES REAL MONEY. There is no exchange key, no order
endpoint and no withdrawal path anywhere in this package; a "position" is a
row in a local SQLite file, priced against real market data. The simulation
is for learning what a strategy would have done.

Separate database from the alert store: the two domains share nothing, and a
schema change to one should not risk the other's rows. Same data directory,
so one folder holds everything the server remembers.

Storage location, in order:
  TRADINGVIEW_MCP_PAPER_DB    explicit path to the database file
  TRADINGVIEW_MCP_DATA_DIR    directory to put ``paper_trading.sqlite3`` in
  ~/.tradingview-mcp/paper_trading.sqlite3
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

OPEN = "open"
CLOSED = "closed"

SPOT = "spot"
FUTURES = "futures"


def db_path() -> Path:
    explicit = os.environ.get("TRADINGVIEW_MCP_PAPER_DB")
    if explicit:
        return Path(explicit).expanduser()
    data_dir = os.environ.get("TRADINGVIEW_MCP_DATA_DIR")
    base = Path(data_dir).expanduser() if data_dir else Path.home() / ".tradingview-mcp"
    return base / "paper_trading.sqlite3"


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10.0)
    conn.row_factory = sqlite3.Row
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
                CREATE TABLE IF NOT EXISTS wallet (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    currency        TEXT    NOT NULL,
                    starting_balance REAL   NOT NULL,
                    cash            REAL    NOT NULL,
                    fee_pct         REAL    NOT NULL,
                    max_leverage    REAL    NOT NULL,
                    created_at      TEXT    NOT NULL,
                    reset_count     INTEGER NOT NULL DEFAULT 0,
                    note            TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT    NOT NULL,
                    market          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,
                    quantity        REAL    NOT NULL,
                    entry_price     REAL    NOT NULL,
                    notional        REAL    NOT NULL,
                    leverage        REAL    NOT NULL,
                    margin          REAL    NOT NULL,
                    stop_loss       REAL,
                    take_profit     REAL,
                    status          TEXT    NOT NULL,
                    opened_at       TEXT    NOT NULL,
                    opened_ms       INTEGER NOT NULL,
                    entry_fee       REAL    NOT NULL DEFAULT 0,
                    exit_fee        REAL    NOT NULL DEFAULT 0,
                    funding_paid    REAL    NOT NULL DEFAULT 0,
                    closed_at       TEXT,
                    closed_ms       INTEGER,
                    exit_price      REAL,
                    realised_pnl    REAL,
                    close_reason    TEXT,
                    note            TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)"
            )
            conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        _initialised.add(key)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ── Wallet ────────────────────────────────────────────────────────────────────

def get_wallet() -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM wallet WHERE id = 1").fetchone()
    return dict(row) if row else None


def reset_wallet(balance: float, currency: str, fee_pct: float,
                 max_leverage: float, note: Optional[str] = None) -> Dict[str, Any]:
    """Create or wipe the wallet.

    A reset deletes every position, open and closed. That is the point of a
    practice wallet — but it is unrecoverable, so the caller confirms it.
    """
    with _connect() as conn:
        previous = conn.execute("SELECT reset_count FROM wallet WHERE id = 1").fetchone()
        count = (previous["reset_count"] + 1) if previous else 0
        with conn:
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM wallet")
            conn.execute(
                """
                INSERT INTO wallet (id, currency, starting_balance, cash, fee_pct,
                                    max_leverage, created_at, reset_count, note)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (currency, float(balance), float(balance), float(fee_pct),
                 float(max_leverage), _now(), count, note),
            )
        row = conn.execute("SELECT * FROM wallet WHERE id = 1").fetchone()
    return dict(row)


def set_cash(amount: float) -> None:
    with _connect() as conn:
        conn.execute("UPDATE wallet SET cash = ? WHERE id = 1", (float(amount),))


def adjust_cash(delta: float) -> None:
    with _connect() as conn:
        conn.execute("UPDATE wallet SET cash = cash + ? WHERE id = 1", (float(delta),))


# ── Positions ─────────────────────────────────────────────────────────────────

def add_position(**fields: Any) -> Dict[str, Any]:
    columns = (
        "symbol", "market", "side", "quantity", "entry_price", "notional",
        "leverage", "margin", "stop_loss", "take_profit", "status",
        "opened_at", "opened_ms", "entry_fee", "note",
    )
    values = [fields.get(c) for c in columns]
    with _connect() as conn:
        cur = conn.execute(
            f"INSERT INTO positions ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            values,
        )
        row = conn.execute("SELECT * FROM positions WHERE id = ?",
                           (cur.lastrowid,)).fetchone()
    return dict(row)


def get_position(position_id: int) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM positions WHERE id = ?",
                           (position_id,)).fetchone()
    return dict(row) if row else None


def list_positions(status: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM positions"
    params: list = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC"
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def update_position(position_id: int, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as conn:
        conn.execute(
            f"UPDATE positions SET {assignments} WHERE id = ?",
            list(fields.values()) + [position_id],
        )


def close_position(position_id: int, exit_price: float, realised_pnl: float,
                   reason: str, exit_fee: float, funding_paid: float,
                   closed_ms: Optional[int] = None) -> None:
    closed_ms = closed_ms if closed_ms is not None else now_ms()
    from datetime import datetime as _dt
    closed_at = _dt.fromtimestamp(closed_ms / 1000.0, tz=timezone.utc).isoformat(
        timespec="seconds"
    )
    with _connect() as conn:
        conn.execute(
            """
            UPDATE positions
               SET status = ?, exit_price = ?, realised_pnl = ?, close_reason = ?,
                   exit_fee = ?, funding_paid = ?, closed_at = ?, closed_ms = ?
             WHERE id = ?
            """,
            (CLOSED, float(exit_price), float(realised_pnl), reason,
             float(exit_fee), float(funding_paid), closed_at, closed_ms,
             position_id),
        )
