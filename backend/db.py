"""SQLite storage for bookings and human-escalation tickets."""
import sqlite3
from pathlib import Path
from typing import Literal

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, name TEXT, phone TEXT,
    model TEXT, city TEXT, date TEXT, km INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, summary TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


def _path(path: Path | None) -> Path:
    return Path(path) if path else config.DB_PATH


def _connect(path: Path | None) -> sqlite3.Connection:
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def init_db(path: Path | None = None) -> None:
    _connect(path).close()


def add_booking(kind: Literal["test_drive", "service"], path: Path | None = None, **fields) -> int:
    cols = ["name", "phone", "model", "city", "date", "km"]
    with _connect(path) as con:
        cur = con.execute(
            f"INSERT INTO bookings (kind, {', '.join(cols)}) VALUES (?, {', '.join('?' * len(cols))})",
            [kind] + [fields.get(c) for c in cols])
        return cur.lastrowid


def list_bookings(path: Path | None = None) -> list[dict]:
    with _connect(path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM bookings ORDER BY id")]


def add_ticket(summary: str, path: Path | None = None) -> int:
    with _connect(path) as con:
        return con.execute("INSERT INTO tickets (summary) VALUES (?)", [summary]).lastrowid


def list_tickets(path: Path | None = None) -> list[dict]:
    with _connect(path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM tickets ORDER BY id")]
