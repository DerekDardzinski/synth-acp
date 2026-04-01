"""Shared SQLite schema and helpers."""

from __future__ import annotations

SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,
    task        TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    to_agent    TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  INTEGER NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'chat',
    reply_to    INTEGER REFERENCES messages(id),
    delivered_at INTEGER
);
CREATE TABLE IF NOT EXISTS agent_commands (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    from_agent  TEXT NOT NULL,
    command     TEXT NOT NULL,
    payload     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    created_at  INTEGER NOT NULL
);
"""


def ensure_schema_sync(conn) -> None:
    """Execute schema DDL on a synchronous sqlite3 connection."""
    conn.executescript(SCHEMA)


async def ensure_schema_async(conn) -> None:
    """Execute schema DDL on an aiosqlite connection."""
    await conn.executescript(SCHEMA)
