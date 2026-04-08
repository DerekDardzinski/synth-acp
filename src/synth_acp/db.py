"""Shared SQLite schema and helpers."""

from __future__ import annotations

SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,
    task        TEXT,
    acp_session_id TEXT,
    harness     TEXT,
    agent_mode  TEXT,
    cwd         TEXT
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


_EXPIRE_SQL = (
    "DELETE FROM agents WHERE status = 'restorable'"
    " AND registered < strftime('%s', 'now', '-{days} days') * 1000"
)
_EXPIRE_ORPHAN_MESSAGES = (
    "DELETE FROM messages WHERE session_id NOT IN (SELECT session_id FROM agents)"
)
_EXPIRE_ORPHAN_COMMANDS = (
    "DELETE FROM agent_commands WHERE session_id NOT IN (SELECT session_id FROM agents)"
)


def expire_old_sessions_sync(conn, max_age_days: int = 30) -> None:
    """Delete restorable agents older than *max_age_days* and orphaned rows."""
    days = int(max_age_days)
    conn.execute(_EXPIRE_SQL.format(days=days))
    conn.execute(_EXPIRE_ORPHAN_MESSAGES)
    conn.execute(_EXPIRE_ORPHAN_COMMANDS)
    conn.commit()


async def expire_old_sessions_async(conn, max_age_days: int = 30) -> None:
    """Async variant of :func:`expire_old_sessions_sync`."""
    days = int(max_age_days)
    await conn.execute(
        "DELETE FROM agents WHERE status = 'restorable'"
        f" AND registered < strftime('%s', 'now', '-{days} days') * 1000"
    )
    await conn.execute(
        "DELETE FROM messages WHERE session_id NOT IN (SELECT session_id FROM agents)"
    )
    await conn.execute(
        "DELETE FROM agent_commands WHERE session_id NOT IN (SELECT session_id FROM agents)"
    )
    await conn.commit()
