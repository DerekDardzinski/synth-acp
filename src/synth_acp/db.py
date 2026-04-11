"""Shared SQLite schema and helpers."""

from __future__ import annotations

SCHEMA = """\
CREATE TABLE IF NOT EXISTS agents (
    agent_id    TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    registered  INTEGER NOT NULL,
    parent      TEXT,
    task        TEXT,
    acp_session_id TEXT,
    harness     TEXT,
    agent_mode  TEXT,
    cwd         TEXT,
    PRIMARY KEY (agent_id, session_id)
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
CREATE TABLE IF NOT EXISTS ui_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ui_events_replay
    ON ui_events (session_id, agent_id, seq);
"""


def ensure_schema_sync(conn) -> None:
    """Execute schema DDL on a synchronous sqlite3 connection."""
    conn.executescript(SCHEMA)
    _migrate_schema_sync(conn)


async def ensure_schema_async(conn) -> None:
    """Execute schema DDL on an aiosqlite connection."""
    await conn.executescript(SCHEMA)
    await _migrate_schema_async(conn)


def _migrate_schema_sync(conn) -> None:
    """Apply one-time migrations for existing databases.

    Detects the old agent_id-only primary key and recreates the table with
    the correct composite (agent_id, session_id) key, preserving all rows.
    """
    cur = conn.execute("PRAGMA table_info(agents)")
    cols = {row[1]: row[5] for row in cur.fetchall()}  # name -> pk position
    pk_cols = [name for name, pk in cols.items() if pk > 0]
    if pk_cols == ["agent_id"]:
        conn.executescript("""
            ALTER TABLE agents RENAME TO agents_old;
            CREATE TABLE agents (
                agent_id    TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                registered  INTEGER NOT NULL,
                parent      TEXT,
                task        TEXT,
                acp_session_id TEXT,
                harness     TEXT,
                agent_mode  TEXT,
                cwd         TEXT,
                PRIMARY KEY (agent_id, session_id)
            );
            INSERT OR IGNORE INTO agents SELECT * FROM agents_old;
            DROP TABLE agents_old;
        """)
        conn.commit()


async def _migrate_schema_async(conn) -> None:
    """Async variant of :func:`_migrate_schema_sync`."""
    cur = await conn.execute("PRAGMA table_info(agents)")
    rows = await cur.fetchall()
    cols = {row[1]: row[5] for row in rows}
    pk_cols = [name for name, pk in cols.items() if pk > 0]
    if pk_cols == ["agent_id"]:
        await conn.executescript("""
            ALTER TABLE agents RENAME TO agents_old;
            CREATE TABLE agents (
                agent_id    TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'active',
                registered  INTEGER NOT NULL,
                parent      TEXT,
                task        TEXT,
                acp_session_id TEXT,
                harness     TEXT,
                agent_mode  TEXT,
                cwd         TEXT,
                PRIMARY KEY (agent_id, session_id)
            );
            INSERT OR IGNORE INTO agents SELECT * FROM agents_old;
            DROP TABLE agents_old;
        """)
        await conn.commit()


_EXPIRE_SQL = (
    "DELETE FROM agents WHERE status = 'restorable'"
    " AND registered < ?"
)
_EXPIRE_ORPHAN_MESSAGES = (
    "DELETE FROM messages WHERE session_id NOT IN (SELECT session_id FROM agents)"
)
_EXPIRE_ORPHAN_COMMANDS = (
    "DELETE FROM agent_commands WHERE session_id NOT IN (SELECT session_id FROM agents)"
)
_EXPIRE_ORPHAN_UI_EVENTS = (
    "DELETE FROM ui_events WHERE session_id NOT IN (SELECT session_id FROM agents)"
)


def _cutoff_ms(max_age_days: int) -> int:
    """Return a millisecond timestamp for *max_age_days* ago."""
    import time

    return int((time.time() - int(max_age_days) * 86400) * 1000)


def expire_old_sessions_sync(conn, max_age_days: int = 30) -> None:
    """Delete restorable agents older than *max_age_days* and orphaned rows."""
    conn.execute(_EXPIRE_SQL, (_cutoff_ms(max_age_days),))
    conn.execute(_EXPIRE_ORPHAN_MESSAGES)
    conn.execute(_EXPIRE_ORPHAN_COMMANDS)
    conn.execute(_EXPIRE_ORPHAN_UI_EVENTS)
    conn.commit()


async def expire_old_sessions_async(conn, max_age_days: int = 30) -> None:
    """Async variant of :func:`expire_old_sessions_sync`."""
    await conn.execute(_EXPIRE_SQL, (_cutoff_ms(max_age_days),))
    await conn.execute(_EXPIRE_ORPHAN_MESSAGES)
    await conn.execute(_EXPIRE_ORPHAN_COMMANDS)
    await conn.execute(_EXPIRE_ORPHAN_UI_EVENTS)
    await conn.commit()
