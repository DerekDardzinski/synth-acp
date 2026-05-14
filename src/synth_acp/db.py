"""Shared SQLite schema and helpers."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

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
CREATE INDEX IF NOT EXISTS idx_ui_events_user_prompts
    ON ui_events (session_id, event_type, seq);
"""


SESSION_EMBEDDINGS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS session_embeddings (
    session_id  TEXT PRIMARY KEY,
    text_hash   TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  INTEGER NOT NULL
);
"""


def ensure_schema_sync(conn) -> None:
    """Execute schema DDL on a synchronous sqlite3 connection."""
    conn.executescript(SCHEMA)
    conn.executescript(SESSION_EMBEDDINGS_SCHEMA)
    _migrate_schema_sync(conn)


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


_EXPIRE_SQL = (
    "DELETE FROM agents WHERE status IN ('restorable', 'active')"
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
    return int((time.time() - int(max_age_days) * 86400) * 1000)


def expire_old_sessions_sync(conn, max_age_days: int = 30) -> None:
    """Delete restorable agents older than *max_age_days* and orphaned rows."""
    conn.execute(_EXPIRE_SQL, (_cutoff_ms(max_age_days),))
    conn.execute(_EXPIRE_ORPHAN_MESSAGES)
    conn.execute(_EXPIRE_ORPHAN_COMMANDS)
    conn.execute(_EXPIRE_ORPHAN_UI_EVENTS)
    conn.commit()


# ------------------------------------------------------------------
# Embedding helpers
# ------------------------------------------------------------------


def store_embedding_sync(
    conn: sqlite3.Connection, session_id: str, text_hash: str, embedding_blob: bytes
) -> None:
    """Upsert a session embedding. embedding_blob is 1536 bytes (384 x float32)."""
    conn.execute(
        "INSERT OR REPLACE INTO session_embeddings (session_id, text_hash, embedding, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, text_hash, embedding_blob, int(time.time() * 1000)),
    )
    conn.commit()


def load_all_embeddings_sync(conn: sqlite3.Connection) -> list[tuple[str, bytes]]:
    """Return all (session_id, embedding_blob) pairs."""
    return conn.execute("SELECT session_id, embedding FROM session_embeddings").fetchall()


def get_unembedded_sessions_sync(conn: sqlite3.Connection) -> list[str]:
    """Return session_ids that exist in agents (restorable) but not in session_embeddings."""
    rows = conn.execute(
        "SELECT session_id FROM agents WHERE status IN ('restorable', 'active') "
        "GROUP BY session_id "
        "EXCEPT SELECT session_id FROM session_embeddings"
    ).fetchall()
    return [r[0] for r in rows]


def _build_embedding_text(session: dict) -> str:
    """Compose text for embedding from session metadata.

    Concatenates: first_messages + agent names + cwd basename + tasks.
    Truncated to ~200 words to stay under 256 tokens.
    """
    parts: list[str] = []
    for msg in session.get("first_messages", []):
        parts.append(msg)
    agents = session.get("agents", [])
    if agents:
        parts.append(", ".join(agents))
    cwd = session.get("cwd")
    if cwd:
        parts.append(Path(cwd).name)
    for task in session.get("tasks", []):
        parts.append(task)
    text = "\n".join(parts)
    words = text.split()
    if len(words) > 200:
        words = words[:200]
    return " ".join(words)


def _text_hash(text: str) -> str:
    """SHA256 hex digest of text for staleness detection."""
    return hashlib.sha256(text.encode()).hexdigest()

