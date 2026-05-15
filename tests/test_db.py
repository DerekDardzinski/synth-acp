"""Tests for synth_acp.db embedding helpers."""

from __future__ import annotations

import sqlite3

from synth_acp.db import (
    ensure_schema_sync,
    get_unembedded_agents_sync,
    load_all_embeddings_sync,
    store_embedding_sync,
)


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    ensure_schema_sync(conn)
    return conn


class TestSessionEmbeddingsSchema:
    def test_ensure_schema_creates_session_embeddings_table(self) -> None:
        conn = _make_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='session_embeddings'"
        ).fetchall()
        assert tables == [("session_embeddings",)]

    def test_migration_drops_old_embeddings_table(self) -> None:
        """Old single-PK schema is detected and recreated with agent_id."""
        conn = sqlite3.connect(":memory:")
        # Create old schema without agent_id
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                registered INTEGER NOT NULL,
                parent TEXT,
                task TEXT,
                acp_session_id TEXT,
                harness TEXT,
                agent_mode TEXT,
                cwd TEXT,
                PRIMARY KEY (agent_id, session_id)
            );
            CREATE TABLE IF NOT EXISTS session_embeddings (
                session_id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO session_embeddings (session_id, text_hash, embedding, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("sess-1", "hash-1", b"\x00" * 1536, 1000),
        )
        conn.commit()
        # Run migration
        ensure_schema_sync(conn)
        # Verify new schema has agent_id column
        cur = conn.execute("PRAGMA table_info(session_embeddings)")
        col_names = {row[1] for row in cur.fetchall()}
        assert "agent_id" in col_names
        # Old data is gone (table was dropped and recreated)
        rows = conn.execute("SELECT * FROM session_embeddings").fetchall()
        assert rows == []

    def test_migration_preserves_new_schema(self) -> None:
        """Migration is idempotent on already-migrated DBs."""
        conn = _make_conn()
        # Insert a row with new schema
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-1", b"\x00" * 1536)
        # Re-run ensure_schema_sync (simulates restart)
        ensure_schema_sync(conn)
        # Row should survive
        result = load_all_embeddings_sync(conn)
        assert len(result) == 1
        assert result[0] == ("sess-1", "agent-1", b"\x00" * 1536)


class TestEmbeddingCRUD:
    def test_store_and_load_with_agent_id(self) -> None:
        """Roundtrip with agent_id returns correct tuple."""
        conn = _make_conn()
        blob = b"\x00" * 1536
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-abc", blob)
        result = load_all_embeddings_sync(conn)
        assert result == [("sess-1", "agent-1", blob)]

    def test_store_embedding_upserts_on_same_key(self) -> None:
        """Second store with same (session_id, agent_id) replaces, not duplicates."""
        conn = _make_conn()
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-1", b"\x01" * 1536)
        store_embedding_sync(conn, "sess-1", "agent-1", "hash-2", b"\x02" * 1536)
        result = load_all_embeddings_sync(conn)
        assert len(result) == 1
        assert result[0] == ("sess-1", "agent-1", b"\x02" * 1536)

    def test_get_unembedded_agents_returns_missing_pairs(self) -> None:
        """Only (session_id, agent_id) pairs without embeddings are returned."""
        conn = _make_conn()
        # Insert two agents in same session
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a1", "sess-1", "restorable", 1000),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a2", "sess-1", "restorable", 2000),
        )
        conn.commit()
        # Embed only a1
        store_embedding_sync(conn, "sess-1", "a1", "hash-1", b"\x00" * 1536)
        # Only a2 should be returned
        result = get_unembedded_agents_sync(conn)
        assert result == [("sess-1", "a2")]

    def test_load_all_embeddings_ordered_by_session_id(self) -> None:
        """Results are ordered by session_id for reduceat grouping."""
        conn = _make_conn()
        # Insert in reverse order
        store_embedding_sync(conn, "sess-b", "a1", "h1", b"\x01" * 1536)
        store_embedding_sync(conn, "sess-a", "a1", "h2", b"\x02" * 1536)
        result = load_all_embeddings_sync(conn)
        assert result[0][0] == "sess-a"
        assert result[1][0] == "sess-b"
