"""Tests for synth_acp.db embedding helpers."""

from __future__ import annotations

import sqlite3

from synth_acp.db import (
    _build_embedding_text,
    ensure_schema_sync,
    get_unembedded_sessions_sync,
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


class TestEmbeddingCRUD:
    def test_store_and_load_embedding_roundtrip(self) -> None:
        conn = _make_conn()
        blob = b"\x00" * 1536
        store_embedding_sync(conn, "sess-1", "hash-abc", blob)
        result = load_all_embeddings_sync(conn)
        assert result == [("sess-1", blob)]

    def test_get_unembedded_sessions_returns_sessions_without_embeddings(self) -> None:
        conn = _make_conn()
        # Insert two restorable sessions
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a1", "sess-1", "restorable", 1000),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES (?, ?, ?, ?)",
            ("a2", "sess-2", "restorable", 2000),
        )
        conn.commit()
        # Embed only sess-1
        store_embedding_sync(conn, "sess-1", "hash-1", b"\x00" * 1536)
        # Only sess-2 should be returned
        result = get_unembedded_sessions_sync(conn)
        assert result == ["sess-2"]


class TestBuildEmbeddingText:
    def test_build_embedding_text_concatenates_fields(self) -> None:
        session = {
            "first_messages": ["fix the login bug", "also check auth"],
            "agents": ["orchestrator", "worker-1"],
            "cwd": "/home/user/my-project",
            "tasks": ["implement auth flow"],
        }
        text = _build_embedding_text(session)
        assert "fix the login bug" in text
        assert "also check auth" in text
        assert "orchestrator" in text
        assert "worker-1" in text
        assert "my-project" in text
        assert "implement auth flow" in text

    def test_build_embedding_text_truncates_to_200_words(self) -> None:
        session = {
            "first_messages": [" ".join(["word"] * 300)],
            "agents": ["agent-1"],
            "cwd": "/tmp/test",
            "tasks": ["a task"],
        }
        text = _build_embedding_text(session)
        assert len(text.split()) <= 200
