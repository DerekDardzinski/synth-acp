"""Tests for synth-mcp server tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Create a test database path."""
    return tmp_path / "test.db"


@pytest.fixture()
def _env(db_path: Path) -> None:
    """Patch environment variables for the MCP server module."""
    with (
        patch("synth_acp.mcp.server.SESSION_ID", "sess-1"),
        patch("synth_acp.mcp.server.DB_PATH", str(db_path)),
        patch("synth_acp.mcp.server.AGENT_ID", "agent-a"),
    ):
        yield  # type: ignore[misc]


def _register_agents(db_path: Path, *agent_ids: str) -> None:
    """Register agents directly in the database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS agents ("
        "agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'active', registered INTEGER NOT NULL);"
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, to_agent TEXT NOT NULL, body TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', created_at INTEGER NOT NULL, "
        "claimed_at INTEGER);"
    )
    for aid in agent_ids:
        conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered) VALUES (?, 'sess-1', 'active', 1000)",
            (aid,),
        )
    conn.commit()
    conn.close()


@pytest.mark.usefixtures("_env")
class TestSendMessage:
    def test_send_message_when_broadcast_expands_to_individual_rows(self, db_path: Path) -> None:
        _register_agents(db_path, "agent-a", "agent-b", "agent-c")
        from synth_acp.mcp.server import send_message

        result = json.loads(send_message(to_agent="*", body="hello all"))
        assert len(result["message_ids"]) == 2

        # Verify rows exist for agent-b and agent-c, not agent-a
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT to_agent FROM messages ORDER BY to_agent").fetchall()
        conn.close()
        targets = [r[0] for r in rows]
        assert "agent-a" not in targets
        assert "agent-b" in targets
        assert "agent-c" in targets
