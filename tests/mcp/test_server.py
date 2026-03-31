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
        "kind TEXT NOT NULL DEFAULT 'chat', "
        "reply_to INTEGER REFERENCES messages(id), "
        "delivered_at INTEGER);"
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


def _init_full_schema(db_path: Path) -> None:
    """Create the full schema including agent_commands table."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS agents ("
        "agent_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'active', registered INTEGER NOT NULL, "
        "parent TEXT, task TEXT);"
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, to_agent TEXT NOT NULL, body TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', created_at INTEGER NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'chat', "
        "reply_to INTEGER REFERENCES messages(id), "
        "delivered_at INTEGER);"
        "CREATE TABLE IF NOT EXISTS agent_commands ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
        "from_agent TEXT NOT NULL, command TEXT NOT NULL, payload TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', error TEXT, "
        "created_at INTEGER NOT NULL);"
    )
    conn.commit()
    conn.close()


def _register_agents_full(db_path: Path, agents: list[tuple[str, str | None, str | None]]) -> None:
    """Register agents with parent and task fields.

    Args:
        db_path: Path to the database.
        agents: List of (agent_id, parent, task) tuples.
    """
    _init_full_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    for aid, parent, task in agents:
        conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered, parent, task) "
            "VALUES (?, 'sess-1', 'active', 1000, ?, ?)",
            (aid, parent, task),
        )
    conn.commit()
    conn.close()


@pytest.mark.usefixtures("_env")
class TestLaunchAgent:
    def test_launch_agent_when_called_inserts_pending_command(self, db_path: Path) -> None:
        _init_full_schema(db_path)
        from synth_acp.mcp.server import launch_agent

        result = json.loads(
            launch_agent(
                agent_id="worker-1",
                agent_name="implementor",
                harness="kiro",
                cwd="/tmp",
                task="Fix auth",
                message="Start working",
            )
        )
        assert result["ok"] is True
        assert result["agent_id"] == "worker-1"

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT from_agent, command, payload, status FROM agent_commands WHERE session_id = 'sess-1'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "agent-a"
        assert row[1] == "launch"
        assert row[3] == "pending"
        payload = json.loads(row[2])
        assert payload["agent_id"] == "worker-1"
        assert payload["agent_name"] == "implementor"
        assert payload["harness"] == "kiro"
        assert payload["cwd"] == "/tmp"
        assert payload["task"] == "Fix auth"
        assert payload["message"] == "Start working"

    def test_launch_agent_when_at_capacity_returns_queued(self, db_path: Path) -> None:
        _register_agents_full(db_path, [("agent-a", None, None)])
        from synth_acp.mcp.server import launch_agent

        with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
            result = json.loads(
                launch_agent(
                    agent_id="worker-1",
                    agent_name="implementor",
                    harness="kiro",
                )
            )
        assert result["ok"] is True
        assert result["queued"] is True

        # Command is still written regardless
        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT command FROM agent_commands").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "launch"


@pytest.mark.usefixtures("_env")
class TestListAgents:
    def test_list_agents_when_agents_have_parent_includes_parent_and_task(
        self, db_path: Path
    ) -> None:
        _register_agents_full(
            db_path,
            [
                ("orchestrator", None, None),
                ("worker-1", "orchestrator", "Fix auth"),
                ("worker-2", "orchestrator", "Write tests"),
            ],
        )
        from synth_acp.mcp.server import list_agents

        result = json.loads(list_agents())
        by_id = {a["agent_id"]: a for a in result}
        assert by_id["worker-1"]["parent"] == "orchestrator"
        assert by_id["worker-1"]["task"] == "Fix auth"
        assert by_id["orchestrator"]["parent"] is None


@pytest.mark.usefixtures("_env")
class TestGetVisibleAgents:
    def test_get_visible_agents_when_local_mode_returns_family_only(self, db_path: Path) -> None:
        # Agent tree: coordinator→{auth, db, api}, auth→{helper}
        _register_agents_full(
            db_path,
            [
                ("coordinator", None, None),
                ("auth", "coordinator", "Auth work"),
                ("db", "coordinator", "DB work"),
                ("api", "coordinator", "API work"),
                ("helper", "auth", "Help auth"),
            ],
        )
        from synth_acp.mcp.server import _get_visible_agents

        # Test for "auth": should see coordinator (parent), db, api (siblings), helper (child)
        with (
            patch("synth_acp.mcp.server.AGENT_ID", "auth"),
            patch("synth_acp.mcp.server.COMMUNICATION_MODE", "LOCAL"),
        ):
            conn = sqlite3.connect(str(db_path))
            visible = set(_get_visible_agents(conn))
            conn.close()
        assert visible == {"coordinator", "db", "api", "helper"}

        # Test for "coordinator": should see auth, db, api (children only, no parent, no grandchildren)
        with (
            patch("synth_acp.mcp.server.AGENT_ID", "coordinator"),
            patch("synth_acp.mcp.server.COMMUNICATION_MODE", "LOCAL"),
        ):
            conn = sqlite3.connect(str(db_path))
            visible = set(_get_visible_agents(conn))
            conn.close()
        assert visible == {"auth", "db", "api"}


@pytest.mark.usefixtures("_env")
class TestSendMessageVisibility:
    def test_send_message_when_local_mode_and_target_not_visible_returns_error(
        self, db_path: Path
    ) -> None:
        _register_agents_full(
            db_path,
            [
                ("agent-a", "coordinator", None),
                ("unrelated", None, None),
            ],
        )
        from synth_acp.mcp.server import send_message

        with patch("synth_acp.mcp.server.COMMUNICATION_MODE", "LOCAL"):
            result = json.loads(send_message(to_agent="unrelated", body="hello"))

        assert "error" in result
        assert "Agent not visible" in result["error"]

        # Verify no message row inserted
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT * FROM messages").fetchall()
        conn.close()
        assert len(rows) == 0
