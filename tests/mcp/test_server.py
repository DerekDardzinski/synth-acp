"""Tests for synth-mcp server tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from synth_acp.mcp.server import create_mcp_server


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def _init_schema(db_path: Path) -> None:
    """Create the full schema."""
    from synth_acp.db import ensure_schema_sync
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_schema_sync(conn)
    conn.commit()
    conn.close()


def _register_agents(db_path: Path, agents: list[tuple[str, str | None, str | None]]) -> None:
    """Register agents with parent and task fields."""
    _init_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    for aid, parent, task in agents:
        conn.execute(
            "INSERT OR IGNORE INTO agents (agent_id, session_id, status, registered, parent, task) "
            "VALUES (?, 'sess-1', 'active', 1000, ?, ?)",
            (aid, parent, task),
        )
    conn.commit()
    conn.close()


def _get_tool(mcp_server, name: str):
    """Extract a tool function from a FastMCP server by name."""
    for tool in mcp_server._tool_manager._tools.values():
        if tool.fn.__name__ == name:
            return tool.fn
    raise KeyError(f"Tool {name!r} not found")


@pytest.fixture()
async def mcp_factory(db_path: Path):
    """Yield a factory that creates MCP servers and closes them all after the test."""
    servers = []

    def _make(db: str | None = None, session: str = "sess-1", agent: str = "agent-a", **kw):
        s = create_mcp_server(str(db or db_path), session, agent, **kw)
        servers.append(s)
        return s

    yield _make
    for s in servers:
        await s.close_db()


class TestSendMessage:
    async def test_send_message_when_broadcast_expands_to_individual_rows(self, db_path: Path, mcp_factory) -> None:
        _register_agents(db_path, [("agent-a", None, None), ("agent-b", None, None), ("agent-c", None, None)])
        server = mcp_factory()
        send_message = _get_tool(server, "send_message")

        result = json.loads(await send_message(to_agent="*", body="hello all"))
        assert len(result["message_ids"]) == 2

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT to_agent FROM messages ORDER BY to_agent").fetchall()
        conn.close()
        targets = [r[0] for r in rows]
        assert "agent-a" not in targets
        assert "agent-b" in targets
        assert "agent-c" in targets


class TestLaunchAgent:
    async def test_launch_agent_when_called_inserts_pending_command(self, db_path: Path, mcp_factory) -> None:
        _init_schema(db_path)
        server = mcp_factory()
        launch_agent = _get_tool(server, "launch_agent")

        result = json.loads(
            await launch_agent(
                agent_id="worker-1",
                harness="kiro",
                cwd="/tmp",
                agent_mode="kiro_planner",
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
        assert payload["harness"] == "kiro"

    async def test_launch_agent_when_at_capacity_returns_queued(self, db_path: Path, mcp_factory) -> None:
        _register_agents(db_path, [("agent-a", None, None)])
        server = mcp_factory()
        launch_agent = _get_tool(server, "launch_agent")

        with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
            result = json.loads(
                await launch_agent(agent_id="worker-1", harness="kiro", message="Start working")
            )
        assert result["ok"] is True
        assert result["queued"] is True


class TestListAgents:
    async def test_list_agents_when_agents_have_parent_includes_parent_and_task(
        self, db_path: Path, mcp_factory,
    ) -> None:
        _register_agents(
            db_path,
            [
                ("orchestrator", None, None),
                ("worker-1", "orchestrator", "Fix auth"),
                ("worker-2", "orchestrator", "Write tests"),
                ("agent-a", None, None),
            ],
        )
        server = mcp_factory()
        list_agents = _get_tool(server, "list_agents")

        result = json.loads(await list_agents())
        by_id = {a["agent_id"]: a for a in result}
        assert by_id["worker-1"]["parent"] == "orchestrator"
        assert by_id["worker-1"]["task"] == "Fix auth"


class TestMcpStartupValidation:
    def test_main_exits_with_missing_env_vars(self, monkeypatch) -> None:
        monkeypatch.delenv("SYNTH_SESSION_ID", raising=False)
        monkeypatch.delenv("SYNTH_DB_PATH", raising=False)
        monkeypatch.delenv("SYNTH_AGENT_ID", raising=False)
        from synth_acp.mcp.server import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

class TestMcpConnectionSafety:
    async def test_send_message_closes_conn_on_visibility_error(self, db_path: Path, mcp_factory) -> None:
        """Connection must survive when _get_visible_agents_async raises."""
        _register_agents(db_path, [("agent-a", None, None)])
        server = mcp_factory()
        send_message = _get_tool(server, "send_message")

        with patch(
            "synth_acp.models.visibility.get_visible_agents_async",
            side_effect=RuntimeError("db corruption"),
        ), pytest.raises(RuntimeError, match="db corruption"):
            await send_message(to_agent="agent-b", body="hi")

        # Verify DB is still accessible — leaked fd would cause issues
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("SELECT 1")

    async def test_list_agents_closes_conn_on_register_error(self, db_path: Path, mcp_factory) -> None:
        """Connection must survive when _ensure_registered raises."""
        _init_schema(db_path)
        server = mcp_factory()
        list_agents = _get_tool(server, "list_agents")

        # Force an error by corrupting the persistent connection after it's created
        # First call succeeds and creates the connection
        with patch(
            "synth_acp.models.visibility.get_visible_agents_async",
            side_effect=RuntimeError("connect failed"),
        ), pytest.raises(RuntimeError, match="connect failed"):
            await list_agents()

        # Original DB still accessible
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("SELECT 1")