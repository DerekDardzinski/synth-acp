"""Tests for synth-mcp server tools."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, patch

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


class TestSendMessage:
    async def test_send_message_when_broadcast_expands_to_individual_rows(self, db_path: Path) -> None:
        _register_agents(db_path, [("agent-a", None, None), ("agent-b", None, None), ("agent-c", None, None)])
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
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
    async def test_launch_agent_when_called_inserts_pending_command(self, db_path: Path) -> None:
        _init_schema(db_path)
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        launch_agent = _get_tool(server, "launch_agent")

        result = json.loads(
            await launch_agent(
                agent_id_param="worker-1",
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

    async def test_launch_agent_when_at_capacity_returns_queued(self, db_path: Path) -> None:
        _register_agents(db_path, [("agent-a", None, None)])
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        launch_agent = _get_tool(server, "launch_agent")

        with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
            result = json.loads(
                await launch_agent(agent_id_param="worker-1", harness="kiro")
            )
        assert result["ok"] is True
        assert result["queued"] is True


class TestListAgents:
    async def test_list_agents_when_agents_have_parent_includes_parent_and_task(
        self, db_path: Path
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
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        list_agents = _get_tool(server, "list_agents")

        result = json.loads(await list_agents())
        by_id = {a["agent_id"]: a for a in result}
        assert by_id["worker-1"]["parent"] == "orchestrator"
        assert by_id["worker-1"]["task"] == "Fix auth"


class TestDeregisterAgent:
    async def test_deregister_inserts_self_terminate_command(self, db_path: Path) -> None:
        _register_agents(db_path, [("agent-a", None, None)])
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        deregister_agent = _get_tool(server, "deregister_agent")

        result = json.loads(await deregister_agent())
        assert result["status"] == "inactive"

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT command, from_agent FROM agent_commands WHERE session_id = 'sess-1'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "self_terminate"
        assert row[1] == "agent-a"

    async def test_deregister_filters_by_session_id(self, db_path: Path) -> None:
        _register_agents(db_path, [("agent-a", None, None)])
        # Insert agent-a in a different session
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered) VALUES ('agent-a-other', 'other-sess', 'active', 1000)"
        )
        conn.commit()
        conn.close()

        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        deregister_agent = _get_tool(server, "deregister_agent")
        await deregister_agent()

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT status FROM agents WHERE agent_id = 'agent-a-other' AND session_id = 'other-sess'"
        ).fetchone()
        conn.close()
        assert row[0] == "active"  # Other session's agent unaffected


class TestMcpStartupValidation:
    def test_main_exits_with_missing_env_vars(self, monkeypatch) -> None:
        monkeypatch.delenv("SYNTH_SESSION_ID", raising=False)
        monkeypatch.delenv("SYNTH_DB_PATH", raising=False)
        monkeypatch.delenv("SYNTH_AGENT_ID", raising=False)
        from synth_acp.mcp.server import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_main_exits_with_empty_db_path(self, monkeypatch) -> None:
        monkeypatch.setenv("SYNTH_SESSION_ID", "s1")
        monkeypatch.setenv("SYNTH_DB_PATH", "")
        monkeypatch.setenv("SYNTH_AGENT_ID", "a1")
        from synth_acp.mcp.server import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestNotifyCallback:
    async def test_send_message_calls_notify(self, db_path: Path) -> None:
        _register_agents(db_path, [("agent-a", None, None), ("agent-b", None, None)])
        notify = AsyncMock()
        server = create_mcp_server(str(db_path), "sess-1", "agent-a", notify=notify)
        send_message = _get_tool(server, "send_message")

        await send_message(to_agent="agent-b", body="hi")
        notify.assert_awaited()


class TestMcpConnectionSafety:
    async def test_send_message_closes_conn_on_visibility_error(self, db_path: Path) -> None:
        """Connection must close even when _get_visible_agents_async raises."""
        _register_agents(db_path, [("agent-a", None, None)])
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        send_message = _get_tool(server, "send_message")

        with patch(
            "synth_acp.models.visibility.get_visible_agents",
            side_effect=RuntimeError("db corruption"),
        ), pytest.raises(RuntimeError, match="db corruption"):
            await send_message(to_agent="agent-b", body="hi")

        # Verify DB is still accessible — leaked fd would cause issues
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("SELECT 1")

    async def test_list_agents_closes_conn_on_register_error(self, db_path: Path) -> None:
        """Connection must close even when _ensure_registered raises."""
        _init_schema(db_path)
        server = create_mcp_server(str(db_path), "sess-1", "agent-a")
        list_agents = _get_tool(server, "list_agents")

        with patch(
            "synth_acp.mcp.server.aiosqlite.connect",
            side_effect=RuntimeError("connect failed"),
        ), pytest.raises(RuntimeError, match="connect failed"):
            await list_agents()

        # Original DB still accessible
        import aiosqlite

        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("SELECT 1")