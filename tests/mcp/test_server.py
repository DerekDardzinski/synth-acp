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

    async def test_launch_agent_when_at_capacity_returns_error(self, db_path: Path, mcp_factory) -> None:
        _register_agents(db_path, [("agent-a", None, None)])
        server = mcp_factory()
        launch_agent = _get_tool(server, "launch_agent")

        with patch.dict("os.environ", {"SYNTH_MAX_AGENTS": "1"}):
            result = json.loads(
                await launch_agent(agent_id="worker-1", harness="kiro", message="Start working")
            )
        assert "error" in result
        assert "Max agents" in result["error"]


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


def _retire(
    db_path: Path,
    agent_id: str,
    *,
    parent: str | None = None,
    task: str | None = None,
    retired_from: str | None = None,
    retired_at: int | None = None,
) -> None:
    """Insert one inactive agent: a handoff predecessor if retired_from is set."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR IGNORE INTO agents "
        "(agent_id, session_id, status, registered, parent, task, retired_from, retired_at) "
        "VALUES (?, 'sess-1', 'inactive', 1000, ?, ?, ?, ?)",
        (agent_id, parent, task, retired_from, retired_at),
    )
    conn.commit()
    conn.close()


class TestListAgentsResurrectable:
    """The surface an agent reads to pick which past self to wake.

    Every failure here is silent and acts on the wrong agent: a wrong generations_back
    wakes the wrong generation, a missing lineage_of makes the count ambiguous when the
    caller can see more than one lineage, and a leaked row offers a revival the broker
    would reject.
    """

    async def test_predecessors_are_numbered_newest_first_within_their_lineage(
        self, db_path: Path, mcp_factory,
    ) -> None:
        _register_agents(db_path, [("root", None, "current work")])
        _retire(db_path, "root.hdead0001", task="older", retired_from="root", retired_at=100)
        _retire(db_path, "root.hdead0002", task="newer", retired_from="root", retired_at=200)
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        by_id = {a["agent_id"]: a for a in result}
        assert by_id["root"]["status"] == "active"
        assert by_id["root"]["is_self"] is True
        assert by_id["root"]["generations_back"] is None
        # generations_back 1 is the DIRECT predecessor, which is the most recently retired.
        assert (by_id["root.hdead0002"]["status"], by_id["root.hdead0002"]["generations_back"]) == (
            "retired", 1,
        )
        assert (by_id["root.hdead0001"]["status"], by_id["root.hdead0001"]["generations_back"]) == (
            "retired", 2,
        )
        assert by_id["root.hdead0002"]["lineage_of"] == "root"
        # Self first, then live agents, then inactive newest-retired first.
        assert [a["agent_id"] for a in result] == [
            "root", "root.hdead0002", "root.hdead0001",
        ]

    async def test_terminated_child_is_listed_without_a_lineage(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """A terminated agent and a retired past self are both revivable but are not the
        same thing, and only one of them has a generation to count."""
        _register_agents(db_path, [("root", None, None)])
        _retire(db_path, "kid", parent="root", task="phase 1")
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        kid = next(a for a in result if a["agent_id"] == "kid")
        assert kid["status"] == "terminated"
        assert kid["lineage_of"] is None
        assert kid["generations_back"] is None
        assert kid["task"] == "phase 1"

    async def test_a_launched_childs_predecessor_is_counted_against_the_child(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """The caller sees two lineages: its own and its child's.  generations_back is
        meaningless without lineage_of naming which one it counts within."""
        _register_agents(db_path, [("root", None, None), ("worker", "root", "build")])
        _retire(
            db_path, "worker.hdead0003", parent="root", task="build",
            retired_from="worker", retired_at=300,
        )
        _retire(db_path, "root.hdead0004", task="plan", retired_from="root", retired_at=400)
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        by_id = {a["agent_id"]: a for a in result}
        assert by_id["worker.hdead0003"]["lineage_of"] == "worker"
        assert by_id["worker.hdead0003"]["generations_back"] == 1
        assert by_id["root.hdead0004"]["lineage_of"] == "root"
        assert by_id["root.hdead0004"]["generations_back"] == 1

    async def test_unrelated_retired_agents_are_absent(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """Listing a row the broker would refuse to resurrect is a dead end the agent
        cannot detect without trying it."""
        _register_agents(db_path, [("root", None, None), ("stranger", None, None)])
        _retire(
            db_path, "stranger.hdead0005",
            retired_from="stranger", retired_at=500,
        )
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        assert [a["agent_id"] for a in result if a["status"] != "active"] == []

    async def test_an_unauthorized_inactive_parent_is_not_marked_resurrectable(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """LOCAL visibility adds `parent` with NO status filter, unlike its children and
        sibling branches, so a dead parent reaches the listing without ever passing an
        authorization check.  Reporting it as revivable sends the agent to a
        resurrect_agent call the broker refuses."""
        _register_agents(db_path, [("worker", "root", "build")])
        _retire(db_path, "root", task="orchestrating")
        server = mcp_factory(agent="worker", communication_mode="LOCAL")

        result = json.loads(await _get_tool(server, "list_agents")())

        root = next(a for a in result if a["agent_id"] == "root")
        assert root["status"] == "terminated"
        assert root["resurrectable"] is False
        # And the caller's own revivable rows are still marked true.
        worker = next(a for a in result if a["agent_id"] == "worker")
        assert worker["resurrectable"] is False

    async def test_authorized_rows_are_marked_resurrectable(
        self, db_path: Path, mcp_factory,
    ) -> None:
        _register_agents(db_path, [("root", None, None)])
        _retire(db_path, "kid", parent="root", task="phase 1")
        _retire(db_path, "root.hdead0007", retired_from="root", retired_at=700)
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        by_id = {a["agent_id"]: a for a in result}
        assert by_id["kid"]["resurrectable"] is True
        assert by_id["root.hdead0007"]["resurrectable"] is True
        assert by_id["root"]["resurrectable"] is False

    async def test_a_resurrected_predecessor_keeps_lineage_but_loses_its_generation(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """Observed in a live session: once resurrected, a predecessor is active again.

        It still belongs to its lineage, so lineage_of stays — that is how a caller tells
        an active past self apart from an unrelated agent. generations_back goes null
        because it numbers revival candidates and this one is no longer a candidate.
        """
        _register_agents(db_path, [("root", None, None)])
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO agents "
            "(agent_id, session_id, status, registered, task, retired_from, retired_at) "
            "VALUES ('root.hdead0008', 'sess-1', 'active', 1000, 'past work', 'root', 800)"
        )
        conn.commit()
        conn.close()
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        pred = next(a for a in result if a["agent_id"] == "root.hdead0008")
        assert pred["status"] == "active"
        assert pred["lineage_of"] == "root"
        assert pred["generations_back"] is None
        assert pred["resurrectable"] is False

    async def test_live_agents_keep_their_existing_shape(
        self, db_path: Path, mcp_factory,
    ) -> None:
        """No live agent's status changes wording, and none gains a lineage."""
        _register_agents(
            db_path, [("root", None, None), ("worker", "root", "build")]
        )
        server = mcp_factory(agent="root")

        result = json.loads(await _get_tool(server, "list_agents")())

        assert all(a["status"] == "active" for a in result)
        assert all(a["lineage_of"] is None for a in result)
        assert all(a["generations_back"] is None for a in result)


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
        """Connection must survive when get_visible_agents raises."""
        _register_agents(db_path, [("agent-a", None, None)])
        server = mcp_factory()
        send_message = _get_tool(server, "send_message")

        with patch(
            "synth_acp.mcp.server.get_visible_agents",
            side_effect=RuntimeError("db corruption"),
        ), pytest.raises(RuntimeError, match="db corruption"):
            await send_message(to_agent="agent-b", body="hi")

        # Verify DB is still accessible — leaked fd would cause issues
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()

    async def test_list_agents_closes_conn_on_register_error(self, db_path: Path, mcp_factory) -> None:
        """Connection must survive when _ensure_registered raises."""
        _init_schema(db_path)
        server = mcp_factory()
        list_agents = _get_tool(server, "list_agents")

        # Force an error by patching get_visible_agents to raise
        with patch(
            "synth_acp.mcp.server.get_visible_agents",
            side_effect=RuntimeError("connect failed"),
        ), pytest.raises(RuntimeError, match="connect failed"):
            await list_agents()

        # Original DB still accessible
        conn = sqlite3.connect(str(db_path))
        conn.execute("SELECT 1")
        conn.close()

class TestHandoff:
    async def test_handoff_records_a_pending_command_and_returns_acceptance(
        self, db_path: Path, mcp_factory
    ) -> None:
        """The payload must carry agent_id, or the broker's self-service authorization
        check rejects the handoff for a reason the calling agent never sees."""
        _init_schema(db_path)
        notified: list[bool] = []

        async def notify() -> None:
            notified.append(True)

        server = mcp_factory(agent="worker", notify=notify)
        handoff = _get_tool(server, "handoff")
        brief = "Context: mid-refactor.\n\nNext: finish db.py."

        result = json.loads(await handoff(handoff_message=brief))

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT id, from_agent, command, payload, status FROM agent_commands"
            ).fetchone()
        finally:
            conn.close()
        assert result == {"accepted": True, "command_id": row[0]}
        assert (row[1], row[2], row[4]) == ("worker", "handoff", "pending")
        payload = json.loads(row[3])
        assert payload == {"agent_id": "worker", "handoff_message": brief}
        assert notified == [True]

    async def test_handoff_reports_acceptance_not_completion(
        self, db_path: Path, mcp_factory
    ) -> None:
        """It must NOT copy launch_agent's poll. That loop treats any status other than
        'pending' as terminal, so a handoff sitting at 'processing' would be reported as
        SUCCESS on the first 0.3s poll -- before its successor exists. And carrying out a
        handoff kills the process group running this very tool call, so no completion
        response could be delivered at all.
        """
        _init_schema(db_path)
        server = mcp_factory(agent="worker")
        handoff = _get_tool(server, "handoff")

        result = json.loads(await handoff(handoff_message="brief"))

        # Acceptance only: no ok/error verdict is claimed either way.
        assert result["accepted"] is True
        assert "ok" not in result
        assert "error" not in result

    def test_handoff_does_not_poll_for_a_terminal_status(self) -> None:
        """Structural guard: the source of the handoff tool must contain no poll loop.
        A behavioral test cannot distinguish 'polled and got lucky' from 'never polled'."""
        import ast
        import inspect

        import synth_acp.mcp.server as server_module

        src = inspect.getsource(server_module)
        fn = next(
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "handoff"
        )
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.For | ast.While)]
        assert "sleep" not in ast.dump(fn)
