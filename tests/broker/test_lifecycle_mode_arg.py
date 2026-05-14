"""Tests for mode_arg CLI flag injection in AgentLifecycle."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentConfig
from synth_acp.models.config import HarnessEntry, SessionConfig


def _make_entry(mode_arg: str | None = "--agent") -> HarnessEntry:
    return HarnessEntry(
        identity="kiro",
        name="Kiro CLI",
        short_name="kiro",
        binary_names=["kiro-cli"],
        run_cmd="kiro-cli acp",
        mode_arg=mode_arg,
    )


def _make_lifecycle(
    agent_id: str,
    agent_mode: str | None,
    mode_arg: str | None,
    *,
    db_path: Path = Path("/tmp/unused.db"),
) -> AgentLifecycle:
    config = SessionConfig(
        project="test",
    )
    reg = AgentRegistry()
    lc = AgentLifecycle(
        config, reg, AsyncMock(), db_path=db_path, session_id="s1"
    )
    lc._harness_registry = [_make_entry(mode_arg)]
    return lc


def _mock_session() -> MagicMock:
    s = MagicMock()
    s.run = AsyncMock()
    s.run_restored = AsyncMock()
    s.state = "idle"
    s.set_session_created_callback = MagicMock()
    return s


class TestModeArgLaunch:
    async def test_mode_arg_and_agent_mode_appends_flag(self) -> None:
        lc = _make_lifecycle("a", agent_mode="code", mode_arg="--agent")
        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=_mock_session()) as mock_cls:
            await lc.launch("a", adhoc_config=AgentConfig(agent_id="a", harness="kiro", agent_mode="code"))
            assert mock_cls.call_args.kwargs["args"] == ["acp", "--agent", "code"]

    async def test_no_mode_arg_leaves_cmd_unchanged(self) -> None:
        lc = _make_lifecycle("a", agent_mode="code", mode_arg=None)
        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=_mock_session()) as mock_cls:
            await lc.launch("a", adhoc_config=AgentConfig(agent_id="a", harness="kiro", agent_mode="code"))
            assert mock_cls.call_args.kwargs["args"] == ["acp"]

    async def test_no_agent_mode_leaves_cmd_unchanged(self) -> None:
        lc = _make_lifecycle("a", agent_mode=None, mode_arg="--agent")
        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=_mock_session()) as mock_cls:
            await lc.launch("a", adhoc_config=AgentConfig(agent_id="a", harness="kiro"))
            assert mock_cls.call_args.kwargs["args"] == ["acp"]


class TestModeArgHandleLaunchCommand:
    async def test_mode_arg_appended_in_handle_launch_command(self, tmp_path: Path) -> None:
        lc = _make_lifecycle("x", agent_mode=None, mode_arg="--agent", db_path=tmp_path / "t.db")

        # Ensure schema exists for the DB writes
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "t.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema_sync(conn)
        conn.close()

        with (
            patch("synth_acp.broker.lifecycle.ACPSession", return_value=_mock_session()) as mock_cls,
            patch.object(lc, "update_command_status", new_callable=AsyncMock),
        ):
            await lc.handle_launch_command(
                cmd_id=1,
                from_agent="parent",
                data={"agent_id": "child", "harness": "kiro", "agent_mode": "code"},
            )
            assert mock_cls.call_args.kwargs["args"] == ["acp", "--agent", "code"]


class TestModeArgRestore:
    async def test_mode_arg_appended_in_restore(self) -> None:
        lc = _make_lifecycle("x", agent_mode=None, mode_arg="--agent")
        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=_mock_session()) as mock_cls:
            await lc.restore(
                agent_id="r1",
                acp_session_id="sess-1",
                harness="kiro",
                agent_mode="code",
                cwd="/tmp",
                parent=None,
            )
            assert mock_cls.call_args.kwargs["args"] == ["acp", "--agent", "code"]
