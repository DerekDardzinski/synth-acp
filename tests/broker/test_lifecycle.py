"""Tests for AgentLifecycle."""

from __future__ import annotations

import ast
import asyncio
import logging
import sqlite3
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synth_acp.broker.lifecycle import (
    AgentLifecycle,
    DiscoveredAgent,
    format_available_agents,
)
from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentConfig, AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentHandedOff, BrokerError, HookFired

pytestmark = pytest.mark.usefixtures("available_harness_binaries")


def _config(*ids: str) -> SessionConfig:
    return SessionConfig(
        project="test",
    )


class TestTaskCleanup:
    async def test_run_task_removed_after_agent_exits(self) -> None:
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.agent_id = "a"

        async def fake_run() -> None:
            pass

        mock_session.run = fake_run
        reg.register("a", mock_session)

        task = lc._make_run_task("a", mock_session)
        lc._tasks["a"] = task
        await task
        await asyncio.sleep(0)  # Let done callback fire
        assert "a" not in lc._tasks


class TestPromptGuard:
    async def test_prompt_rejects_non_idle_agent(self) -> None:
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.BUSY
        reg.register("a", mock_session)

        await lc.prompt("a", "hello")
        assert any(isinstance(e, BrokerError) for e in events)


class TestLifecycleShutdown:
    async def test_shutdown_terminates_all_then_cancels_tasks(self) -> None:
        """Shutdown must force_kill all agents, then cancel remaining tasks."""
        config = _config("busy-agent", "idle-agent")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        busy = AsyncMock()
        busy.state = AgentState.BUSY
        busy.agent_id = "busy-agent"
        busy.force_kill = MagicMock()
        reg.register("busy-agent", busy)

        idle = AsyncMock()
        idle.state = AgentState.IDLE
        idle.agent_id = "idle-agent"
        idle.force_kill = MagicMock()
        reg.register("idle-agent", idle)

        await lc.shutdown()

        busy.force_kill.assert_called_once()
        idle.force_kill.assert_called_once()

    async def test_shutdown_terminates_agents_concurrently(self) -> None:
        """force_kill is sync so shutdown should complete near-instantly for N agents."""
        config = _config("a", "b", "c")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        killed: list[str] = []
        for aid in ("a", "b", "c"):
            session = AsyncMock()
            session.state = AgentState.IDLE
            session.agent_id = aid
            session.force_kill = MagicMock(side_effect=lambda _aid=aid: killed.append(_aid))
            reg.register(aid, session)

        t0 = asyncio.get_event_loop().time()
        await lc.shutdown()
        elapsed = asyncio.get_event_loop().time() - t0

        assert set(killed) == {"a", "b", "c"}
        # force_kill is sync — shutdown should be well under 1s
        assert elapsed < 0.5, f"Shutdown took {elapsed:.2f}s — unexpectedly slow"

    async def test_terminate_times_out_on_unresponsive_agent(self, tmp_path: Path) -> None:
        """If session.terminate() hangs, lifecycle must not block forever."""
        config = _config("stuck")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        lc._terminate_timeout = 0.1  # Fast timeout for testing

        stuck_session = AsyncMock()
        stuck_session.state = AgentState.IDLE
        stuck_session.agent_id = "stuck"

        async def hang_forever() -> None:
            await asyncio.sleep(60)

        stuck_session.terminate = hang_forever
        reg.register("stuck", stuck_session)

        # Ensure schema exists for the terminate DB writes
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema_sync(conn)
        conn.close()

        t0 = asyncio.get_event_loop().time()
        await lc.terminate("stuck")
        elapsed = asyncio.get_event_loop().time() - t0
        assert elapsed < 1.0


class TestResurrect:
    """Tests for handle_resurrect_command."""

    async def _make_lifecycle(self, tmp_path: Path) -> tuple[AgentLifecycle, list]:
        config = _config()
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.execute("PRAGMA journal_mode=WAL")
        ensure_schema_sync(conn)
        conn.close()
        return lc, events

    def _insert_agent(
        self,
        lc: AgentLifecycle,
        agent_id: str,
        *,
        status: str = "inactive",
        parent: str | None = None,
        harness: str = "kiro",
        acp_session_id: str | None = None,
        cwd: str = "/tmp",
        retired_from: str | None = None,
    ) -> None:
        import sqlite3

        conn = sqlite3.connect(str(lc._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent, harness, acp_session_id, cwd, retired_from) "
            "VALUES (?, ?, ?, 1000, ?, ?, ?, ?, ?)",
            (agent_id, lc._session_id, status, parent, harness, acp_session_id, cwd, retired_from),
        )
        conn.commit()
        conn.close()

    def _insert_command(self, lc: AgentLifecycle, cmd_id: int = 1) -> int:
        import sqlite3

        conn = sqlite3.connect(str(lc._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO agent_commands (id, session_id, from_agent, command, payload, status, created_at) "
            "VALUES (?, ?, 'test', 'resurrect', '{}', 'pending', 1000)",
            (cmd_id, lc._session_id),
        )
        conn.commit()
        conn.close()
        return cmd_id

    def _get_command_status(self, lc: AgentLifecycle, cmd_id: int) -> tuple[str, str | None]:
        import sqlite3

        conn = sqlite3.connect(str(lc._db_path))
        row = conn.execute(
            "SELECT status, error FROM agent_commands WHERE id = ?", (cmd_id,)
        ).fetchone()
        conn.close()
        return (row[0], row[1]) if row else ("not_found", None)

    def _get_agent_status(self, lc: AgentLifecycle, agent_id: str) -> str | None:
        import sqlite3

        conn = sqlite3.connect(str(lc._db_path))
        row = conn.execute(
            "SELECT status FROM agents WHERE agent_id = ? AND session_id = ?",
            (agent_id, lc._session_id),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    async def test_resurrect_rejects_wrong_parent(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        self._insert_agent(lc, "child", parent="parent-agent")
        cmd_id = self._insert_command(lc)

        await lc.handle_resurrect_command(cmd_id, "other-agent", {"agent_id": "child"})

        status, error = self._get_command_status(lc, cmd_id)
        assert status == "rejected"
        assert "Not authorized" in (error or "")

    async def test_resurrect_authorizes_a_root_agents_predecessor(self, tmp_path: Path) -> None:
        """The whole point of retired_from.

        A handoff never rewrites parent, so a ROOT agent's predecessor keeps parent NULL
        and no caller can ever match on parentage.  Before retired_from existed, the
        successor holding the original id was told "Not authorized" for its own past self.
        """
        lc, _ = await self._make_lifecycle(tmp_path)
        self._insert_agent(
            lc, "root.hdead0001",
            status="inactive", parent=None, retired_from="root",
            acp_session_id="sess-1",
        )
        cmd_id = self._insert_command(lc)
        lc.restore = AsyncMock()

        await lc.handle_resurrect_command(cmd_id, "root", {"agent_id": "root.hdead0001"})

        assert self._get_command_status(lc, cmd_id) == ("processed", None)
        assert self._get_agent_status(lc, "root.hdead0001") == "active"

    async def test_resurrect_rejects_a_predecessor_of_a_different_lineage(
        self, tmp_path: Path
    ) -> None:
        """retired_from widens authorization by exactly one relation, not to any caller.

        Silent failure if this regresses: any agent could revive any other agent's past
        self, and the revived agent would appear in a lineage it never belonged to.
        """
        lc, _ = await self._make_lifecycle(tmp_path)
        self._insert_agent(
            lc, "stranger.hdead0002",
            status="inactive", parent=None, retired_from="stranger",
        )
        cmd_id = self._insert_command(lc)

        await lc.handle_resurrect_command(cmd_id, "root", {"agent_id": "stranger.hdead0002"})

        status, error = self._get_command_status(lc, cmd_id)
        assert status == "rejected"
        assert "Not authorized" in (error or "")
        assert self._get_agent_status(lc, "stranger.hdead0002") == "inactive"

    async def test_resurrect_rejects_non_inactive_agent(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        self._insert_agent(lc, "child", status="active", parent="parent-agent")
        cmd_id = self._insert_command(lc)

        await lc.handle_resurrect_command(cmd_id, "parent-agent", {"agent_id": "child"})

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "not inactive" in errors[0].message

    async def test_resurrect_success_calls_restore_and_updates_status(self, tmp_path: Path) -> None:
        lc, events = await self._make_lifecycle(tmp_path)
        self._insert_agent(
            lc, "child",
            status="inactive",
            parent="parent-agent",
            harness="kiro",
            acp_session_id="sess-123",
            cwd="/tmp",
        )
        cmd_id = self._insert_command(lc)

        lc.restore = AsyncMock()

        await lc.handle_resurrect_command(cmd_id, "parent-agent", {"agent_id": "child"})

        lc.restore.assert_awaited_once_with(
            agent_id="child",
            acp_session_id="sess-123",
            harness="kiro",
            agent_mode=None,
            cwd="/tmp",
            parent="parent-agent",
        )
        status, error = self._get_command_status(lc, cmd_id)
        assert status == "processed"
        assert error is None
        assert self._get_agent_status(lc, "child") == "active"


class TestPromptLock:
    async def test_concurrent_prompts_serialize(self) -> None:
        """Two concurrent prompt() calls for the same agent serialize — second sees BUSY."""
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.agent_id = "a"

        async def fake_prompt(text: str) -> None:
            # Simulate the session going BUSY during prompt
            mock_session.state = AgentState.BUSY
            await asyncio.sleep(0.05)

        mock_session.prompt = fake_prompt
        reg.register("a", mock_session)

        t1 = asyncio.create_task(lc.prompt("a", "first"))
        # Small yield to let t1 acquire the lock and start
        await asyncio.sleep(0)
        t2 = asyncio.create_task(lc.prompt("a", "second"))
        await asyncio.gather(t1, t2)

        # Second call waited for lock, then saw BUSY state → emitted warning
        warnings = [e for e in events if isinstance(e, BrokerError) and "cannot prompt" in e.message]
        assert len(warnings) == 1

    async def test_different_agents_no_contention(self) -> None:
        """Prompts to different agents run concurrently (no cross-agent blocking)."""
        config = _config("a", "b")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        for aid in ("a", "b"):
            mock_session = AsyncMock()
            mock_session.state = AgentState.IDLE
            mock_session.agent_id = aid
            reg.register(aid, mock_session)

        # Patch prompt to block so we can observe concurrency
        original_prompt = lc.prompt
        lock_acquired_at: dict[str, float] = {}

        async def timed_prompt(agent_id: str, text: str) -> None:
            lock_acquired_at[agent_id] = asyncio.get_event_loop().time()
            await original_prompt(agent_id, text)

        t1 = asyncio.create_task(timed_prompt("a", "hello"))
        t2 = asyncio.create_task(timed_prompt("b", "hello"))
        await asyncio.gather(t1, t2)

        # Both agents were prompted (no blocking between them)
        assert "a" in lock_acquired_at
        assert "b" in lock_acquired_at



class TestStartupHookForDynamicChild:
    async def test_startup_hook_fires_for_dynamic_child(self, tmp_path: Path) -> None:
        """on_agent_startup must prepend context for dynamically launched agents."""
        from unittest.mock import patch

        from synth_acp.models.events import HookFired

        config = _config("orchestrator")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        submitted: list[tuple] = []

        async def mock_submit(agent_id: str, text: str, source: str, from_agent: str | None) -> None:
            submitted.append((agent_id, text, source, from_agent))

        lc.set_submit_prompt(mock_submit)

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ensure_schema_sync(conn)
        conn.close()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session), \
             patch("synth_acp.broker.lifecycle.load_startup_context", return_value="<ctx>{agent_id},{parent_id},{task},{harness}</ctx>\n\n"):
            await lc.handle_launch_command(
                cmd_id=1,
                from_agent="orchestrator",
                data={
                    "agent_id": "child-1",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Do work",
                    "message": "Hello child",
                },
            )

        # Verify startup context was prepended with correct slots
        assert len(submitted) == 1
        enqueued_msg = submitted[0][1]
        assert enqueued_msg.startswith("<ctx>child-1,orchestrator,Do work,kiro</ctx>")
        assert enqueued_msg.endswith("Hello child")

        hook_events = [e for e in events if isinstance(e, HookFired) and e.hook_name == "on_agent_startup"]
        assert len(hook_events) == 1

    async def test_startup_hook_inactive_skips_injection_but_marks_prompted(self, tmp_path: Path) -> None:
        """active=False skips context injection but still marks agent as first-prompted."""
        from unittest.mock import patch

        from synth_acp.models.config import HooksConfig, SettingsConfig, StartupHookConfig

        config = SessionConfig(
            project="test",
            settings=SettingsConfig(hooks=HooksConfig(on_agent_startup=StartupHookConfig(active=False))),
        )
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        submitted: list[tuple] = []

        async def mock_submit(agent_id: str, text: str, source: str, from_agent: str | None) -> None:
            submitted.append((agent_id, text, source, from_agent))

        lc.set_submit_prompt(mock_submit)

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ensure_schema_sync(conn)
        conn.close()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()
        mock_session.state = AgentState.IDLE

        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
            await lc.handle_launch_command(
                cmd_id=1,
                from_agent="orchestrator",
                data={
                    "agent_id": "child-1",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Do work",
                    "message": "Hello child",
                },
            )

        # No startup context prepended
        assert len(submitted) == 1
        enqueued_msg = submitted[0][1]
        assert enqueued_msg == "Hello child"

        # Agent is marked as first-prompted — subsequent prompt() won't inject
        assert "child-1" in lc._first_prompted

        # Verify no double-injection on next prompt
        reg.register("child-1", mock_session)

        async def fake_prompt(text: str) -> None:
            pass

        mock_session.prompt = fake_prompt
        result = await lc.prompt("child-1", "second message")
        assert result is True
        # prompt task was created with the raw text (no startup context)
        # Since _first_prompted contains child-1, no injection happens

    async def test_startup_hook_fires_for_root_agent_on_first_prompt(self) -> None:
        """on_agent_startup must prepend context for root agents with parent_id='', task=''."""
        from unittest.mock import patch

        from synth_acp.models.events import HookFired

        config = _config("root-agent")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.agent_id = "root-agent"
        reg.register("root-agent", mock_session)

        with patch("synth_acp.broker.lifecycle.load_startup_context", return_value="<ctx>{agent_id},{parent_id},{task}</ctx>\n\n"):
            await lc.prompt("root-agent", "Hello root")

        # session.prompt is called with the prepended text
        mock_session.prompt.assert_called_once_with("<ctx>root-agent,,</ctx>\n\nHello root")

        hook_events = [e for e in events if isinstance(e, HookFired) and e.hook_name == "on_agent_startup"]
        assert len(hook_events) == 1


class TestFireMessageHookActiveFlag:
    async def test_fire_message_hook_respects_active_flag(self, tmp_path: Path) -> None:
        """_fire_message_hook must skip when hook.active=False."""
        from synth_acp.models.config import MessageHook

        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ensure_schema_sync(conn)
        conn.close()

        hook = MessageHook(active=False, recipients="mesh", template="Agent {agent_id} joined.")
        await lc._fire_message_hook(hook, "a", "task", "parent", "on_agent_join")

        # No HookFired events — hook was inactive
        from synth_acp.models.events import HookFired

        hook_events = [e for e in events if isinstance(e, HookFired)]
        assert len(hook_events) == 0


# ---------------------------------------------------------------------------
# Race condition reproducers: lifecycle serialization
# ---------------------------------------------------------------------------


class TestLifecycleSerialization:
    """Verify that resurrect, terminate, and handle_launch_command are serialized via agent_lock."""

    async def test_concurrent_resurrect_same_agent_orphans_a_session(
        self, tmp_path: Path,
    ) -> None:
        """Two concurrent resurrect() calls must serialize — only one session constructed."""
        config = _config("a")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "synth.db", session_id="s1")

        fetch_started = asyncio.Event()
        fetch_unblock = asyncio.Event()
        fetch_count = 0

        async def stub_db_op(fn: object) -> object:
            nonlocal fetch_count
            fetch_count += 1
            if fetch_count <= 2:
                fetch_started.set()
                await fetch_unblock.wait()
                return ("acp-session-1", "kiro", None, ".", None, "", "inactive")
            return None

        lc._db_op = stub_db_op  # type: ignore[method-assign]

        constructed: list[int] = []

        async def stub_restore(*args: object, **kwargs: object) -> None:
            agent_id = str(kwargs.get("agent_id") or args[0])
            sess = AsyncMock()
            sess.state = AgentState.IDLE
            sess.agent_id = agent_id
            constructed.append(id(sess))
            reg.register(agent_id, sess)

            async def noop() -> None:
                pass

            lc._tasks[agent_id] = asyncio.create_task(noop(), name=f"run-{agent_id}")

        lc.restore = stub_restore  # type: ignore[method-assign]

        old_session = AsyncMock()
        old_session.state = AgentState.TERMINATED
        reg.register("a", old_session)

        t1 = asyncio.create_task(lc.resurrect("a"))
        t2 = asyncio.create_task(lc.resurrect("a"))

        await asyncio.wait_for(fetch_started.wait(), timeout=1.0)
        for _ in range(5):
            await asyncio.sleep(0)

        fetch_unblock.set()
        await asyncio.gather(t1, t2, return_exceptions=True)

        assert len(constructed) == 1, (
            f"Bug: resurrect() not serialized. {len(constructed)} sessions constructed."
        )

    async def test_concurrent_terminate_same_agent(self, tmp_path: Path) -> None:
        """Two concurrent terminate() calls must serialize — only one calls session.terminate()."""
        config = _config("a")
        reg = AgentRegistry()
        events: list[object] = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "synth.db", session_id="s1")

        terminate_started = asyncio.Event()
        terminate_unblock = asyncio.Event()
        terminate_call_count = 0

        session = AsyncMock()
        session.state = AgentState.IDLE

        async def slow_terminate() -> None:
            nonlocal terminate_call_count
            terminate_call_count += 1
            terminate_started.set()
            await terminate_unblock.wait()
            session.state = AgentState.TERMINATED

        session.terminate = slow_terminate
        reg.register("a", session)

        async def stub_db_op(fn: object) -> object:
            return None

        lc._db_op = stub_db_op  # type: ignore[method-assign]

        t1 = asyncio.create_task(lc.terminate("a"))
        t2 = asyncio.create_task(lc.terminate("a"))

        await asyncio.wait_for(terminate_started.wait(), timeout=1.0)
        terminate_unblock.set()
        await asyncio.gather(t1, t2, return_exceptions=True)

        assert terminate_call_count == 1, (
            f"Bug: terminate() not serialized. session.terminate() called {terminate_call_count} times."
        )

    async def test_concurrent_handle_launch_command_same_agent(self, tmp_path: Path) -> None:
        """Two concurrent handle_launch_command() calls must serialize — only one succeeds."""
        config = _config("a")
        reg = AgentRegistry()
        events: list[object] = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "synth.db", session_id="s1")

        db_started = asyncio.Event()
        db_unblock = asyncio.Event()
        db_call_count = 0

        async def slow_db_op(fn: object) -> object:
            nonlocal db_call_count
            db_call_count += 1
            if db_call_count == 1:
                db_started.set()
                await db_unblock.wait()
            return None

        lc._db_op = slow_db_op  # type: ignore[method-assign]

        from synth_acp.harnesses import HarnessEntry

        lc._harness_registry = [
            HarnessEntry(
                identity="kiro",
                name="Kiro CLI",
                short_name="kiro",
                binary_names=["kiro-cli"],
                run_cmd="echo",
                mode_arg=None,
            )
        ]

        command_statuses: list[tuple[int, str]] = []

        async def track_status(cmd_id: int, status: str, error: str | None = None) -> None:
            command_statuses.append((cmd_id, status))

        lc.update_command_status = track_status  # type: ignore[method-assign]

        data = {"agent_id": "new-agent", "harness": "kiro", "cwd": "."}

        t1 = asyncio.create_task(lc.handle_launch_command(1, "parent", data))
        t2 = asyncio.create_task(lc.handle_launch_command(2, "parent", data))

        await asyncio.wait_for(db_started.wait(), timeout=1.0)
        for _ in range(5):
            await asyncio.sleep(0)

        db_unblock.set()
        await asyncio.gather(t1, t2, return_exceptions=True)

        # Clean up spawned run tasks
        tasks = list(lc._tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        processed = [s for s in command_statuses if s[1] == "processed"]
        rejected = [s for s in command_statuses if s[1] == "rejected"]
        assert len(processed) == 1, (
            f"Bug: handle_launch_command() not serialized. Statuses: {command_statuses}"
        )
        assert len(rejected) == 1, (
            f"Bug: second launch should be rejected. Statuses: {command_statuses}"
        )


class TestSetConfigOption:
    async def test_set_config_option_delegates_to_session(self) -> None:
        """set_config_option must call session.set_config_option with correct args."""
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        reg.register("a", mock_session)

        await lc.set_config_option("a", "mode", "architect")

        mock_session.set_config_option.assert_awaited_once_with("mode", "architect")

    async def test_set_config_option_when_not_idle_emits_error(self) -> None:
        """set_config_option on a non-idle agent must emit BrokerError and not call session."""
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.BUSY
        reg.register("a", mock_session)

        await lc.set_config_option("a", "effort", "high")

        mock_session.set_config_option.assert_not_awaited()
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "cannot change config option" in errors[0].message


class TestSetAgent:
    async def test_set_agent_calls_fork_with_agent_and_updates_db(self, tmp_path: Path) -> None:
        """set_agent must call fork_with_agent and persist new session_id to DB.
        Silent failure: DB not updated means session restore uses stale session_id."""
        import sqlite3

        from synth_acp.db import ensure_schema_sync

        db_path = tmp_path / "synth.db"
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=db_path, session_id="s1")

        # Seed DB with agent row
        with sqlite3.connect(str(db_path)) as conn:
            ensure_schema_sync(conn)
            conn.execute(
                "INSERT INTO agents (agent_id, session_id, status, registered, harness, acp_session_id) "
                "VALUES (?, ?, 'active', 1000, 'claude', 'old-acp-sess')",
                ("a", "s1"),
            )
            conn.commit()

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session.fork_with_agent = AsyncMock(return_value="new-acp-sess")
        reg.register("a", mock_session)

        await lc.set_agent("a", "code-planner")

        mock_session.fork_with_agent.assert_awaited_once_with("code-planner")

        # Verify DB updated
        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                "SELECT acp_session_id FROM agents WHERE agent_id = ? AND session_id = ?",
                ("a", "s1"),
            ).fetchone()
        assert row[0] == "new-acp-sess"

    async def test_set_agent_emits_error_when_not_idle(self) -> None:
        """set_agent on a non-idle agent must emit BrokerError and not call fork.
        Silent failure: fork attempted on busy agent causes undefined behavior."""
        config = _config("a")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.BUSY
        mock_session.fork_with_agent = AsyncMock()
        reg.register("a", mock_session)

        await lc.set_agent("a", "code-planner")

        mock_session.fork_with_agent.assert_not_awaited()
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "cannot switch agent" in errors[0].message


class TestAvailableAgents:
    def test_format_available_agents_populated(self) -> None:
        """Populated list renders compact two-space lines; empty description drops the em dash."""
        agents = [
            DiscoveredAgent(
                qualified_name="plan", name="plan", description="Planning agent", source="user"
            ),
            DiscoveredAgent(qualified_name="x", name="x", description="", source="user"),
        ]
        result = format_available_agents(agents)
        lines = result.split("\n")
        assert lines == ["  - plan — Planning agent", "  - x"]
        # No-description agent must not emit a dangling em dash.
        assert "x —" not in result

    def test_format_available_agents_empty(self) -> None:
        """Empty list renders the graceful no-configurations line, never an empty block."""
        assert (
            format_available_agents([])
            == "  (no named agent configurations are available in this harness)"
        )

    async def test_prompt_first_prompt_injects_available_agents(self) -> None:
        """First prompt renders the discovered configs into the {available_agents} slot."""
        from unittest.mock import patch

        config = _config("a")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")

        mock_session = AsyncMock()
        mock_session.state = AgentState.IDLE
        mock_session._cwd = ""
        reg.register("a", mock_session)
        reg.set_harness("a", "kiro")

        lc.get_discovered_agents = lambda _agent_id: [  # type: ignore[method-assign]
            DiscoveredAgent(
                qualified_name="plan", name="plan", description="Planning agent", source="user"
            )
        ]

        # Patch the template source so the test is hermetic and does not depend
        # on the developer's live ~/.synth/context.md (mirrors sibling tests).
        with patch(
            "synth_acp.broker.lifecycle.load_startup_context",
            return_value="<ctx>{available_agents}</ctx>\n\n",
        ):
            assert await lc.prompt("a", "hello") is True
            task = lc._tasks.get("prompt-a")
            if task:
                await task

        rendered = mock_session.prompt.call_args[0][0]
        assert "  - plan — Planning agent" in rendered
        assert rendered.endswith("hello")


class TestGetDiscoveredAgents:
    async def test_caches_by_harness_identity(self) -> None:
        """Discovery runs once per harness identity; subsequent calls hit the cache.

        Silent failure: a filesystem scan runs on every startup-context render."""
        from unittest.mock import patch as _patch

        from synth_acp.harnesses import HarnessEntry

        config = _config("agent-1")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")
        lc._harness_registry = [
            HarnessEntry(
                identity="claude",
                name="Claude Code",
                short_name="claude",
                binary_names=["claude"],
                run_cmd="claude acp",
            )
        ]
        reg.set_harness("agent-1", "claude")

        fake_agents = [
            DiscoveredAgent(qualified_name="planner", name="planner", description="", source="user")
        ]

        with _patch(
            "synth_acp.broker.lifecycle.discover_agents", return_value=fake_agents
        ) as mock_discover:
            result1 = lc.get_discovered_agents("agent-1")
            result2 = lc.get_discovered_agents("agent-1")

        assert result1 == fake_agents
        assert result2 == fake_agents
        assert mock_discover.call_count == 1

    def test_returns_empty_for_unknown_harness(self) -> None:
        """Unknown harness short_name resolves to [] without raising."""
        config = _config("agent-1")
        reg = AgentRegistry()

        async def sink(e: object) -> None:
            pass

        lc = AgentLifecycle(config, reg, sink, db_path=Path("/tmp/unused.db"), session_id="s1")
        reg.set_harness("agent-1", "nonexistent")

        assert lc.get_discovered_agents("agent-1") == []


class TestExpireOldSessions:
    """Tests for expire_old_sessions scheduling.

    Every stub that parks the worker does so on a threading.Event with a bounded
    wait, released in a finally block, with the task drained before the test
    returns.  Cancelling the asyncio task does NOT interrupt the to_thread
    worker, so an unreleased stub strands an executor thread and hangs
    interpreter shutdown for the whole suite.
    """

    @staticmethod
    def _lifecycle(tmp_path: Path) -> AgentLifecycle:
        async def sink(e: object) -> None:
            pass

        return AgentLifecycle(
            _config("a"), AgentRegistry(), sink,
            db_path=tmp_path / "session.db", session_id="s1",
        )

    async def test_returns_before_work_completes(self, tmp_path: Path) -> None:
        """Awaiting the method must await the scheduling only, never the expiry."""
        lc = self._lifecycle(tmp_path)
        gate = threading.Event()

        def slow(conn: object, **kwargs: object) -> None:
            gate.wait(timeout=1.0)

        with patch("synth_acp.broker.lifecycle.expire_old_sessions_sync", slow):
            try:
                await lc.expire_old_sessions()
                assert lc._expiry_task is not None
                assert not lc._expiry_task.done()
            finally:
                gate.set()
                await asyncio.wait([lc._expiry_task], timeout=1.0)

    async def test_task_retained_and_runs_dry_run(self, tmp_path: Path) -> None:
        """The task is held on _expiry_task and can only preview, never delete."""
        lc = self._lifecycle(tmp_path)
        calls: list[dict[str, object]] = []

        def record(conn: object, **kwargs: object) -> str:
            calls.append(kwargs)
            return "report"

        with patch("synth_acp.broker.lifecycle.expire_old_sessions_sync", record):
            await lc.expire_old_sessions()
            assert isinstance(lc._expiry_task, asyncio.Task)
            await lc._expiry_task

        assert calls == [{"dry_run": True}]

    async def test_failure_is_logged_not_propagated(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A corrupt or locked database must not take startup down."""
        lc = self._lifecycle(tmp_path)

        def boom(conn: object, **kwargs: object) -> None:
            raise sqlite3.OperationalError("database is locked")

        with (
            patch("synth_acp.broker.lifecycle.expire_old_sessions_sync", boom),
            caplog.at_level(logging.WARNING, logger="synth_acp.broker.lifecycle"),
        ):
            await lc.expire_old_sessions()
            assert lc._expiry_task is not None
            await lc._expiry_task

        assert lc._expiry_task.exception() is None
        assert "Session expiry preview failed" in caplog.text

    async def test_shutdown_cancels_expiry_task_without_error_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Shutdown must not hang on a long expiry, nor report cancellation as an error."""
        lc = self._lifecycle(tmp_path)
        gate = threading.Event()

        def slow(conn: object, **kwargs: object) -> None:
            gate.wait(timeout=1.0)

        with (
            patch("synth_acp.broker.lifecycle.expire_old_sessions_sync", slow),
            caplog.at_level(logging.WARNING, logger="synth_acp.broker.lifecycle"),
        ):
            try:
                await lc.expire_old_sessions()
                task = lc._expiry_task
                assert task is not None
                await lc.shutdown()
                assert task.cancelled()
            finally:
                gate.set()
                await asyncio.wait([task], timeout=1.0)

        assert caplog.records == []


# ── Agent handoff ──


def _seed_handoff_db(db_path: Path, session_id: str, agent_id: str = "worker") -> None:
    from synth_acp.db import ensure_schema_sync

    conn = sqlite3.connect(str(db_path))
    try:
        ensure_schema_sync(conn)
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent, task, "
            "harness, agent_mode, cwd, acp_session_id) "
            "VALUES (?, ?, 'active', 100, 'boss', 'the task', 'kiro', NULL, '.', 'acp-1')",
            (agent_id, session_id),
        )
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent) "
            "VALUES ('kid', ?, 'active', 101, ?)",
            (session_id, agent_id),
        )
        conn.commit()
    finally:
        conn.close()


def _handoff_lifecycle(tmp_path: Path, events: list) -> tuple[AgentLifecycle, AgentRegistry, MagicMock]:
    """A lifecycle whose handoff runs for real against a temp database.

    Only session construction and run-task creation are replaced, so nothing spawns a
    subprocess; every decision the handoff makes stays real.
    """
    async def sink(e: object) -> None:
        events.append(e)

    reg = AgentRegistry()
    lc = AgentLifecycle(
        _config("worker"), reg, sink, db_path=tmp_path / "synth.db", session_id="s1"
    )
    _seed_handoff_db(tmp_path / "synth.db", "s1")

    predecessor = MagicMock()
    predecessor.state = AgentState.IDLE
    predecessor.agent_id = "worker"
    predecessor.force_kill = MagicMock()
    predecessor.terminate = AsyncMock()
    predecessor.rename = MagicMock()
    reg.register("worker", predecessor)
    reg.set_parent("worker", "boss")
    reg.set_parent("kid", "worker")
    reg.set_harness("worker", "kiro")

    def _build(_cfg: object, _entry: object) -> MagicMock:
        session = MagicMock(state=AgentState.INITIALIZING)
        # run/run_restored must be awaitable: restore() wraps them in create_task.
        session.run = AsyncMock()
        session.run_restored = AsyncMock()
        return session

    def _run_task(_aid: str, _session: object) -> asyncio.Task:
        return asyncio.create_task(asyncio.sleep(0))

    lc._build_session = MagicMock(side_effect=_build)
    lc._make_run_task = MagicMock(side_effect=_run_task)
    return lc, reg, predecessor


class TestHandoff:
    async def test_handoff_retires_predecessor_and_starts_successor_under_original_id(
        self, tmp_path: Path
    ) -> None:
        events: list = []
        lc, reg, predecessor = _handoff_lifecycle(tmp_path, events)

        result = await lc.handoff("worker", "HANDOFF BRIEF")

        assert result.successor_started is True
        assert result.error is None
        assert result.retired_agent_id.startswith("worker.h")
        assert len(result.retired_agent_id) == len("worker.h") + 8

        # force_kill, never terminate(): terminate warns and continues after 5s, leaving
        # a live subprocess that can still write to the database being renamed.
        predecessor.force_kill.assert_called_once()
        predecessor.terminate.assert_not_awaited()

        # The successor occupies the original id; the predecessor is reachable under the
        # retired one, and its own registry entry was moved, not copied.
        assert reg.has_session("worker") is True
        assert reg.get_session("worker") is not predecessor

        handed_off = [e for e in events if isinstance(e, AgentHandedOff)]
        assert len(handed_off) == 1
        assert handed_off[0].agent_id == "worker"
        assert handed_off[0].retired_agent_id == result.retired_agent_id
        assert handed_off[0].parent == "boss"
        assert handed_off[0].task == "the task"

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            assert conn.execute(
                "SELECT status, acp_session_id FROM agents WHERE agent_id = ?",
                (result.retired_agent_id,),
            ).fetchone() == ("inactive", "acp-1")
            assert conn.execute(
                "SELECT status, acp_session_id FROM agents WHERE agent_id = 'worker'"
            ).fetchone() == ("active", None)
        finally:
            conn.close()

    async def test_handoff_never_unregisters_or_restores(self, tmp_path: Path) -> None:
        """unregister() pops the lock the handoff is holding, so mutual exclusion would
        silently evaporate; restore() replays the full conversation the feature exists to
        discard, and suppresses the startup hook."""
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        reg.unregister = MagicMock(side_effect=AssertionError("unregister must not be called"))
        lc.restore = AsyncMock(side_effect=AssertionError("restore must not be called"))

        result = await lc.handoff("worker", "brief")

        assert result.successor_started is True
        lc.restore.assert_not_awaited()

    async def test_handoff_message_reaches_the_successor_verbatim(self, tmp_path: Path) -> None:
        """The predecessor's brief is the successor's ONLY inherited context, so it must
        not be summarized, truncated or reformatted -- only prefixed with startup context
        the same way a launched child's initial message is."""
        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)
        seeded: list[str] = []

        class _State:
            async def drain_agent_journal(self, agent_id: str) -> None: ...

            def apply_handoff_rekey(self, result: object) -> None: ...

            def seed_first_prompt(self, agent_id: str, text: str, from_agent: str) -> None:
                seeded.append(text)

        lc.set_handoff_state(_State())
        brief = "Context: I was mid-refactor.\n\nNext: finish db.py, then run the gate."

        await lc.handoff("worker", brief)

        assert len(seeded) == 1
        assert seeded[0].endswith(brief)

    async def test_handoff_succeeds_at_max_agents(self, tmp_path: Path) -> None:
        """Both max-agent gates would reject a handoff exactly when a tree is full, which
        is when it is needed most, and the operation is net-zero."""
        import os

        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)
        with patch.dict(os.environ, {"SYNTH_MAX_AGENTS": "1"}):
            result = await lc.handoff("worker", "brief")
        assert result.successor_started is True

    async def test_a_failed_successor_reports_without_raising_and_clears_the_reservation(
        self, tmp_path: Path
    ) -> None:
        """The rename is durable and the retired id is real, so this reports rather than
        raises. A reservation left behind would refuse every later prompt to that id for
        the rest of the session, and the only symptom would be prompts going nowhere."""
        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)
        lc._build_session = MagicMock(side_effect=RuntimeError("no binary"))

        result = await lc.handoff("worker", "brief")

        assert result.successor_started is False
        assert result.error is not None and "no binary" in result.error
        assert result.retired_agent_id.startswith("worker.h")
        assert "worker" not in lc._reserved_first_prompt


class TestHandoffCommand:
    @staticmethod
    def _cmd(lc: AgentLifecycle, payload: str = '{"agent_id": "worker"}') -> int:
        conn = sqlite3.connect(str(lc._db_path))
        try:
            cur = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, "
                "created_at) VALUES ('s1', 'worker', 'handoff', ?, 'processing', 100)",
                (payload,),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    @staticmethod
    def _status(lc: AgentLifecycle, cmd_id: int) -> tuple:
        conn = sqlite3.connect(str(lc._db_path))
        try:
            return conn.execute(
                "SELECT status, error FROM agent_commands WHERE id = ?", (cmd_id,)
            ).fetchone()
        finally:
            conn.close()

    async def test_rejects_when_agent_id_is_not_the_caller(self, tmp_path: Path) -> None:
        """Self-service: an agent hands ITSELF off. This is deliberately not the
        parent-derived check terminate and resurrect use."""
        events: list = []
        lc, _, predecessor = _handoff_lifecycle(tmp_path, events)
        cmd_id = self._cmd(lc, '{"agent_id": "someone-else"}')

        await lc.handle_handoff_command(cmd_id, "worker", {"agent_id": "someone-else"})

        status, error = self._status(lc, cmd_id)
        assert status == "rejected"
        assert "only hand itself off" in error
        predecessor.force_kill.assert_not_called()
        assert [e for e in events if isinstance(e, BrokerError)] == []

    async def test_rename_error_is_rejected_without_a_broker_error(self, tmp_path: Path) -> None:
        """Nothing was changed and nothing is durable, so surfacing a BrokerError would
        tell the user something broke when the database is byte-identical."""
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        reg.unregister("worker")  # no live session -> AgentRenameError
        cmd_id = self._cmd(lc)

        await lc.handle_handoff_command(cmd_id, "worker", {"agent_id": "worker"})

        status, error = self._status(lc, cmd_id)
        assert status == "rejected"
        assert "no live session" in error
        assert [e for e in events if isinstance(e, BrokerError)] == []

    async def test_successor_failure_emits_a_broker_error_naming_both_ids(
        self, tmp_path: Path
    ) -> None:
        """handle_resurrect_command reports 'processed' even when resurrect() bailed out
        with a BrokerError, because it infers success from a None return. Taking a typed
        result is what keeps this method from repeating that."""
        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)
        lc._build_session = MagicMock(side_effect=RuntimeError("no binary"))
        cmd_id = self._cmd(lc)

        await lc.handle_handoff_command(cmd_id, "worker", {"agent_id": "worker"})

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        message = errors[0].message
        assert "worker" in message
        assert ".h" in message
        assert "resurrectable" in message
        status, error = self._status(lc, cmd_id)
        assert status == "rejected"
        assert error is not None and "no binary" in error

    async def test_successful_handoff_is_processed(self, tmp_path: Path) -> None:
        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)
        cmd_id = self._cmd(lc)

        await lc.handle_handoff_command(cmd_id, "worker", {"agent_id": "worker"})

        assert self._status(lc, cmd_id) == ("processed", None)
        assert [e for e in events if isinstance(e, BrokerError)] == []

    async def test_shutting_down_rejects_without_starting_anything(self, tmp_path: Path) -> None:
        events: list = []
        lc, _, predecessor = _handoff_lifecycle(tmp_path, events)
        lc._shutting_down = True
        cmd_id = self._cmd(lc)

        await lc.handle_handoff_command(cmd_id, "worker", {"agent_id": "worker"})

        status, _ = self._status(lc, cmd_id)
        assert status == "rejected"
        predecessor.force_kill.assert_not_called()
        assert [e for e in events if isinstance(e, BrokerError)] == []


class TestHandoffShutdownOrdering:
    async def test_shutdown_waits_for_a_registered_successor_and_never_cancels(
        self, tmp_path: Path
    ) -> None:
        """A handoff PAST its own shutdown check has already registered its successor, so
        shutdown must await it before taking the force-kill and _tasks snapshots --
        otherwise the successor is started behind a completed kill pass and survives as an
        orphaned agent subprocess. The owned task is never cancelled: cancelling it
        between the committed rename and the in-memory re-key is the partial-state failure
        this design removed."""
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        released = asyncio.Event()
        registered = asyncio.Event()
        successors: list = []

        def _build(cfg: object, entry: object) -> MagicMock:
            s = MagicMock(state=AgentState.INITIALIZING)
            successors.append(s)
            return s

        lc._build_session = MagicMock(side_effect=_build)

        async def sink(event: object) -> None:
            events.append(event)
            # Fired at the very end of handoff, so registration has already happened.
            if isinstance(event, HookFired):
                registered.set()
                await released.wait()

        lc._sink = sink

        handoff = asyncio.create_task(lc.handoff("worker", "brief"))
        shutdown: asyncio.Task | None = None
        try:
            await asyncio.wait_for(registered.wait(), timeout=1.0)
            assert reg.get_session("worker") is successors[0], (
                "successor must be registered by now"
            )

            shutdown = asyncio.create_task(lc.shutdown())
            await asyncio.sleep(0)
            assert lc._shutting_down is True
            assert not shutdown.done(), "shutdown must wait for the in-flight handoff"

            released.set()
            result = await asyncio.wait_for(handoff, timeout=1.0)
            await asyncio.wait_for(shutdown, timeout=1.0)
        finally:
            released.set()
            pending = [task for task in (handoff, shutdown) if task is not None]
            for task in pending:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        assert result.successor_started is True
        assert not handoff.cancelled()
        # The successor was in the kill snapshot, which is the point of the ordering.
        successors[0].force_kill.assert_called_once()

    async def test_a_handoff_still_before_the_check_starts_no_successor(
        self, tmp_path: Path
    ) -> None:
        """The mirror case: a handoff that has not reached its check when the flag goes up
        aborts and starts nothing, so there is no successor to leak."""
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        released = asyncio.Event()

        class _State:
            async def drain_agent_journal(self, agent_id: str) -> None:
                lc._shutting_down = True  # shutdown began while we were blocked
                await released.wait()

            def apply_handoff_rekey(self, result: object) -> None: ...

            def seed_first_prompt(self, agent_id: str, text: str, from_agent: str) -> None: ...

        lc.set_handoff_state(_State())
        handoff = asyncio.create_task(lc.handoff("worker", "brief"))
        await asyncio.sleep(0)
        released.set()
        result = await handoff

        assert result.successor_started is False
        assert "shutting down" in (result.error or "")
        lc._make_run_task.assert_not_called()


class TestHandoffHasNoCancellationMachinery:
    """Criterion 21a-i: five mechanisms were deleted when the whole command became an
    owned task nothing cancels. Their return would mean the ownership boundary was
    misplaced again, and nothing else would notice -- the code would still pass every
    behavioral test above.
    """

    _CHECKED = (
        ("lifecycle.py", "handoff"),
        ("lifecycle.py", "handle_handoff_command"),
        ("broker.py", "apply_handoff_rekey"),
        ("broker.py", "drain_agent_journal"),
    )

    @staticmethod
    def _fn(filename: str, name: str) -> ast.AST:
        src = Path(__file__).resolve().parents[2] / "src" / "synth_acp" / "broker" / filename
        return next(
            n
            for n in ast.walk(ast.parse(src.read_text()))
            if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef) and n.name == name
        )

    def test_no_shield_no_result_no_cancelled_handler(self) -> None:
        offenders: list[str] = []
        for filename, name in self._CHECKED:
            fn = self._fn(filename, name)
            for node in ast.walk(fn):
                if isinstance(node, ast.Attribute) and node.attr in {"shield", "result"}:
                    offenders.append(f"{filename}:{name} uses .{node.attr}")
                if isinstance(node, ast.ExceptHandler) and "CancelledError" in ast.dump(node):
                    offenders.append(f"{filename}:{name} handles CancelledError")
        assert offenders == []

    def test_no_inner_only_task_in_the_handoff_path(self) -> None:
        """The handoff itself must not spawn a task for part of its own work. Creating the
        ONE long-lived serialized command tail is a different thing and is not checked
        here."""
        offenders: list[str] = []
        for name in ("handoff", "handle_handoff_command"):
            fn = self._fn("lifecycle.py", name)
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "create_task"
                ):
                    offenders.append(f"lifecycle.py:{name} creates a task")
        assert offenders == []


class TestHandoffPostCommitFailures:
    async def test_a_failing_startup_hook_emission_reports_instead_of_raising(
        self, tmp_path: Path
    ) -> None:
        """The emission happens AFTER the rename committed, so raising would leave the
        caller unable to settle its command, the reservation installed -- refusing every
        later prompt to this id -- and the required BrokerError never emitted."""
        events: list = []
        lc, _, _ = _handoff_lifecycle(tmp_path, events)

        async def sink(event: object) -> None:
            events.append(event)
            if isinstance(event, HookFired):
                raise RuntimeError("hook sink failed")

        lc._sink = sink

        result = await lc.handoff("worker", "brief")

        assert result.successor_started is False
        assert result.error is not None and "hook sink failed" in result.error
        assert result.retired_agent_id.startswith("worker.h")
        assert "worker" not in lc._reserved_first_prompt


class TestHandoffPredecessorResurrection:
    async def test_terminating_the_successor_preserves_the_predecessor(
        self, tmp_path: Path
    ) -> None:
        """Root AC 5. Terminating the successor must not orphan or destroy the retired
        predecessor, which is the row that carries the lineage authorizing its revival.

        The failure is silent in the worst way: `terminate` marks the SUCCESSOR inactive,
        so a run that also cleared the predecessor's retired_from or deleted its row would
        leave a session that still looks healthy while the predecessor has quietly become
        unreachable — resurrect would answer "Not authorized" for an agent that was
        revivable a moment earlier.
        """
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)

        result = await lc.handoff("worker", "brief")
        retired = result.retired_agent_id
        # The successor session comes from the shared builder, which only makes the run
        # coroutines awaitable; terminate() is awaited by lifecycle.terminate.
        reg.get_session("worker").terminate = AsyncMock()

        await lc.terminate("worker")

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            # The predecessor keeps status, ACP session and lineage, so it is still
            # resurrectable by whoever holds 'worker'.
            assert conn.execute(
                "SELECT status, retired_from, acp_session_id FROM agents "
                "WHERE agent_id = ? AND session_id = 's1'",
                (retired,),
            ).fetchone() == ("inactive", "worker", "acp-1")
            assert conn.execute(
                "SELECT status FROM agents WHERE agent_id = 'worker' AND session_id = 's1'"
            ).fetchone() == ("inactive",)
        finally:
            conn.close()

    async def test_the_predecessor_is_resurrectable_with_its_full_transcript(
        self, tmp_path: Path
    ) -> None:
        """Root AC 2. The retired row is 'inactive' with its acp_session_id intact, which
        is exactly what resurrect() requires, and the rename moved its journal, so its
        transcript comes back under the new id while the original id starts clean."""
        from synth_acp.broker.broker import ACPBroker

        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            conn.executemany(
                "INSERT INTO ui_events (session_id, agent_id, seq, event_type, payload, "
                "created_at) VALUES ('s1', 'worker', ?, 'UserPromptSubmitted', ?, 100)",
                [(0, '{"agent_id": "worker", "text": "first"}'),
                 (1, '{"agent_id": "worker", "text": "second"}')],
            )
            conn.commit()
        finally:
            conn.close()

        result = await lc.handoff("worker", "brief")
        retired = result.retired_agent_id
        reg.unregister(retired)  # the predecessor's process is gone

        await lc.resurrect(retired)

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            assert conn.execute(
                "SELECT status FROM agents WHERE agent_id = ?", (retired,)
            ).fetchone() == ("active",)
        finally:
            conn.close()
        assert reg.has_session(retired) is True

        journal = await ACPBroker.load_journal(
            MagicMock(_db_path=tmp_path / "synth.db"), retired, "s1"
        )
        assert [e.text for e in journal] == ["first", "second"]
        assert await ACPBroker.load_journal(
            MagicMock(_db_path=tmp_path / "synth.db"), "worker", "s1"
        ) == []


class TestSuccessorAuthority:
    async def test_the_successor_can_terminate_its_inherited_child(
        self, tmp_path: Path
    ) -> None:
        """Criterion 12. handle_terminate_command authorizes by comparing from_agent
        against the child's parent pointer. Rewriting that pointer to the retired id would
        make the successor's legitimate command fail "Not authorized" while agents.parent
        correctly still said the original id -- two stores disagreeing, with no error."""
        events: list = []
        lc, reg, _ = _handoff_lifecycle(tmp_path, events)
        child = MagicMock()
        child.state = AgentState.IDLE
        child.terminate = AsyncMock()
        reg.register("kid", child)

        await lc.handoff("worker", "brief")

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            cur = conn.execute(
                "INSERT INTO agent_commands (session_id, from_agent, command, payload, status, "
                "created_at) VALUES ('s1', 'worker', 'terminate', '{}', 'processing', 100)"
            )
            conn.commit()
            cmd_id = cur.lastrowid
        finally:
            conn.close()

        await lc.handle_terminate_command(cmd_id, "worker", {"agent_id": "kid"})

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            status, error = conn.execute(
                "SELECT status, error FROM agent_commands WHERE id = ?", (cmd_id,)
            ).fetchone()
        finally:
            conn.close()
        assert (status, error) == ("processed", None)


class TestHarnessBinaryPreflight:
    """A harness whose spawn program is absent must be reported, never spawned.

    The failure this prevents is the one that motivated the check: a package-manager
    wrapper in the spawn path starts fine and then stalls resolving what to run, so
    create_subprocess_exec never raises and the session sits in INITIALIZING with no
    error for the life of the process.
    """

    def _lifecycle(self, events: list) -> AgentLifecycle:
        async def sink(e: object) -> None:
            events.append(e)

        return AgentLifecycle(
            _config(), AgentRegistry(), sink, db_path=Path("/tmp/unused.db"), session_id="s1"
        )

    def test_message_names_the_program_and_the_install_command(self) -> None:
        lc = self._lifecycle([])
        entry = next(e for e in lc._harness_registry if e.short_name == "claude")
        with patch("synth_acp.broker.lifecycle.shutil.which", return_value=None):
            msg = lc._check_harness_binary(entry)
        assert msg is not None
        assert "claude-agent-acp" in msg
        assert "npm install -g @agentclientprotocol/claude-agent-acp" in msg

    def test_checks_the_spawned_program_not_the_harness_tool(self) -> None:
        """binary_names is 'claude'; the program spawned is 'claude-agent-acp'. A check
        against binary_names would pass on a machine that cannot launch."""
        lc = self._lifecycle([])
        entry = next(e for e in lc._harness_registry if e.short_name == "claude")
        with patch(
            "synth_acp.broker.lifecycle.shutil.which",
            side_effect=lambda n: "/usr/bin/claude" if n == "claude" else None,
        ):
            assert lc._check_harness_binary(entry) is not None

    async def test_launch_emits_error_and_registers_no_session(self) -> None:
        events: list = []
        lc = self._lifecycle(events)
        cfg = AgentConfig(agent_id="a", harness="claude", cwd="/tmp")
        with patch("synth_acp.broker.lifecycle.shutil.which", return_value=None):
            await lc.launch("a", adhoc_config=cfg)
        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "claude-agent-acp" in errors[0].message
        assert not lc._registry.has_session("a")
        assert "a" not in lc._tasks


class TestHarnessBinaryPreflightAcrossEveryLaunchPath:
    """Four call sites construct a session; each must check the spawn program first.

    The silent failure: deleting the check from any one path leaves the other three
    passing, and that path spawns a program that is not there. For a package-manager
    wrapper that does not even raise -- it starts, then stalls resolving what to run --
    so the agent sits in INITIALIZING with no error for the life of the process.
    """

    async def test_command_launch_rejects_and_registers_no_session(self, tmp_path: Path) -> None:
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            ensure_schema_sync(conn)
            # The command row is created by the MCP server, not by the lifecycle, which
            # only UPDATEs it. Without it there is nothing for the rejection to land on.
            conn.execute(
                "INSERT INTO agent_commands (id, session_id, from_agent, command, payload, "
                "status, created_at) VALUES (1, 's1', 'parent', 'launch_agent', '{}', "
                "'pending', 100)"
            )
            conn.commit()
        finally:
            conn.close()

        reg = AgentRegistry()
        lc = AgentLifecycle(
            _config(), reg, sink, db_path=tmp_path / "synth.db", session_id="s1"
        )
        with patch("synth_acp.broker.lifecycle.shutil.which", return_value=None):
            await lc.handle_launch_command(
                1, "parent", {"agent_id": "kid", "harness": "claude", "cwd": str(tmp_path)}
            )

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "claude-agent-acp" in errors[0].message
        assert not reg.has_session("kid")

        # The MCP caller's durable command record must say why, not just go quiet.
        conn = sqlite3.connect(str(tmp_path / "synth.db"))
        try:
            status, error = conn.execute(
                "SELECT status, error FROM agent_commands WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        assert status == "rejected"
        assert "claude-agent-acp" in error

    async def test_handoff_reports_it_and_starts_no_successor(self, tmp_path: Path) -> None:
        """The rename is already durable here, so the error must be reported rather than
        raised, and no successor may be constructed."""
        events: list = []
        lc, _reg, _predecessor = _handoff_lifecycle(tmp_path, events)

        with patch("synth_acp.broker.lifecycle.shutil.which", return_value=None):
            result = await lc.handoff("worker", "BRIEF")

        # The seeded agent row is kiro, and handoff resolves the harness from that row
        # rather than from the registry -- so this also pins that the check is generic
        # over harnesses rather than special-cased for claude.
        assert result.successor_started is False
        assert "kiro-cli" in (result.error or "")
        assert result.retired_agent_id.startswith("worker.h")
        lc._build_session.assert_not_called()

    async def test_restore_reports_it_and_registers_no_session(self, tmp_path: Path) -> None:
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        reg = AgentRegistry()
        lc = AgentLifecycle(
            _config(), reg, sink, db_path=tmp_path / "synth.db", session_id="s1"
        )
        with patch("synth_acp.broker.lifecycle.shutil.which", return_value=None):
            await lc.restore(
                agent_id="old",
                acp_session_id="acp-1",
                harness="claude",
                agent_mode=None,
                cwd=str(tmp_path),
                parent=None,
            )

        errors = [e for e in events if isinstance(e, BrokerError)]
        assert len(errors) == 1
        assert "Cannot restore" in errors[0].message
        assert "claude-agent-acp" in errors[0].message
        assert not reg.has_session("old")


class TestEveryHarnessInheritsByDefault:
    """The widened environment is deliberate and applies to EVERY harness, not just claude.

    Recorded because it reverses prior behavior: before this, every harness subprocess
    received the ACP SDK's six-variable environment. The reasoning is harness-agnostic --
    a harness is a program the user already runs interactively in this same shell, and
    handing it less is a silent divergence from how they invoke it directly -- so a change
    that quietly re-narrowed one harness would be a regression, not a safety measure.
    """

    def _lifecycle(self, harness_env: dict) -> AgentLifecycle:
        async def sink(_e: object) -> None:
            return None

        config = SessionConfig(project="p", settings={"harness_env": harness_env})
        return AgentLifecycle(
            config, AgentRegistry(), sink, db_path=Path("/tmp/unused.db"), session_id="s1"
        )

    @pytest.mark.parametrize("short_name", ["kiro", "claude", "opencode", "gemini"])
    def test_parent_variable_reaches_the_child_with_no_policy_configured(
        self, short_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SYNTH_PROBE_MARKER", "present")
        monkeypatch.setenv("npm_config_registry", "denied")
        lc = self._lifecycle({})
        entry = next(e for e in lc._harness_registry if e.short_name == short_name)

        env = lc._resolve_harness_env(entry)

        assert env is not None
        assert env["SYNTH_PROBE_MARKER"] == "present"
        assert "npm_config_registry" not in env

    def test_a_strict_policy_on_one_harness_leaves_the_others_inheriting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silent failure guarded: a per-harness policy leaking across harnesses, which
        would strip an environment the user only meant to narrow for one."""
        monkeypatch.setenv("SYNTH_PROBE_MARKER", "present")
        lc = self._lifecycle({"claude": {"inherit": ["HOME"]}})

        claude = next(e for e in lc._harness_registry if e.short_name == "claude")
        kiro = next(e for e in lc._harness_registry if e.short_name == "kiro")

        assert "SYNTH_PROBE_MARKER" not in (lc._resolve_harness_env(claude) or {})
        assert (lc._resolve_harness_env(kiro) or {})["SYNTH_PROBE_MARKER"] == "present"
