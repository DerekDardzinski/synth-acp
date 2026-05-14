"""Tests for AgentLifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from synth_acp.broker.lifecycle import AgentLifecycle
from synth_acp.broker.registry import AgentRegistry
from synth_acp.models.agent import AgentState
from synth_acp.models.config import SessionConfig
from synth_acp.models.events import BrokerError


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
    ) -> None:
        import sqlite3

        conn = sqlite3.connect(str(lc._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO agents (agent_id, session_id, status, registered, parent, harness, acp_session_id, cwd) "
            "VALUES (?, ?, ?, 1000, ?, ?, ?, ?)",
            (agent_id, lc._session_id, status, parent, harness, acp_session_id, cwd),
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


class TestInitialPromptDeliveredAtEnqueue:
    async def test_initial_prompt_delivered_emitted_at_enqueue(self, tmp_path: Path) -> None:
        """handle_launch_command must emit InitialPromptDelivered immediately on enqueue."""
        from unittest.mock import patch

        from synth_acp.models.events import HookFired, InitialPromptDelivered

        config = _config("orchestrator")
        reg = AgentRegistry()
        events: list = []

        async def sink(e: object) -> None:
            events.append(e)

        lc = AgentLifecycle(config, reg, sink, db_path=tmp_path / "test.db", session_id="s1")
        lc._enqueue_raw = MagicMock()
        lc._enqueue_pending = MagicMock()

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ensure_schema_sync(conn)
        conn.close()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session):
            await lc.handle_launch_command(
                cmd_id=1,
                from_agent="orchestrator",
                data={
                    "agent_id": "worker-1",
                    "harness": "kiro",
                    "cwd": ".",
                    "task": "Fix bug",
                    "message": "Please fix the auth bug",
                },
            )

        prompt_events = [e for e in events if isinstance(e, InitialPromptDelivered)]
        assert len(prompt_events) == 1
        assert prompt_events[0].agent_id == "worker-1"
        assert prompt_events[0].from_agent == "orchestrator"
        assert prompt_events[0].text == "Please fix the auth bug"

        hook_events = [e for e in events if isinstance(e, HookFired) and e.hook_name == "on_agent_startup"]
        assert len(hook_events) == 1
        assert hook_events[0].agent_id == "worker-1"


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
        lc._enqueue_raw = MagicMock()
        lc._enqueue_pending = MagicMock()

        import sqlite3

        from synth_acp.db import ensure_schema_sync

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        ensure_schema_sync(conn)
        conn.close()

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        with patch("synth_acp.broker.lifecycle.ACPSession", return_value=mock_session), \
             patch("synth_acp.broker.lifecycle.load_startup_context", return_value="<ctx>{agent_id},{parent_id},{task}</ctx>\n\n"):
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
        enqueued_msg = lc._enqueue_raw.call_args[0][1]
        assert enqueued_msg.startswith("<ctx>child-1,orchestrator,Do work</ctx>")
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
        lc._enqueue_raw = MagicMock()
        lc._enqueue_pending = MagicMock()

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
        enqueued_msg = lc._enqueue_raw.call_args[0][1]
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
