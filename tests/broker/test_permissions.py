"""Tests for PermissionEngine rule lookup and persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from synth_acp.broker.permissions import PermissionEngine
from synth_acp.models.permissions import PermissionDecision, PermissionRule


class TestPermissionEngine:
    def test_check_when_no_rule_returns_none(self, tmp_path: Path) -> None:
        db = tmp_path / "synth.db"
        engine = PermissionEngine(db, session_id="sess-1")

        assert engine.check("agent-1", "execute", "sess-1") is None

    async def test_check_when_allow_always_stored_returns_allow_always(self, tmp_path: Path) -> None:
        db = tmp_path / "synth.db"
        engine = PermissionEngine(db, session_id="sess-1")
        await engine.persist_async(
            PermissionRule(
                agent_id="agent-1",
                tool_kind="execute",
                session_id="sess-1",
                decision=PermissionDecision.allow_always,
            )
        )

        assert engine.check("agent-1", "execute", "sess-1") == PermissionDecision.allow_always

    async def test_persist_when_called_writes_to_sqlite(self, tmp_path: Path) -> None:
        db = tmp_path / "synth.db"
        engine = PermissionEngine(db, session_id="sess-1")
        await engine.persist_async(
            PermissionRule(
                agent_id="agent-1",
                tool_kind="execute",
                session_id="sess-1",
                decision=PermissionDecision.allow_always,
            )
        )

        conn = sqlite3.connect(str(db))
        row = conn.execute("SELECT agent_id, tool_kind, session_id, decision FROM rules").fetchone()
        conn.close()

        assert row == ("agent-1", "execute", "sess-1", "allow_always")

    async def test_check_when_different_session_returns_none(self, tmp_path: Path) -> None:
        db = tmp_path / "synth.db"
        engine = PermissionEngine(db, session_id="sess-1")
        await engine.persist_async(
            PermissionRule(
                agent_id="agent-1",
                tool_kind="execute",
                session_id="sess-1",
                decision=PermissionDecision.allow_always,
            )
        )

        assert engine.check("agent-1", "execute", "sess-2") is None

    async def test_init_when_called_starts_with_empty_cache(self, tmp_path: Path) -> None:
        db = tmp_path / "synth.db"
        # Persist a rule via one engine instance
        engine1 = PermissionEngine(db, session_id="sess-1")
        await engine1.persist_async(
            PermissionRule(
                agent_id="agent-1",
                tool_kind="execute",
                session_id="sess-1",
                decision=PermissionDecision.allow_always,
            )
        )

        # New engine instance should NOT pre-load from SQLite
        engine2 = PermissionEngine(db, session_id="sess-1")
        assert engine2.check("agent-1", "execute", "sess-1") is None
