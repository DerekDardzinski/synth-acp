"""Tests for PlanBlock widget rendering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from acp.schema import PlanEntry

from synth_acp.models.config import SessionConfig
from synth_acp.ui.app import SynthApp
from synth_acp.ui.widgets.plan_block import PlanBlock


def _make_config() -> SessionConfig:
    return SessionConfig(
        project="test",
        agents=[{"agent_id": "a1", "harness": "kiro"}],
    )


def _make_broker() -> MagicMock:
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()

    async def _events():
        return
        yield

    broker.events = _events
    return broker


async def _get_feed(app: SynthApp) -> object:
    await app.select_agent("a1")
    return app._panels["a1"]


class TestPlanBlock:
    async def test_plan_block_renders_entries_with_status_icons(self) -> None:
        """Completed entry has ✓ and strike class; pending has · ."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            entries = [
                PlanEntry(content="Done task", status="completed", priority="medium"),
                PlanEntry(content="Todo task", status="pending", priority="medium"),
            ]
            await feed.update_plan(entries)
            blocks = app.query(PlanBlock)
            assert len(blocks) == 1
            statics = blocks[0].query(".plan-entry")
            assert len(statics) == 2
            assert "completed" in statics[0].classes
            assert "✓" in str(statics[0].content)
            assert "pending" in statics[1].classes
            assert "·" in str(statics[1].content)

    async def test_conversation_feed_update_plan_replaces_existing(self) -> None:
        """Second update_plan removes the first PlanBlock."""
        app = SynthApp(_make_broker(), _make_config())
        async with app.run_test(headless=True, size=(120, 40)):
            feed = await _get_feed(app)
            entries1 = [PlanEntry(content="Step 1", status="pending", priority="medium")]
            entries2 = [PlanEntry(content="Step 2", status="in_progress", priority="medium")]
            await feed.update_plan(entries1)
            await feed.update_plan(entries2)
            blocks = app.query(PlanBlock)
            assert len(blocks) == 1
            entry_widgets = blocks[0].query(".plan-entry")
            assert "Step 2" in str(entry_widgets[0].content)
