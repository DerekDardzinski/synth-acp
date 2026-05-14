"""Tests for broker event models."""

from __future__ import annotations

from synth_acp.models.events import ToolCallUpdated


class TestToolCallUpdatedParentField:
    def test_defaults_none(self) -> None:
        """Backward compat: existing callers not passing parent_tool_call_id must still work."""
        evt = ToolCallUpdated(
            agent_id="a",
            tool_call_id="tc-1",
            title="Edit",
            kind="edit",
            status="pending",
        )
        assert evt.parent_tool_call_id is None

    def test_set_value(self) -> None:
        """Field name must match exactly — typo on frozen model silently drops the value."""
        evt = ToolCallUpdated(
            agent_id="a",
            tool_call_id="tc-1",
            title="Edit",
            kind="edit",
            status="pending",
            parent_tool_call_id="parent-tc-1",
        )
        assert evt.parent_tool_call_id == "parent-tc-1"
