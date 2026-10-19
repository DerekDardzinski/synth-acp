"""Tests for broker event models."""

from __future__ import annotations

from synth_acp.models.events import PermissionRequested, ToolCallUpdated


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


class TestPermissionRequestedJournalCompat:
    def test_row_journaled_without_tool_identity_loads(self) -> None:
        """Restore replays ui_events with model_validate_json; a required field would
        make every pre-existing PermissionRequested row fail to load."""
        payload = (
            '{"agent_id": "a", "request_id": "r1", "title": "ls", "kind": "execute", "options": []}'
        )
        evt = PermissionRequested.model_validate_json(payload)
        assert evt.tool_identity is None


class TestAgentHandedOffUnion:
    def test_agent_handed_off_is_in_the_system_event_union(self) -> None:
        """The union is the declared broker-to-frontend surface. An event class left out
        of it is invisible to every consumer that switches on that type, and nothing
        else in the suite would notice the omission.
        """
        from typing import get_args

        from synth_acp.models.events import AgentHandedOff, SystemEvent

        assert AgentHandedOff in get_args(SystemEvent.__value__)
