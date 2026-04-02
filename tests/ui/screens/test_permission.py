"""Tests for PermissionBar and permission routing in SynthApp."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from acp.schema import PermissionOption

from synth_acp.models.events import PermissionRequested
from synth_acp.ui.screens.permission import PermissionBar


def _opt(option_id: str, name: str, kind: str) -> PermissionOption:
    """Create a PermissionOption."""
    return PermissionOption(option_id=option_id, name=name, kind=kind)


# ── PermissionBar unit tests ──


class TestPermissionBarHotkey:
    def test_permission_screen_when_hotkey_pressed_selects_matching_kind(self) -> None:
        """action_select_kind triggers _do_select with the correct index."""
        options = [
            _opt("o1", "Allow once", "allow_once"),
            _opt("o2", "Reject once", "reject_once"),
        ]
        bar = PermissionBar("agent-1", "Test", options)
        bar._rows = [MagicMock(), MagicMock()]
        bar._confirmed = False

        with patch.object(bar, "_do_select") as mock_do:
            bar.action_select_kind("allow_once")

        mock_do.assert_called_once_with(0)


class TestPermissionBarEscape:
    def test_permission_screen_when_escape_pressed_dismisses_reject_once(self) -> None:
        """action_cancel finds first reject_once option and calls _do_select."""
        options = [
            _opt("o1", "Allow once", "allow_once"),
            _opt("o2", "Reject once", "reject_once"),
        ]
        bar = PermissionBar("agent-1", "Test", options)
        bar._rows = [MagicMock(), MagicMock()]
        bar._confirmed = False

        with patch.object(bar, "_do_select") as mock_do:
            bar.action_cancel()

        mock_do.assert_called_once_with(1)

    def test_permission_screen_when_escape_no_reject_option_dismisses_empty(self) -> None:
        """action_cancel posts Resolved with empty string when no reject_once exists."""
        options = [
            _opt("o1", "Allow once", "allow_once"),
            _opt("o2", "Allow always", "allow_always"),
        ]
        bar = PermissionBar("agent-1", "Test", options)
        bar._rows = [MagicMock(), MagicMock()]

        with patch.object(bar, "_resolve") as mock_resolve:
            bar.action_cancel()

        mock_resolve.assert_called_once_with("")


# ── App routing tests ──


def _make_app():
    """Create a SynthApp with mock broker for routing tests."""
    from synth_acp.models.config import SessionConfig
    from synth_acp.ui.app import SynthApp

    config = SessionConfig(
        project="test",
        agents=[{"agent_id": "a", "harness": "kiro"}],
    )
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    broker._pending_permissions = {}
    broker.is_permission_pending = MagicMock(return_value=False)
    broker.get_agent_parent = MagicMock(return_value=None)

    async def _events():
        return
        yield

    broker.events = _events
    return SynthApp(broker, config)


class TestReplayPermissionSkip:
    async def test_replay_event_when_permission_resolved_skips_modal(self) -> None:
        """Replayed PermissionRequested with no pending entry does not mount bar."""
        app = _make_app()
        feed = MagicMock()
        feed.add_chunk = AsyncMock()
        event = PermissionRequested(
            agent_id="a",
            request_id="r1",
            title="Test",
            kind="edit",
            options=[_opt("o1", "Allow", "allow_once")],
        )
        # agent_id NOT in _pending_permissions → should skip
        with patch.object(app, "_mount_permission_bar") as mock_mount:
            await app._replay_event(feed, event)

        mock_mount.assert_not_called()


class TestRoutePermissionMount:
    async def test_route_event_when_permission_requested_pushes_modal_and_dispatches(self) -> None:
        """PermissionRequested triggers _mount_permission_bar on the feed."""
        from synth_acp.models.commands import RespondPermission

        app = _make_app()
        feed = MagicMock()
        event = PermissionRequested(
            agent_id="a",
            request_id="r1",
            title="Test",
            kind="edit",
            options=[_opt("o1", "Allow", "allow_once")],
        )

        with patch.object(app, "_mount_permission_bar") as mock_mount:
            await app._route_event_to_feed(feed, event)

        mock_mount.assert_called_once_with(feed, event)

        # Test the Resolved message handler dispatches to broker
        message = PermissionBar.Resolved("a", "o1")
        await app.on_permission_bar_resolved(message)

        app.broker.handle.assert_called_once_with(
            RespondPermission(agent_id="a", option_id="o1")
        )
