"""Tests for SynthApp broker event bridge and CLI mode selection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synth_acp.models.config import SessionConfig
from synth_acp.models.events import AgentStateChanged, BrokerEvent
from synth_acp.ui.app import SynthApp
from synth_acp.ui.messages import BrokerEventMessage


def _make_config(*agent_ids: str) -> SessionConfig:
    """Create a minimal SessionConfig."""
    return SessionConfig(
        project="test",
        agents=[{"id": aid, "cmd": ["echo"]} for aid in agent_ids],
    )


def _make_broker_mock(events: list[BrokerEvent] | None = None) -> MagicMock:
    """Create a mock broker with an async events() iterator."""
    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()

    async def _events():
        for e in events or []:
            yield e

    broker.events = _events
    return broker


class TestConsumeEvents:
    async def test_consume_broker_events_when_event_emitted_posts_message(self) -> None:
        event = AgentStateChanged(agent_id="a", old_state="idle", new_state="busy")
        broker = _make_broker_mock([event])
        config = _make_config("a")
        app = SynthApp(broker, config)

        posted: list[BrokerEventMessage] = []
        app.post_message = MagicMock(side_effect=lambda m: posted.append(m))  # type: ignore[method-assign]

        await app._consume_broker_events()

        assert len(posted) == 1
        assert isinstance(posted[0], BrokerEventMessage)
        assert posted[0].event is event


class TestCLIModeSelection:
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_main_when_headless_flag_calls_async_run(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".synth.toml"
        config_file.write_text('project = "s"\n\n[[agents]]\nid = "a"\ncmd = ["echo"]\n')

        with (
            patch("synth_acp.cli.asyncio.run") as mock_run,
            patch(
                "synth_acp.cli.sys.argv",
                ["synth", "-c", str(config_file), "--headless"],
            ),
            pytest.raises(SystemExit, match="0"),
        ):
            from synth_acp.cli import main

            main()

        mock_run.assert_called_once()
        # Close the unawaited coroutine to suppress RuntimeWarning
        mock_run.call_args[0][0].close()

    def test_main_when_default_calls_tui(self, tmp_path: Path) -> None:
        config_file = tmp_path / ".synth.toml"
        config_file.write_text('project = "s"\n\n[[agents]]\nid = "a"\ncmd = ["echo"]\n')

        with (
            patch("synth_acp.cli._run_tui") as mock_tui,
            patch(
                "synth_acp.cli.sys.argv",
                ["synth", "-c", str(config_file)],
            ),
            pytest.raises(SystemExit, match="0"),
        ):
            from synth_acp.cli import main

            main()

        mock_tui.assert_called_once()
