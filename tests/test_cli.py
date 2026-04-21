"""Tests for CLI input parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from acp.schema import PermissionOption

from synth_acp.cli import _build_transient_config, parse_input, parse_permission_response


class TestParseInput:
    """Tests for parse_input."""

    def test_parse_input_when_at_prefix_extracts_agent_and_text(self) -> None:
        assert parse_input("@kiro refactor auth", None) == ("kiro", "refactor auth")

    def test_parse_input_when_bare_text_uses_default(self) -> None:
        assert parse_input("hello", "kiro") == ("kiro", "hello")

    def test_parse_input_when_select_command_returns_none(self) -> None:
        assert parse_input("/select kiro-auth", None) is None

    def test_parse_input_when_no_default_and_bare_text_raises(self) -> None:
        with pytest.raises(ValueError, match="No default agent set"):
            parse_input("hello", None)


class TestParsePermissionResponse:
    """Tests for parse_permission_response."""

    def test_parse_permission_response_when_numeric_returns_option_id(self) -> None:
        options = [
            PermissionOption(option_id="allow-once", name="Allow once", kind="allow_once"),
            PermissionOption(option_id="allow-always", name="Always allow", kind="allow_always"),
            PermissionOption(option_id="reject-once", name="Reject", kind="reject_once"),
            PermissionOption(option_id="reject-always", name="Always reject", kind="reject_always"),
        ]
        assert parse_permission_response("2", options) == "allow-always"



class TestBuildTransientConfig:
    """Tests for _build_transient_config."""

    def test_build_transient_config_when_called_sets_absolute_cwd(self) -> None:
        """Transient config must resolve cwd to an absolute path, not '.'."""
        config = _build_transient_config("kiro", None, None)
        agent = config.agents[0]
        assert agent.cwd != "."
        assert Path(agent.cwd).is_absolute()
