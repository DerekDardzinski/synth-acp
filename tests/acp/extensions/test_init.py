"""Tests for the ACP extension registry in synth_acp.acp.extensions."""

from __future__ import annotations

from typing import ClassVar

import pytest

from synth_acp.acp.extensions import (
    AcpExtension,
    UnknownExtensionError,
    get_extension,
    register_extension,
)
from synth_acp.acp.extensions.claude import ClaudeExtension


class TestRegistry:
    def test_register_extension_rejects_duplicate_name(self) -> None:
        """A second module reusing a registered name must fail loudly, not silently
        replace that engine's behavior for every session."""

        class Impostor(AcpExtension):
            name: ClassVar[str] = "claude"

        with pytest.raises(ValueError, match="claude"):
            register_extension(Impostor)
        assert type(get_extension("claude")) is ClaudeExtension

    def test_get_extension_unknown_name_raises(self) -> None:
        """A TOML naming an unregistered extension must not fall back to anything."""
        with pytest.raises(UnknownExtensionError, match="kiro_v9") as exc_info:
            get_extension("kiro_v9")
        assert isinstance(exc_info.value, KeyError)
