"""Tests for harness registry loader."""

from __future__ import annotations

from synth_acp.harnesses import load_harness_registry


class TestLoadHarnessRegistry:
    def test_load_harness_registry_returns_known_harnesses(self) -> None:
        entries = load_harness_registry()
        short_names = {e.short_name for e in entries}
        assert short_names >= {"kiro", "claude", "gemini", "opencode"}
