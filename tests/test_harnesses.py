"""Tests for harness registry loader."""

from __future__ import annotations

from synth_acp.harnesses import load_harness_registry


class TestLoadHarnessRegistry:
    def test_load_harness_registry_returns_entries_with_required_fields(self) -> None:
        entries = load_harness_registry()
        assert len(entries) >= 1
        for entry in entries:
            assert entry.identity
            assert entry.short_name
            assert entry.binary_names
            assert entry.run_cmd

    def test_steer_protocol_declared_only_for_kiro(self) -> None:
        """Steering is opt-in per harness; a stray declaration would steer a harness
        whose payload shape differs and silently fail with -32602."""
        by_short_name = {e.short_name: e.steer_protocol for e in load_harness_registry()}
        assert by_short_name["kiro"] == "kiro"
        assert by_short_name["claude"] is None
        assert by_short_name["opencode"] is None
        assert by_short_name["gemini"] is None

    def test_spawn_commands_are_unchanged_and_only_claude_needs_an_install_hint(self) -> None:
        """Two claims the npx removal rests on, pinned structurally.

        Silent failure guarded: someone changes another harness's run_cmd to a program
        that differs from its binary_names. The pre-spawn check and auto-detect then
        start gating that harness on a second binary nobody declared an install command
        for, so it silently disappears from auto-detect instead of erroring.
        """
        by_short_name = {e.short_name: e for e in load_harness_registry()}

        assert by_short_name["kiro"].run_cmd == "kiro-cli acp"
        assert by_short_name["opencode"].run_cmd == "opencode acp"
        assert by_short_name["gemini"].run_cmd == "gemini --experimental-acp"
        assert by_short_name["claude"].run_cmd == "claude-agent-acp"

        for short_name in ("kiro", "opencode", "gemini"):
            entry = by_short_name[short_name]
            assert entry.run_cmd.split()[0] in entry.binary_names, short_name
            assert entry.install_hint is None, short_name

        claude = by_short_name["claude"]
        assert claude.run_cmd.split()[0] not in claude.binary_names
        assert claude.install_hint == "npm install -g @agentclientprotocol/claude-agent-acp"

    def test_no_harness_spawns_through_a_package_manager(self) -> None:
        """A package manager in the spawn path resolves over the network on every launch
        and, with an unreachable registry, was measured writing nothing at all for 30s --
        which synth can only render as a permanent INITIALIZING."""
        for entry in load_harness_registry():
            assert entry.run_cmd.split()[0] not in {"npx", "npm", "pnpm", "yarn", "bunx", "uvx"}
