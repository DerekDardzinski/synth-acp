"""Tests for harness registry loader."""

from __future__ import annotations

import pytest

from synth_acp.acp.extensions import get_extension
from synth_acp.harnesses import UnknownHarnessVariantError, load_harness_registry


class TestLoadHarnessRegistry:
    def test_load_harness_registry_returns_entries_with_required_fields(self) -> None:
        entries = load_harness_registry()
        assert len(entries) >= 1
        for entry in entries:
            assert entry.identity
            assert entry.short_name
            assert entry.binary_names
            assert entry.run_cmd

    def test_steering_declared_only_for_supported_harnesses(self) -> None:
        """Steering is opt-in per harness; a stray declaration would call a payload
        shape the adaptor does not serve and silently fall back to the queue."""
        by_short_name = {
            e.short_name: get_extension(e.extension).capabilities.steer
            for e in load_harness_registry()
        }
        assert by_short_name["kiro"] is True
        assert by_short_name["codex"] is True
        assert by_short_name["claude"] is True
        assert by_short_name["opencode"] is False
        assert by_short_name["gemini"] is False

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

    def test_default_variants_resolve(self) -> None:
        """With no harness_variant, kiro must keep its v2 command line and extension:
        losing mode_arg launches Kiro without --agent, losing the extension loses
        steering and context usage."""
        by_short_name = {e.short_name: e for e in load_harness_registry()}
        kiro = by_short_name["kiro"]
        assert (kiro.run_cmd, kiro.mode_arg, kiro.extension, kiro.variant) == (
            "kiro-cli acp",
            "--agent",
            "kiro_v2",
            "v2",
        )
        assert (by_short_name["claude"].extension, by_short_name["claude"].variant) == (
            "claude",
            None,
        )
        assert by_short_name["opencode"].extension == "opencode"
        assert by_short_name["gemini"].extension == "standard"

    def test_v3_variant_resolves(self) -> None:
        """v3 exits 2 on --agent, so a mode_arg leaking in from v2 would stop every launch."""
        kiro = next(e for e in load_harness_registry({"kiro": "v3"}) if e.short_name == "kiro")
        assert (kiro.run_cmd, kiro.mode_arg, kiro.extension, kiro.variant) == (
            "kiro-cli acp --agent-engine=v3 --auth-method=cli",
            None,
            "kiro_v3",
            "v3",
        )

    def test_unknown_variant_raises_naming_harness_requested_and_valid(self) -> None:
        """A typo must fail, not silently fall back to the default engine."""
        with pytest.raises(UnknownHarnessVariantError) as exc_info:
            load_harness_registry({"kiro": "nope"})
        message = str(exc_info.value)
        assert "'kiro'" in message
        assert "'nope'" in message
        assert "v2" in message

    @pytest.mark.parametrize(
        ("variants", "harness", "requested"),
        [({"claude": "v9"}, "claude", "v9"), ({"kiro": ""}, "kiro", "")],
    )
    def test_variant_the_harness_cannot_honor_raises(
        self, variants: dict[str, str], harness: str, requested: str
    ) -> None:
        """A variant for a harness without variants, or an empty name, must not be ignored."""
        with pytest.raises(UnknownHarnessVariantError) as exc_info:
            load_harness_registry(variants)
        message = str(exc_info.value)
        assert f"'{harness}'" in message
        assert f"'{requested}'" in message

    def test_codex_entry_resolves(self) -> None:
        """A TOML typo would drop codex, or run it through the standard extension,
        which cannot steer."""
        codex = next(e for e in load_harness_registry() if e.short_name == "codex")
        assert (codex.run_cmd, codex.binary_names, codex.extension, codex.variant) == (
            "codex-acp",
            ["codex-acp"],
            "codex",
            None,
        )

    def test_every_bundled_extension_is_registered(self) -> None:
        """A TOML extension typo would otherwise surface only when an agent launches."""
        for entry in load_harness_registry():
            assert get_extension(entry.extension).name == entry.extension
