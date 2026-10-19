"""Tests for synth_acp.acp.extensions.kiro_v3.

Payloads are copied from wire logs of kiro-cli 2.24.0 (KAS 0.66.8), /tmp/v3probe/*.jsonl.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from acp.schema import SessionInfoUpdate, UserMessageChunk

from synth_acp.acp.extensions.kiro_v3 import KiroV3Extension
from synth_acp.models.extensions import ContextUsage, ProfileUnavailable, SteerOutcome


def _context_usage(pct: Any) -> SessionInfoUpdate:
    return SessionInfoUpdate.model_validate(
        {
            "sessionUpdate": "session_info_update",
            "_meta": {
                "kiro": {
                    "kind": "context_usage",
                    "usagePercentage": pct,
                    "contextUsage": {"usagePercentage": pct},
                }
            },
        }
    )


class TestSessionOptions:
    async def test_shell_type_request_is_answered_with_shell(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without this answer v3 session/new fails (-32601), and {} fails with -32603."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        ext = KiroV3Extension()
        answer = await ext.on_ext_request(
            "kiro/terminal/shell_type", {"sessionId": "sess_42d0f0f5"}
        )
        assert answer == {"shellType": "/bin/zsh"}
        assert await ext.on_ext_request("kiro/other", {}) is None

    def test_session_meta_by_op(self) -> None:
        """modeId replaces the --agent flag v3 rejects; noReplay must be under kiro,
        because a top-level noReplay is ignored and the whole history replays."""
        ext = KiroV3Extension()
        assert ext.session_meta("new", "plan") == {"kiro": {"modeId": "plan"}}
        assert ext.session_meta("new", None) == {}
        assert ext.session_meta("load", None) == {"kiro": {"noReplay": True}}
        assert ext.session_meta("resume", "plan") == {}
        assert ext.session_meta("fork", "plan") == {}


class TestSubprocessEnv:
    """v3 hides an MCP server's tools unless the agent profile names it or the server is
    listed in ASBX_KIRO_MANDATORY_MCPS (kiro-cli 2.29.1)."""

    def test_synth_servers_become_mandatory(self) -> None:
        """Without the variable, profiles lacking @synth-mcp silently lose every synth tool."""
        env = KiroV3Extension().subprocess_env({"HOME": "/home/u"}, ["synth-mcp"])
        assert env == {"HOME": "/home/u", "ASBX_KIRO_MANDATORY_MCPS": "synth-mcp"}

    def test_existing_value_is_appended_to_without_duplicates(self) -> None:
        """Overwriting would silently drop servers the user's tooling made mandatory."""
        env = KiroV3Extension().subprocess_env(
            {"ASBX_KIRO_MANDATORY_MCPS": "corp-mcp, synth-mcp"}, ["synth-mcp", "extra-mcp"]
        )
        assert env == {"ASBX_KIRO_MANDATORY_MCPS": "corp-mcp,synth-mcp,extra-mcp"}


class TestInterpretUpdate:
    def test_replay_marker_sets_is_replay(self) -> None:
        update = UserMessageChunk.model_validate(
            {
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": "Reply with exactly the word OK."},
                "_meta": {
                    "kiro": {
                        "userMessageTag": "prompt_2e36f2c2",
                        "messageId": "262482e6",
                        "timestamp": "2026-09-24T16:00:00.590Z",
                        "replay": True,
                    }
                },
            }
        )
        assert KiroV3Extension().interpret_update(update).is_replay is True

    def test_context_usage_becomes_percent_signal(self) -> None:
        hints = KiroV3Extension().interpret_update(_context_usage(16.2115))
        assert hints.is_replay is False
        assert hints.signals == (ContextUsage(used=16, size=100),)

    @pytest.mark.parametrize("pct", [True, "x", float("nan"), -1, 101])
    def test_invalid_usage_gives_no_signal(self, pct: Any) -> None:
        """NaN would clamp to 100 and fire the handoff nudge on no data; a bool is an int."""
        assert KiroV3Extension().interpret_update(_context_usage(pct)).signals == ()


class TestProfileNotifications:
    def test_not_found_with_skip_record(self) -> None:
        """The only signal that a SHScienceAgentKit profile was rejected for 'vibe'."""
        signals = KiroV3Extension().on_ext_notification(
            "kiro/customAgent/not_found",
            {
                "sessionId": "sess_86882e75",
                "requestedAgent": "code-researcher",
                "fallbackAgent": "vibe",
                "skipped": {
                    "path": "/Users/u/.kiro/agents/local-SHScienceAgentKit-code-researcher.json",
                    "reasonCode": "cli_only_agent",
                    "error": "Agent profile uses fields this agent engine does not support: "
                    "allowedTools, toolsSettings",
                },
            },
        )
        assert signals == (
            ProfileUnavailable(
                requested="code-researcher",
                fallback="vibe",
                reason="cli_only_agent: Agent profile uses fields this agent engine "
                "does not support: allowedTools, toolsSettings",
            ),
        )

    def test_config_error_names_the_file(self) -> None:
        signals = KiroV3Extension().on_ext_notification(
            "kiro/customAgent/config_error",
            {
                "sessionId": "sess_1",
                "path": "/Users/u/.kiro/agents/broken.json",
                "reasonCode": "invalid_json",
                "error": "Unexpected token",
            },
        )
        assert signals == (
            ProfileUnavailable(
                requested="broken.json", fallback=None, reason="invalid_json: Unexpected token"
            ),
        )

    def test_malformed_params_give_no_signal(self) -> None:
        ext = KiroV3Extension()
        assert ext.on_ext_notification("kiro/customAgent/not_found", {"sessionId": "s"}) == ()
        assert (
            ext.on_ext_notification(
                "kiro/customAgent/not_found",
                {"requestedAgent": "a", "fallbackAgent": "vibe", "skipped": "oops"},
            )
            == ()
        )


class TestSteer:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (
                {"queued": False, "messageId": "steer-1", "dropped": "epoch_changed"},
                SteerOutcome(status="dropped", message_id="steer-1"),
            ),
            (
                {"queued": True, "messageId": "steer-2"},
                SteerOutcome(status="queued", message_id="steer-2"),
            ),
        ],
    )
    async def test_steer_result_maps_to_outcome(
        self, result: dict[str, Any], expected: SteerOutcome
    ) -> None:
        """queued:false counted as delivered would lose the message silently."""
        conn = AsyncMock()
        conn.ext_method.return_value = result
        outcome = await KiroV3Extension().steer(conn, "sess_1", "hello")
        assert outcome == expected
        conn.ext_method.assert_awaited_once_with(
            "session/steer", {"sessionId": "sess_1", "message": "hello"}
        )


class TestPermissionHints:
    def test_mcp_tool_identity(self) -> None:
        meta = {
            "kiro": {
                "toolId": "mcp_probe_echo_echo",
                "agentManagesTrust": True,
                "consent": {
                    "capability": "mcp",
                    "resource": "probe-echo/echo",
                    "askType": "implicit",
                    "workspaceRoot": "/tmp/v3probe/ws",
                },
                "consentRound": 1,
                "mcpTool": {
                    "version": 1,
                    "identity": {"serverName": "probe-echo", "toolName": "echo"},
                },
            }
        }
        hints = KiroV3Extension().permission_hints(None, meta)
        assert hints.tool_identity == "probe-echo/echo"

    def test_consent_resource_when_no_mcp_tool(self) -> None:
        meta = {"kiro": {"consent": {"capability": "mcp", "resource": "synth-mcp/list_agents"}}}
        assert KiroV3Extension().permission_hints(None, meta).tool_identity == (
            "synth-mcp/list_agents"
        )

    def test_shell_consent_gives_no_identity(self) -> None:
        """A shell resource is a command line, not a tool identity."""
        meta = {"kiro": {"consent": {"capability": "shell", "resource": "echo probe123"}}}
        assert KiroV3Extension().permission_hints(None, meta).tool_identity is None
