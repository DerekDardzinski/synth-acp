"""Tests for the Codex (codex-acp) extension.

Payloads are copied from the wire measured on codex-acp 2.0.0 (bundled codex 0.158.0).
"""

from __future__ import annotations

from typing import Any

import pytest
from acp.schema import SessionInfoUpdate, ToolCallUpdate

from synth_acp.acp.extensions.codex import CodexExtension
from synth_acp.models.extensions import TurnActivity


def _thread_status(status: Any) -> SessionInfoUpdate:
    return SessionInfoUpdate.model_validate(
        {"sessionUpdate": "session_info_update", "_meta": {"codex": {"threadStatus": status}}}
    )


class _Conn:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        return self.answer


class TestSteer:
    @pytest.mark.parametrize(
        ("outcome", "status"),
        [("injected", "delivered"), ("startedNewTurn", "started_turn"), ("failed", "failed")],
    )
    async def test_steer_sends_prompt_blocks_and_maps_outcomes(
        self, outcome: str, status: str
    ) -> None:
        """A wrong method or payload key makes every steer fall back to the queue;
        mapping startedNewTurn to delivered hides the turn Codex started on its own."""
        conn = _Conn({"outcome": outcome})
        result = await CodexExtension().steer(conn, "s", "hi")
        assert conn.calls == [
            ("session/steering", {"sessionId": "s", "prompt": [{"type": "text", "text": "hi"}]})
        ]
        assert (result.status, result.message_id) == (status, None)


class TestInterpretUpdate:
    @pytest.mark.parametrize(
        ("status", "signals"),
        [
            (
                {"type": "active", "activeFlags": ["waitingOnApproval"]},
                (TurnActivity(active=True),),
            ),
            ({"type": "idle"}, (TurnActivity(active=False),)),
            ({"type": "systemError"}, (TurnActivity(active=False),)),
            ({"type": "notLoaded"}, (TurnActivity(active=False),)),
            ("idle", ()),
            ({"type": "somethingNew"}, ()),
        ],
    )
    def test_interpret_update_maps_thread_status(
        self, status: Any, signals: tuple[TurnActivity, ...]
    ) -> None:
        """An unmapped idle leaves a Codex-started turn BUSY forever; a malformed
        payload must yield nothing rather than raise."""
        assert CodexExtension().interpret_update(_thread_status(status)).signals == signals

    def test_update_without_codex_meta_gives_nothing(self) -> None:
        update = SessionInfoUpdate.model_validate(
            {"sessionUpdate": "session_info_update", "title": "t"}
        )
        assert CodexExtension().interpret_update(update).signals == ()


class TestPermissionHints:
    def test_permission_hints_identity(self) -> None:
        """Without an identity every synth-mcp call prompts the user; a shell command
        passed off as an MCP tool would be auto-approved."""
        ext = CodexExtension()
        from_input = ToolCallUpdate(
            tool_call_id="c1",
            raw_input={"server": "synth-mcp", "tool": "send_message", "arguments": {}},
        )
        from_title = ToolCallUpdate(tool_call_id="c2", title="mcp.synth-mcp.send_message")
        shell = ToolCallUpdate(
            tool_call_id="c3",
            title="Run command",
            raw_input={"command": "echo probe > probe.txt", "cwd": "/tmp/codexprobe/ws"},
            kind="execute",
        )
        flagged = {"is_mcp_tool_approval": True}
        assert ext.permission_hints(from_input, {}).tool_identity == "synth-mcp/send_message"
        assert ext.permission_hints(from_title, flagged).tool_identity == "synth-mcp/send_message"
        assert ext.permission_hints(shell, flagged).tool_identity is None
        assert ext.permission_hints(from_title, {}).tool_identity is None
