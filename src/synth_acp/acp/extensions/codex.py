"""Codex (codex-acp) extension.

Wire facts below were measured on codex-acp 2.0.0 with its bundled codex 0.158.0.
Sessions, prompts, permissions, modes, config options, usage and resume are standard
ACP.  The proprietary parts are:

- In-turn steering: ``_session/steering`` {sessionId, prompt: ContentBlock[]} answers
  ``{"outcome": "injected" | "startedNewTurn" | "failed"}``.  ``startedNewTurn`` means
  the adaptor found no live turn and started its own ("detached") turn with the text;
  synth receives no PromptResponse for it.  There is no opt-out (``idleBehavior`` is
  ignored), so synth tracks that turn through ``TurnActivity`` instead.
- Turn activity: ``session_info_update`` with ``_meta.codex.threadStatus.type`` of
  ``active`` (any ``activeFlags``), ``idle``, ``systemError`` or ``notLoaded``.  It carries
  no turn id.  An ordinary turn's ``idle`` arrives just before its PromptResponse.
- MCP permissions: the request carries only {toolCallId, kind, status} plus request
  ``_meta.is_mcp_tool_approval``.  The earlier tool_call for the same id carries title
  ``mcp.<server>.<tool>`` and rawInput {server, tool, arguments}; ACPSession fills the
  missing fields from that tool_call before calling ``permission_hints``.
- agent_mode takes Codex mode ids (read-only, workspace-write, agent, agent-full-access)
  through the standard set_session_mode path; there are no agent profiles to discover.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from synth_acp.acp.extensions.base import AcpExtension
from synth_acp.models.extensions import (
    ExtensionCapabilities,
    PermissionHints,
    SteerOutcome,
    TurnActivity,
    UpdateHints,
)

# The SDK prepends the leading underscore itself; the wire method is _session/steering.
_STEER_METHOD = "session/steering"
_ENDED_STATUSES = frozenset({"idle", "systemError", "notLoaded"})
# Server names may contain '-' but not '.'; the tool name is the remainder.
_MCP_TITLE = re.compile(r"^mcp\.([^.]+)\.(.+)$")


class CodexExtension(AcpExtension):
    """codex-acp: _session/steering, threadStatus turn activity, MCP permission identity."""

    name = "codex"
    capabilities = ExtensionCapabilities(steer=True)

    def interpret_update(self, update: Any) -> UpdateHints:
        """Map ``_meta.codex.threadStatus`` on a session_info_update onto TurnActivity."""
        if getattr(update, "session_update", None) != "session_info_update":
            return UpdateHints()
        field_meta = getattr(update, "field_meta", None)
        codex_meta = field_meta.get("codex") if isinstance(field_meta, dict) else None
        status = codex_meta.get("threadStatus") if isinstance(codex_meta, dict) else None
        status_type = status.get("type") if isinstance(status, dict) else None
        if status_type == "active":
            return UpdateHints(signals=(TurnActivity(active=True),))
        if status_type in _ENDED_STATUSES:
            return UpdateHints(signals=(TurnActivity(active=False),))
        return UpdateHints()

    def permission_hints(self, tool_call: Any, meta: dict[str, Any]) -> PermissionHints:
        """``"<server>/<tool>"`` for an MCP tool permission; None otherwise."""
        return PermissionHints(tool_identity=_mcp_identity(tool_call, meta))

    async def steer(self, conn: Any, session_id: str, text: str) -> SteerOutcome:
        """Send the text as one text block; map the adaptor's outcome.

        ``injected`` is ``delivered``, ``startedNewTurn`` is ``started_turn``, and
        anything else (``failed``, an unknown session, a malformed answer) is ``failed``
        so the broker queues the message.
        """
        result = await conn.ext_method(
            _STEER_METHOD,
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )
        outcome = result.get("outcome") if isinstance(result, dict) else None
        if outcome == "injected":
            return SteerOutcome(status="delivered")
        if outcome == "startedNewTurn":
            return SteerOutcome(status="started_turn")
        return SteerOutcome(status="failed")


def _mcp_identity(tool_call: Any, meta: Mapping[str, Any]) -> str | None:
    """'<server>/<tool>' from raw_input or an 'mcp.<server>.<tool>' title; None otherwise.

    The title is trusted only when the request is flagged ``is_mcp_tool_approval``: a
    Codex shell tool_call's title is the command text, so a command literally named
    ``mcp.synth-mcp.x`` must not pass as the MCP tool ``synth-mcp/x``.
    """
    raw_input = getattr(tool_call, "raw_input", None)
    if isinstance(raw_input, dict):
        server, tool = raw_input.get("server"), raw_input.get("tool")
        if isinstance(server, str) and isinstance(tool, str):
            return f"{server}/{tool}"
    if meta.get("is_mcp_tool_approval") is not True:
        return None
    title = getattr(tool_call, "title", None)
    match = _MCP_TITLE.match(title) if isinstance(title, str) else None
    return f"{match.group(1)}/{match.group(2)}" if match else None
