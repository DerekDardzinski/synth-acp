"""Kiro CLI v3 engine (``kiro-cli acp --agent-engine=v3``) extension.

Wire facts below were measured on kiro-cli 2.24.0 (Kiro Agent Server 0.66.8).  v3 has
no ``_kiro.dev/metadata``: context usage arrives as a ``session_info_update`` whose
``_meta.kiro.kind`` is ``"context_usage"``.  It asks the client for its shell during
session/new, takes the agent profile as ``_meta.kiro.modeId`` (it rejects the ``--agent``
flag), marks replayed updates with ``_meta.kiro.replay``, and loads without replay when
asked with ``_meta.kiro.noReplay``.  Client MCP servers survive a mode switch, so no
reload is needed.  Profiles are ``~/.kiro/agents/*.json``, as for v2.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from synth_acp.acp.extensions.base import AcpExtension, SessionOp
from synth_acp.acp.extensions.kiro_v2 import _discover_kiro
from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.extensions import (
    ContextUsage,
    ExtensionCapabilities,
    PermissionHints,
    ProfileUnavailable,
    Signal,
    SteerOutcome,
    UpdateHints,
)

log = logging.getLogger(__name__)

# Method names arrive with the SDK-stripped leading underscore.
_SHELL_TYPE_REQUEST = "kiro/terminal/shell_type"
_PROFILE_NOT_FOUND = "kiro/customAgent/not_found"
_PROFILE_CONFIG_ERROR = "kiro/customAgent/config_error"

# Kiro reports context fullness as a percentage and no token counts, so usage is
# expressed as used/100; every consumer reads only the ratio (see kiro_v2).
_PERCENT_SCALE = 100


class KiroV3Extension(AcpExtension):
    """kiro-cli v3 engine: _meta.kiro session options, context_usage updates, session/steer."""

    name = "kiro_v3"
    capabilities = ExtensionCapabilities(steer=True, no_replay_load=True)

    def session_meta(self, op: SessionOp, agent_mode: str | None) -> dict[str, Any]:
        """``kiro.modeId`` on session/new, ``kiro.noReplay`` on session/load.

        noReplay must sit under ``_meta.kiro``: a top-level noReplay is ignored
        (measured).  resume and fork carry nothing; v3 does not advertise resume.
        """
        if op == "new":
            return {"kiro": {"modeId": agent_mode}} if agent_mode else {}
        if op == "load":
            return {"kiro": {"noReplay": True}}
        return {}

    def interpret_update(self, update: Any) -> UpdateHints:
        """Read the replay marker and context_usage readings from ``_meta.kiro``."""
        field_meta = getattr(update, "field_meta", None)
        if not isinstance(field_meta, dict):
            return UpdateHints()
        kiro_meta = field_meta.get("kiro")
        if not isinstance(kiro_meta, dict):
            return UpdateHints()
        is_replay = kiro_meta.get("replay") is True
        signals: tuple[Signal, ...] = ()
        if (
            getattr(update, "session_update", None) == "session_info_update"
            and kiro_meta.get("kind") == "context_usage"
        ):
            usage = _context_usage_from_meta(kiro_meta)
            if usage is not None:
                signals = (usage,)
        return UpdateHints(is_replay=is_replay, signals=signals)

    def on_ext_notification(self, method: str, params: dict[str, Any]) -> tuple[Signal, ...]:
        """Map rejected or broken agent profiles onto ProfileUnavailable.

        ``customAgent/not_found`` {requestedAgent, fallbackAgent, skipped?: {path,
        reasonCode, error}} is how v3 reports a profile it will not use, most often
        reasonCode ``cli_only_agent`` for a profile with allowedTools or toolsSettings.
        ``customAgent/config_error`` {path, reasonCode, error} reports a profile file
        v3 could not load.  Malformed params give no signal and a log warning.
        """
        if method == _PROFILE_NOT_FOUND:
            requested = params.get("requestedAgent")
            fallback = params.get("fallbackAgent")
            skipped = params.get("skipped")
            reason = "not found" if skipped is None else _skip_reason(skipped)
            if (
                not isinstance(requested, str)
                or not (fallback is None or isinstance(fallback, str))
                or reason is None
            ):
                log.warning("Malformed %s params: %r", method, params)
                return ()
            return (ProfileUnavailable(requested=requested, fallback=fallback, reason=reason),)
        if method == _PROFILE_CONFIG_ERROR:
            path = params.get("path")
            reason = _skip_reason(params)
            if not isinstance(path, str) or reason is None:
                log.warning("Malformed %s params: %r", method, params)
                return ()
            return (ProfileUnavailable(requested=Path(path).name, fallback=None, reason=reason),)
        return ()

    async def on_ext_request(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Answer ``kiro/terminal/shell_type``; without an answer v3 session/new fails."""
        if method == _SHELL_TYPE_REQUEST:
            return {"shellType": os.environ.get("SHELL") or "/bin/sh"}
        return None

    def permission_hints(self, tool_call: Any, meta: dict[str, Any]) -> PermissionHints:
        """``"<server>/<tool>"`` from ``_meta.kiro.mcpTool`` (version 1) or the MCP consent resource."""
        kiro_meta = meta.get("kiro")
        if not isinstance(kiro_meta, dict):
            return PermissionHints()
        mcp_tool = kiro_meta.get("mcpTool")
        if isinstance(mcp_tool, dict) and mcp_tool.get("version") == 1:
            identity = mcp_tool.get("identity")
            if isinstance(identity, dict):
                server, tool = identity.get("serverName"), identity.get("toolName")
                if isinstance(server, str) and isinstance(tool, str):
                    return PermissionHints(tool_identity=f"{server}/{tool}")
        consent = kiro_meta.get("consent")
        if isinstance(consent, dict) and consent.get("capability") == "mcp":
            resource = consent.get("resource")
            if isinstance(resource, str):
                return PermissionHints(tool_identity=resource)
        return PermissionHints()

    async def steer(self, conn: Any, session_id: str, text: str) -> SteerOutcome:
        """Send ``{"sessionId", "message"}`` to ``session/steer`` (no leading underscore).

        v3 answers ``{"queued": true, "messageId"}`` or, when a cancel races the steer,
        ``{"queued": false, "messageId", "dropped": "epoch_changed"}``.  Anything but
        ``queued: true`` is ``dropped``, so the broker queues the message rather than
        losing it.
        """
        result = await conn.ext_method("session/steer", {"sessionId": session_id, "message": text})
        message_id = result.get("messageId") if isinstance(result, dict) else None
        if not isinstance(message_id, str):
            message_id = None
        if isinstance(result, dict) and result.get("queued") is True:
            return SteerOutcome(status="queued", message_id=message_id)
        return SteerOutcome(status="dropped", message_id=message_id)

    def discover_profiles(self, cwd: Path) -> list[DiscoveredAgent]:
        """Kiro agents from ``~/.kiro/agents/*.json`` (cwd is not used)."""
        return _discover_kiro()


def _context_usage_from_meta(kiro_meta: Mapping[str, Any]) -> ContextUsage | None:
    """Validate usagePercentage and build ContextUsage(size=100, used=round(pct)); None when invalid.

    Rejects bool, non-numeric, non-finite and out-of-range readings rather than clamping
    them: a NaN clamps to 100 and would fire the handoff nudge on no data.
    """
    pct = kiro_meta.get("usagePercentage")
    if isinstance(pct, bool) or not isinstance(pct, int | float):
        return None
    if not math.isfinite(pct) or not 0 <= pct <= _PERCENT_SCALE:
        return None
    return ContextUsage(size=_PERCENT_SCALE, used=round(pct))


def _skip_reason(skipped: Any) -> str | None:
    """``"<reasonCode>: <error>"`` from a profile skip record, or None when malformed."""
    if not isinstance(skipped, dict):
        return None
    code, error = skipped.get("reasonCode"), skipped.get("error")
    if not isinstance(code, str) or not isinstance(error, str):
        return None
    return f"{code}: {error}"
