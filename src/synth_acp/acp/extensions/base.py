"""The standard-ACP extension every harness engine starts from."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.extensions import (
    ExtensionCapabilities,
    PermissionHints,
    ProfileUnavailable,
    Signal,
    SteerOutcome,
    UpdateHints,
)

SessionOp = Literal["new", "load", "resume", "fork"]


class AcpExtension:
    """Plain standard ACP. Subclasses override only what their engine needs.

    Stateless; one instance serves every session of an engine.  ACPSession keeps every
    guard (session id, shutdown, replay, timeouts) and calls these hooks only at the
    seams, so an extension cannot bypass a guard.
    """

    name: ClassVar[str] = "standard"
    capabilities: ClassVar[ExtensionCapabilities] = ExtensionCapabilities()

    def session_meta(self, op: SessionOp, agent_mode: str | None) -> dict[str, Any]:
        """Keyword arguments for conn.new_session / load_session / resume_session / fork_session.

        The SDK places them in ``_meta``. Default ``{}``.
        """
        return {}

    def interpret_update(self, update: Any) -> UpdateHints:
        """Read proprietary data from one session/update payload.

        ``update`` is the SDK update object, with ``field_meta``.  MUST be synchronous
        and must not raise. Default ``UpdateHints()``.
        """
        return UpdateHints()

    def on_ext_notification(self, method: str, params: dict[str, Any]) -> tuple[Signal, ...]:
        """Translate a proprietary notification.

        ``method`` has the SDK-stripped leading underscore.  Called only after
        ACPSession's shutdown, sessionId and replay guards pass. Default ``()``.
        """
        return ()

    async def on_ext_request(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Answer an agent-to-client extension request.

        None means unsupported, and ACPSession replies method_not_found. Default None.
        """
        return None

    def permission_hints(self, tool_call: Any, meta: dict[str, Any]) -> PermissionHints:
        """Tool identity for a session/request_permission.

        ``meta`` is the request-level ``_meta`` the SDK spreads into request_permission
        kwargs. Default ``PermissionHints()``.
        """
        return PermissionHints()

    async def steer(self, conn: Any, session_id: str, text: str) -> SteerOutcome:
        """Deliver text into a running turn.

        ACPSession wraps this in its timeout and error handling; raising is allowed.
        Default ``SteerOutcome(status="unsupported")``.
        """
        return SteerOutcome(status="unsupported")

    def discover_profiles(self, cwd: Path) -> list[DiscoveredAgent]:
        """Agent profiles selectable as agent_mode.

        Must work without a session or connection. Default ``[]``.
        """
        return []

    def check_profile(self, agent_mode: str, cwd: Path) -> ProfileUnavailable | None:
        """Pre-launch validation of agent_mode. Default None (no opinion)."""
        return None
