"""Kiro CLI v2 engine (``kiro-cli acp``) extension.

Kiro v2 reports context usage through the proprietary ``_kiro.dev/metadata``
notification, accepts in-turn steering through ``_session/steer``, and drops its MCP
server connections on a mode switch.  Profiles are ``~/.kiro/agents/*.json``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from synth_acp.acp.extensions.base import AcpExtension
from synth_acp.models.agent import DiscoveredAgent
from synth_acp.models.extensions import (
    ContextUsage,
    ExtensionCapabilities,
    Signal,
    SteerOutcome,
)

log = logging.getLogger(__name__)

# Kiro's proprietary usage notification. The wire method is "_kiro.dev/metadata";
# the acp SDK strips the leading underscore before dispatching to a client, so the
# name matched here has none.
_KIRO_METADATA_NOTIFICATION = "kiro.dev/metadata"

# Denominator used because Kiro reports context fullness as a percentage rather than
# as token counts, so UsageUpdated.used / .size is the true fraction.
_PERCENT_SCALE = 100


class KiroV2Extension(AcpExtension):
    """kiro-cli v2 engine: metadata usage, session/steer, MCP reload after mode switch."""

    name = "kiro_v2"
    capabilities = ExtensionCapabilities(steer=True, reload_after_mode_switch=True)

    def on_ext_notification(self, method: str, params: dict[str, Any]) -> tuple[Signal, ...]:
        """Map ``kiro.dev/metadata`` onto a ContextUsage signal.

        Payload shape, measured on kiro-cli 2.18.1::

            mid-turn:  {sessionId, contextUsagePercentage}
            turn-end:  {sessionId, contextUsagePercentage,
                        meteringUsage: [{value, unit, unitPlural}], turnDurationMs}

        ``contextUsagePercentage`` IS A PERCENTAGE AND KIRO REPORTS NO TOKEN COUNTS, so
        it becomes ``size=100`` with ``used`` the rounded percent.  Every consumer of
        those two fields treats them as a ratio -- the TUI's only use is ``used / size``
        rendered as a percent (``UsageBarVisual._build_label``), and the handoff nudge
        compares the same ratio to a threshold -- so the ratio is exact and the
        resolution is one percent, which is all Kiro reports.  A consumer that read
        ``size`` as a token budget would be wrong for Kiro agents; none does today.

        Any other method, or a reading that is not a finite number, gives no signal.
        """
        if method != _KIRO_METADATA_NOTIFICATION:
            return ()
        raw_pct = params.get("contextUsagePercentage")
        if isinstance(raw_pct, bool) or not isinstance(raw_pct, int | float):
            return ()
        # NaN and the infinities survive json.loads, which the SDK uses verbatim,
        # and the clamp below does not stop them: every comparison against NaN is
        # False, so min(100.0, nan) returns 100.0 and a NaN reading became a false
        # full-context event. A non-finite reading is no reading at all.
        if not math.isfinite(raw_pct):
            return ()
        pct = max(0.0, min(float(_PERCENT_SCALE), float(raw_pct)))

        cost_amount: float | None = None
        cost_currency: str | None = None
        metering = params.get("meteringUsage")
        if isinstance(metering, list) and metering and isinstance(metering[0], dict):
            first = metering[0]
            value = first.get("value")
            if not isinstance(value, bool) and isinstance(value, int | float):
                cost_amount = float(value)
                unit = first.get("unitPlural") or first.get("unit")
                cost_currency = unit if isinstance(unit, str) else None

        return (
            ContextUsage(
                size=_PERCENT_SCALE,
                used=round(pct),
                cost_amount=cost_amount,
                cost_currency=cost_currency,
            ),
        )

    async def steer(self, conn: Any, session_id: str, text: str) -> SteerOutcome:
        """Send ``{"sessionId", "message"}`` to ``session/steer``.

        NO leading underscore: the acp SDK's ``ext_method`` prepends one itself, so
        ``"_session/steer"`` reaches the wire as ``__session/steer`` and returns
        -32601, which reads exactly like the harness not supporting steering.

        Any non-error result counts as ``queued``: Kiro returns an opaque
        ``{"queued": true}`` that does not say which delivery path ran.
        """
        await conn.ext_method("session/steer", {"sessionId": session_id, "message": text})
        return SteerOutcome(status="queued")

    def discover_profiles(self, cwd: Path) -> list[DiscoveredAgent]:
        """Kiro agents from ``~/.kiro/agents/*.json`` (cwd is not used)."""
        return _discover_kiro()


def _discover_kiro() -> list[DiscoveredAgent]:
    """Discover Kiro agents from ~/.kiro/agents/*.json."""
    agents: list[DiscoveredAgent] = []
    agents_dir = Path.home() / ".kiro" / "agents"

    if not agents_dir.is_dir():
        return agents

    for json_file in agents_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Malformed agent JSON: %s", json_file)
            continue

        name = data.get("name")
        if not name:
            log.warning("Missing 'name' field in agent JSON: %s", json_file)
            continue

        agents.append(
            DiscoveredAgent(
                qualified_name=name,
                name=name,
                description=data.get("description", ""),
                source="user",
            )
        )

    return agents
