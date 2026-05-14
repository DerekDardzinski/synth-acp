"""Commands sent from the frontend to the broker."""

from __future__ import annotations

from pydantic import BaseModel

from synth_acp.models.agent import AgentConfig


class BrokerCommand(BaseModel, frozen=True):
    """Base for all commands the frontend sends to the broker."""


class LaunchAgent(BrokerCommand):
    """Launch an agent by ID, or by ad-hoc config."""

    agent_id: str
    config: AgentConfig | None = None


class TerminateAgent(BrokerCommand):
    """Terminate a running agent."""

    agent_id: str


class ResurrectAgent(BrokerCommand):
    """Resurrect a terminated agent."""

    agent_id: str


class SendPrompt(BrokerCommand):
    """Send a user prompt to an agent."""

    agent_id: str
    text: str


class RespondPermission(BrokerCommand):
    """Resolve a pending permission request on an agent."""

    agent_id: str
    request_id: str
    option_id: str


class CancelTurn(BrokerCommand):
    """Cancel the active prompt on an agent."""

    agent_id: str


class SetConfigOption(BrokerCommand):
    """Request the broker to change a session config option.

    Only valid when the agent is IDLE. The broker forwards this to
    lifecycle.set_config_option(), which delegates to session.set_config_option().
    """

    agent_id: str
    config_id: str
    value: str | bool


class SetAgentMode(BrokerCommand):
    """Deprecated: use SetConfigOption(config_id='mode', value=mode_id) instead.

    Still functional for backward compatibility. The broker routes this
    through set_config_option internally.
    """

    agent_id: str
    mode_id: str


class SetAgentModel(BrokerCommand):
    """Deprecated: use SetConfigOption(config_id='model', value=model_id) instead.

    Still functional for backward compatibility. The broker routes this
    through set_config_option internally.
    """

    agent_id: str
    model_id: str


class RestoreSession(BrokerCommand):
    """Restore a previously saved SYNTH session."""

    broker_session_id: str


class HoldDelivery(BrokerCommand):
    """Hold MCP message delivery for an agent. Messages go to UI queue instead."""

    agent_id: str


class ReleaseDelivery(BrokerCommand):
    """Release MCP delivery hold. Future messages deliver directly to agent."""

    agent_id: str
