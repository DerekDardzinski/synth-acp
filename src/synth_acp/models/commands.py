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


class SetAgentMode(BrokerCommand):
    """Request the broker to switch an agent's active mode.

    Only valid when the agent is IDLE. The broker forwards this to
    session.set_mode(), which calls set_session_mode() on the ACP connection.
    The agent confirms via a current_mode_update stream event.
    """

    agent_id: str
    mode_id: str


class SetAgentModel(BrokerCommand):
    """Request the broker to switch an agent's active model.

    Only valid when the agent is IDLE. The broker forwards this to
    session.set_model(), which calls set_session_model() on the ACP connection.
    AgentModelChanged is emitted immediately after the call returns (no ACP
    push notification exists for model changes).
    """

    agent_id: str
    model_id: str
