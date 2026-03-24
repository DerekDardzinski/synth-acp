"""Commands sent from the frontend to the broker."""

from __future__ import annotations

from pydantic import BaseModel


class BrokerCommand(BaseModel, frozen=True):
    """Base for all commands the frontend sends to the broker."""


class LaunchAgent(BrokerCommand):
    """Launch an agent by ID."""

    agent_id: str


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
    option_id: str


class CancelTurn(BrokerCommand):
    """Cancel the active prompt on an agent."""

    agent_id: str
