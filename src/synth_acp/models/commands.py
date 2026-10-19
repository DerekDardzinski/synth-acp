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
    """Send a user prompt to an agent.

    ``steer`` asks for the text to be injected into the agent's running turn, ahead of
    any queued prompts. It applies only while the agent is busy on a harness that can
    steer; otherwise the prompt is delivered or queued as usual.
    """

    agent_id: str
    text: str
    steer: bool = False


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


class RequestProfileSwitch(BrokerCommand):
    """The user confirmed switching agent_id to another agent profile by an agent-written handoff.

    Used for harnesses whose ``ExtensionCapabilities.profile_switch`` is ``"handoff"``:
    the broker asks the agent to call the ``handoff`` MCP tool with ``agent_mode``, and
    appends ``instructions`` verbatim to the successor's opening prompt.

    Attributes:
        agent_id: The agent to switch.
        agent_mode: Qualified name of the target agent profile.
        instructions: Free text the user added for the handoff; None or "" means none.
    """

    agent_id: str
    agent_mode: str
    instructions: str | None = None


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


class HoldQueue(BrokerCommand):
    """Suppress auto-drain for an agent (user is composing)."""

    agent_id: str


class ReleaseQueue(BrokerCommand):
    """Resume auto-drain for an agent."""

    agent_id: str


class DrainQueue(BrokerCommand):
    """User explicitly requested draining the front queue item."""

    agent_id: str


class EditQueueItem(BrokerCommand):
    """User started editing a queued item (blocks drain at that item)."""

    agent_id: str
    item_id: str


class CommitQueueEdit(BrokerCommand):
    """User saved an edit to a queued item."""

    agent_id: str
    item_id: str
    text: str


class DeleteQueueItem(BrokerCommand):
    """User deleted a queued item."""

    agent_id: str
    item_id: str


class SteerQueueItem(BrokerCommand):
    """User moved a queued item out of the queue to steer it into the running turn."""

    agent_id: str
    item_id: str


