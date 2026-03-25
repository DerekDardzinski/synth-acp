"""Textual Message subclasses wrapping BrokerEvent."""

from __future__ import annotations

from textual.message import Message

from synth_acp.models.events import BrokerEvent


class BrokerEventMessage(Message):
    """Wraps any BrokerEvent for Textual message routing.

    Widgets inspect ``event`` type via ``isinstance``.
    """

    def __init__(self, event: BrokerEvent) -> None:
        self.event = event
        super().__init__()
