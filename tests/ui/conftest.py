"""Shared fixtures for UI tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _disable_embedding():
    """Prevent _do_index_sessions from hitting broker._db_path in UI tests.

    When embedding deps are installed, SynthApp.on_mount() calls
    _do_index_sessions() which does sqlite3.connect(str(broker._db_path)).
    With a mock broker, this creates files named after the MagicMock repr.
    """
    with patch("synth_acp.ui.app.embedding_available", return_value=False):
        yield
