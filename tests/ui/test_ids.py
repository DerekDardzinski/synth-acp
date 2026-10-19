from __future__ import annotations

from textual.dom import check_identifiers

from synth_acp.ui.ids import widget_id


class TestWidgetId:
    def test_valid_id_is_unchanged(self) -> None:
        """Silent failure: escaping valid ids breaks every existing #tool-<id> query."""
        assert widget_id("tool", "call_abc-123") == "tool-call_abc-123"

    def test_codex_guardian_id_becomes_a_valid_textual_id(self) -> None:
        """Measured crash: codex-acp's 'guardian_assessment:<uuid>' raised BadIdentifier."""
        wid = widget_id("tool", "guardian_assessment:a4cc05b2-e4b0")
        assert wid == "tool-guardian_assessment_x3a_a4cc05b2-e4b0"
        check_identifiers("id", wid)

    def test_distinct_invalid_characters_give_distinct_ids(self) -> None:
        """Silent failure: two tool calls mapped to one id collide in the DOM."""
        assert widget_id("tool", "a:b") != widget_id("tool", "a.b")
