"""Tests for permission rule keys."""

from __future__ import annotations

from synth_acp.models.permissions import permission_rule_key


class TestPermissionRuleKey:
    def test_identity_keys_the_rule_and_kind_is_the_fallback(self) -> None:
        """Silent failure: an identity-keyed rule collides with a kind-keyed one,
        or the kind is used even though the identity is known."""
        assert permission_rule_key("other", "github/create_issue") == "tool:github/create_issue"
        assert permission_rule_key("execute", None) == "execute"
