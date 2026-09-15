"""Tests for config models, load_config, find_config, hooks, and global config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_acp.models.config import (
    CommunicationMode,
    GlobalConfig,
    HarnessEntry,
    HarnessEnvConfig,
    HooksConfig,
    McpMessageHook,
    MessageHook,
    RawSessionConfig,
    SessionConfig,
    find_config,
    format_mcp_message,
    load_config,
    merge_harness_env,
    normalize_message_kind,
    render_template,
    resolve_harness_env,
)


class TestSessionConfigValidation:
    def test_session_config_when_session_key_coerces_to_project(self):
        config = SessionConfig(
            session="test",
        )
        assert config.project == "test"


class TestRawSessionConfigDeprecation:
    def test_raw_session_config_strips_agents_key(self):
        """Old .synth.json with agents key must parse without error."""
        config = RawSessionConfig(
            project="test",
            agents=[{"agent_id": "a", "harness": "kiro"}],
        )
        assert config.project == "test"
        assert not hasattr(config, "agents")

    def test_raw_session_config_strips_ui_key(self):
        """Old .synth.json with ui key must parse without error."""
        config = RawSessionConfig(
            project="test",
            ui={"web_port": 9000, "theme": "light"},
        )
        assert config.project == "test"
        assert not hasattr(config, "ui")


class TestLoadConfig:
    def test_load_config_when_json_file_parses_correctly(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {"project": "myproject", "agents": [{"agent_id": "kiro", "harness": "kiro"}]}
            )
        )
        config = load_config(config_file)
        assert config.project == "myproject"

    def test_load_config_returns_raw_session_config(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps({"project": "p", "agents": [{"agent_id": "a", "harness": "kiro"}]})
        )
        result = load_config(config_file)
        assert isinstance(result, RawSessionConfig)


class TestFindConfig:
    def test_find_config_when_json_exists_returns_path(self, tmp_path: Path):
        (tmp_path / ".synth.json").write_text(
            '{"project":"t","agents":[{"agent_id":"a","harness":"kiro"}]}'
        )
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == ".synth.json"

    def test_find_config_when_no_config_returns_none(self, tmp_path: Path):
        assert find_config(tmp_path) is None


class TestSettingsConfig:
    def test_load_config_when_settings_has_local_mode_parses_enum(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {
                    "project": "p",
                    "settings": {"communication_mode": "LOCAL"},
                    "agents": [{"agent_id": "a", "harness": "kiro"}],
                }
            )
        )
        config = load_config(config_file)
        assert config.settings.communication_mode == CommunicationMode.LOCAL




class TestHooksConfig:

    def test_hooks_from_json(self, tmp_path: Path):
        config_file = tmp_path / ".synth.json"
        config_file.write_text(
            json.dumps(
                {
                    "project": "p",
                    "settings": {
                        "hooks": {
                            "on_agent_join": {
                                "recipients": "parent",
                                "template": "Agent {agent_id} joined.",
                            }
                        }
                    },
                    "agents": [{"agent_id": "a", "harness": "kiro"}],
                }
            )
        )
        config = load_config(config_file)
        assert config.settings.hooks.on_agent_join.recipients == "parent"
        assert config.settings.hooks.on_agent_join.template == "Agent {agent_id} joined."

    def test_env_override_join_recipients(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SYNTH_JOIN_RECIPIENTS", "family")
        config = RawSessionConfig(
            project="test",
        )
        assert config.settings.hooks.on_agent_join.recipients == "family"

    def test_message_hook_recipients_none_backward_compat(self):
        hook = MessageHook.model_validate({"recipients": "none", "template": "hi"})
        assert hook.active is False
        assert hook.recipients == "parent"  # default after removing 'none'

    def test_hooks_config_ignores_on_agent_prompt(self):
        hooks = HooksConfig.model_validate(
            {
                "on_agent_prompt": {"prepend": "some context"},
                "on_agent_join": {"recipients": "parent", "template": "hi"},
            }
        )
        assert not hasattr(hooks, "on_agent_prompt")
        assert hooks.on_agent_join.recipients == "parent"

    def test_hooks_config_ignores_startup_prepend(self):
        hooks = HooksConfig.model_validate(
            {
                "on_agent_startup": {"active": True, "prepend": "old context"},
            }
        )
        assert hooks.on_agent_startup.active is True


class TestRenderTemplate:
    def test_renders_known_slots(self):
        result = render_template(
            "Hello {agent_id}, parent is {parent_id}", {"agent_id": "a", "parent_id": "lead"}
        )
        assert result == "Hello a, parent is lead"

    def test_unknown_slots_become_empty(self):
        result = render_template("Hello {agent_id} {unknown}", {"agent_id": "a"})
        assert result == "Hello a "


class TestFileIO:
    def test_load_global_config_returns_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", tmp_path / "config.json")
        from synth_acp.models.config import load_global_config

        cfg = load_global_config()
        assert cfg.communication_mode == CommunicationMode.LOCAL
        assert cfg.auto_approve_tools == ["synth-mcp"]

    def test_save_and_load_global_config_roundtrip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", tmp_path)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", tmp_path / "config.json")
        from synth_acp.models.config import load_global_config, save_global_config

        cfg = GlobalConfig(
            default_harness="kiro",
            communication_mode=CommunicationMode.MESH,
            auto_approve_tools=["synth-mcp/send_message"],
        )
        save_global_config(cfg)
        loaded = load_global_config()
        assert loaded.default_harness == "kiro"
        assert loaded.communication_mode == CommunicationMode.MESH
        assert loaded.auto_approve_tools == ["synth-mcp/send_message"]

    def test_ensure_synth_dir_seeds_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        synth_dir = tmp_path / ".synth"
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", synth_dir)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", synth_dir / "config.json")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", synth_dir / "context.md")
        from synth_acp.models.config import ensure_synth_dir

        ensure_synth_dir()
        assert (synth_dir / "config.json").exists()
        assert (synth_dir / "context.md").exists()

    def test_ensure_synth_dir_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        synth_dir = tmp_path / ".synth"
        monkeypatch.setattr("synth_acp.models.config.SYNTH_DIR", synth_dir)
        monkeypatch.setattr("synth_acp.models.config.GLOBAL_CONFIG_PATH", synth_dir / "config.json")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", synth_dir / "context.md")
        from synth_acp.models.config import ensure_synth_dir

        ensure_synth_dir()
        # Modify files
        (synth_dir / "context.md").write_text("custom content")
        ensure_synth_dir()
        # User edits preserved
        assert (synth_dir / "context.md").read_text() == "custom content"

    def test_load_startup_context_reads_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        context_file = tmp_path / "context.md"
        context_file.write_text("my custom context")
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", context_file)
        from synth_acp.models.config import load_startup_context

        assert load_startup_context() == "my custom context"

    def test_load_startup_context_returns_default_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr("synth_acp.models.config.CONTEXT_MD_PATH", tmp_path / "nonexistent.md")
        from synth_acp.models.config import DEFAULT_STARTUP_CONTEXT, load_startup_context

        assert load_startup_context() == DEFAULT_STARTUP_CONTEXT


class TestFormatMcpMessage:
    def test_format_mcp_message_kinds(self):
        """Default templates render the exact historical envelope per kind."""
        hook = McpMessageHook()
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="chat")
            == "[Message from X]: body"
        )
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="request")
            == "[Request from X]: body"
        )
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="response")
            == "[Response from X]: body"
        )
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="system")
            == "[System notification — no action required]: body"
        )

    def test_format_mcp_message_inactive_returns_body(self):
        """active=False delivers the raw body with no prefix."""
        hook = McpMessageHook(active=False)
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="chat")
            == "body"
        )

    def test_format_mcp_message_custom_template(self):
        """A custom template renders all slots (incl. to_agent) and appends body verbatim."""
        hook = McpMessageHook(template="{kind}:{from_agent}>{to_agent} ")
        assert (
            format_mcp_message(hook, from_agent="X", to_agent="Y", body="body", kind="chat")
            == "chat:X>Y body"
        )

    def test_format_mcp_message_custom_system_template(self):
        """The system branch renders the configured system_template, not a hardcoded prefix."""
        hook = McpMessageHook(system_template="SYS {from_agent}->{to_agent} [{kind}]: ")
        assert (
            format_mcp_message(hook, from_agent="A", to_agent="B", body="body", kind="system")
            == "SYS A->B [system]: body"
        )


class TestTemplateValidation:
    @pytest.mark.parametrize("field", ["template", "system_template"])
    @pytest.mark.parametrize(
        "value",
        [
            "{",  # malformed brace
            "{}",  # positional/empty
            "{from_agent:{kind}}",  # nested field / non-empty format spec (empty-probe passes)
            "{from_agent.nope}",  # attribute traversal
            "{from_agent[bad]}",  # index traversal
            "{from_agent!r}",  # conversion
            "{bogus}",  # unknown slot
        ],
    )
    def test_template_validation_rejects(self, field: str, value: str):
        """Malformed/unsafe templates raise ValidationError at construction."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            McpMessageHook.model_validate({field: value})

    @pytest.mark.parametrize("field", ["template", "system_template"])
    def test_template_validation_accepts_all_slot_template(self, field: str):
        """A non-default all-slot template is accepted for both fields."""
        value = "{message_type} {from_agent} {to_agent} {kind}"
        hook = McpMessageHook.model_validate({field: value})
        assert getattr(hook, field) == value


class TestNormalizeMessageKind:
    def test_normalize_message_kind(self):
        assert normalize_message_kind("request") == "request"
        assert normalize_message_kind("response") == "response"
        assert normalize_message_kind("system") == "system"
        assert normalize_message_kind("chat") == "chat"
        assert normalize_message_kind("bogus") == "chat"
        assert normalize_message_kind("") == "chat"


class TestHandoffNudgeConfig:
    """The nudge's three settings and its template's slot set."""

    def test_normalize_message_kind_preserves_notification(self):
        """A notification row must not be narrowed to chat.

        normalize_message_kind is the choke point every delivered row passes
        through, so a missing branch here silently turns the nudge into an
        ordinary chat message with no error anywhere.
        """
        from synth_acp.models.config import normalize_message_kind

        assert normalize_message_kind("notification") == "notification"

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_threshold_outside_unit_interval_rejected(self, bad: float):
        """A threshold of 0 nudges on every report; above 1 never nudges."""
        from pydantic import ValidationError

        from synth_acp.models.config import SettingsConfig

        with pytest.raises(ValidationError):
            SettingsConfig(handoff_nudge_threshold=bad)

    @pytest.mark.parametrize(
        "value",
        [
            "{bogus}",  # unknown slot
            "{from_agent}",  # valid for on_mcp_message, not for the nudge
        ],
    )
    def test_nudge_template_rejects_bad_slots(self, value: str):
        """A bad nudge template must fail at config load, not at delivery."""
        from pydantic import ValidationError

        from synth_acp.models.config import SettingsConfig

        with pytest.raises(ValidationError):
            SettingsConfig(handoff_nudge_template=value)

    def test_nudge_template_accepts_all_slots(self):
        """All three nudge slots are usable together."""
        from synth_acp.models.config import SettingsConfig

        value = "{agent_id} {used} {nudge_threshold}"
        assert SettingsConfig(handoff_nudge_template=value).handoff_nudge_template == value

    def test_format_handoff_nudge_renders_whole_percents(self):
        """Fractions render as whole percents, so the agent reads real numbers."""
        from synth_acp.models.config import format_handoff_nudge

        rendered = format_handoff_nudge(
            "{agent_id} is {used} full, threshold {nudge_threshold}",
            agent_id="agent-1",
            used_fraction=0.626,
            threshold=0.5,
        )
        assert rendered == "agent-1 is 63% full, threshold 50%"

    def test_notification_envelope_names_the_sender(self):
        """A notification takes the ordinary template with a Notification noun.

        Without the _MESSAGE_TYPE_BY_KIND entry the agent sees "[ from synth]"
        with a stray space, which is wrong but raises nothing.
        """
        from synth_acp.models.config import McpMessageHook, format_mcp_message

        assert format_mcp_message(
            McpMessageHook(), from_agent="synth", to_agent="a1", body="body", kind="notification"
        ) == "[Notification from synth]: body"


_CLAUDE = HarnessEntry(
    identity="claude",
    name="Claude Code",
    short_name="claude",
    binary_names=["claude"],
    run_cmd="claude-agent-acp",
    executable_env_var="CLAUDE_CODE_EXECUTABLE",
    clear_env_vars=["CLAUDECODE"],
)

_PLAIN = HarnessEntry(
    identity="plain",
    name="Plain",
    short_name="plain",
    binary_names=["plain"],
    run_cmd="plain acp",
)


def _which(_name: str) -> str | None:
    return None


class TestResolveHarnessEnv:
    """The env a harness subprocess is handed. Each case is a silent-failure mode."""

    def test_inherit_all_forwards_parent_but_drops_denied_prefixes(self) -> None:
        """A denied prefix leaking through resolves child paths against the parent's
        build context; NPM_TOKEN being taken with it turns an install into a 401."""
        env = resolve_harness_env(
            _PLAIN,
            HarnessEnvConfig(),
            {
                "AWS_PROFILE": "dev",
                "npm_config_registry": "https://internal",
                "NPM_TOKEN": "secret",
                "brazil_pkg": "x",
                "LD_LIBRARY_PATH": "/opt/lib",
            },
            which=_which,
        )
        assert env == {"AWS_PROFILE": "dev", "NPM_TOKEN": "secret"}

    def test_denylist_is_case_insensitive(self) -> None:
        env = resolve_harness_env(
            _PLAIN, HarnessEnvConfig(), {"NPM_CONFIG_REGISTRY": "x"}, which=_which
        )
        assert env == {}

    def test_exported_shell_function_is_never_forwarded(self) -> None:
        """A '()' value is a function body, not a value."""
        env = resolve_harness_env(
            _PLAIN,
            HarnessEnvConfig(),
            {"BASH_FUNC_foo%%": "() { echo hi; }", "REAL": "v"},
            which=_which,
        )
        assert env == {"REAL": "v"}

    def test_allowlist_forwards_only_named_and_matched(self) -> None:
        env = resolve_harness_env(
            _PLAIN,
            HarnessEnvConfig(inherit=["AWS_*", "HOME"]),
            {"AWS_PROFILE": "dev", "AWS_REGION": "us-west-2", "HOME": "/h", "OTHER": "n"},
            which=_which,
        )
        assert env == {"AWS_PROFILE": "dev", "AWS_REGION": "us-west-2", "HOME": "/h"}

    def test_explicit_env_overrides_inherited_value_of_same_name(self) -> None:
        env = resolve_harness_env(
            _PLAIN,
            HarnessEnvConfig(env={"ANTHROPIC_MODEL": "opus"}),
            {"ANTHROPIC_MODEL": "haiku"},
            which=_which,
        )
        assert env == {"ANTHROPIC_MODEL": "opus"}

    def test_explicit_env_is_honored_against_the_denylist(self) -> None:
        """Naming a variable states intent; inheriting one does not."""
        env = resolve_harness_env(
            _PLAIN,
            HarnessEnvConfig(env={"npm_config_registry": "https://chosen"}),
            {"npm_config_registry": "https://inherited"},
            which=_which,
        )
        assert env == {"npm_config_registry": "https://chosen"}

    def test_clear_env_vars_wins_over_explicit_env(self) -> None:
        """CLAUDECODE only becomes load-bearing once the parent env is inherited."""
        env = resolve_harness_env(
            _CLAUDE,
            HarnessEnvConfig(env={"CLAUDECODE": "1"}),
            {"CLAUDECODE": "1"},
            which=_which,
        )
        assert env["CLAUDECODE"] == ""

    def test_executable_env_var_resolves_first_matching_binary(self) -> None:
        env = resolve_harness_env(
            _CLAUDE,
            HarnessEnvConfig(inherit=[]),
            {},
            which=lambda name: "/usr/bin/claude" if name == "claude" else None,
        )
        assert env["CLAUDE_CODE_EXECUTABLE"] == "/usr/bin/claude"

    def test_explicit_env_overrides_the_detected_executable_path(self) -> None:
        """A user pointing at a specific build must not be overruled by PATH order."""
        env = resolve_harness_env(
            _CLAUDE,
            HarnessEnvConfig(inherit=[], env={"CLAUDE_CODE_EXECUTABLE": "/opt/claude"}),
            {},
            which=lambda _n: "/usr/bin/claude",
        )
        assert env["CLAUDE_CODE_EXECUTABLE"] == "/opt/claude"

    def test_none_policy_behaves_as_the_default_policy(self) -> None:
        env = resolve_harness_env(_PLAIN, None, {"HOME": "/h"}, which=_which)
        assert env == {"HOME": "/h"}


class TestMergeHarnessEnv:
    """Project-over-global merge. A wrong merge drops a variable silently."""

    def test_env_merges_per_variable_rather_than_replacing_the_block(self) -> None:
        merged = merge_harness_env(
            {"claude": HarnessEnvConfig(env={"CLAUDE_CODE_USE_BEDROCK": "1"})},
            {"claude": HarnessEnvConfig(env={"ANTHROPIC_MODEL": "opus"})},
        )
        assert merged["claude"].env == {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "ANTHROPIC_MODEL": "opus",
        }

    def test_two_allowlists_concatenate_so_a_project_can_widen(self) -> None:
        merged = merge_harness_env(
            {"claude": HarnessEnvConfig(inherit=["AWS_*"])},
            {"claude": HarnessEnvConfig(inherit=["HTTPS_PROXY"])},
        )
        assert merged["claude"].inherit == ["AWS_*", "HTTPS_PROXY"]

    def test_project_allowlist_narrows_a_global_all(self) -> None:
        merged = merge_harness_env(
            {"claude": HarnessEnvConfig(inherit="all")},
            {"claude": HarnessEnvConfig(inherit=["AWS_*"])},
        )
        assert merged["claude"].inherit == ["AWS_*"]

    def test_project_not_setting_inherit_keeps_the_global_posture(self) -> None:
        """The project block sets only env; the global allowlist must survive."""
        merged = merge_harness_env(
            {"claude": HarnessEnvConfig(inherit=["AWS_*"])},
            {"claude": HarnessEnvConfig(env={"X": "1"})},
        )
        assert merged["claude"].inherit == ["AWS_*"]

    def test_harness_present_on_only_one_side_keeps_its_values(self) -> None:
        """Silent failure guarded: a one-sided harness surviving as a key while its
        values are replaced by defaults, which a key-set assertion would accept."""
        merged = merge_harness_env(
            {"kiro": HarnessEnvConfig(env={"A": "1"})},
            {"claude": HarnessEnvConfig(env={"B": "2"})},
        )
        assert merged["kiro"].env == {"A": "1"}
        assert merged["claude"].env == {"B": "2"}
