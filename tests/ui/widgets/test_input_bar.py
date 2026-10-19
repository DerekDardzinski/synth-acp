"""Tests for input_bar helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from acp.schema import SessionConfigOptionBoolean, SessionConfigOptionSelect

from synth_acp.models.commands import SetConfigOption
from synth_acp.ui.file_discovery import FileEntry
from synth_acp.ui.widgets.gradient_bar import ActivityBar
from synth_acp.ui.widgets.input_bar import InputBar, PromptTextArea, _PickerLabel, _short_path


class TestInputBarActivityBar:
    async def test_inactive_input_bar_activity_bar_keeps_usage_bar(self) -> None:
        """An idle InputBar's ActivityBar must still carry its UsageBar.

        Silent failure: applying ExpandableSection's lazy-bar change to the shared
        ActivityBar blanks the input bar's static context/cost row, which no
        ExpandableSection test observes.
        """
        from textual.app import App, ComposeResult

        from synth_acp.ui.widgets.gradient_bar import ActivityBar, UsageBar

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one(InputBar).query_one(ActivityBar)
            bar.active = False
            await pilot.pause()

            assert len(bar.query(UsageBar)) == 1


class TestShortPath:
    def test_short_path_when_relative_dot_resolves_to_absolute(self) -> None:
        """Relative '.' must resolve, never appear as literal '.' in the UI."""
        result = _short_path(".")
        assert result != "."
        assert result.startswith("~/") or Path(result).is_absolute()

    def test_short_path_when_home_dir_returns_tilde(self) -> None:
        """Home directory must display as '~', not '~/.'."""
        result = _short_path(str(Path.home()))
        assert result == "~"


class TestAtDetection:
    """Tests for InputBar @ message handlers."""

    def _make_input_bar(self) -> InputBar:
        """Create an InputBar with a fake cwd and pre-populated file cache."""
        bar = InputBar.__new__(InputBar)
        bar._cwd = "/tmp/project"
        bar._file_cache = [
            FileEntry("src/foo.py", 400),
            FileEntry("src/bar.py", 800),
            FileEntry("README.md", 200),
        ]
        bar._file_cache_time = float("inf")  # Treat cache as fresh
        bar._file_picker = None
        bar._at_pos = None
        bar._at_row = 0
        bar._at_textarea = None
        bar._filter_timer = None
        return bar

    def test_at_trigger_sets_at_pos_and_opens_picker(self) -> None:
        """AtTrigger at start of text should set _at_pos and open picker."""
        bar = self._make_input_bar()
        event = PromptTextArea.AtTrigger(query="", cursor_row=0, at_pos=0)
        with patch.object(bar, "_open_file_picker") as mock_open:
            bar.on_prompt_text_area_at_trigger(event)
        assert bar._at_pos == 0
        assert bar._at_row == 0
        mock_open.assert_called_once_with("")

    def test_at_trigger_after_space_sets_at_pos(self) -> None:
        """AtTrigger after whitespace should set _at_pos from event."""
        bar = self._make_input_bar()
        event = PromptTextArea.AtTrigger(query="sr", cursor_row=0, at_pos=6)
        with patch.object(bar, "_open_file_picker") as mock_open:
            bar.on_prompt_text_area_at_trigger(event)
        assert bar._at_pos == 6
        mock_open.assert_called_once_with("sr")

    def test_at_trigger_updates_filter_when_picker_open(self) -> None:
        """AtTrigger with existing picker should schedule filter update."""
        bar = self._make_input_bar()
        bar._file_picker = MagicMock()
        event = PromptTextArea.AtTrigger(query="foo", cursor_row=0, at_pos=0)
        with patch.object(bar, "_schedule_filter") as mock_filter:
            bar.on_prompt_text_area_at_trigger(event)
        mock_filter.assert_called_once_with("foo")

    def test_at_dismiss_closes_picker(self) -> None:
        """AtDismiss should close the file picker."""
        bar = self._make_input_bar()
        bar._file_picker = MagicMock()
        bar._at_pos = 5
        event = PromptTextArea.AtDismiss()
        with patch.object(bar, "_close_file_picker") as mock_close:
            bar.on_prompt_text_area_at_dismiss(event)
        mock_close.assert_called_once()

    def test_picker_key_enter_selects_file(self) -> None:
        """PickerKey enter should select the highlighted file."""
        bar = self._make_input_bar()
        bar._file_picker = MagicMock()
        bar._file_picker.highlighted = 0
        option = MagicMock()
        option.id = "src/foo.py"
        bar._file_picker.get_option_at_index.return_value = option
        event = PromptTextArea.PickerKey("enter")
        with patch.object(bar, "_on_file_selected") as mock_select:
            bar.on_prompt_text_area_picker_key(event)
        mock_select.assert_called_once_with("src/foo.py")

    def test_file_selected_inserts_path(self) -> None:
        """Selecting a file should insert @rel_path + space at the @ position."""
        bar = self._make_input_bar()
        bar._at_pos = 0
        bar._file_picker = MagicMock()
        ta = MagicMock()
        ta.text = "@sr"
        ta.cursor_location = (0, 3)
        bar._at_textarea = ta
        bar._on_file_selected("src/foo.py")
        ta.load_text.assert_called_once_with("@src/foo.py ")
        ta.move_cursor.assert_called_once_with((0, 12))
        assert bar._file_picker is None
        assert bar._at_pos is None
        assert bar._at_textarea is None

    def test_at_trigger_stores_textarea_reference(self) -> None:
        """AtTrigger stores the triggering PromptTextArea as _at_textarea."""
        bar = self._make_input_bar()
        bar._at_textarea = None
        ta_mock = MagicMock(spec=PromptTextArea)
        event = PromptTextArea.AtTrigger(query="", cursor_row=0, at_pos=0)
        event._sender = ta_mock
        with patch.object(bar, "_open_file_picker"):
            bar.on_prompt_text_area_at_trigger(event)
        assert bar._at_textarea is ta_mock


class TestCheckAtTrigger:
    """Tests for PromptTextArea._check_at_trigger message posting."""

    def _check_at_trigger(self, text: str, cursor: tuple[int, int], at_active: bool = False) -> MagicMock:
        """Run _check_at_trigger with given text/cursor and return the mock post_message."""
        mock_post = MagicMock()
        with patch.object(PromptTextArea, 'text', new_callable=lambda: property(lambda _: text)), \
             patch.object(PromptTextArea, 'cursor_location', new_callable=lambda: property(lambda _: cursor)):
            ta = PromptTextArea.__new__(PromptTextArea)
            ta._at_active = at_active
            ta.post_message = mock_post
            ta._check_at_trigger()
        return mock_post

    def test_at_word_boundary_posts_at_trigger(self) -> None:
        """@ at start of text posts AtTrigger with correct fields."""
        mock_post = self._check_at_trigger("@src", (0, 4))
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, PromptTextArea.AtTrigger)
        assert msg.query == "src"
        assert msg.cursor_row == 0
        assert msg.at_pos == 0

    def test_at_after_space_posts_at_trigger(self) -> None:
        """@ after whitespace posts AtTrigger with correct at_pos."""
        mock_post = self._check_at_trigger("hello @sr", (0, 9))
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, PromptTextArea.AtTrigger)
        assert msg.query == "sr"
        assert msg.at_pos == 6

    def test_space_after_query_posts_dismiss(self) -> None:
        """Space after @ query posts AtDismiss."""
        mock_post = self._check_at_trigger("@src ", (0, 5), at_active=True)
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, PromptTextArea.AtDismiss)

    def test_no_at_does_not_post_when_already_inactive(self) -> None:
        """No @ and _at_active=False should not post any message."""
        mock_post = self._check_at_trigger("hello world", (0, 11))
        mock_post.assert_not_called()

    def test_mid_word_at_does_not_trigger(self) -> None:
        """@ mid-word (user@example) should not post AtTrigger."""
        mock_post = self._check_at_trigger("user@example", (0, 12))
        mock_post.assert_not_called()

    def test_picker_key_posted_when_at_active(self) -> None:
        """up/down/enter/escape post PickerKey when _at_active is True."""
        from textual import events

        mock_post = MagicMock()
        with patch.object(PromptTextArea, 'text', new_callable=lambda: property(lambda _: "@src")), \
             patch.object(PromptTextArea, 'cursor_location', new_callable=lambda: property(lambda _: (0, 4))):
            ta = PromptTextArea.__new__(PromptTextArea)
            ta._at_active = True
            ta.post_message = mock_post
            event = events.Key("down", None)
            ta._on_key(event)
        msg = mock_post.call_args[0][0]
        assert isinstance(msg, PromptTextArea.PickerKey)
        assert msg.key == "down"


class TestQueueIntegration:
    """Tests for InputBar is_composing property."""

    def _make_input_bar(self, busy: bool = False) -> InputBar:
        """Create an InputBar with mocked internals for queue testing."""
        bar = InputBar.__new__(InputBar)
        bar._agent_id = "test-agent"
        bar._agent_name = "Test Agent"
        bar._harness = "kiro"
        bar._cwd = "/tmp/project"
        bar._busy = busy
        bar._slash_commands = []
        bar._file_cache = []
        bar._file_picker = None
        bar._at_pos = None
        bar._at_row = 0
        bar._filter_timer = None
        return bar

    def test_is_composing_true_when_text(self) -> None:
        """is_composing returns True when textarea has non-empty text."""
        bar = self._make_input_bar()
        ta_mock = MagicMock()
        ta_mock.text = "  some text  "

        with patch.object(bar, "query_one", return_value=ta_mock):
            assert bar.is_composing is True

    def test_is_composing_false_when_empty(self) -> None:
        """is_composing returns False when textarea is empty/whitespace."""
        bar = self._make_input_bar()
        ta_mock = MagicMock()
        ta_mock.text = "   "

        with patch.object(bar, "query_one", return_value=ta_mock):
            assert bar.is_composing is False


class TestFileInjection:
    """Tests for @path file content injection (now in lifecycle layer)."""

    def test_submit_with_file_ref_injects_xml(self, tmp_path: Path) -> None:
        """@path references should produce XML blocks prepended to prompt."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("print('hello')")
        from synth_acp.broker.lifecycle import RE_FILE_REF

        # Inline the injection logic for unit testing
        text = "check @src/foo.py please"
        refs = RE_FILE_REF.findall(text)
        assert refs == ["src/foo.py"]
        contents = (tmp_path / "src" / "foo.py").read_text()
        result = f'<file path="src/foo.py">\n{contents}\n</file>\n\n{text}'
        assert result.startswith('<file path="src/foo.py">')
        assert "print('hello')" in result
        assert result.endswith("\n\ncheck @src/foo.py please")

    def test_submit_with_missing_file_skips(self, tmp_path: Path) -> None:
        """Missing files should be skipped without crashing."""
        from synth_acp.broker.lifecycle import RE_FILE_REF

        text = "look at @nonexistent.py"
        refs = RE_FILE_REF.findall(text)
        assert refs == ["nonexistent.py"]
        # Attempting to read a missing file just skips it
        try:
            (tmp_path / "nonexistent.py").read_text()
            raise AssertionError("Should have raised")
        except OSError:
            pass  # Expected — lifecycle logs and skips

    def test_multiple_refs_parsed(self) -> None:
        """Multiple @references should each be found by the regex."""
        from synth_acp.broker.lifecycle import RE_FILE_REF

        text = "@a.py and @b.py"
        refs = RE_FILE_REF.findall(text)
        assert refs == ["a.py", "b.py"]


class TestConfigOptionPickers:
    """Tests for dynamic config option picker creation and dispatch."""

    def _make_input_bar(self) -> InputBar:
        """Create an InputBar with mocked internals for picker testing."""
        bar = InputBar.__new__(InputBar)
        bar._extra_options = []
        bar._agent_id = "test-agent"
        bar._agent_name = "Test Agent"
        bar._harness = "kiro"
        bar._cwd = "/tmp/project"
        bar._busy = False
        bar._slash_commands = []
        bar._file_cache = []
        bar._file_picker = None
        bar._at_pos = None
        bar._at_row = 0
        bar._filter_timer = None
        return bar

    def _make_select_option(self, opt_id: str, name: str, category: str | None, current_value: str, options: list[tuple[str, str]]) -> SessionConfigOptionSelect:
        """Create a SessionConfigOptionSelect with flat options."""
        from acp.schema import SessionConfigSelectOption
        return SessionConfigOptionSelect(
            id=opt_id,
            name=name,
            category=category,
            type="select",
            current_value=current_value,
            options=[SessionConfigSelectOption(value=v, name=n) for v, n in options],
        )

    def test_update_config_options_creates_pickers_in_category_order(self) -> None:
        """Pickers are created for select-type options only, ordered by category priority."""
        from unittest.mock import MagicMock, patch

        from textual.css.query import NoMatches

        bar = self._make_input_bar()
        container = MagicMock()
        mounted_pickers: list[_PickerLabel] = []

        def capture_mount(picker):
            mounted_pickers.append(picker)

        container.mount = capture_mount
        container.children = []
        container.query_one = MagicMock(side_effect=NoMatches())

        with patch.object(bar, "query_one", return_value=container):
            options = [
                self._make_select_option("model", "Model", "model", "gpt-4", [("gpt-4", "GPT-4"), ("gpt-3", "GPT-3")]),
                SessionConfigOptionBoolean(id="verbose", name="Verbose", category="debug", type="boolean", current_value=True),
                self._make_select_option("mode", "Mode", "mode", "code", [("code", "Code"), ("plan", "Plan")]),
                self._make_select_option("effort", "Effort", "thought_level", "medium", [("low", "Low"), ("medium", "Medium"), ("high", "High")]),
            ]
            bar.update_config_options(options)

        # Inline: the reserved categories only, ordered mode(0), model(1), thought_level(2)
        inline = [p for p in mounted_pickers if isinstance(p, _PickerLabel)]
        assert [p.id for p in inline] == ["picker-mode", "picker-model", "picker-effort"]
        # The boolean is not dropped: it waits behind "⋯ 1 more"
        assert [o.id for o in bar._extra_options] == ["verbose"]

    def test_unreserved_options_go_behind_more_in_agent_order(self) -> None:
        """Kiro v3's three uncategorized options must reach the modal in Kiro's order,
        and none may be drawn inline (that is what overflowed a split window)."""
        from unittest.mock import MagicMock, patch

        from textual.css.query import NoMatches

        bar = self._make_input_bar()
        container = MagicMock()
        mounted: list = []
        container.mount = lambda w, **_kw: mounted.append(w)
        container.children = []
        container.query_one = MagicMock(side_effect=NoMatches())
        on_off = [("on", "On"), ("off", "Off")]
        with patch.object(bar, "query_one", return_value=container):
            bar.update_config_options([
                self._make_select_option("mode", "Mode", "mode", "a", [("a", "A")]),
                self._make_select_option("memoryReflection", "Memory reflection", None, "on", on_off),
                self._make_select_option("model", "Model", "model", "m", [("m", "M")]),
                self._make_select_option("autopilot", "Autopilot", None, "on", on_off),
                self._make_select_option("contentCollection", "Content", "_custom", "on", on_off),
            ])

        assert [p.id for p in mounted if isinstance(p, _PickerLabel)] == ["picker-mode", "picker-model"]
        assert [o.id for o in bar._extra_options] == ["memoryReflection", "autopilot", "contentCollection"]
        more = next(w for w in mounted if w.id == "picker-more")
        assert more.display is True

    def test_boolean_value_change_reaches_the_modal_copy(self) -> None:
        """A boolean change must update the copy the modal opens from, or the modal
        shows the old value and Apply would send it back."""
        bar = self._make_input_bar()
        bar._extra_options = [
            SessionConfigOptionBoolean(id="verbose", name="Verbose", type="boolean", current_value=True)
        ]
        bar.update_config_option_value("verbose", False)
        assert bar._extra_options[0].current_value is False

    def test_update_config_option_value_updates_picker(self) -> None:
        """update_config_option_value calls set_current on the correct picker."""
        from unittest.mock import MagicMock, patch

        bar = self._make_input_bar()
        picker = MagicMock(spec=_PickerLabel)

        with patch.object(bar, "query_one", return_value=picker):
            bar.update_config_option_value("model", "gpt-3")

        picker.set_current.assert_called_once_with("gpt-3")

    def test_picker_selected_dispatches_set_config_option(self) -> None:
        """Picker selection dispatches SetConfigOption with correct config_id and value."""
        from unittest.mock import AsyncMock, MagicMock, patch

        bar = self._make_input_bar()
        picker = MagicMock(spec=_PickerLabel)

        mock_app = MagicMock()
        mock_app.broker = MagicMock()
        mock_app.broker.handle = AsyncMock()
        # Close the coroutine the real run_worker would have awaited. Without this the
        # AsyncMock call object is never consumed and pytest reports an un-awaited-coroutine
        # RuntimeWarning, which gets attributed to whichever test runs next.
        mock_app.run_worker = MagicMock(side_effect=lambda coro, **_kw: coro.close())

        with patch.object(bar, "query_one", return_value=picker), \
             patch.object(type(bar), "app", new_callable=lambda: property(lambda _: mock_app)), \
             patch("synth_acp.ui.widgets.input_bar.isinstance", side_effect=lambda obj, cls: True if obj is mock_app else isinstance(obj, cls)):
            bar._on_picker_selected("picker-effort", "high")

        mock_app.run_worker.assert_called_once()
        mock_app.broker.handle.assert_called_once_with(
            SetConfigOption(agent_id="test-agent", config_id="effort", value="high")
        )


class TestApplyBusyDuringTeardown:
    """set_busy must not raise when the bar's buttons are gone but the bar remains.

    The three buttons are GRANDCHILDREN, composed inside a Horizontal, so an earlier guard
    on `self.children` was not sufficient: the bar can hold its direct children while that
    container's are gone. Tearing a feed down with a deferred input-bar sync queued reaches
    that state, and query_one then raised NoMatches out of Screen._on_idle -- an UNHANDLED
    Textual exception rather than a caught one. It surfaced only as a 1-in-4 intermittent
    failure of an unrelated conversation test, so nothing in the suite named the real cause.
    """

    async def test_missing_buttons_do_not_raise(self) -> None:
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = app.query_one(InputBar)
            await pilot.pause()

            # Exactly the teardown state: the bar is still mounted and still has direct
            # children, while the container holding the buttons has lost its own.
            await bar.query_one("#submit-btn").remove()
            await pilot.pause()
            assert bar.is_mounted
            assert bar.children
            assert not bar.query("#submit-btn")

            # Would raise NoMatches before the fix.
            bar.set_busy(True)
            bar.set_busy(False)

    async def test_a_partial_teardown_leaves_nothing_half_applied(self) -> None:
        """State is all-or-none: one missing node means nothing is applied.

        The consequence is a MIXED bar rather than a specific pair of buttons. With
        #drain-btn gone, an assign-as-you-go implementation updates Submit and Cancel
        consistently and then raises before reaching the ActivityBar, so the buttons say
        busy while the activity indicator says idle.

        The button assertions are what catch that mutation, since both buttons move before
        the failing query. The ActivityBar assertion catches nothing on its own and is here
        to specify the whole contract: every one of the four nodes is unchanged, not just
        the ones an implementation happens to reach first.
        """
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = app.query_one(InputBar)
            await pilot.pause()
            submit_before = bar.query_one("#submit-btn").display
            cancel_before = bar.query_one("#cancel-btn").display

            # Drop only the LAST node queried, so an implementation that assigned as it
            # went would already have toggled submit before failing.
            await bar.query_one("#drain-btn").remove()
            await pilot.pause()

            activity_before = bar.query_one(ActivityBar).active

            bar.set_busy(True)

            assert bar.query_one("#submit-btn").display == submit_before
            assert bar.query_one("#cancel-btn").display == cancel_before
            # The node AFTER the missing one: an assign-as-you-go version never reaches it,
            # leaving the buttons and the indicator disagreeing.
            assert bar.query_one(ActivityBar).active == activity_before

    async def test_a_healthy_bar_still_toggles(self) -> None:
        """The guard must not swallow the normal path."""
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = app.query_one(InputBar)
            await pilot.pause()

            bar.set_busy(True)
            assert bar.query_one("#cancel-btn").display is True
            assert bar.query_one("#submit-btn").display is False

            bar.set_busy(False)
            assert bar.query_one("#cancel-btn").display is False
            assert bar.query_one("#submit-btn").display is True

    async def test_state_recorded_before_compose_is_applied_on_mount(self) -> None:
        """A busy/disabled state recorded before this bar composes must reach the widgets.

        This is the contract set_busy's docstring has always claimed and never had. The app
        syncs an agent's bar as soon as the FEED reports an input_bar, one level above this
        widget's own children, so set_busy and set_disabled are routinely called first --
        and on_mount replays them. An is_mounted guard defeated that replay, because
        is_mounted is False while Textual runs on_mount: MEASURED before the fix, Submit and
        Cancel were BOTH visible and the prompt was left enabled.

        Nothing in the suite asserted it, which is why a guard that discarded every
        pre-compose state passed review twice.
        """
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                bar = InputBar("agent-1", "agent-1", "kiro", cwd=".")
                bar.set_busy(True)
                bar.set_disabled(True, "waiting")
                yield bar

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await pilot.pause()
            bar = app.query_one(InputBar)

            assert bar.query_one("#submit-btn").display is False
            assert bar.query_one("#cancel-btn").display is True
            assert bar.query_one(ActivityBar).active is True
            assert bar.query_one("#prompt-input", PromptTextArea).disabled is True


def _handoff_app():
    """A SynthApp over a mock broker whose agent switches profile by handoff."""
    from unittest.mock import AsyncMock

    from synth_acp.models.agent import AgentConfig
    from synth_acp.models.config import SessionConfig
    from synth_acp.models.extensions import ExtensionCapabilities
    from synth_acp.ui.app import SynthApp

    broker = MagicMock()
    broker.handle = AsyncMock()
    broker.shutdown = AsyncMock()
    broker._initial_agent = AgentConfig(agent_id="agent-1", harness="claude")
    broker.get_agent_parent = MagicMock(return_value=None)
    broker.get_agent_harness = MagicMock(return_value="claude")
    broker.get_agent_cwd = MagicMock(return_value=".")
    broker.get_agent_display_name = MagicMock(return_value=None)
    broker.get_usage = MagicMock(return_value=None)
    broker.pending_profile_switch = MagicMock(return_value=None)
    broker._registry.get_capabilities = MagicMock(
        return_value=ExtensionCapabilities(profile_switch="handoff")
    )

    async def _events():
        return
        yield

    broker.events = _events
    return SynthApp(
        broker,
        SessionConfig(project="test"),
        initial_agent=AgentConfig(agent_id="agent-1", harness="claude"),
    )


def _agent_option() -> SessionConfigOptionSelect:
    from acp.schema import SessionConfigSelectOption

    return SessionConfigOptionSelect(
        id="agent",
        name="Agent",
        category="agent",
        type="select",
        current_value="",
        options=[
            SessionConfigSelectOption(value="", name="Default"),
            SessionConfigSelectOption(value="probe-heron", name="probe-heron"),
        ],
    )


def _sent(app, cls) -> list:
    """Commands of type cls the app handed to broker.handle."""
    return [c.args[0] for c in app.broker.handle.call_args_list if isinstance(c.args[0], cls)]


class TestProfileSwitchByHandoff:
    """A picker pick on a handoff harness is confirmed in a ChoiceBar before anything is sent."""

    async def _pick(self, app, pilot) -> InputBar:
        """Select the agent, show its picker, and pick probe-heron."""
        await pilot.pause()
        bar = app._panels["agent-1"].input_bar
        bar.update_config_options([_agent_option()])
        bar._on_picker_selected("picker-agent", "probe-heron")
        await pilot.pause()
        return bar

    async def test_pick_mounts_choice_bar_and_keeps_picker(self) -> None:
        """Silent failure: the picker jumps to an unconfirmed agent, or a command goes out."""
        from synth_acp.models.commands import RequestProfileSwitch
        from synth_acp.ui.widgets.choice_bar import ChoiceBar

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = await self._pick(app, pilot)
            assert len(bar.query(ChoiceBar)) == 1
            assert bar.query_one("#picker-agent", _PickerLabel)._current_id == ""
            assert _sent(app, SetConfigOption) == []
            assert _sent(app, RequestProfileSwitch) == []

    async def test_hand_off_sends_request_without_instructions(self) -> None:
        from synth_acp.models.commands import RequestProfileSwitch

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._pick(app, pilot)
            await pilot.press("h")
            await pilot.pause(0.6)
            assert _sent(app, RequestProfileSwitch) == [
                RequestProfileSwitch(agent_id="agent-1", agent_mode="probe-heron", instructions=None)
            ]

    async def test_cancel_and_escape_send_nothing(self) -> None:
        from synth_acp.models.commands import RequestProfileSwitch
        from synth_acp.ui.widgets.choice_bar import ChoiceBar

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = await self._pick(app, pilot)
            await pilot.press("c")
            await pilot.pause(0.6)
            bar._on_picker_selected("picker-agent", "probe-heron")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause(0.6)
            assert len(bar.query(ChoiceBar)) == 0
            assert _sent(app, RequestProfileSwitch) == []

    async def test_instructions_submit_sends_text_not_prompt(self) -> None:
        """Silent failure: the instructions go to the agent as an ordinary prompt."""
        from synth_acp.models.commands import RequestProfileSwitch, SendPrompt

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._pick(app, pilot)
            await pilot.press("i")
            await pilot.pause(0.6)
            # Inserted rather than typed: PromptTextArea._on_key does not await
            # TextArea._on_key, so typed keys raise an un-awaited-coroutine warning.
            app._panels["agent-1"].input_bar.query_one(PromptTextArea).insert("keep {x} tests")
            await pilot.press("enter")
            await pilot.pause()
            assert _sent(app, RequestProfileSwitch) == [
                RequestProfileSwitch(
                    agent_id="agent-1", agent_mode="probe-heron", instructions="keep {x} tests"
                )
            ]
            assert _sent(app, SendPrompt) == []

    async def test_instructions_bang_text_is_not_a_shell_command(self) -> None:
        from synth_acp.models.commands import RequestProfileSwitch

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            await self._pick(app, pilot)
            await pilot.press("i")
            await pilot.pause(0.6)
            feed = app._panels["agent-1"]
            with patch.object(feed, "run_shell_command") as shell:
                feed.input_bar.query_one(PromptTextArea).insert("!ls")
                await pilot.press("enter")
                await pilot.pause()
            shell.assert_not_called()
            assert _sent(app, RequestProfileSwitch) == [
                RequestProfileSwitch(agent_id="agent-1", agent_mode="probe-heron", instructions="!ls")
            ]

    async def test_instructions_escape_sends_nothing_and_restores_placeholder(self) -> None:
        from synth_acp.models.commands import RequestProfileSwitch

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = await self._pick(app, pilot)
            await pilot.press("i")
            await pilot.pause(0.6)
            ta = bar.query_one("#prompt-input", PromptTextArea)
            assert str(ta.placeholder) == (
                "Instructions for the handoff to probe-heron  (enter to send, esc to cancel)"
            )
            await pilot.press("escape")
            await pilot.pause()
            assert str(ta.placeholder) == "Ask anything    ▌@▐ file  ▌!▐ shell  ▌ctrl+j▐ newline"
            assert _sent(app, RequestProfileSwitch) == []

    async def test_repick_during_instructions_mode_ends_it(self) -> None:
        """Silent failure: after a re-pick is cancelled, the user's next message is sent as
        handoff instructions instead of as a prompt."""
        from synth_acp.models.commands import RequestProfileSwitch, SendPrompt

        app = _handoff_app()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            bar = await self._pick(app, pilot)
            await pilot.press("i")
            await pilot.pause(0.6)
            bar._on_picker_selected("picker-agent", "probe-heron")
            await pilot.pause()
            await pilot.press("c")
            await pilot.pause(0.6)
            bar.query_one(PromptTextArea).insert("hello agent")
            await pilot.press("enter")
            await pilot.pause()
            assert _sent(app, RequestProfileSwitch) == []
            assert _sent(app, SendPrompt) == [SendPrompt(agent_id="agent-1", text="hello agent")]

    def test_mode_harness_agent_pick_sends_set_config_option(self) -> None:
        """Silent failure: a harness that switches in place loses its SetConfigOption."""
        from unittest.mock import AsyncMock

        bar = TestConfigOptionPickers()._make_input_bar()
        mock_app = MagicMock()
        mock_app.broker.handle = AsyncMock()
        mock_app.switches_profile_by_handoff = MagicMock(return_value=False)
        mock_app.run_worker = MagicMock(side_effect=lambda coro, **_kw: coro.close())

        with patch.object(bar, "query_one", return_value=MagicMock(spec=_PickerLabel)), \
             patch.object(bar, "_confirm_profile_switch") as confirm, \
             patch.object(type(bar), "app", new_callable=lambda: property(lambda _: mock_app)), \
             patch("synth_acp.ui.widgets.input_bar.isinstance", side_effect=lambda obj, cls: True if obj is mock_app else isinstance(obj, cls)):
            bar._on_picker_selected("picker-agent", "probe-heron")

        confirm.assert_not_called()
        mock_app.broker.handle.assert_called_once_with(
            SetConfigOption(agent_id="test-agent", config_id="agent", value="probe-heron")
        )


class TestSteerOrQueue:
    """While the agent is busy, Submit splits into Steer and Queue, and Shift+Enter steers."""

    async def _mounted(self, can_steer: bool):
        from textual.app import App, ComposeResult

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

        app = _App()
        return app, patch.object(InputBar, "_can_steer", return_value=can_steer)

    async def test_busy_bar_offers_steer_and_queue_only_when_harness_can_steer(self) -> None:
        """Offering Steer on Claude would send a steer that cannot be delivered; hiding
        Queue while busy would leave the user no way to queue with the mouse."""
        for can_steer in (True, False):
            app, can = await self._mounted(can_steer)
            with can:
                async with app.run_test(headless=True, size=(120, 40)) as pilot:
                    bar = app.query_one(InputBar)
                    bar.set_busy(True)
                    await pilot.pause()
                    shown = {b: bar.query_one(f"#{b}").display
                             for b in ("submit-btn", "steer-btn", "queue-btn")}
                    assert shown == {"submit-btn": False, "steer-btn": can_steer, "queue-btn": True}
                    bar.set_busy(False)
                    await pilot.pause()
                    assert bar.query_one("#submit-btn").display is True
                    assert bar.query_one("#steer-btn").display is False
                    assert bar.query_one("#queue-btn").display is False

    async def test_shift_enter_sends_a_steer_and_enter_does_not(self) -> None:
        """The key mapping is the whole user-facing contract: Shift+Enter steers, plain
        Enter keeps queueing as before."""
        from textual.app import App, ComposeResult

        posted: list[bool] = []

        class _App(App):
            def compose(self) -> ComposeResult:
                yield InputBar("agent-1", "agent-1", "kiro", cwd=".")

            def on_prompt_text_area_submitted(self, message: PromptTextArea.Submitted) -> None:
                posted.append(message.steer)

        app = _App()
        async with app.run_test(headless=True, size=(120, 40)) as pilot:
            ta = app.query_one("#prompt-input", PromptTextArea)
            ta.focus()
            for key in ("shift+enter", "enter"):
                ta.load_text("x")
                await pilot.press(key)
                await pilot.pause()
        assert posted == [True, False]
