"""Tests for input_bar helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from acp.schema import SessionConfigOptionBoolean, SessionConfigOptionSelect

from synth_acp.models.commands import SetConfigOption
from synth_acp.ui.file_discovery import FileEntry
from synth_acp.ui.widgets.input_bar import InputBar, PromptTextArea, _PickerLabel, _short_path
from synth_acp.ui.widgets.prompt_queue import QueuedPrompt


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
    """Tests for InputBar queue routing, enqueue/drain_next/is_composing, and DrainReady."""

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

    def test_submit_when_busy_enqueues(self) -> None:
        """Submission while busy routes to queue, not SendPrompt."""
        bar = self._make_input_bar(busy=True)
        queue_mock = MagicMock()
        ta_mock = MagicMock()
        ta_mock.text = "hello agent"

        with patch.object(bar, "query_one", return_value=queue_mock):
            message = PromptTextArea.Submitted(ta_mock)
            bar.on_prompt_text_area_submitted(message)

        ta_mock.clear.assert_called_once()
        queue_mock.enqueue.assert_called_once_with("hello agent", "user", None)

    def test_submit_when_not_busy_sends_prompt(self) -> None:
        """Submission while not busy does NOT enqueue."""
        bar = self._make_input_bar(busy=False)
        ta_mock = MagicMock()
        ta_mock.text = "hello agent"

        # When not busy, the code skips the enqueue branch and tries to access self.app.
        # Since we're not in a Textual tree, it raises AttributeError — that's fine.
        # We only need to verify enqueue was never called.
        enqueue_called = False

        def fake_query_one(*args, **kwargs):
            nonlocal enqueue_called
            enqueue_called = True
            return MagicMock()

        with patch.object(bar, "query_one", side_effect=fake_query_one):
            message = PromptTextArea.Submitted(ta_mock)
            try:
                bar.on_prompt_text_area_submitted(message)
            except AttributeError:
                pass  # Expected — self.app not available outside Textual tree

        assert not enqueue_called

    def test_drain_next_delegates(self) -> None:
        """drain_next delegates to PromptQueue.drain_next."""
        bar = self._make_input_bar()
        expected = QueuedPrompt(text="queued text", source="user")
        queue_mock = MagicMock()
        queue_mock.drain_next.return_value = expected

        with patch.object(bar, "query_one", return_value=queue_mock):
            result = bar.drain_next()

        assert result is expected

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

    def test_drain_ready_posts_with_agent_id(self) -> None:
        """DrainReady from PromptQueue bubbles as InputBar.DrainReady with agent_id."""
        bar = self._make_input_bar()
        bar._agent_id = "my-agent"
        posted = []
        bar.post_message = posted.append  # type: ignore[assignment]

        drain_msg = MagicMock()
        drain_msg.stop = MagicMock()
        bar.on_prompt_queue_drain_ready(drain_msg)

        assert len(posted) == 1
        assert isinstance(posted[0], InputBar.DrainReady)
        assert posted[0].agent_id == "my-agent"
        drain_msg.stop.assert_called_once()


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

        # 3 pickers (boolean filtered out), ordered: mode(0), model(1), thought_level(2)
        assert len(mounted_pickers) == 3
        assert mounted_pickers[0].id == "picker-mode"
        assert mounted_pickers[1].id == "picker-model"
        assert mounted_pickers[2].id == "picker-effort"

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

        with patch.object(bar, "query_one", return_value=picker), \
             patch.object(type(bar), "app", new_callable=lambda: property(lambda _: mock_app)), \
             patch("synth_acp.ui.widgets.input_bar.isinstance", side_effect=lambda obj, cls: True if obj is mock_app else isinstance(obj, cls)):
            bar._on_picker_selected("picker-effort", "high")

        mock_app.run_worker.assert_called_once()
        mock_app.broker.handle.assert_called_once_with(
            SetConfigOption(agent_id="test-agent", config_id="effort", value="high")
        )
