"""Pulsating gradient bar widget."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.color import Color, Gradient
from textual.css.query import NoMatches
from textual.css.styles import RulesMap
from textual.reactive import reactive
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions, Visual
from textual.widget import Widget
from textual.widgets import Static


class GradientBarVisual(Visual):

    def __init__(
        self,
        gradient: Gradient,
        character: str = "━",
        get_time: Callable[[], float] = monotonic,
    ) -> None:
        self.character = character
        self.get_time = get_time
        self._gradient = gradient
        self._cache: dict[tuple, list[Segment]] = {}

    def _make_segments(self, width: int, background: object) -> list[Segment]:
        key = (width, background)
        if key not in self._cache:
            self._cache[key] = [
                Segment(
                    self.character,
                    RichStyle.from_color(
                        self._gradient.get_rich_color((offset / width) % 1),
                        background,
                    ),
                )
                for offset in range(width * 2)
            ]
        return self._cache[key]

    def render_strips(
        self,
        width: int,
        height: int | None,  # noqa: ARG002
        style: Style,
        options: RenderOptions,  # noqa: ARG002
    ) -> list[Strip]:
        time = self.get_time()
        segments = self._make_segments(width, style.rich_style.bgcolor)
        offset = width - int((time % 2.0) / 2.0 * width)
        return [Strip(segments[offset : offset + width], cell_length=width)]

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:  # noqa: ARG002
        return container_width

    def get_height(self, rules: RulesMap, width: int) -> int:  # noqa: ARG002
        return 1


class GradientBar(Widget):
    """An animated gradient line that reacts to theme changes."""

    def on_mount(self) -> None:
        self.auto_refresh = 1 / 15
        self._gradient = self._build_gradient()
        self._visual = GradientBarVisual(self._gradient)

    def on_app_theme_changed(self) -> None:
        self._gradient = self._build_gradient()
        self._visual = GradientBarVisual(self._gradient)
        self.refresh()

    def _build_gradient(self) -> Gradient:
        theme = self.app.current_theme
        primary = Color.parse(theme.primary)
        secondary = Color.parse(theme.secondary or theme.primary)
        return Gradient.from_colors(
            primary.hex,
            secondary.hex,
            primary.hex,
        )

    def render(self) -> GradientBarVisual:
        return self._visual


class ActivityBar(Widget):
    """Animated gradient bar with static fallback to prevent layout shift.

    Set ``active`` to toggle between animated gradient and static placeholder.
    """

    DEFAULT_CSS = """
    ActivityBar {
        height: 1;
        hatch: none;
    }
    ActivityBar > GradientBar {
        height: 1;
        hatch: none;
    }
    ActivityBar > .activity-bar-bg {
        height: 1;
        background: transparent;
        display: none;
    }
    """

    active: reactive[bool] = reactive(True)

    def compose(self):
        yield GradientBar()
        yield Static("", classes="activity-bar-bg")

    def watch_active(self, value: bool) -> None:
        """Toggle gradient vs static fallback."""
        try:
            self.query_one(GradientBar).display = value
            self.query_one(".activity-bar-bg").display = not value
        except NoMatches:
            pass
