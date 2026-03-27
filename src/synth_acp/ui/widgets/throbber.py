"""Pulsating gradient throbber widget."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.color import Color, Gradient
from textual.css.styles import RulesMap
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions, Visual
from textual.widget import Widget


class ThrobberVisual(Visual):

    def __init__(
        self,
        gradient: Gradient,
        character: str = "━",
        get_time: Callable[[], float] = monotonic,
    ) -> None:
        self.character = character
        self.get_time = get_time
        self._gradient = gradient
        self._cache: dict[tuple[int], list[Segment]] = {}

    def _make_segments(self, width: int, background: object) -> list[Segment]:
        key = (width,)
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
        offset = width - int((time % 1.0) * width)
        return [Strip(segments[offset : offset + width], cell_length=width)]

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:  # noqa: ARG002
        return container_width

    def get_height(self, rules: RulesMap, width: int) -> int:  # noqa: ARG002
        return 1


class Throbber(Widget):
    """A throbbing gradient line that reacts to theme changes."""

    def on_mount(self) -> None:
        self.auto_refresh = 1 / 15
        self._gradient = self._build_gradient()

    def on_app_theme_changed(self) -> None:
        self._gradient = self._build_gradient()

    def _build_gradient(self) -> Gradient:
        base = Color.parse(self.app.current_theme.primary)
        shades = [
            base.darken(0.4),
            base.darken(0.2),
            base,
            base.lighten(0.3),
            base.lighten(0.5),
            base.lighten(0.3),
            base,
            base.darken(0.2),
            base.darken(0.4),
        ]
        return Gradient.from_colors(*[s.hex for s in shades])

    def render(self) -> ThrobberVisual:
        return ThrobberVisual(self._gradient)
