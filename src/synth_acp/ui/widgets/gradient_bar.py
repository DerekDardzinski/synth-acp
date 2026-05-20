"""Pulsating gradient bar widget."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.color import Color, Gradient
from textual.css.styles import RulesMap
from textual.reactive import reactive
from textual.strip import Strip
from textual.style import Style
from textual.visual import RenderOptions, Visual
from textual.widget import Widget


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
        offset = width - int((time % 1.0) * width)
        return [Strip(segments[offset : offset + width], cell_length=width)]

    def get_optimal_width(self, rules: RulesMap, container_width: int) -> int:  # noqa: ARG002
        return container_width

    def get_height(self, rules: RulesMap, width: int) -> int:  # noqa: ARG002
        return 1


class UsageBarVisual(Visual):
    """Single-row usage progress bar with percentage and cost label."""

    def __init__(
        self,
        used: int,
        size: int,
        cost_text: str,
        fill_color: Color,
        empty_color: Color,
        label_color: Color,
        fill_char: str = "━",
        empty_char: str = "─",
    ) -> None:
        self.used = used
        self.size = size
        self.cost_text = cost_text
        self.fill_color = fill_color
        self.empty_color = empty_color
        self.label_color = label_color
        self.fill_char = fill_char
        self.empty_char = empty_char

    def _build_label(self) -> str:
        """Build the right-aligned label text."""
        if self.size > 0:
            pct = int(self.used / self.size * 100)
            if self.cost_text:
                return f"{pct}% {self.cost_text}"
            return f"{pct}%"
        if self.cost_text:
            return self.cost_text
        return ""

    def render_strips(
        self,
        width: int,
        height: int | None,  # noqa: ARG002
        style: Style,
        options: RenderOptions,  # noqa: ARG002
    ) -> list[Strip]:
        """Render the bar with filled/empty chars and right-aligned label."""
        label = self._build_label()
        if not label and self.used == 0 and self.size == 0:
            return [Strip([])]

        if label:
            bar_width = max(0, width - len(label) - 1)
        else:
            bar_width = width

        pct = self.used / self.size if self.size > 0 else 0.0
        filled = min(int(pct * bar_width), bar_width)
        empty = bar_width - filled

        bg = style.rich_style.bgcolor

        segments: list[Segment] = []
        if filled > 0:
            segments.append(
                Segment(
                    self.fill_char * filled,
                    RichStyle.from_color(self.fill_color.rich_color, bg),
                )
            )
        if empty > 0:
            segments.append(
                Segment(
                    self.empty_char * empty,
                    RichStyle.from_color(self.empty_color.rich_color, bg),
                )
            )
        if label:
            segments.append(
                Segment(
                    " " + label,
                    RichStyle.from_color(self.label_color.rich_color, bg),
                )
            )

        return [Strip(segments, cell_length=width)]

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
        self.app.theme_changed_signal.subscribe(self, self._on_theme_changed)

    def on_show(self) -> None:
        """Resume animation timer when widget becomes visible."""
        self.auto_refresh = 1 / 15

    def on_hide(self) -> None:
        """Pause animation timer when widget is hidden."""
        self.auto_refresh = None

    def on_unmount(self) -> None:
        self.app.theme_changed_signal.unsubscribe(self)

    def _on_theme_changed(self, theme: object) -> None:  # noqa: ARG002
        self._gradient = self._build_gradient()
        self._visual = GradientBarVisual(self._gradient)
        self.refresh()

    def _build_gradient(self) -> Gradient:
        theme = self.app.current_theme
        primary = Color.parse(theme.primary)
        secondary = Color.parse(theme.secondary or theme.primary)
        accent = Color.parse(theme.accent or theme.secondary or theme.primary)

        def variants(c: Color) -> tuple[str, str, str]:
            dark = c.darken(0.05)
            light = c.lighten(0.05)
            return dark.hex, c.hex, light.hex

        p_d, p, p_l = variants(primary)
        s_d, s, s_l = variants(secondary)
        a_d, a, a_l = variants(accent)

        return Gradient.from_colors(
            p_d, p, p_l,
            s_d, s, s_l,
            a_d, a, a_l,
            p_d,
        )

    def render(self) -> GradientBarVisual:
        return self._visual


class UsageBar(Widget):
    """Progress bar showing context window usage.

    Call update() to set new values. Responds to theme changes.
    Height is managed by the parent ActivityBar CSS (.activity-bar-bg).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._used: int = 0
        self._context_size: int = 0
        self._cost_text: str = ""
        self._visual: UsageBarVisual | None = None

    def on_mount(self) -> None:
        """Subscribe to theme changes."""
        self.app.theme_changed_signal.subscribe(self, self._on_theme_changed)

    def on_unmount(self) -> None:
        self.app.theme_changed_signal.unsubscribe(self)

    def update(self, used: int, size: int, cost_text: str = "") -> None:
        """Update the usage bar with new values.

        Args:
            used: Tokens currently used in context window.
            size: Total context window size (0 = unknown).
            cost_text: Formatted cost string (e.g. "$1.23").
        """
        self._used = used
        self._context_size = size
        self._cost_text = cost_text
        self._rebuild_visual()
        self.refresh()

    def _rebuild_visual(self) -> None:
        """Rebuild the visual with current values and theme colors."""
        pct = self._used / self._context_size if self._context_size > 0 else 0.0
        fill_color = self._pick_color(pct)
        theme = self.app.current_theme
        empty_color = Color.parse(theme.surface or theme.background or "#333333")
        label_color = Color.parse(theme.foreground or "#e0e0e0")
        self._visual = UsageBarVisual(
            used=self._used,
            size=self._context_size,
            cost_text=self._cost_text,
            fill_color=fill_color,
            empty_color=empty_color,
            label_color=label_color,
        )

    def _pick_color(self, pct: float) -> Color:
        """Return success/warning/error color based on usage percentage.

        <70% -> success, 70-90% -> warning, >=90% -> error.
        """
        theme = self.app.current_theme
        if pct >= 0.9:
            return Color.parse(theme.error)
        if pct >= 0.7:
            return Color.parse(theme.warning)
        return Color.parse(theme.success)

    def _on_theme_changed(self, theme: object) -> None:  # noqa: ARG002
        self._rebuild_visual()
        self.refresh()

    def render(self) -> UsageBarVisual | str:
        """Return the visual, or empty string if no data."""
        if self._visual is not None and (self._context_size > 0 or self._cost_text):
            return self._visual
        return ""


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
        display: none;
    }
    ActivityBar > .activity-bar-bg {
        height: 1;
        display: block;
    }
    ActivityBar.activity-active > GradientBar {
        display: block;
    }
    ActivityBar.activity-active > .activity-bar-bg {
        display: none;
    }
    """

    active: reactive[bool] = reactive(True, toggle_class="activity-active")

    def compose(self):
        yield GradientBar()
        yield UsageBar(classes="activity-bar-bg")

    def update_usage(self, used: int, size: int, cost_text: str = "") -> None:
        """Update the usage bar display.

        Args:
            used: Tokens used in context (0 = no data).
            size: Total context window tokens (0 = unknown).
            cost_text: Pre-formatted cost string (e.g. "$1.23").

        When active=True (busy), the gradient shows regardless of usage data.
        When active=False (idle), the usage bar shows with these values.
        """
        self.query_one(UsageBar).update(used, size, cost_text)
