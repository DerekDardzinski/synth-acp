# synth-acp UI Performance Optimizations

Implementation spec for the `feature/ui-perf-optimization` branch of `synth-acp`. All changes target the Textual TUI layer under `src/synth_acp/ui/`. All APIs referenced below are verified against Textual 8.2.x source (`textual/layouts/stream.py`, `textual/widget.py`, `textual/signal.py`, etc.).

Work on the `feature/ui-perf-optimization` branch. Do NOT create a new branch.

---

## Priority Table

| # | Change | File(s) | Impact | Effort |
|---|--------|---------|--------|--------|
| 1 | `layout: stream` on `.conv-scroll` | `app.tcss` | 🔴 High | Trivial |
| 2 | `anchor()` replaces manual `_follow` tracking | `conversation.py`, `app.py` | 🔴 High | Low |
| 3 | Cache `AgentTile` refs, avoid `query_one()` in hot paths | `app.py` | 🟠 Medium | Low |
| 4 | Prune stale entries from `_turns` list | `conversation.py` | 🟠 Medium | Low |
| 5 | `Signal` for tile ↔ feed coordination | `app.py`, `agent_list.py` | 🟠 Medium | Medium |
| 6 | `widget.batch()` for multi-step mount/remove ops | `conversation.py` | 🟡 Low-Med | Low |
| 7 | `Lazy` mount for heavy `ToolCallBlock` children | `tool_call.py` | 🟡 Low-Med | Medium |
| 8 | `RichLog` for large shell output | `tool_call.py` | 🟡 Low-Med | High |
| 9 | `reactive(toggle_class=...)` / `var` for boolean state | `agent_list.py`, feeds | 🟢 Low | Low |
| 10 | `mount_compose()` for dynamic widget building | `tool_call.py` | 🟢 Low | Low |
| 11 | Verify `textual-speedups` is active at runtime | env / entry point | 🟢 Low | Trivial |
| 12 | Fix `AgentTile` double-click bug via `add_content()` | `app.py` | 🔴 Bug fix | Trivial |

---

## 1. `layout: stream` on `.conv-scroll`

**File:** `app.tcss`

Textual ships an undocumented layout mode called `stream` (see `textual/layouts/stream.py`), designed for LLM chat-style UIs. It's a stripped-down vertical layout that skips layers, absolute positioning, CSS extrema, overlay handling, and non-TCSS styles.

The critical difference from `layout: vertical`: **stream caches per-widget placements and only recomputes from the first "dirty" widget downward** (see the `pre_populate` loop at lines 63–73 of `stream.py`). For `ConversationFeed`, where existing turns never move and new ones only append at the bottom, layout cost per append becomes O(1) instead of O(N).

```css
/* app.tcss — one line change */
.conv-scroll {
    layout: stream;   /* was: default layout: vertical */
    height: 1fr;
    overflow-y: auto;
}
```

The constraints: all children must be full-width and `height: auto`. `TurnContainer` already satisfies both. No absolute positioning or layers are used in the feed.

---

## 2. `anchor()` — Replace Manual `_follow` Tracking

**Files:** `conversation.py`, `app.py`

`ScrollableContainer.anchor()` pins scroll to the bottom whenever new content is mounted, and self-manages the user's ability to scroll away and back:

```python
def anchor(self, anchor: bool = True) -> None:
    """An anchored widget will stay scrolled to the bottom when new content
    is added, until the user moves the scroll position."""
    self._anchored = anchor
    if anchor:
        self.scroll_end(immediate=True, animate=False)
```

When the user scrolls up, Textual internally calls `release_anchor()`. When they scroll back to the bottom, `_check_anchor()` re-engages it automatically — no `on_scroll` handler or `_follow` flag needed.

**What to remove from `ConversationFeed`:**
- `self._follow: bool` instance variable
- `_is_at_bottom()` method
- `_scroll_to_bottom()` method
- The `_follow` update in `on_scroll` (keep the visibility timer dispatch)

**What to add:**
```python
def on_mount(self) -> None:
    self._scroll = self.query_one(".conv-scroll", ScrollableContainer)
    self._scroll.anchor()  # engaged from start, auto-releases on user scroll

# Anywhere you currently check self._follow, replace with:
if self._scroll and self._scroll.is_anchored:
    ...
```

**In `app.py`**, replace the `set_timer(0.2, scroll_end)` call after session replay with double-nested `call_after_refresh` — frame-perfect with no time dependency:

```python
# Replace: self.set_timer(0.2, lambda: feed._scroll.scroll_end(...))
feed.call_after_refresh(
    lambda: feed.call_after_refresh(
        lambda: feed._scroll.scroll_end(animate=False) if feed._scroll else None
    )
)
```

If you enable `anchor()` on mount, the scroll position is handled automatically and the timer is unnecessary entirely.

---

## 3. Cache `AgentTile` Refs — Avoid `query_one()` in Hot Paths

**File:** `app.py`

Every `AgentStateChanged` event triggers:
```python
tile = self.query_one(f"#tile-{css_id(event.agent_id)}", AgentTile)
```

`query_one()` traverses the full widget tree on each call. With N agents receiving frequent state updates, this is O(N × events) tree traversals. The app already maintains `self._panels` (a `dict[str, ConversationFeed]`) — use the same pattern for tiles:

```python
# Add to SynthApp.__init__:
self._tiles: dict[str, AgentTile] = {}

# When a tile is created:
self._tiles[agent_id] = tile

# In event handlers, replace query_one() with:
tile = self._tiles.get(event.agent_id)
if tile is None:
    return

# On termination:
self._tiles.pop(agent_id, None)
```

O(1) dict lookup replaces O(N) tree traversal per event.

---

## 4. Prune Stale Entries from `_turns` List

**File:** `conversation.py`

`self._turns` is appended to in `_start_turn()` but never pruned. Two problems:

1. If a `TurnContainer` is removed, calling `turn.virtual_region` on a detached widget returns garbage geometry — causing `_update_turn_visibility` to make incorrect show/hide decisions.
2. Over long sessions the list grows unbounded and the visibility loop gets slower.

One-line fix at the top of `_update_turn_visibility`:

```python
def _update_turn_visibility(self) -> None:
    self._turns = [t for t in self._turns if t.is_mounted]  # prune detached
    if self._scroll is None:
        return
    # ... rest of existing logic unchanged
```

---

## 5. `Signal` for Tile ↔ Feed Coordination

**Files:** `app.py`, `agent_list.py`

High-frequency UI coordination — e.g., the `AgentTile` activity bar reacting the instant a stream chunk arrives — currently flows through the broker event bus → app-level dispatch → DOM message bubbling. DOM messages are processed in queue order, adding latency at high throughput.

`Signal` (from `textual.signal`) bypasses the message queue entirely:

```python
from textual.signal import Signal

# In ConversationFeed.__init__:
self.streaming_signal: Signal[bool] = Signal(self, "streaming")

# When streaming starts (first chunk received):
self.streaming_signal.publish(True)

# When streaming stops (message finalized):
self.streaming_signal.publish(False)

# In AgentTile, after the feed is mounted:
feed.streaming_signal.subscribe(self, self._on_streaming_changed, immediate=True)

def _on_streaming_changed(self, streaming: bool) -> None:
    self._activity_bar.set_active(streaming)
```

`immediate=True` invokes the callback synchronously on the same asyncio tick as the publish. Signals use `WeakKeyDictionary` for subscriptions, so dead subscribers (terminated agents) are auto-pruned on the next publish.

---

## 6. `widget.batch()` for Multi-Step Mount/Remove Operations

**File:** `conversation.py`

`app.batch_update()` (used during replay) is a sync context manager that defers renders but doesn't lock the widget. For runtime operations where you simultaneously mount a new turn *and* hide/remove an old one, `widget.batch()` acquires the container lock and batches renders atomically:

```python
# When adding a new turn while removing a hidden old one:
async with self._scroll.batch():
    await self._scroll.mount(new_turn)
    await old_hidden_turn.remove()
    # Single layout pass after both ops — no interleaving
```

Use this anywhere in `ConversationFeed` where multiple `await mount()`/`await remove()` calls on the same container happen in sequence.

---

## 7. `textual.lazy.Lazy` for Heavy `ToolCallBlock` Children

**File:** `tool_call.py`

`ToolCallBlock.compose()` mounts all children synchronously: `CopyButton`, header `Static`, location label, raw input `Label`, `Markdown`, `DiffView`, output scroll. For tool calls with large diffs or long output, this blocks the event loop while the feed is still trying to scroll.

Wrap the expensive children in `Lazy` to defer them to the next frame:

```python
from textual.lazy import Lazy

def compose(self):
    yield CopyButton(lambda: "\n".join(self._copyable_parts))
    yield Static(self._build_markup(), id="tc-header")
    for w in self._location_widgets(self._initial_locations):
        yield w                  # cheap — keep sync
    for w in self._raw_input_widgets(self._initial_raw_input):
        yield Lazy(w)            # syntax-highlighted — defer
    for w in self._diff_widgets(self._initial_diffs):
        yield Lazy(w)            # DiffView is the heaviest — always defer
```

The header and location appear immediately; the diff/output loads in the next frame. The feed stays responsive during the mount.

---

## 8. `RichLog` for Large Shell Output

**File:** `tool_call.py`

Large shell output is currently rendered as a syntax-highlighted `Label` inside a `VerticalScroll`. The nested `VerticalScroll` creates an extra scrollable container with its own layout and scroll state inside an already-scrolling `ConversationFeed` — unnecessary overhead for content that mostly just needs to be read, not interactively scrolled within.

`RichLog` is purpose-built for appending lines efficiently and handles internal scrolling without the nested container overhead:

```python
from textual.widgets import RichLog

# In _raw_output_widgets():
log = RichLog(id="tc-raw-output", highlight=True, markup=False, max_lines=2000)
log.write(text)
return [Rule(line_style="dashed", id="tc-output-sep"), log]
```

`max_lines=2000` caps memory for runaway output. `RichLog` also supports incremental `write()` calls after mount, so partial tool output can be streamed in as it arrives.

---

## 9. `reactive(toggle_class=...)` and `var` for Boolean State

**Files:** `agent_list.py`, widget files

For boolean state that only needs to toggle a CSS class, the `toggle_class` parameter on `reactive` eliminates the `watch_*` boilerplate:

```python
# Instead of:
is_busy: reactive[bool] = reactive(False)
def watch_is_busy(self, busy: bool) -> None:
    self.set_class(busy, "tile-busy")

# Write:
is_busy: reactive[bool] = reactive(False, toggle_class="tile-busy")
```

For internal tracking state that needs `watch_*` callbacks but must not trigger repaint or layout (e.g., `_follow`, internal counters), use `var`:

```python
from textual.reactive import var

# Triggers watch_selected_agent() but costs zero renders on its own:
selected_agent: var[str] = var("")
```

`var` is a `Reactive` subclass with `repaint=False` and `layout=False`.

---

## 10. `mount_compose()` for Dynamic Widget Building

**File:** `tool_call.py`

`ToolCallBlock` builds widget lists via methods returning `list[Widget]` then calls `mount_all()`. `mount_compose()` accepts a `ComposeResult` generator directly — more readable, and ensures `on_mount` lifecycle hooks fire correctly for all children:

```python
# Instead of:
widgets = self._location_widgets(locations)
await self.mount_all(widgets)

# Write a generator:
def _compose_location(self, loc: ToolCallLocation) -> ComposeResult:
    label = f"{loc.path}:{loc.line}" if loc.line is not None else loc.path
    yield Static(label, id="tc-location", markup=False)

await self.mount_compose(self._compose_location(loc))
```

Particularly useful in `update_content()` where multiple widget types are conditionally mounted.

---

## 11. Verify `textual-speedups` Is Active

**File:** env / entry point

`textual-speedups>=0.2.1` is in `pyproject.toml`. This package replaces `textual.geometry.Region`, `Size`, `Offset`, and `Spacing` with native C extension types — the most frequently allocated objects in any layout pass. Verify it's engaged:

```python
from textual.geometry import Region
print(type(Region).__module__)
# "builtins"         → speedups active ✓
# "textual.geometry" → not loaded ✗
```

If not active, add `import textual_speedups` to the entry point before any other Textual imports, or confirm the package is installed in the active venv (`uv run python -c "import textual_speedups"`).

---

## 12. Fix `AgentTile` Double-Click Bug

**File:** `app.py`

Clicking an `AgentTile` for an agent that hasn't been visited before requires two clicks to switch the feed. The root cause is a race in `select_agent` between mounting the feed and the `ContentSwitcher` recognising it as a child.

The sequence on first click:

1. `agent_id not in self._panels` → feed is created and mounted with `await self.query_one("#right").mount(feed)`
2. `self.selected_agent = agent_id` triggers `watch_selected_agent`
3. `watch_selected_agent` calls `switcher.get_child_by_id(feed_id)` — but the feed was mounted onto `#right` (the `ContentSwitcher` element), not registered through `ContentSwitcher`'s own API, so `get_child_by_id` raises, hits `except: return`, and the switcher never updates
4. Second click skips the mount block entirely and goes straight to setting `selected_agent` — this time the feed is already a registered child, so it works

The fix is to use `ContentSwitcher.add_content()` instead of raw `mount()`. `add_content` registers the widget as a proper switcher child with `display=False`, so `get_child_by_id` finds it immediately when `watch_selected_agent` fires:

```python
# In select_agent(), replace:
await self.query_one("#right").mount(feed)

# With:
await self.query_one("#right", ContentSwitcher).add_content(feed, set_current=False)
```

`set_current=False` keeps the current feed visible during the mount — the switcher updates on the subsequent `self.selected_agent = agent_id` line as before. One-line change, fixes the bug entirely.

---

## How These Fit Together

**`layout: stream` + viewport visibility (already in branch):** The visibility toggle (`display: none`) eliminates off-screen turns from layout entirely. Stream layout then makes the remaining visible turns cheaper to lay out. Together they make long sessions O(viewport) instead of O(all turns).

**`anchor()` + existing `Markdown.get_stream()`:** Anchor handles scroll position without polling; `get_stream()` (already used in `AgentMessage`) handles token coalescing without manual rate limiting. Adding `anchor()` completes the picture — the streaming UX becomes smooth at any token rate without application code managing scroll position.

**`Signal` + cached tile refs:** Both reduce the cost of high-frequency state propagation. Signals bypass the DOM message queue; cached refs bypass widget tree traversal. Together they make per-event overhead nearly flat instead of O(N agents).

---

## Important Notes

- **Do not touch `agent_message.py` streaming** — it already uses `Markdown.get_stream()` correctly for live token coalescing, and `_coalesce_events()` handles the replay path.
- **Do not touch `input_bar.py` git branch detection** — it already uses `asyncio.to_thread()` via `run_worker(exclusive=True)` to run git operations off the event loop.
- **There is no `on_scroll_end` event in Textual 8.2.x** — `action_scroll_end` is a keyboard binding for the `End` key, not a scroll-gesture-settled event. The existing timer-debounce in `on_scroll` is the correct pattern for deferring `_update_turn_visibility`.
