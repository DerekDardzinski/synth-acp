"""Time the first real diff MOUNT in a fresh interpreter, with and without the pre-warm.

Run as a SUBPROCESS by the perf gate. It must be a fresh interpreter each time: the Pygments
lexer index and lexer modules are cached process-globally, so any earlier highlight in the
pytest session warms them and an in-process comparison reads ~1ms both ways and passes
vacuously.

Usage: python -m tests._prewarm_probe {cold|warm}
Prints one line: ``<milliseconds> <rendered|unrendered>``
"""

from __future__ import annotations

import asyncio
import sys
import time

from synth_acp.models.events import ToolCallDiff


async def _measure(*, prewarm: bool) -> tuple[float, bool]:
    from synth_acp.ui.app import SynthApp
    from synth_acp.ui.widgets.diff_view import prewarm_highlighting
    from synth_acp.ui.widgets.tool_call import DiffState
    from tests.conftest import drain_pending_diffs
    from tests.ui.widgets.test_conversation import _make_broker, _make_config

    # The app starts its OWN pre-warm worker unconditionally, so a baseline that merely
    # skips calling prewarm_highlighting() is not cold — the worker warms the cache while
    # the measurement runs. The baseline must therefore disable the feature itself, which is
    # what makes this a with-versus-without comparison.
    if prewarm:
        prewarm_highlighting()

    diff = ToolCallDiff(
        path="src/module.py",
        old_text="\n".join(f"def old_{n}(x):\n    return x + {n}" for n in range(60)),
        new_text="\n".join(f"def new_{n}(x):\n    return x * {n}" for n in range(60)),
    )

    if not prewarm:
        SynthApp._prewarm_highlighting = lambda *_: None

    app = SynthApp(_make_broker(), _make_config())
    async with app.run_test(headless=True, size=(120, 40)):
        await app.select_agent("a1")
        feed = app._panels["a1"]

        started = time.perf_counter()
        await feed.add_tool_call("tc-1", "Edit", "edit", "completed", diffs=[diff])
        await drain_pending_diffs(app, timeout=120.0)
        elapsed_ms = (time.perf_counter() - started) * 1000

        block = feed._tool_call_blocks["tc-1"]
        rendered = any(record.state is DiffState.RENDERED for record in block._diff_states.values())
    return elapsed_ms, rendered


def main() -> None:
    mode = sys.argv[1]
    elapsed_ms, rendered = asyncio.run(_measure(prewarm=(mode == "warm")))
    print(f"{elapsed_ms:.1f} {'rendered' if rendered else 'unrendered'}")


if __name__ == "__main__":
    main()
