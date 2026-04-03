from __future__ import annotations

import sys

import pytest

from synth_acp.terminal.manager import Command, TerminalProcess


@pytest.fixture
def tmp_cwd(tmp_path: object) -> str:
    """Provide a temporary directory path as string."""
    return str(tmp_path)


class TestTerminalProcess:
    """Tests for TerminalProcess PTY spawning and output buffering."""

    async def test_terminal_process_when_echo_command_returns_output(
        self, tmp_cwd: str
    ) -> None:
        """PTY spawning and output buffering work — output captured and exit code set."""
        proc = TerminalProcess(Command("echo", ["hello"], {}, tmp_cwd))
        await proc.start()
        rc, _ = await proc.wait_for_exit()
        state = proc.tool_state
        assert "hello" in state.output
        assert state.return_code == 0
        assert rc == 0

    async def test_terminal_process_when_killed_sets_return_code(
        self, tmp_cwd: str
    ) -> None:
        """Kill signal delivered — process doesn't leak."""
        proc = TerminalProcess(Command("sleep", ["60"], {}, tmp_cwd))
        await proc.start()
        assert proc.kill() is True
        rc, _ = await proc.wait_for_exit()
        # Killed process has non-None return code (negative on signal)
        assert rc is not None or _ is not None
        # Second kill returns False — process already dead
        assert proc.kill() is False

    async def test_terminal_process_when_output_exceeds_limit_truncates(
        self, tmp_cwd: str
    ) -> None:
        """Byte limit enforced — unbounded output doesn't blow token budget."""
        proc = TerminalProcess(
            Command(
                sys.executable, ["-c", "print('x' * 200)"], {}, tmp_cwd
            ),
            output_byte_limit=100,
        )
        await proc.start()
        await proc.wait_for_exit()
        state = proc.tool_state
        assert state.truncated is True
        assert len(state.output.encode("utf-8")) <= 100

    async def test_terminal_process_when_output_written_calls_on_output_per_write(
        self, tmp_cwd: str
    ) -> None:
        """on_output called per read, not batched to exit — UI gets live updates."""
        chunks: list[str] = []

        async def collect(text: str) -> None:
            chunks.append(text)

        proc = TerminalProcess(
            Command(
                sys.executable,
                ["-c", "import time; print('a'); time.sleep(0.1); print('b')"],
                {},
                tmp_cwd,
            ),
        )
        proc.on_output = collect
        await proc.start()
        await proc.wait_for_exit()
        # Callback was invoked with string data
        assert any("a" in c for c in chunks)
        assert any("b" in c for c in chunks)
        # Called more than once (not just a single batch at exit)
        assert len(chunks) >= 2
