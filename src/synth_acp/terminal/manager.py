from __future__ import annotations

import asyncio
import codecs
import fcntl
import logging
import os
import pty
import shlex
import struct
import termios
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from synth_acp.terminal.shell_read import shell_read

log = logging.getLogger(__name__)

BUFFER_SIZE = 64 * 1024 * 2


@dataclass
class Command:
    """A command and corresponding environment.

    Args:
        command: Command to run.
        args: List of arguments.
        env: Environment variables.
        cwd: Current working directory.
    """

    command: str
    args: list[str]
    env: Mapping[str, str]
    cwd: str

    def __str__(self) -> str:
        return shlex.join([self.command, *self.args]).strip("'")


@dataclass
class ToolState:
    """Snapshot of terminal process state.

    Args:
        output: Decoded output text.
        truncated: Whether output was truncated to byte limit.
        return_code: Process exit code, or None if still running.
        signal: Signal name if killed by signal.
    """

    output: str
    truncated: bool
    return_code: int | None = None
    signal: str | None = None


class TerminalProcess:
    """Async PTY process manager with raw byte buffering.

    Spawns a command in a pseudo-terminal, reads output, and provides
    callbacks for data and exit events. Zero Textual dependency.

    Args:
        command: The command to execute.
        output_byte_limit: Max bytes to retain in output buffer.
    """

    def __init__(
        self, command: Command, *, output_byte_limit: int | None = None
    ) -> None:
        self._command = command
        self._output_byte_limit = output_byte_limit
        self._output: deque[bytes] = deque()
        self._output_bytes_count = 0
        self._process: asyncio.subprocess.Process | None = None
        self._shell_fd: int | None = None
        self._return_code: int | None = None
        self._released = False
        self._ready_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

        self.on_output: Callable[[str], Awaitable[None]] | None = None
        """Async callback invoked with decoded text after each PTY read."""
        self.on_exit: Callable[[int | None], None] | None = None
        """Sync callback invoked by finalize() with the return code."""

    @property
    def return_code(self) -> int | None:
        """Process exit code, or None if still running."""
        return self._return_code

    @property
    def released(self) -> bool:
        """Whether release() has been called."""
        return self._released

    @property
    def tool_state(self) -> ToolState:
        """Current output snapshot."""
        output, truncated = self.get_output()
        return ToolState(
            output=output, truncated=truncated, return_code=self._return_code
        )

    async def start(self) -> None:
        """Spawn the process as a background task and wait until ready."""
        self._task = asyncio.create_task(
            self._run_wrapper(), name=f"TerminalProcess {self._command}"
        )
        await self._ready_event.wait()

    async def _run_wrapper(self) -> None:
        """Wrapper that ensures _exit_event is always set."""
        try:
            await self._run()
        except Exception:
            log.error("TerminalProcess._run failed", exc_info=True)
        finally:
            self._exit_event.set()

    async def _run(self) -> None:
        """Spawn PTY, read output loop, finalize on EOF."""
        master, slave = pty.openpty()
        self._shell_fd = master

        flags = fcntl.fcntl(master, fcntl.F_GETFL)
        fcntl.fcntl(master, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        command = self._command
        environment = os.environ | command.env

        if " " in command.command:
            run_command = command.command
        else:
            run_command = f"{command.command} {shlex.join(command.args)}"

        shell = os.environ.get("SHELL", "sh")

        try:
            process = self._process = await asyncio.create_subprocess_exec(
                shell, "-c", run_command,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=environment,
                cwd=command.cwd,
            )
        except Exception:
            self._ready_event.set()
            raise

        self._ready_event.set()
        os.close(slave)

        reader = asyncio.StreamReader(BUFFER_SIZE)
        protocol = asyncio.StreamReaderProtocol(reader)
        loop = asyncio.get_event_loop()
        transport, _ = await loop.connect_read_pipe(
            lambda: protocol, os.fdopen(master, "rb", 0)
        )

        unicode_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                data = await shell_read(reader, BUFFER_SIZE)
                if process_data := unicode_decoder.decode(data, final=not data):
                    self._record_output(data)
                    if self.on_output is not None:
                        await self.on_output(process_data)
                if not data:
                    break
        finally:
            transport.close()

        self.finalize()
        self._return_code = await process.wait()

    def kill(self) -> bool:
        """Send SIGKILL to the subprocess.

        Returns:
            True if killed, False if already exited or no process.
        """
        if self._return_code is not None:
            return False
        if self._process is None:
            return False
        try:
            self._process.kill()
        except Exception:
            return False
        return True

    def release(self) -> None:
        """Mark the terminal as released."""
        self._released = True

    async def wait_for_exit(self) -> tuple[int | None, str | None]:
        """Wait for the process to exit.

        Returns:
            Tuple of (return_code, signal).
        """
        await self._exit_event.wait()
        return (self._return_code, None)

    def get_output(self) -> tuple[str, bool]:
        """Get buffered output, truncated to byte limit.

        Returns:
            Tuple of (output_text, truncated).
        """
        output_bytes = b"".join(self._output)
        truncated = False
        if (
            self._output_byte_limit is not None
            and len(output_bytes) > self._output_byte_limit
        ):
            truncated = True
            output_bytes = output_bytes[-self._output_byte_limit :]
            for offset, byte_value in enumerate(output_bytes):
                if (byte_value & 0b11000000) != 0b10000000:
                    if offset:
                        output_bytes = output_bytes[offset:]
                    break

        return output_bytes.decode("utf-8", "replace"), truncated

    def resize_pty(self, width: int, height: int) -> None:
        """Set PTY window size via TIOCSWINSZ ioctl.

        No-op if PTY fd is not open.

        Args:
            width: Terminal width in columns.
            height: Terminal height in rows.
        """
        if self._shell_fd is None:
            return
        try:
            size = struct.pack("HHHH", height, width, 0, 0)
            fcntl.ioctl(self._shell_fd, termios.TIOCSWINSZ, size)
        except OSError:
            pass

    def finalize(self) -> None:
        """Mark process as exited and invoke on_exit callback."""
        if self.on_exit is not None:
            self.on_exit(self._return_code)

    def _record_output(self, data: bytes) -> None:
        """Buffer raw bytes, enforcing byte limit.

        Args:
            data: Raw bytes from PTY read.
        """
        self._output.append(data)
        self._output_bytes_count += len(data)

        if self._output_byte_limit is None:
            return

        while self._output_bytes_count > self._output_byte_limit and self._output:
            oldest = self._output[0]
            oldest_len = len(oldest)
            if self._output_bytes_count - oldest_len < self._output_byte_limit:
                break
            self._output.popleft()
            self._output_bytes_count -= oldest_len
