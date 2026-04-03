#!/usr/bin/env python3
"""Diagnostic: test if kiro-cli sends terminal/create RPCs when terminal=True.

Spawns kiro-cli with terminal capability enabled, sends a prompt that should
trigger a shell command, and logs all terminal RPC calls and session updates.

Usage:
    uv run python scripts/test_terminal.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from acp import Client, spawn_agent_process, text_block
from acp.schema import (
    AllowedOutcome,
    CreateTerminalResponse,
    KillTerminalResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class TerminalTestClient(Client):
    """ACP client that logs all terminal RPCs and session updates."""

    def __init__(self) -> None:
        self._terminal_count = 0
        self._terminal_output: dict[str, list[str]] = {}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        su = getattr(update, "session_update", None) or getattr(update, "sessionUpdate", None)
        # Log all update types
        log(f"[UPDATE] type={su}")

        # Log tool call details
        if su in ("tool_call", "tool_call_update"):
            kind = getattr(update, "kind", None)
            title = getattr(update, "title", None)
            status = getattr(update, "status", None)
            tool_call_id = getattr(update, "toolCallId", None) or getattr(update, "tool_call_id", None)
            log(f"  tool_call_id={tool_call_id} kind={kind} title={title} status={status}")

            # Check for terminal content
            for item in getattr(update, "content", None) or []:
                item_type = getattr(item, "type", None)
                if item_type == "terminal":
                    tid = getattr(item, "terminal_id", None) or getattr(item, "terminalId", None)
                    log(f"  *** TERMINAL CONTENT: terminal_id={tid}")
                else:
                    log(f"  content item type={item_type}")

            raw_output = getattr(update, "raw_output", None)
            if raw_output is not None:
                log(f"  raw_output keys={list(raw_output.keys()) if isinstance(raw_output, dict) else type(raw_output)}")
                if isinstance(raw_output, dict):
                    for k, v in raw_output.items():
                        val_str = str(v)[:500]
                        log(f"  raw_output[{k}]={val_str}")

        # Log agent messages
        if su == "agent_message_chunk":
            content = getattr(update, "content", None)
            if content:
                text = getattr(content, "text", None)
                if text:
                    log(f"  text={text[:100]}...")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: Any = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        self._terminal_count += 1
        tid = f"test-terminal-{self._terminal_count}"
        log(f"[TERMINAL/CREATE] command={command} args={args} cwd={cwd} output_byte_limit={output_byte_limit}")
        log(f"  → returning terminal_id={tid}")
        self._terminal_output[tid] = []
        return CreateTerminalResponse(terminal_id=tid)

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> TerminalOutputResponse:
        log(f"[TERMINAL/OUTPUT] terminal_id={terminal_id}")
        # Return empty output since we're not actually running the command
        return TerminalOutputResponse(output="(test client - no real output)", truncated=False)

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> KillTerminalResponse:
        log(f"[TERMINAL/KILL] terminal_id={terminal_id}")
        return KillTerminalResponse()

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> ReleaseTerminalResponse:
        log(f"[TERMINAL/RELEASE] terminal_id={terminal_id}")
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> WaitForTerminalExitResponse:
        log(f"[TERMINAL/WAIT_FOR_EXIT] terminal_id={terminal_id}")
        return WaitForTerminalExitResponse(exit_code=0)

    async def request_permission(self, options: Any, session_id: str, tool_call: Any, **kwargs: Any) -> Any:
        log(f"[PERMISSION] title={getattr(tool_call, 'title', '?')} kind={getattr(tool_call, 'kind', '?')}")
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=options[0].option_id, outcome="selected")
        )

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> Any:
        return {"content": ""}

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> Any:
        return None

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        log(f"[EXT_NOTIFICATION] method={method}")

    def on_connect(self, conn: Any) -> None:
        pass


async def main() -> None:
    client = TerminalTestClient()

    log("=== Starting kiro-cli with terminal=True ===")
    async with spawn_agent_process(client, "kiro-cli", "acp", cwd=".") as (conn, proc):
        await conn.initialize(
            protocol_version=1,
            client_capabilities={
                "fs": {"readTextFile": True, "writeTextFile": True},
                "terminal": True,
            },
            client_info={"name": "terminal-test", "version": "0.1"},
        )
        session = await conn.new_session(cwd=".")
        log(f"Session: {session.session_id}")

        prompt = "Run `echo hello world` in the terminal and tell me what it outputs."
        log(f"\n=== Sending prompt: {prompt} ===\n")

        resp = await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block(prompt)],
        )

        log(f"\n=== DONE ===")
        log(f"stop_reason={resp.stop_reason}")
        log(f"Terminals created: {client._terminal_count}")
        for tid, output in client._terminal_output.items():
            log(f"  {tid}: {len(output)} output chunks")


if __name__ == "__main__":
    asyncio.run(main())
