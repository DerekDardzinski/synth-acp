#!/usr/bin/env python3
"""Quick script to test what session_update types kiro-cli emits."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from acp import Client, spawn_agent_process


class TestClient(Client):
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        su = getattr(update, "session_update", None) or getattr(update, "sessionUpdate", None)
        print(f"[UPDATE] type={su}", file=sys.stderr)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        print(f"[EXT_NOTIFICATION] method={method} params={params}", file=sys.stderr)

    async def request_permission(
        self, options: Any, session_id: str, tool_call: Any, **kwargs: Any
    ) -> Any:
        from acp import RequestPermissionResponse
        from acp.schema import AllowedOutcome

        # Auto-allow everything
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id=options[0].option_id, outcome="selected")
        )

    async def read_text_file(self, path: str, session_id: str, **kwargs: Any) -> Any:
        return None

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> Any:
        return None

    async def create_terminal(self, command: str, session_id: str, **kwargs: Any) -> Any:
        return None

    async def terminal_output(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None

    async def wait_for_terminal_exit(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None

    async def kill_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None

    async def release_terminal(self, session_id: str, terminal_id: str, **kwargs: Any) -> Any:
        return None


async def main() -> None:
    client = TestClient()
    async with spawn_agent_process(client, "kiro-cli", "acp", cwd=".") as (
        conn,
        proc,
    ):
        await conn.initialize(
            protocol_version=1,
            client_capabilities={
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            client_info={"name": "test", "version": "0.1"},
        )
        session = await conn.new_session(cwd=".")
        print(f"Session: {session.session_id}", file=sys.stderr)

        from acp import text_block

        resp = await conn.prompt(
            session_id=session.session_id,
            prompt=[text_block("Say hello in one sentence")],
        )
        print(f"\n[DONE] Response received", file=sys.stderr)
        print(f"[RESP] stop_reason={resp.stop_reason}", file=sys.stderr)
        print(f"[RESP] usage={resp.usage}", file=sys.stderr)
        if resp.usage:
            for name in resp.usage.model_fields:
                print(f"  {name}={getattr(resp.usage, name, None)}", file=sys.stderr)

        # Check available slash commands
        # Commands come via ext_notification above — check stderr output


if __name__ == "__main__":
    asyncio.run(main())
