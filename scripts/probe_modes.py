#!/usr/bin/env python3
"""Probe kiro-cli ACP new_session() for mode support."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from acp import Client, spawn_agent_process


class ProbeClient(Client):
    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        pass

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        pass

    async def request_permission(self, options: Any, session_id: str, tool_call: Any, **kwargs: Any) -> Any:
        from acp import RequestPermissionResponse
        from acp.schema import AllowedOutcome
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
    agent = sys.argv[1] if len(sys.argv) > 1 else "coder"
    print(f"Spawning kiro-cli acp --agent {agent} ...", file=sys.stderr)

    async with spawn_agent_process(
        ProbeClient(), "kiro-cli", "acp", "--agent", agent, cwd="."
    ) as (conn, proc):
        await conn.initialize(
            protocol_version=1,
            client_capabilities={
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
            client_info={"name": "probe", "version": "0.1"},
        )
        session = await conn.new_session(cwd=".")

        print(f"\nsession_id: {session.session_id}")
        print(f"modes field: {session.modes}")

        if session.modes:
            print(f"\n  current_mode_id: {session.modes.current_mode_id}")
            print(f"  available_modes ({len(session.modes.available_modes)}):")
            for m in session.modes.available_modes:
                print(f"    - id={m.id!r}  name={m.name!r}  desc={m.description!r}")
        else:
            print("\n  (agent does not advertise modes)")

        # Dump all non-None fields for completeness
        print(f"\nAll set fields: {session.model_fields_set}")
        dumped = session.model_dump(exclude_none=True, by_alias=True)
        for k, v in dumped.items():
            if k != "sessionId":
                print(f"  {k}: {v}")

        proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
