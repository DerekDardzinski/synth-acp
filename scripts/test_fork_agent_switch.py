"""Test: fork session with different agent to verify agent switching + history continuity.

Spawns a Claude Code session with no custom agent (default), asks it a question with a
secret word, then forks the session with a different agent (code-planner), and asks
the forked session what its role is and what the secret word was.

Usage:
    uv run python scripts/test_fork_agent_switch.py
"""

from __future__ import annotations

import asyncio
import os

from acp import text_block
from acp.client.connection import ClientSideConnection
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    FileSystemCapabilities,
    Implementation,
    PermissionOption,
    RequestPermissionResponse,
    ToolCallUpdate,
    UsageUpdate,
)
from acp.stdio import spawn_agent_process


class SimpleClient:
    """Minimal ACP client that auto-approves permissions and collects text."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self._text_parts: list[str] = []

    def on_connect(self, conn: ClientSideConnection) -> None:
        pass

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        if isinstance(update, UsageUpdate):
            return
        if isinstance(update, AgentMessageChunk):
            if hasattr(update, "content") and hasattr(update.content, "text"):
                self._text_parts.append(update.content.text)

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: object,
    ) -> RequestPermissionResponse:
        # Auto-approve: pick the first "allow" option
        for opt in options:
            if opt.kind.value in ("allow", "allow_always"):
                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", option_id=opt.option_id)
                )
        # Fallback: pick first option
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=options[0].option_id)
        )

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: object):
        return None

    async def read_text_file(self, path: str, session_id: str, **kwargs: object):
        return None

    def get_text_and_reset(self) -> str:
        text = "".join(self._text_parts)
        self._text_parts.clear()
        return text


async def prompt_and_collect(
    conn: ClientSideConnection, client: SimpleClient, session_id: str, text: str
) -> str:
    """Send a prompt and return the collected text response."""
    client._text_parts.clear()
    print(f"\n{'='*60}")
    print(f"PROMPT: {text}")
    print(f"{'='*60}")
    response = await conn.prompt(prompt=[text_block(text)], session_id=session_id)
    result = client.get_text_and_reset()
    print(f"\nRESPONSE ({response.stop_reason}):")
    print(result[:2000] if result else "(no text collected)")
    return result


async def main() -> None:
    cmd = "npx"
    args = ["@agentclientprotocol/claude-agent-acp"]
    cwd = os.getcwd()

    # Resolve claude binary — the adaptor needs this to authenticate
    import shutil
    claude_bin = os.environ.get("CLAUDE_CODE_EXECUTABLE") or shutil.which("claude")
    if not claude_bin:
        print("ERROR: 'claude' binary not found. Set CLAUDE_CODE_EXECUTABLE.")
        return
    env = {**os.environ, "CLAUDE_CODE_EXECUTABLE": claude_bin}

    client = SimpleClient()

    print("Starting Claude Code ACP session (no custom agent)...")
    print(f"CWD: {cwd}")
    print(f"CLAUDE_CODE_EXECUTABLE: {claude_bin}")

    async with spawn_agent_process(client, cmd, *args, cwd=cwd, env=env) as (conn, proc):
        # Initialize
        init = await conn.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=True,
            ),
            client_info=Implementation(name="synth-test", version="0.1.0"),
        )
        print(f"Initialized. Fork supported: {init.agent_capabilities.session_capabilities.fork is not None}")

        # Create session with NO custom agent
        session = await conn.new_session(cwd=cwd)
        session_id = session.session_id
        client.session_id = session_id
        print(f"Session created: {session_id}")

        # Step 1: Ask the agent what its role is and give it a secret word
        await prompt_and_collect(
            conn, client, session_id,
            "The secret word is 'BANANA'. Remember it. Briefly: what is your role? Are you a custom agent or default Claude? Answer in 2-3 sentences max."
        )

        # Step 2: Fork the session with a different agent
        print(f"\n{'#'*60}")
        print("FORKING SESSION with agent: local-SHScienceAgentKit-all:code-planner")
        print(f"{'#'*60}")

        fork_response = await conn.fork_session(
            session_id=session_id,
            cwd=cwd,
            claudeCode={"options": {"agent": "local-SHScienceAgentKit-all:code-planner"}},
        )
        new_session_id = fork_response.session_id
        client.session_id = new_session_id
        print(f"Fork successful! New session ID: {new_session_id}")
        if fork_response.config_options:
            print(f"Config options: {[o.id for o in fork_response.config_options]}")

        # Step 3: Ask the forked session about its role and the secret word
        await prompt_and_collect(
            conn, client, new_session_id,
            "Two questions: 1) What is your role/identity — are you a custom agent with a specific system prompt? 2) What was the secret word I told you earlier? Answer briefly."
        )

        print(f"\n{'#'*60}")
        print("TEST COMPLETE")
        print(f"{'#'*60}")
        print(f"\nOriginal session: {session_id}")
        print(f"Forked session:   {new_session_id}")
        print("\nIf the forked session knows 'BANANA' and identifies as code-planner,")
        print("then fork-based agent switching works with history continuity.")


if __name__ == "__main__":
    asyncio.run(main())
