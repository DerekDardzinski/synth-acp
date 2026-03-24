# ACP Python SDK Reference

Source: `github.com/agentclientprotocol/python-sdk`
PyPI: `agent-client-protocol`

## Key Imports

```python
# Spawn and manage an agent subprocess
from acp import spawn_agent_process

# Content block helpers
from acp import text_block, start_tool_call, update_tool_call, tool_content

# Client interface (implement this to receive agent callbacks)
from acp.interfaces import Client

# Schema models (Pydantic v2)
from acp.schema import (
    McpServerStdio, EnvVariable, ClientCapabilities, FileSystemCapabilities,
    PermissionOption, RequestPermissionResponse, AllowedOutcome, DeniedOutcome,
    ReadTextFileResponse, WriteTextFileResponse, CreateTerminalResponse,
    TerminalOutputResponse, WaitForTerminalExitResponse,
    TextContentBlock, PromptResponse,
)

# Contrib helpers
from acp.contrib.session_state import SessionAccumulator
from acp.contrib.tool_calls import ToolCallTracker
from acp.contrib.permissions import PermissionBroker
```

## Client Interface

The `Client` protocol defines callbacks the SDK invokes when the agent sends
requests/notifications:

```python
class Client(Protocol):
    async def session_update(self, session_id, update, **kwargs) -> None: ...
    async def request_permission(self, options, session_id, tool_call, **kwargs) -> RequestPermissionResponse: ...
    async def read_text_file(self, path, session_id, **kwargs) -> ReadTextFileResponse: ...
    async def write_text_file(self, content, path, session_id, **kwargs) -> WriteTextFileResponse | None: ...
    async def create_terminal(self, command, session_id, **kwargs) -> CreateTerminalResponse: ...
    async def terminal_output(self, session_id, terminal_id, **kwargs) -> TerminalOutputResponse: ...
    async def release_terminal(self, session_id, terminal_id, **kwargs) -> ReleaseTerminalResponse | None: ...
    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs) -> WaitForTerminalExitResponse: ...
    async def kill_terminal(self, session_id, terminal_id, **kwargs) -> KillTerminalResponse | None: ...
    def on_connect(self, conn: Agent) -> None: ...
```

SYNTH implements `session_update` and `request_permission`. The fs/terminal
methods are not implemented (capabilities declared as False).

## spawn_agent_process

```python
async with spawn_agent_process(client, binary, *args, cwd=cwd) as (conn, proc):
    # conn is the Agent interface (for sending requests TO the agent)
    # proc is asyncio.subprocess.Process
    await conn.initialize(protocol_version=1, client_capabilities=...)
    session = await conn.new_session(cwd=cwd, mcp_servers=[...])
    await conn.prompt(session_id=session.session_id, prompt=[text_block("Hello")])
    # session_update() and request_permission() are called as callbacks
```

## SessionAccumulator (contrib)

```python
accumulator = SessionAccumulator()
accumulator.apply(notification)  # feed every SessionNotification
snapshot = accumulator.snapshot()  # immutable SessionSnapshot

# snapshot contains:
# - snapshot.tool_calls: merged tool call state
# - snapshot.user_messages: ordered message history
# - snapshot.plan: current plan entries
```

Handles out-of-order tool_call_update (arriving before tool_call_start),
automatic reset on session change, and deep-copy safety.

## Agent Interface (for sending requests TO the agent)

```python
class Agent(Protocol):
    async def initialize(self, protocol_version, client_capabilities=None, client_info=None, **kwargs) -> InitializeResponse: ...
    async def new_session(self, cwd, mcp_servers=None, **kwargs) -> NewSessionResponse: ...
    async def load_session(self, cwd, session_id, mcp_servers=None, **kwargs) -> LoadSessionResponse | None: ...
    async def prompt(self, prompt, session_id, **kwargs) -> PromptResponse: ...
    async def cancel(self, session_id, **kwargs) -> None: ...
```
