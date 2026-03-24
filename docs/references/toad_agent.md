# Toad ACP Agent Implementation Reference

Source: `github.com/batrachianai/toad` — `src/toad/acp/agent.py`

Toad is a single-agent ACP TUI. This file is the most complete reference for
"how to be an ACP client in Python." SYNTH's `ACPSession` serves the same role
but for multiple agents, using the `agent-client-protocol` SDK instead of Toad's
hand-rolled JSON-RPC.

## Key Patterns

### Subprocess Management

Toad spawns the agent as a subprocess with stdin/stdout piped, then reads
JSON-RPC lines from stdout in a loop:

```python
process = await asyncio.create_subprocess_shell(
    command, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=env, cwd=cwd,
    limit=10 * 1024 * 1024,
)
while line := await process.stdout.readline():
    agent_data = json.loads(line)
    # dispatch as JSON-RPC request or response
```

SYNTH uses `spawn_agent_process()` from the SDK instead, which wraps this.

### ACP Handshake

```python
async def run(self):
    await self.acp_initialize()       # initialize with capabilities
    if self.session_id is None:
        await self.acp_new_session()  # create new session
    else:
        await self.acp_load_session() # resume existing session
```

Initialize declares client capabilities:
```python
client_capabilities = {
    "fs": {"readTextFile": True, "writeTextFile": True},
    "terminal": True,
}
```

SYNTH declares fs and terminal as False since target harnesses use their own tools.

### Session Update Handling

Toad uses match/case on the `sessionUpdate` discriminator field:

```python
match update:
    case {"sessionUpdate": "agent_message_chunk", "content": {"type": type, "text": text}}:
        self.post_message(messages.Update(type, text))
    case {"sessionUpdate": "tool_call", "toolCallId": tool_call_id}:
        self.tool_calls[tool_call_id] = update
        self.post_message(messages.ToolCall(update))
    case {"sessionUpdate": "tool_call_update", "toolCallId": tool_call_id}:
        # Merge update into existing tool call
        # Handle edge case: update arrives before start
```

SYNTH uses `SessionAccumulator.apply()` from the SDK instead of manual tracking.

### Permission Handling

Toad creates a Future, bundles it into a Textual message, posts it, and awaits:

```python
async def rpc_request_permission(self, sessionId, options, toolCall, _meta=None):
    result_future: asyncio.Future[Answer] = asyncio.Future()
    tool_call = deepcopy(self.tool_calls.get(tool_call_id, toolCall))
    message = messages.RequestPermission(options, tool_call, result_future)
    self.post_message(message)
    await result_future
    return {"outcome": {"optionId": result_future.result().id, "outcome": "selected"}}
```

SYNTH uses the same Future pattern but routes through the broker (for auto-resolve)
before reaching the UI.

### Filesystem Methods

Toad implements these as simple file I/O scoped to project root:

```python
def rpc_read_text_file(self, sessionId, path, line=None, limit=None):
    read_path = self.project_root_path / path
    text = read_path.read_text(encoding="utf-8", errors="ignore")
    return {"content": text}

def rpc_write_text_file(self, sessionId, path, content):
    write_path = self.project_root_path / path
    write_path.write_text(content, encoding="utf-8", errors="ignore")
```

SYNTH does not implement these (capabilities declared as False).

### Terminal Methods

Toad implements full terminal lifecycle: create, output, kill, wait_for_exit, release.
Each terminal is a subprocess tracked by ID. SYNTH does not implement these.

### Process Exit Handling

After the stdout read loop ends:
```python
if process.returncode:
    fail_details = (await process.stderr.read()).decode("utf-8", "replace")
    self.post_message(AgentFail(f"Agent returned failure code: {process.returncode}",
                                 details=fail_details))
```

SYNTH handles this in the `finally` block of `ACPSession.run()`.
