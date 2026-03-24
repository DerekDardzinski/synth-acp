# ACP Protocol Types Reference

Source: `github.com/batrachianai/toad` — `src/toad/acp/protocol.py`
and `agent-client-protocol` SDK — `src/acp/schema.py`

This documents the key ACP types relevant to SYNTH. The SDK provides these as
Pydantic models in `acp.schema`; Toad defines them as TypedDicts.

## Session Updates (agent → client notifications)

All session updates have a `sessionUpdate` discriminator field.

| Type | Discriminator | Key Fields |
|---|---|---|
| `UserMessageChunk` | `"user_message_chunk"` | `content: ContentBlock` |
| `AgentMessageChunk` | `"agent_message_chunk"` | `content: ContentBlock` |
| `AgentThoughtChunk` | `"agent_thought_chunk"` | `content: ContentBlock` |
| `ToolCallStart` | `"tool_call"` | `toolCallId, title, kind, status, content, locations, rawInput` |
| `ToolCallProgress` / `ToolCallUpdate` | `"tool_call_update"` | `toolCallId, title?, kind?, status?, content?` |
| `Plan` | `"plan"` | `entries: list[PlanEntry]` |
| `AvailableCommandsUpdate` | `"available_commands_update"` | `availableCommands: list[AvailableCommand]` |
| `CurrentModeUpdate` | `"current_mode_update"` | `currentModeId: str` |

## Content Blocks

```
ContentBlock = TextContent | ImageContent | AudioContent
             | EmbeddedResourceContent | ResourceLinkContent
```

Most common: `TextContent` with `type="text"` and `text: str`.

## Tool Call Content

Tool calls carry specialized content types:

| Type | Discriminator | Purpose |
|---|---|---|
| `ToolCallContentContent` | `type="content"` | Generic content block |
| `ToolCallContentDiff` | `type="diff"` | File diff: `path, oldText, newText` |
| `ToolCallContentTerminal` | `type="terminal"` | Terminal reference: `terminalId` |

## Tool Kinds

```
ToolKind = "read" | "edit" | "delete" | "move" | "search"
         | "execute" | "think" | "fetch" | "switch_mode" | "other"
```

## Tool Call Status

```
ToolCallStatus = "pending" | "in_progress" | "completed" | "failed"
```

## Permission Types

```python
class PermissionOption:
    kind: "allow_once" | "allow_always" | "reject_once" | "reject_always"
    name: str           # human-readable label
    optionId: str       # opaque ID to return in response

class RequestPermissionResponse:
    outcome: OutcomeSelected | OutcomeCancelled

class OutcomeSelected:
    outcome: "selected"
    optionId: str

class OutcomeCancelled:
    outcome: "cancelled"
```

## MCP Server Config (for session/new)

```python
class McpServerStdio:
    name: str              # human-readable name
    command: str           # executable path
    args: list[str]        # command-line arguments
    env: list[EnvVariable] # NOT a dict

class EnvVariable:
    name: str
    value: str
```

## Client Capabilities (declared in initialize)

```python
class ClientCapabilities:
    fs: FileSystemCapabilities | None
    terminal: bool | None

class FileSystemCapabilities:
    read_text_file: bool | None   # alias: readTextFile
    write_text_file: bool | None  # alias: writeTextFile
```

## Session Lifecycle Methods

| Method | Direction | Purpose |
|---|---|---|
| `initialize` | client → agent | Handshake, declare capabilities |
| `session/new` | client → agent | Create new session with CWD and MCP servers |
| `session/load` | client → agent | Resume existing session by ID |
| `session/prompt` | client → agent | Send user message, starts a turn |
| `session/cancel` | client → agent | Cancel active turn |
| `session/update` | agent → client | Stream response chunks, tool calls, plans |
| `session/request_permission` | agent → client | Request approval for sensitive operation |

## Prompt Response

```python
class PromptResponse:
    stopReason: "end_turn" | "max_tokens" | "max_turn_requests" | "refusal" | "cancelled"
```
