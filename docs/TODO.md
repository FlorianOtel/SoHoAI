---
title: "SoHoAI — Future work and deferred tasks"
created_at: 2026-05-04--17-30
created_by: Claude Code (Claude Sonnet 4.6)
context: >
  Tracks deferred implementation work that is understood, scoped, and intentionally
  left for a future session. Each entry includes the motivation, the known approach,
  estimated effort, and the blocker or reason for deferral.
---

# SoHoAI — Future work

---

## [2026-05-04] Tool-use support on the local-model path (`/v1/messages` → LiteLLM)

**Status**: Deferred — understood and scoped, not yet implemented.

### Background

`/v1/messages` has two routing paths (see `docs/proxy-functionality.md §2`):

1. **Transparent forward** (Anthropic models): the full request body is forwarded
   byte-for-byte to `api.anthropic.com`. Tools, `tool_use`, `tool_result`, and
   `cache_control` are preserved. Full Claude Code tool loop works.

2. **LiteLLM conversion** (local models, currently only `gemma-4-e4b`): the request
   is converted Anthropic→OpenAI format before routing through LiteLLM to llama-server.
   **Current limitation**: our conversion code in `_anthropic_messages_litellm()` only
   preserves `text` content blocks. Everything else is silently dropped.

### What is stripped and why it matters

| Dropped field | Effect on the local-model sub-agent |
|---|---|
| `tools` array | Model has no knowledge of available tools — cannot call Read/Write/Bash |
| `tool_use` blocks in assistant messages | Model loses its own previous tool-call requests from conversation history |
| `tool_result` blocks in user messages | Model loses file contents and command outputs returned by tool calls |
| `cache_control` markers | No Anthropic-side prompt caching (irrelevant for local model, but noted) |

**Important**: LiteLLM itself fully supports tool use. The stripping is entirely in
*our* Anthropic→OpenAI conversion code, not in LiteLLM. LiteLLM correctly translates
OpenAI `tool_calls` to the provider's native format.

### Required implementation (~70 lines total in `_anthropic_messages_litellm()`)

**1. Convert `tools` array (Anthropic → OpenAI format)**

Anthropic format:
```json
[{
  "name": "Read",
  "description": "Read a file",
  "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
}]
```

OpenAI format:
```json
[{
  "type": "function",
  "function": {
    "name": "Read",
    "description": "Read a file",
    "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}
  }
}]
```

Conversion: rename `input_schema` → `parameters`, wrap in `{"type": "function", "function": {...}}`. ~10 lines.

**2. Convert `tool_use` blocks in assistant messages (Anthropic → OpenAI)**

Anthropic assistant message:
```json
{"role": "assistant", "content": [
  {"type": "text", "text": "I will read the file."},
  {"type": "tool_use", "id": "toolu_abc", "name": "Read", "input": {"file_path": "main.py"}}
]}
```

OpenAI assistant message:
```json
{"role": "assistant", "content": "I will read the file.",
 "tool_calls": [{"id": "toolu_abc", "type": "function",
                 "function": {"name": "Read", "arguments": "{\"file_path\": \"main.py\"}"}}]}
```

Conversion: separate text and tool_use blocks; serialize `input` dict → JSON string for `arguments`. ~20 lines.

**3. Convert `tool_result` blocks in user messages (Anthropic → OpenAI)**

Anthropic user message:
```json
{"role": "user", "content": [
  {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "def main(): ..."}
]}
```

OpenAI tool message:
```json
{"role": "tool", "tool_call_id": "toolu_abc", "content": "def main(): ..."}
```

Conversion: change role to `"tool"`, rename `tool_use_id` → `tool_call_id`, extract content string.
Handle mixed user messages (text + tool_result) by splitting into separate messages. ~20 lines.

**4. Convert streaming response for tool use (OpenAI → Anthropic SSE)**

LiteLLM returns tool use in streaming as OpenAI `tool_calls` deltas:
```
choices[0].delta.tool_calls[0].function.arguments = '{"file'
```

Need to emit Anthropic SSE events instead of `text_delta`:
```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"Read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":"}}
```

~20 lines in the streaming loop.

### Unknown: Gemma 4 tool-call reliability

llama-server exposes the OpenAI-compatible tools API and Gemma 4 E4B has
instruction-following capability, but tool-call reliability at Q8_0 with complex
multi-tool conversations is untested. May require:
- Grammar-constrained generation (`--grammar` in llama-server) for reliable JSON output
- Prompt engineering / system-prompt injection to improve tool adherence
- Evaluation harness to measure tool-call success rate

### When to implement

Implement when a concrete use case requires a local-model sub-agent to use file-access
or bash tools (e.g. an Actor that reads and modifies files without paying Haiku API cost).
Until then, use `claude-haiku-4-5` (transparent forward, full tool support, ~$0.01/session).
