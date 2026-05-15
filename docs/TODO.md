---
title: "SoHoAI — Future work and deferred tasks"
created_at: 2026-05-04--17-30
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-15--19-54
context: >
  Tracks deferred implementation work that is understood, scoped, and intentionally
  left for a future session. Each entry includes the motivation, the known approach,
  estimated effort, and the blocker or reason for deferral.
---

# SoHoAI — Future work

---

## Usage telemetry — Stage 2 (future claude-orchestra session)

SoHoAI Stage 1 telemetry pipeline is live (`usage_events` table, `GET /v1/usage/stats`).
See `docs/Telemetry.md` for the full Stage 2 spec.

Required work in a future claude-orchestra branch:
- Inject `X-Orchestra-Session-ID` header on every `/v1/messages` call during `/brain`/`/duo` sessions
- Migrate T2 cost calculation from `pricing.yaml` to `litellm.completion_cost()`
- After 4+ weeks of SoHoAI data: query `GET /v1/usage/stats?session_id=` as primary cost source (fall back to JSONL parsing)
- Update `telemetry-report.sh` to annotate SoHoAI-sourced vs JSONL-parsed sessions

---

## Phase 3 — MCP integration + Web UI + Auth (in progress)

**Google OAuth2 (OIDC) authentication middleware** — family members authenticate with separate Google accounts within the same Google Family Group. User identity from JWT is mapped to `owner` via `config.yaml` `users:` section; Qdrant search filtered by ownership.

**MCP gateway in orchestrator** — `mcp_gateway.py` stub ready. Plan: delegate tool calls to specialized MCP servers (filesystem, web search, calendar, weather).

**Initial MCP tool servers**:
- Filesystem (done): `nfs_files_mcp_server.py` exposes Gin-AI filesystem
- Web search (planned)
- Calendar (planned)

**Web frontend**: custom FastAPI + HTMX/React with server-managed history (chat_id, Redis, KV cache). Supports multi-user concurrent sessions without state collision (each user's Redis key is independent).

**OpenAI-compatible response format** for Open WebUI integration. Currently using custom `ChatResponse` model with `chat_id`, `model_used`, `message`, `rag_sources`, `rag_mode_used` fields.

**Offline resilience**: locally cached session tokens with multi-hour TTL; CLI local API key fallback for when Anthropic API is unreachable.

---

## Phase 4 — Image search + RL (future)

**CLIP model** (openai/clip-vit-base-patch32) on Server 2 GPU. Family photo ingestion → CLIP embeddings → separate Qdrant `images` collection (same Qdrant instance as `documents`). Text-to-image similarity search for family photo library.

**RL training data pipeline**: export conversations with feedback signals (thumbs up/down on turns) as DPO-format JSONL for training with TRL framework. Goal: fine-tune a smaller local model (e.g. Qwen3.5-4B or similar) on real family conversations to reduce API cost over time.

---

## [2026-05-04] Tool-use support on the LiteLLM path (`/v1/messages` → LiteLLM)

**Status: IMPLEMENTED 2026-05-10** — see smoke harness `utils/tool_use_smoke_test.py`.

### Background

`/v1/messages` has two routing paths (see `docs/Model-routing.md §2`):

1. **Transparent forward** (`anthropic/*` models): the full request body is forwarded
   byte-for-byte to `api.anthropic.com`. Tools, `tool_use`, `tool_result`, and
   `cache_control` are preserved. Full Claude Code tool loop works.

2. **LiteLLM conversion** (`internal/*` and `ollama-cloud/*` models): the request
   is converted Anthropic→OpenAI format before routing through LiteLLM.
   The conversion now preserves `tools`, `tool_use`, `tool_result`, and `cache_control` markers.

   Affected models (all five):
   - `internal/qwen3-4b` → llama-server (Server 2)
   - `ollama-cloud/deepseek-v4-pro`, `ollama-cloud/kimi-k2.6`, `ollama-cloud/glm-5.1`,
     `ollama-cloud/qwen3-coder-next` → Ollama cloud (`https://ollama.com/v1`)

### Implementation note

Tool-use support was implemented in `main.py` via four helper functions: `_convert_tools()` (wraps
Anthropic tools in OpenAI format), `_convert_assistant_message()` (splits text and tool_use blocks),
`_convert_user_message()` (handles tool_result and image blocks), and extensions to the streaming
SSE emitter (`anthropic_event_stream()`) to emit `content_block_start/delta/stop` events for tool
calls. The non-streaming path also extracts `tool_calls` from the LiteLLM response and returns
Anthropic-format `tool_use` content blocks. Streaming responses emit proper Anthropic SSE events
including `message_delta` with `stop_reason: "tool_use"` on tool-calling responses.

### What was previously stripped and why it matters (now forwarded)

| Field | Effect on the local-model sub-agent | Status |
|---|---|---|
| `tools` array | Model has knowledge of available tools — can call Read/Write/Bash | ✅ Forwarded |
| `tool_use` blocks in assistant messages | Model sees its own previous tool-call requests from conversation history | ✅ Forwarded |
| `tool_result` blocks in user messages | Model sees file contents and command outputs returned by tool calls | ✅ Forwarded |
| `cache_control` markers | No Anthropic-side prompt caching (irrelevant for local model, but noted) | ✅ Forwarded |

**Important**: LiteLLM itself fully supports tool use. The stripping was entirely in
*our* Anthropic→OpenAI conversion code, not in LiteLLM. LiteLLM correctly translates
OpenAI `tool_calls` to the provider's native format.

### Implementation detail (~70 lines total in `_anthropic_messages_litellm()`)

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

Emits Anthropic SSE events:
```
event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_abc","name":"Read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"file_path\":"}}
```

~20 lines in the streaming loop.

### Per-model tool-call reliability status

Once the conversion was implemented, reliability per model is now being validated:

- **`internal/qwen3-4b`**: llama-server exposes the OpenAI tools API; Qwen3.5-4B
  has instruction-following capability but complex multi-tool JSON output reliability is
  **informational-only in this session**. May need grammar-constrained generation (`--grammar`)
  or prompt engineering if unreliable in practice. See deferred entry (a) below.
- **`ollama-cloud/deepseek-v4-pro`** and **`ollama-cloud/qwen3-coder-next`**: both
  support OpenAI function calling natively and are expected to be reliable for tool use.
- **`ollama-cloud/kimi-k2.6`**: gating target in `utils/tool_use_smoke_test.py`; **PASS (streaming, 2026-05-10)**.
- **`ollama-cloud/glm-5.1`**: gating target in `utils/tool_use_smoke_test.py`; **PASS (streaming, 2026-05-10)**.

---

## Tool-use deferred: internal/qwen3-4b broader reliability verification

**Synthetic smoke status (2026-05-10):** PASS on both streaming and non-streaming legs of `utils/tool_use_smoke_test.py` with Gemma 4 E4B. Qwen3.5-4B replaced Gemma 4 E4B on 2026-05-15 — tool-use validation carried over to the new model. Initial validation needed with Qwen3.5-4B to match Gemma's baseline.

### Background / rationale

The synthetic smoke harness exercises a single, simple tool with one string argument. Real Claude Code subagent traffic is more demanding: multi-tool catalogues (Read/Write/Bash/Glob/Grep/Edit/TodoWrite/Agent), parallel tool calls, deeply-nested JSON arguments, multi-turn tool chains, and 10K+ token system prompts. Qwen3.5-4B may degrade on those workloads even though a predecessor passed smoke. Without broader validation we cannot recommend `internal/qwen3-4b` as an Actor-tier subagent in claude-orchestra.

### Goal

Validate Qwen3.5's tool-call reliability on representative claude-orchestra workloads (full Claude Code tool catalogue, multi-turn, parallel calls) before recommending it for any Actor-tier or research-tier subagent. If reliability holds, fold Qwen into the recommended-models list in `docs/Model-routing.md §3`. If it degrades, scope grammar-constrained generation (Step c) precisely against the failure modes observed.

### What needs to be done

1. Configure a non-trivial claude-orchestra subagent (e.g. a research-tier explore agent) to use `model: internal/qwen3-4b` in its frontmatter. Run a full `/duo` or `/brain` session against a real (not synthetic) task that requires Read/Glob/Grep/Bash.
2. Capture: (a) tool-call success rate, (b) any malformed `arguments` JSON observed, (c) latency per turn, (d) whether the model hits context window limits faster than expected.
3. Re-run with a multi-tool prompt that should provoke parallel tool calls. Record whether Qwen serializes (good) or attempts and fails parallelism (bad).
4. If reliability ≥ 95 % across (1)-(3): update `docs/Model-routing.md §3` to drop the "unvalidated" qualifier, add Qwen to the recommended-Actor list with cost ≈ $0, and mark this entry complete. If reliability < 95 %: enumerate the failure classes observed and use them to scope Step c (grammar-constrained generation) precisely instead of speculatively.

---

## [2026-05-10] Tool-use deferred: full Claude Code tool catalogue + parallel tool calls

**Status: PARTIALLY DONE (2026-05-10)** — Parallel tool calls validated. Real orchestra session still open.

### What is done (2026-05-10)

**Parallel tool calls — DONE.** `utils/tool_use_smoke_test.py --parallel` added and run successfully with 3 simultaneous tools (`get_file_size`, `get_file_owner`, `get_file_permissions`). Results:
- All 4 ollama-cloud models (`qwen3-coder-next`, `deepseek-v4-pro`, `kimi-k2.6`, `glm-5.1`): **PASS** on both streaming and non-streaming legs.
- `internal/qwen3-4b`: **INFO FAIL** parallel (2026-05-15) — single-tool PASS (both stream/no-stream); parallel test gets 1 tool_use block instead of 5. Expected 4B-class model limitation, not a proxy bug.

### Remaining open item

**Real claude-orchestra Actor session** — A synthetic harness cannot replicate actual Claude Code sub-agent traffic: the full tool catalogue (Read/Write/Edit/Bash/Glob/Grep/TodoWrite/Agent), deeply-nested JSON arguments, multi-turn tool chains, and 10K+ token system prompts. No real `/duo` or `/brain` session has been run with the local model (Qwen3.5-4B) or any ollama-cloud model as Actor.

### Goal

Run at least one real `/duo` or `/brain` session with an ollama-cloud model (e.g. `qwen3-coder-next`) configured as Actor against a non-trivial task requiring Read/Glob/Grep/Bash. Validate tool-call success rate and argument correctness under real orchestra load.

### What needs to be done

1. Configure a claude-orchestra sub-agent (actor or research-explore) with `model: internal/qwen3-4b` in its frontmatter. Run a full `/duo` or `/brain` session against a real task that requires Read/Glob/Grep/Bash (e.g. a small refactor or codebase exploration task).
2. Record: (a) tool-call success rate, (b) any malformed `arguments` JSON, (c) latency per turn, (d) whether multi-turn tool chains complete correctly.
3. If reliability ≥ 95 %: update `docs/Model-routing.md §4.3` to remove the "remaining open item" note and mark this entry complete. If reliability < 95 %: document failure classes and open a targeted follow-up.

---

## Tool-use deferred: image-block conversion live validation

### Background / rationale

`_convert_user_message()` includes conversion logic for Anthropic `image` content blocks:
- `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<b64>"}}` → `{"type":"image_url","image_url":{"url":"data:image/png;base64,<b64>"}}`
- `{"type":"image","source":{"type":"url","url":"<url>"}}` → `{"type":"image_url","image_url":{"url":"<url>"}}`

However, no vision-capable model is available in the current stack:
- `internal/qwen3-4b`: text-only, no mmproj or vision encoder
- `ollama-cloud/*`: currently text-only SKUs; Ollama may offer vision SKUs in the future
- `anthropic/claude-*`: not on the LiteLLM path, so not affected by this conversion

Image blocks in user messages would be forwarded to a non-vision model, which would either fail or silently ignore them.

### Goal

Validate that image-block conversion and forwarding work correctly when a vision-capable model becomes available.

### What needs to be done

1. Provision Qwen3.5 or similar with multimodal support (e.g. llama.cpp with a compatible CLIP/SigLIP mmproj and Server 2 GPU)
   or wait for an Ollama cloud vision SKU.
2. Add a smoke leg to `tool_use_smoke_test.py` that sends a small test image (e.g. a 32×32 PNG with a recognizable pattern)
   and a prompt asking the model to describe its contents.
3. Assert that the response text acknowledges the image content (e.g. contains "image" or a description of the pattern).
4. Merge the vision-capable model config and re-run smoke; if image-block conversion passes, mark this entry complete.

---

## Tool-use deferred: grammar-constrained generation fallback for Qwen3.5

**Status (2026-05-15):** **likely unnecessary**, gated on the broader reliability check above. Predecessor (Gemma 4 E4B) passed synthetic smoke without any grammar constraint. Only revisit if Qwen3.5's broader workload validation (Step 13 in this section above) shows malformed `arguments` JSON or wrong tool selection. Keeping the entry for traceability.

### Background / rationale

If Qwen3.5's broader validation shows malformed tool-call JSON on real workloads, grammar-constrained generation is a proven technique to enforce valid JSON output. LiteLLM and llama.cpp both support the `--grammar` parameter (GBNF format). This would be a targeted fallback: only applied to `internal/qwen3-4b` when the request includes `tools`.

Implementation cost: ~15–20 lines in the `kv_cache.py` native llama-server path to inject a grammar parameter into the `/completion` request; ~10 lines to define or fetch the JSON grammar spec.

### Goal

ONLY if Qwen3.5's real-workload validation shows failures, implement grammar-constrained generation to bring it to ≥ 95 % tool-call reliability. Skip this work entirely if validation passes.

### What needs to be done

1. Confirm from the broader workload validation above that Qwen3.5 reliability is < 95 % (synthetic-smoke pass alone is NOT a trigger).
2. Obtain or generate a GBNF grammar for the OpenAI function-calling JSON schema (tools + arguments structure).
3. Modify `kv_cache.py` `_call_native_llama()` to inject `grammar: <spec>` into the llama-server `/completion` request when `model.startswith("internal/")` and the request includes `tools`.
4. Re-run the broader workload validation; assert Qwen3.5 now produces valid JSON on all tool calls.
5. Measure inference latency impact (grammar can add 5–20% overhead); document findings.

---

## Tool-use deferred: disable_parallel_tool_use semantic parity

### Background / rationale

The Anthropic API accepts `disable_parallel_tool_use: true` to force tool calls to be serialized (one at a time)
instead of parallelized (multiple tool calls in one turn). This is forwarded to LiteLLM as `parallel_tool_calls: false`.

However, not all providers enforce this semantic. For example, a provider might silently ignore the `parallel_tool_calls: false`
directive and emit tool calls in parallel anyway. Currently there is no validation that the provider actually respects
the serialization request.

### Goal

Verify that each provider (Ollama cloud and llama-server) respects `parallel_tool_calls: false` when forwarded.

### What needs to be done

1. Create a test prompt that requires multiple independent tool calls (e.g. "Read file A, then read file B, then read file C").
2. Send the request with `parallel_tool_calls: false` to each target model.
3. Assert that the response contains a single tool_use block (or a sequence of separate turns), not multiple tool_use blocks in one response.
4. Document the per-model behavior: does the model serialize or parallelize?
5. If a provider silently ignores the directive, escalate to a follow-up task to decide: accept the limitation, add a workaround
   (e.g. inject constraint in system prompt), or deprecate that model for parallel-sensitive workloads.

---

## Tool-use deferred: is_error semantic parity across providers

### Background / rationale

The Anthropic API accepts `is_error: true` in `tool_result` content blocks to indicate that a tool call failed. The conversion
adds an `[ERROR] ` prefix to the content string (a best-effort marker that some models may recognize). However, the standard
OpenAI API does not have a native `is_error` field; instead, errors are typically indicated by error content or by the caller
injecting `[ERROR]` prefix into the text.

Different providers may respond differently to this `[ERROR]` prefix. For example:
- Ollama cloud models may recognize the prefix and adjust their error recovery behavior.
- llama-server (Qwen3.5) may not have special handling and just treats it as literal text.

Currently the behavior is undocumented.

### Goal

Validate that the `[ERROR]` prefix is handled sensibly by each provider and document any model-specific quirks.

### What needs to be done

1. Craft a two-turn scenario: Turn 1 calls a hypothetical tool; Turn 2 returns an error result marked with `is_error: true`
   (which becomes `[ERROR] tool failed` in the content string).
2. Send to each target model (`ollama-cloud/qwen3-coder-next`, `ollama-cloud/deepseek-v4-pro`, `internal/qwen3-4b`).
3. Observe and document: does the model recognize the error flag? Does it retry the tool call or adjust its response?
4. If a model completely ignores error signals, consider injecting a more verbose error message in the system prompt
   or as a separate user message context.

---

## Tool-use deferred: extract conversion helpers to a dedicated module

### Background / rationale

The four helper functions (`_convert_tools()`, `_convert_tool_choice()`, `_convert_assistant_message()`, `_convert_user_message()`)
are currently defined in `main.py` above `_anthropic_messages_litellm()`. Today they are only called by that one function.
However, as the codebase grows, a second consumer may appear (e.g. a future `/proxy/v1/messages` endpoint for Cline that
also needs Anthropic→OpenAI conversion).

Extracting to a dedicated module `anthropic_openai_convert.py` at the top level would improve code organization and
re-usability. However, until a second consumer exists, the extraction is premature (YAGNI principle).

### Goal

When a second consumer of the Anthropic→OpenAI conversion appears, extract the four helpers and their unit tests
to `anthropic_openai_convert.py` for re-use.

### What needs to be done

1. Monitor the codebase for use cases that need Anthropic→OpenAI conversion outside `_anthropic_messages_litellm()`.
2. When a second consumer appears: create `anthropic_openai_convert.py` with the four helper functions and comprehensive
   unit tests (cover edge cases: empty tools, mixed content types, image blocks, tool errors, etc.).
3. Update `main.py` to import from the new module instead of defining locally.
4. Run the existing smoke harness to ensure no regression.
5. Trigger: when Cline or another integration needs to use the same conversion logic.
