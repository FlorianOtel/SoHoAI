---
title: "SoHoAI Model routing — Cline and Claude Code integration"
created_at: 2026-05-04--14-50
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-15--11-49
context: >
  SoHoAI exposes two stateless pass-through paths built on the same LiteLLM Router.
  One is OpenAI-compatible for Cline VSCode plugin. The other is Anthropic-compatible
  for Claude Code (ANTHROPIC_BASE_URL), with model-aware routing: Anthropic models use
  a transparent HTTP forward (full fidelity including tools and caching); local models
  (gemma-4-e4b) and Ollama cloud models use the LiteLLM conversion path with tool-use
  support (implementation complete 2026-05-10). Synthetic smoke harness
  (utils/tool_use_smoke_test.py) passed end-to-end for all 3 target models including
  Gemma 4 E4B on both streaming and non-streaming. New in this update (2026-05-10):
  gateway model discovery with claude-code-* alias scheme for claude-orchestra integration.
  See docs/claude-orchestra-handoff.md for deployment runbook and tier recommendations.
  Update 2026-05-11: ollama-cloud/* 503/timeout guard — deepseek-v4-pro has a documented
  ~70% failure rate on Ollama shared inference. Silent LiteLLM fallback to another model
  was rejected (causes per-request model oscillation mid-agent-task). Instead: 30s
  request_timeout for ollama-cloud/* and HTTP 529 overloaded_error on failure so Claude
  Code surfaces a clean error. --parallel smoke test extended to 5 simultaneous tools.
---

# SoHoAI model routing — Cline and Claude Code integration

Both proxy paths are **stateless**: no Redis, no SQLite, no RAG, no KV cache, no rolling
summarization. The caller manages its own conversation history.

---

## 0. LLM routing overview

SoHoAI routes conversation inference across two model tiers: **external** (Claude Sonnet 4.6 via Anthropic API, primary cloud default) and **internal** (Gemma 4 E4B 7.52B Q8_0 on Server 2, fallback/summarization). Routing logic is implemented in `router.py`. The default is external (Sonnet 4.6); if Anthropic becomes unreachable, the router automatically falls back to local Gemma. Rolling summarization at ~100K tokens uses internal (Gemma 4) to keep per-token costs predictable and persists summaries to SQLite for cold-resume recovery.

**External (Sonnet 4.6) path** goes through LiteLLM with prompt caching enabled — cache_control markers are injected on the system message (long-lived anchor) and on `messages[-2]` (rolling prefix anchor), reducing input cost by ~90% on cache hits. Prompt caching is active only on the Anthropic-compatible path; the local path uses Anthropic prompt caching instead.

**Specialist (Gemma 4) path** bypasses LiteLLM and calls llama-server's native `/completion` endpoint directly, which is mandatory to pass `slot_id` for KV cache targeting. This path is used only on fallback (Anthropic down), for rolling summarization operations, and for background/offline tasks. Rolling summarization erases the KV slot before calling Gemma, and both summarization and the subsequent main inference start cold sequentially on the same slot. Prompt caching is irrelevant for local inference (no API cost).

**Design rationale**: Sonnet 4.6 is now the interactive default (2026-04-22 flip). At ~50–100 turns/day for a 4-user family, Sonnet with prompt caching costs ~$30–60/mo — tolerable — while delivering substantially better reasoning and tool-use fidelity than a 4B local model. Gemma's role shifted from "default inferencer" to "specialized worker" without removing the infrastructure.

**LiteLLM stays as the routing + fallback layer**, handling OpenAI/Anthropic API differences and executing the fallback chain `external → internal` (reversed direction from pre-flip implementation). **Internal bypasses LiteLLM** — native `/completion` is required to pass `slot_id` for KV cache targeting. The branch point is at `main.py:333-347`.

**Tool-use on both paths** uses an XML sentinel: `<tool_call>{"name":"search_documents","args":{"query":"…"}}</tool_call>`. Both external and internal paths emit and parse this sentinel. Sonnet handles it reliably. Native Anthropic `tools=[...]` + `tool_use` content blocks remain a deferred follow-up; unifying the two paths isn't blocking current quality.

---

## 1. Cline VSCode plugin — OpenAI-compatible path

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/proxy/v1/models` | OpenAI-compatible model list — all 8 public IDs from `_PROXY_EXPOSED_MODELS` (including `anthropic/*`) |
| GET | `/proxy/v1/model/info` | LiteLLM-compatible model info (`max_input_tokens`, `context_window`) — Cline reads this to set its context-window display |
| POST | `/proxy/v1/chat/completions` | Stateless OpenAI chat completions, streaming supported |

### Model mapping (`_PROXY_EXPOSED_MODELS` in `main.py`)

`_PROXY_EXPOSED_MODELS` is an **identity map** — every public ID equals its LiteLLM
`model_name` in `config.yaml`. No translation layer; `config.yaml` is the single source
of truth.

| Public ID (= config.yaml model_name) | Path | Backend | Context window |
|---|---|---|---|
| `internal/gemma-4-e4b` | LiteLLM | llama-server, Server 2, Gemma 4 E4B Q8_0 | 110,024 |
| `anthropic/claude-haiku-4-5` | Transparent forward | Anthropic API | 200,000 |
| `anthropic/claude-sonnet-4-6` | Transparent forward | Anthropic API (Gemma fallback if down) | 200,000 |
| `anthropic/claude-opus-4-7` | Transparent forward | Anthropic API | 1,000,000 |
| `ollama-cloud/deepseek-v4-pro` | LiteLLM | Ollama cloud (`https://ollama.com/v1`) | **~70% 503 rate** — see §2.3 |
| `ollama-cloud/kimi-k2.6` | LiteLLM | Ollama cloud | — |
| `ollama-cloud/glm-5.1` | LiteLLM | Ollama cloud | — |
| `ollama-cloud/qwen3-coder-next` | LiteLLM | Ollama cloud | — |

**Why `/proxy/v1/models` exposes all 8 — including `anthropic/*`:** Cline builds its
model dropdown entirely from this endpoint; it has no hardcoded built-in model list.
Exposing `anthropic/*` here simply tells Cline those models are selectable — no
duplication risk. This is the key difference from `GET /v1/models` (the Claude Code
discovery endpoint, §4.5): that endpoint excludes `anthropic/*` because Claude Code
already has those IDs in its native built-in list and would show each model twice if
the gateway also returned them.

`_resolve_proxy_model()` also accepts legacy bare names (`gemma-4-e4b`, `claude-sonnet-4-6`
etc.) via `_LEGACY_ALIASES` for backward compat with existing Cline configs.

**Ollama cloud models are reasoning models** (DeepSeek V4 Pro, Kimi K2.6, GLM-5.1
in particular). They spend a variable number of tokens on internal reasoning before
emitting visible output. Use `max_tokens ≥ 500` for these models; requests with low
limits (e.g. `max_tokens=20`) will hit the limit during the thinking phase and return
empty `content[0].text` with `stop_reason: max_tokens`.

### Cline configuration

In Cline VSCode settings, choose **LiteLLM** as the provider (not "OpenAI Compatible"):

```
Base URL : http://192.168.1.93:8000/proxy
API Key  : sohoai-local  (any non-empty string)
Model    : gemma-4-e4b  (local, 110K ctx)  OR  claude-sonnet-4-6  (cloud, 200K ctx)
```

API key must be non-empty — Cline's client-side gate rejects an empty string regardless
of server-side validation.

### Verify proxy model info

```bash
curl http://192.168.1.93:8000/proxy/v1/model/info | python3 -m json.tool | grep -E "model_name|max_input_tokens"
```

### Shared llama-server slot race

Both SoHoAI's `internal` path (summarization, offline fallback) and Cline's
`gemma-4-e4b` proxy path hit the same llama-server on Server 2. A Cline request that
lands on a slot between SoHoAI's `restore → inference → save` sequence will corrupt that
chat's KV state. Accepted risk: SoHoAI's Gemma usage is rare post-flip, and the
corruption self-heals on the next turn.

---

## 2. Claude Code — Anthropic Messages API path

### Endpoint

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/messages` | Anthropic Messages API with model-aware routing |

### Configuration

Set in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.1.93:8000",
    "ANTHROPIC_API_KEY": "<your-real-anthropic-api-key>"
  }
}
```

Claude Code sends the real API key in the `x-api-key` header on every request. The
proxy extracts it and forwards it to `api.anthropic.com` on the transparent path.

### Model-aware routing

The proxy inspects the `model` field and branches on two paths:

```
REQUEST arrives at POST /v1/messages
  │
  ├─ model starts with "internal/" (internal/gemma-4-e4b, future internal/*)
  │     → _anthropic_messages_litellm()
  │     → Anthropic→OpenAI format conversion (tools, tool_use, tool_result forwarded)
  │     → LiteLLM Router → llama-server (Server 2)
  │     → tools forwarded; Gemma synthetic-smoke PASS, broader workload validation pending
  │
  ├─ model starts with "ollama-cloud/" (all 4 Ollama cloud models)
  │     → _anthropic_messages_litellm() [same conversion path]
  │     → Anthropic→OpenAI format conversion (tools, tool_use, tool_result forwarded)
  │     → LiteLLM Router → https://ollama.com/v1 (OLLAMA_API_KEY from .env)
  │     → tools forwarded
  │     → NOTE: reasoning models — use max_tokens ≥ 500
  │     → NOTE: Claude Code system prompt causes model to identify as Claude
  │
  └─ model starts with "anthropic/" (or unresolved)
        → "anthropic/" prefix stripped before forwarding
        → _anthropic_messages_forward()
        → transparent HTTP forward to api.anthropic.com/v1/messages
        → full fidelity: tools, history, caching, streaming all preserved
```

### §2.1 Transparent forward path (Anthropic models)

`_anthropic_messages_forward()` in `main.py` acts as a pure HTTP relay:

- Forwards the **exact request body bytes** — no parsing, no modification
- Forwards **Anthropic-specific headers**: `x-api-key`, `anthropic-version`, `anthropic-beta`
- Streams the **raw response bytes** back — no re-encoding of SSE events

**What this preserves and why it matters:**

| Preserved | Consequence |
|---|---|
| `tools` array | Model knows all Claude Code tools (Read, Write, Bash, Agent, etc.) on every turn |
| `tool_use` blocks in assistant messages | Full multi-turn tool-call history reaches the model; it knows what it already requested |
| `tool_result` blocks in user messages | File contents, bash outputs, and sub-agent results from past turns are visible to the model |
| `cache_control: ephemeral` markers | Anthropic caches the system prompt + rolling message prefix; after turn 1 the ~100K token context costs ~10% per turn instead of 100% |
| `anthropic-beta` headers | Features like `interleaved-thinking`, extended output, etc. work unchanged |
| Native SSE event types | Claude Code's Ink UI receives `content_block_start {type:tool_use}` events and renders compact one-line tool-call status instead of scrolling file contents |

**Without transparent forward** (the old single LiteLLM path): cache_control was
stripped → no caching → every turn paid full input-token cost; tools were stripped →
model could not call Read/Write/Bash → Claude Code fell back to injecting file contents
as inline text that scrolled on screen.

### §2.2 LiteLLM path (internal/* and ollama-cloud/*)

`_anthropic_messages_litellm()` in `main.py` converts the Anthropic-format request to
OpenAI format and routes through LiteLLM Router to the target provider.

**Tool use is now supported on this path** (implemented 2026-05-10).

The conversion now handles full Anthropic→OpenAI transformation:
- `tools` array: wrapped in OpenAI `{"type":"function","function":{...}}` format
- `tool_use` blocks (assistant): converted to OpenAI `tool_calls` with serialized `arguments`
- `tool_result` blocks (user): converted to OpenAI `role:"tool"` messages
- Streaming responses: emit proper Anthropic SSE `content_block_start/delta/stop` events for tool calls
- Image blocks (user): converted to OpenAI `image_url` format for vision-capable models
- `cache_control` markers: forwarded (irrelevant for local inference, but preserved)

**What is now forwarded and how the model can use it:**

| Forwarded | What the model receives | Use case |
|---|---|---|
| `tools` array | Full knowledge of Read/Write/Bash tools | Model can request file reads, code writes, bash execution |
| `tool_use` blocks (assistant) | Complete prior tool-call requests from history | Model reasons about previous attempts and iterations |
| `tool_result` blocks (user) | File contents and command outputs from past turns | Model uses retrieved data to refine answers |
| `cache_control` markers | Forwarded to provider (has no effect on llama-server) | Prepared for future local models with caching support |

**Gemma reliability — current state**: `internal/gemma-4-e4b` **passed the synthetic two-turn smoke harness** (single tool, one string argument) on both streaming and non-streaming legs (2026-05-10). Broader claude-orchestra workload validation (full Claude Code tool catalogue, multi-turn, parallel calls) is the remaining open item — see `docs/TODO.md`. Grammar-constrained generation (deferred Step c) is therefore likely unnecessary; revisit only if broader validation surfaces malformed `arguments` JSON.

For `ollama-cloud/deepseek-v4-pro` and `ollama-cloud/qwen3-coder-next`: both support OpenAI
function calling natively and are expected to be reliable for tool use **when the endpoint
is reachable** — see §2.3 for the reliability caveat on deepseek-v4-pro specifically.

**Appropriate use cases for the LiteLLM path:**
- Full Claude Code sessions with ollama-cloud models (qwen3-coder-next recommended; deepseek-v4-pro usable but unreliable — see §2.3)
- Sub-agents using ollama-cloud models that need to read files, write code, or run bash commands
- Multi-turn tool call chains with cloud models (zero API cost vs Anthropic)
- Summarization tasks (`internal/gemma-4-e4b` already used by `maybe_summarize()`)
- Offline/background drafting when Anthropic is unreachable (`internal/gemma-4-e4b`)
- Exploratory prompting with Ollama cloud models
- Cost-sensitive deployments where cloud model cost is critical (Ollama cloud ≈ $0 vs Anthropic)

**Inappropriate use cases:**
- `internal/gemma-4-e4b` for tool-requiring workloads (until reliability is validated)
- Tasks where tool-call reliability is mission-critical and cannot tolerate failures
- `ollama-cloud/deepseek-v4-pro` for mission-critical or long-running agent tasks given its current instability

**Claude Code identity note:** When using `ollama-cloud/*` models via `claude --model`,
Claude Code injects its own system prompt ("You are Claude Code..."). Ollama cloud models
follow this persona and will self-identify as Claude. The actual model is confirmed by
the status bar and server logs — not the model's self-description.

### §2.3 Ollama cloud reliability, timeout backoff, and 503 guard

**deepseek-v4-pro has a documented ~70% HTTP 500/503 failure rate** on Ollama's shared
inference tier since its April 2026 release (GitHub issues #15832, #15934). Ollama shared
inference carries **no SLA**. The other three models (kimi-k2.6, glm-5.1, qwen3-coder-next)
are substantially more reliable.

**Why not a silent fallback?** A LiteLLM Router fallback (deepseek → Sonnet) was
considered and rejected: LiteLLM fallback is per-request, so if deepseek recovers on the
next turn it resumes answering. Within a single sub-agent task this produces alternating
models mid-task — worse than a clean failure. It also incurs unexpected cost when the user
explicitly chose a $0 model.

**Timeout and backoff (updated 2026-05-15):** Instead of a single 30s timeout that immediately
surfaces as HTTP 529, the proxy now uses a **3-step increasing-timeout backoff** in
`SmartRouter.complete()` (`router.py`):

| Attempt | Timeout | Log |
|---------|---------|-----|
| 1 | 30 s | (silent) |
| 2 | 60 s | WARNING: `ollama-cloud … timed out on attempt 1/3, retrying (timeout=60s)…` |
| 3 | 90 s | WARNING: `ollama-cloud … timed out on attempt 2/3, retrying (timeout=90s)…` |
| exhausted | — | ERROR: `ollama-cloud … all 3 attempts timed out` → HTTP 529 |

Worst-case total: 30+60+90 = 180 s — within CC's 300 s httpx limit. Only `litellm.Timeout`
triggers a retry; auth errors and 4xx responses propagate immediately.

**How final failures are reported:**

| Path | behavior after all 3 attempts exhausted |
|------|----------------------------------------|
| Streaming (`stream: true`) | SSE `error` event emitted in-stream; HTTP 200 (stream already open) |
| Non-streaming | HTTP **529** `overloaded_error` with Anthropic-format JSON body |

```json
{
  "type": "error",
  "error": {
    "type": "overloaded_error",
    "message": "Model 'ollama-cloud/deepseek-v4-pro' is temporarily unavailable (Ollama cloud overloaded or timed out). Retry or switch models with /model. ..."
  }
}
```

HTTP 529 is Anthropic's own "overloaded" code — Claude Code renders it as a clean
user-visible error rather than an opaque crash. The message includes a `/model` hint.

**Recommendation:** For interactive Claude Code sessions that need reliability, use
`ollama-cloud/qwen3-coder-next` (coding) or an `anthropic/*` model. Reserve deepseek-v4-pro
for exploratory/disposable sessions where occasional failures are acceptable.

### §2.4 Subagent blocking for `claude-code-*` sessions

When Claude Code uses a `claude-code-*` (Ollama Cloud) model as its main model, it
internally auto-spawns `claude-haiku-4-5-20251001` for lightweight background tasks.
These would otherwise be transparently forwarded to Anthropic — incurring unexpected cost
while the user expects a $0 Ollama session.

**Guard in place (2026-05-15):** `proxy.blocked_models` in `config.yaml` contains
`claude-haiku-4-5-20251001`. In `anthropic_messages()`, any request whose `model` is in
this list returns HTTP 400 (`not_supported_error`) before routing. CC does not retry 400s
and falls back to the main model.

```yaml
proxy:
  blocked_models:
    - claude-haiku-4-5-20251001
```

Add future versioned Haiku model names here as needed.

### §2.5 Tool_use ID sanitization

Ollama Cloud models (confirmed: kimi-k2.6) return tool call IDs in the format
`functions.{name}:{index}` — e.g. `functions.Bash:38`. These contain `.` and `:` which
violate Anthropic's required pattern `^[a-zA-Z0-9_-]+$`. Without sanitization, CC stores
these IDs in its conversation history and the next call to an Anthropic-native model fails
with HTTP 400.

**Guard in place (2026-05-15):** `_sanitize_tool_use_id()` in `main.py` replaces invalid
characters with `_` (`functions.Bash:38` → `functions_Bash_38`). Applied in both the
streaming and non-streaming paths of `_anthropic_messages_litellm()`. Substitution is
deterministic, so the ID round-trips correctly when CC sends it back as `tool_use_id`.

---

## 3. Sub-agent mechanics and model-tier routing

### How Claude Code dispatches sub-agents

Claude Code dispatches sub-agents via its built-in `Agent` tool (used by `/duo`,
`/brain`, and other pipelines). Each sub-agent is defined in a markdown file under
`~/.claude/agents/` or the project's `.claude/agents/`. The relevant frontmatter fields:

```yaml
model: claude-haiku-4-5          # which model to use for this agent
description: "Actor — implementation"
tools: [Read, Write, Bash, Glob, Grep]
```

When a sub-agent runs:
1. Claude Code spawns a new Claude Code process (headless, non-interactive)
2. That process reads the `model` field from the agent's frontmatter
3. It calls `POST /v1/messages` with `"model": "claude-haiku-4-5"` to whatever
   `ANTHROPIC_BASE_URL` is configured (inherited from `~/.claude/settings.json`)
4. SoHoAI proxy receives the request, sees `claude-haiku-4-5` → transparent forward → Anthropic Haiku

**Note**: `api_base_url` is NOT a documented Claude Code agent frontmatter field.
All agents share the same `ANTHROPIC_BASE_URL`. The `model` field is the only
supported mechanism for per-agent model selection. Routing is determined by whatever
model name the sub-agent sends in its request body.

### /duo pipeline model tiers

```
Parent session (Brain + Planner) — Sonnet 4.6
  source: settings.json "model": "sonnet[1m]"
  request: POST /v1/messages {"model": "claude-sonnet-4-6", ...}
  routing: transparent forward → Anthropic Sonnet 4.6
  tools: ✅ full (Read, Write, Bash, Agent, Glob, Grep, ...)
  caching: ✅ full (system prompt + rolling prefix cached after turn 1)

Actor subagent — Haiku 4.5
  source: agent frontmatter "model: claude-haiku-4-5"
  request: POST /v1/messages {"model": "claude-haiku-4-5", ...}
  routing: transparent forward → Anthropic Haiku 4.5
  tools: ✅ full (same tool loop as parent)
  caching: ✅ full (independent cache slot from parent)
```

Both are independent Anthropic API calls with independent prompt-cache slots.

### /brain pipeline model tiers

```
Brain    — claude-opus-4-7   → transparent forward → Anthropic Opus 4.7   (tools ✅, cache ✅)
Planner  — claude-sonnet-4-6 → transparent forward → Anthropic Sonnet 4.6 (tools ✅, cache ✅)
Actor    — claude-haiku-4-5  → transparent forward → Anthropic Haiku 4.5  (tools ✅, cache ✅)
Reviewer — claude-sonnet-4-6 → transparent forward → Anthropic Sonnet 4.6 (tools ✅, cache ✅)
```

Each tier has its own Anthropic API call, its own context window, and its own
independent prompt-cache slot. The Brain's long research context does not inflate
the Actor's costs.

### Local sub-agent (gemma-4-e4b) — tool-use status

A sub-agent with `model: internal/gemma-4-e4b` or `model: ollama-cloud/*` routes via the LiteLLM local path.
**Tool use is now supported on this path** (implemented 2026-05-10). All three target models — `ollama-cloud/qwen3-coder-next`, `ollama-cloud/deepseek-v4-pro`, and `internal/gemma-4-e4b` — passed the synthetic two-turn smoke (`utils/tool_use_smoke_test.py`) on both streaming and non-streaming.

For `ollama-cloud/*` models (deepseek-v4-pro, qwen3-coder-next), such an agent can:
- Read files (Read tool supported, smoke-validated)
- Write or edit files (Write/Edit tools supported)
- Run bash commands (Bash tool supported)
- Use sub-agents (Agent tool supported)
- Perform complex multi-turn tool call chains

For `internal/gemma-4-e4b`, such an agent passed the simple-tool smoke. Real-world claude-orchestra workloads (full Claude Code tool catalogue, parallel tool calls, deeply-nested arguments) are the remaining open question — see `docs/TODO.md` "Tool-use deferred: internal/gemma-4-e4b broader reliability verification".

Recommended Actor-tier model selection:
- **Anthropic transparent forward** (`anthropic/claude-haiku-4-5`): the safest choice for production-grade Actor tasks that require reliable tools. Cost ≈ $0.01/session.
- **Ollama cloud** (`ollama-cloud/qwen3-coder-next` for coding tasks, `ollama-cloud/deepseek-v4-pro` for reasoning): smoke-validated, cost ≈ $0/session. Reasoning models need `max_tokens ≥ 500`.
- **Local Gemma** (`internal/gemma-4-e4b`): smoke-validated, cost = $0, but real-world reliability not yet measured. Recommended for cost-sensitive non-critical Actor work; not yet recommended for primary tool-using subagents until broader validation completes.

---

## 4. Claude Code Integration

This section consolidates how Claude Code uses SoHoAI as a transparent proxy — both as
the primary interactive client and as a sub-agent dispatcher.

### 4.1 Two paths

| Path | Config | Endpoint | Use case |
|------|--------|----------|----------|
| **Direct Anthropic** | `ANTHROPIC_BASE_URL=http://192.168.1.93:8000` in `~/.claude/settings.json` | `POST /v1/messages` | Interactive Claude Code sessions; all `anthropic/*` models |
| **Sub-agent / proxy** | `api_base_url: http://192.168.1.93:8000/proxy` in agent frontmatter | `POST /proxy/v1/chat/completions` | Stateless sub-agent dispatch; `internal/*` + `anthropic/*` + `ollama-cloud/*` |

**Direct Anthropic path** (transparent forward): the proxy relays the exact request bytes
to `api.anthropic.com`. Tools, `tool_use`, `tool_result`, `cache_control`, and
`anthropic-beta` headers are fully preserved. Prompt caching, streaming, and native SSE
event types all work unchanged.

**Sub-agent / proxy path** (`/proxy/v1/chat/completions`): stateless OpenAI-compatible
pass-through. Sub-agents in claude-orchestra (`/brain`, `/duo`) use this path when
given an `api_base_url` override in their frontmatter. The full model list is available.

### 4.2 Available models for sub-agents

All models exposed via `_PROXY_EXPOSED_MODELS` in `main.py`:

| Model ID | Path | Backend | Notes |
|----------|------|---------|-------|
| `internal/gemma-4-e4b` | LiteLLM conversion | llama-server, Server 2 | $0/session; tool-use smoke PASS; broader validation pending |
| `anthropic/claude-haiku-4-5` | Transparent forward | Anthropic API | Safest Actor-tier choice; ~$0.01/session |
| `anthropic/claude-sonnet-4-6` | Transparent forward | Anthropic API | Default interactive model |
| `anthropic/claude-opus-4-7` | Transparent forward | Anthropic API | Brain tier in /brain pipeline |
| `ollama-cloud/deepseek-v4-pro` | LiteLLM conversion | Ollama cloud | Reasoning model; `max_tokens ≥ 500`; **~70% 503 rate** — see §2.3; not recommended for critical tasks |
| `ollama-cloud/kimi-k2.6` | LiteLLM conversion | Ollama cloud | Reasoning model; `max_tokens ≥ 500`; tool-use smoke PASS |
| `ollama-cloud/glm-5.1` | LiteLLM conversion | Ollama cloud | Reasoning model; `max_tokens ≥ 500`; tool-use smoke PASS |
| `ollama-cloud/qwen3-coder-next` | LiteLLM conversion | Ollama cloud | Coding model; standard `max_tokens`; tool-use smoke PASS; **recommended** for $0 coding tasks |

### 4.3 Tool calling via sub-agents (LiteLLM path)

When a sub-agent uses an `internal/*` or `ollama-cloud/*` model, Claude Code sends
requests in Anthropic Messages API format. `_anthropic_messages_litellm()` in `main.py`
converts them to OpenAI format for LiteLLM using four helpers:

| Helper | Converts |
|--------|---------|
| `_convert_tools()` | Anthropic `tools` array → OpenAI `{"type":"function","function":{...}}` |
| `_convert_tool_choice()` | Anthropic `tool_choice` → OpenAI `tool_choice` |
| `_convert_assistant_message()` | `tool_use` content blocks → OpenAI `tool_calls` with JSON `arguments` |
| `_convert_user_message()` | `tool_result` blocks → OpenAI `role:"tool"` messages; `image` blocks → `image_url` |

Streaming responses emit proper Anthropic SSE `content_block_start/delta/stop` events
for tool calls, including `message_delta` with `stop_reason: "tool_use"`.

**Validation status:**

| Model | Single-tool smoke | 5-tool parallel smoke | Notes |
|-------|------------------|-----------------------|-------|
| `internal/gemma-4-e4b` | PASS (streaming + non-streaming, 2026-05-10) | INFO: 1/5 tools only | Broader workload validation pending |
| `ollama-cloud/qwen3-coder-next` | PASS (streaming, 2026-05-10) | PASS (streaming, 2026-05-11) | — |
| `ollama-cloud/deepseek-v4-pro` | PASS (streaming, 2026-05-10) | FAIL — live 503 (2026-05-11) | Reasoning model: `max_tokens ≥ 500`; see §2.3 |
| `ollama-cloud/kimi-k2.6` | PASS (streaming, 2026-05-10) | PASS (streaming, 2026-05-11) | Reasoning model: `max_tokens ≥ 500` |
| `ollama-cloud/glm-5.1` | PASS (streaming, 2026-05-10) | PASS (streaming, 2026-05-11) | Reasoning model: `max_tokens ≥ 500` |

Smoke harness: `utils/tool_use_smoke_test.py`
- Standard: `--server http://192.168.1.93:8000` — 2-turn, single tool
- Parallel: `--parallel` — 2-turn, 5 simultaneous tools (get_file_size, get_file_owner,
  get_file_permissions, get_file_modification_time, get_file_line_count)

**Remaining open item**: real claude-orchestra sub-agent session with full Claude Code tool
catalogue (Read/Write/Edit/Bash/Glob/Grep) and parallel sub-agent dispatch. The 5-tool
parallel smoke covers parallel tool calls in a single turn; multi-agent parallelism
remains unvalidated. See `docs/TODO.md`.

### 4.4 Identity caveat

When an `ollama-cloud/*` model receives a Claude Code sub-agent request, it also
receives Claude Code's system prompt ("You are Claude Code…"). These models follow
the persona and **self-identify as "Claude"**. The actual model is confirmed by:
- The status bar model indicator in the Claude Code UI
- Server logs on Server 1 (`uvicorn` stdout shows `model=ollama-cloud/...`)

Model self-description is not a reliable indicator of which model is handling the request.

### 4.5 Gateway model discovery and `claude-code-*` alias scheme

SoHoAI's `GET /v1/models` endpoint publishes the models that Claude Code cannot discover
on its own — non-Anthropic models only. The transformation is performed by
`_claude_code_alias_for()` in `main.py` and reversed by `_claude_code_alias_to_public()`
before routing to the backend. Bijection correctness is validated by `utils/alias_bijection_test.py`.

**`GET /v1/models` does NOT include `anthropic/*` models.**

Rationale: Claude Code already has `claude-haiku-4-5`, `claude-sonnet-4-6`, and
`claude-opus-4-7` in its built-in native model list. When `ANTHROPIC_BASE_URL` points at
the SoHoAI gateway, these are routed transparently through `POST /v1/messages` — no
discovery entry is needed. Including them in `/v1/models` caused two problems:

1. **Duplicate picker entries**: each Anthropic model appeared twice (once from the native list, once from gateway discovery).
2. **Misleading metadata**: the gateway's copy carried hardcoded fallback values (`context_window`, `max_tokens`, display labels synthesized by `_display_name_for()`) rather than real Anthropic API metadata.

**Models returned by `GET /v1/models` (5 total — non-Anthropic only):**

| Public ID | claude-code-* alias | Backend | Notes |
|---|---|---|---|
| `internal/gemma-4-e4b` | `claude-code-gemma-4-e4b` | llama-server, Server 2 | $0/session |
| `ollama-cloud/deepseek-v4-pro` | `claude-code-deepseek-v4-pro` | Ollama cloud | Reasoning; `max_tokens ≥ 500` |
| `ollama-cloud/kimi-k2.6` | `claude-code-kimi-k2.6` | Ollama cloud | Reasoning; `max_tokens ≥ 500` |
| `ollama-cloud/glm-5.1` | `claude-code-glm-5.1` | Ollama cloud | Reasoning; `max_tokens ≥ 500` |
| `ollama-cloud/qwen3-coder-next` | `claude-code-qwen3-coder-next` | Ollama cloud | Coding; standard `max_tokens` |

Anthropic models (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`) are still
fully accessible — they appear in the picker from Claude Code's native built-in list and
route through the gateway transparently via `ANTHROPIC_BASE_URL`.

**Contrast with `/proxy/v1/models` (Cline path):** that endpoint still returns all 8
public IDs from `_PROXY_EXPOSED_MODELS` (including `anthropic/*`). The distinction matters
because Cline manages its own model list and does not have a native built-in list that
would conflict.

Claude Code's model picker and subagent frontmatter read from `/v1/models`. Both the picker
(when `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`) and manual subagent `model:` fields
accept the full `claude-code-*` identifier. The proxy's `_resolve_proxy_model()` function
accepts both the legacy public_id forms (e.g., `ollama-cloud/deepseek-v4-pro`) and the
new aliases for backward compatibility with existing configurations.

**Token counting endpoint (`POST /v1/messages/count_tokens`):**

Claude Code sends pre-flight token estimates before dispatching requests. The endpoint handles two cases:
1. **Anthropic models** (`claude-*`): request is forwarded transparently to `api.anthropic.com/v1/messages/count_tokens`; response is returned as-is.
2. **LiteLLM-routed models** (`claude-code-*` aliases): `litellm.token_counter()` estimates token count for the target model. Unknown tokenizers fall back to `len(text) // 4`. Both paths return `{input_tokens, output_tokens}` for consistency.

This closes a 404 that occurred before the endpoint was available — Claude Code's pre-flight checks now succeed, enabling faster interactive feedback.

**Activation:**

Enable gateway model discovery by setting `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` in `~/.claude/settings.json` (requires Claude Code v2.1.129+). See `docs/claude-orchestra-handoff.md` §5.2 for the exact settings.json snippet and §5 for the full post-merge deployment runbook.

---

## 5. Alternative - Anthropic passthrough — and comparison


The problem: LiteLLM's standard proxy endpoint (`/v1/messages`) strips/transforms certain Anthropic-specific elements (cache_control markers, tools array, anthropic-beta headers) when it converts requests. 

This causes tool calls to be stripped—forcing Claude Code to fall back on inline file injection—and strips cache_control markers, eliminating prompt caching and incurring full costs every turn.


The general recommendation from practitioners is: Only use LiteLLM as a fallback router or when wrapping an OpenAI-compatible local endpoint that needs format translation.

For the Brain/Planner/Reviewer tiers in claude-orchestra that use Anthropic models, pointing directly at `api.anthropic.com` or LiteLLM's `/anthropic` passthrough is the correct path. 


LiteLLM's passthrough URL approach uses `ANTHROPIC_BASE_URL=http://localhost:8000/anthropic` to route Claude Code directly to LiteLLM's built-in passthrough endpoint, which forwards requests transparently without any transformation. SoHoAI's solution implements the same transparent relay by creating a custom HTTP endpoint that forwards the exact request body bytes without modification.

The real distinction is that SoHoAI operates at the server level with model-aware routing—Anthropic requests get forwarded transparently while local models get converted through LiteLLM—whereas the LiteLLM passthrough is purely Anthropic-focused with no branching logic, where the client -- e.g. claude code -- would use a custom URL, with no ability for the LiteLLM to do mode-aware routing 


### 5.1 In-depth comparison

Both are two implementations of the same fix. In more detail: 

The root cause in both cases is identical: LiteLLM's standard `/v1/messages` endpoint **transforms** requests — it parses, modifies, and re-serialises them. In doing so it strips `cache_control` markers, drops the `tools` array, and does not forward `anthropic-beta` headers. Claude Code, receiving responses from this path, loses the SSE event types it needs for compact tool rendering and falls back to injecting file contents as scrolling inline text.

**LiteLLM's `/anthropic` passthrough** fixes this by telling LiteLLM: skip transformation entirely, relay bytes to `api.anthropic.com` unchanged.

**SoHoAI's `_anthropic_messages_forward()`** fixes this by implementing the same relay directly inside the FastAPI app — exact body bytes forwarded, `x-api-key` / `anthropic-version` / `anthropic-beta` headers preserved, raw SSE bytes streamed back.

### 5.2 Why SoHoAI's approach is strictly more capable

The critical difference is the **model-aware branching** at the `POST /v1/messages` handler:

```
model = "claude-*"  →  _anthropic_messages_forward()   (transparent relay)
model = "gemma-4-e4b"  →  _anthropic_messages_litellm()  (LiteLLM conversion)
```

LiteLLM's `/anthropic` passthrough endpoint has no such branching — it can only forward to Anthropic. It cannot route local models. SoHoAI's implementation handles both paths behind the same `ANTHROPIC_BASE_URL`, which is what allows `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`, and `gemma-4-e4b` to all be valid `model:` values in agent frontmatter while sharing one endpoint configuration in `settings.json`.

### 5.3 Observation regarding caching — the one finding worth noting

Section §6 below contains an important empirical result: prompt caching is **not currently active** on your account — `cache_creation_input_tokens: 0` and `inference_geo: not_available` on both proxy and direct Anthropic calls. 

The transparent forward is correctly preserving `cache_control` markers; the limitation is account-tier or region on Anthropic's side. 

This means the cost tables in §6 are the target state once caching activates, not the current state. The transparent forward implementation is correct and future-proof — no changes needed when caching becomes available.



---

## 6. Prompt caching and cost implications

### How Anthropic prompt caching works (transparent path only)

Claude Code injects `cache_control: {type: ephemeral}` markers on:
1. The system prompt (CLAUDE.md + project context — typically 20K–100K tokens)
2. `messages[-2]` — the penultimate message (rolling prefix breakpoint)

On the transparent forward path these markers reach Anthropic unchanged.
After turn 1, Anthropic reports `cache_read_input_tokens` instead of `input_tokens`
for the cached portion — the cached tokens cost ~10% of uncached input price.

On the LiteLLM local path, cache_control markers are stripped by the conversion.
This is irrelevant because local inference has no per-token API cost.

**Note — account-tier / region dependency (verified 2026-05-04):** Prompt caching
requires Anthropic to serve the request from a caching-enabled inference region.
Smoke testing showed `cache_creation_input_tokens: 0` and `inference_geo: not_available`
on both proxy and direct Anthropic calls with the current API key — indicating caching
is not active for this account tier or routing configuration. The proxy forwards
`cache_control` markers correctly; the limitation is on Anthropic's side. If caching
becomes available (account upgrade, region change), no proxy changes are needed.

### Approximate cost per tier (Anthropic API pricing, May 2026)

| Model | Input (uncached) | Input (cached read) | Output |
|---|---|---|---|
| Opus 4.7 | $15/MTok | $1.50/MTok | $75/MTok |
| Sonnet 4.6 | $3/MTok | $0.30/MTok | $15/MTok |
| Haiku 4.5 | $0.80/MTok | $0.08/MTok | $4/MTok |
| Gemma 4 (local) | $0 | $0 | $0 |

### /duo session cost breakdown (typical task, 5 parent turns + 1 actor execution)

**Parent (Sonnet 4.6) — transparent forward, prompt caching active**

| Turn | Input (system+history) | Cache hit? | Effective input cost |
|---|---|---|---|
| 1 (cold) | ~30K tokens | ❌ | 30K × $3/MTok = $0.090 |
| 2 | ~32K (30K cached + 2K new) | ✅ 30K cached | 2K×$3 + 30K×$0.30 = $0.015 |
| 3–5 | ~34K (growing) | ✅ rolling cache | ~$0.015/turn |
| Output | ~500 tok/turn × 5 | — | 2.5K × $15/MTok = $0.038 |
| **Parent total** | | | **~$0.18** |

**Actor (Haiku 4.5) — transparent forward, narrower context**

| Turn | Input | Cache hit? | Cost |
|---|---|---|---|
| 1 (cold) | ~5K tokens | ❌ | 5K × $0.80/MTok = $0.004 |
| 2–3 | cached | ✅ | ~$0.001/turn |
| Output | ~1K tok × 2 | — | 2K × $4/MTok = $0.008 |
| **Actor total** | | | **~$0.015** |

**Session total (approx)**: **~$0.20**

**Without transparent forward (broken LiteLLM path, pre-fix)**:
- cache_control stripped → every turn cold → 30K × $3/MTok × 5 = $0.45 input alone
- Tools broken → task fails or Claude Code injects files inline (even larger context)
- Effective cost 3–5× higher and task likely incomplete

### /brain session cost (Opus 4.7 Brain tier)

| Tier | Model | Turns | Est. cost |
|---|---|---|---|
| Brain | Opus 4.7 | 3–5 research turns | $0.50–$1.20 |
| Planner | Sonnet 4.6 | 1–2 planning turns | $0.05–$0.10 |
| Actor | Haiku 4.5 | implementation | $0.01–$0.03 |
| Reviewer | Sonnet 4.6 | 1 review turn | $0.03–$0.05 |
| **Total** | | | **$0.60–$1.40** |

Prompt caching cuts the Opus Brain cost by ~70% after the first turn (large research
context is re-read cached). Without caching (broken path), costs would be 3–4× higher.

---

## 7. LiteLLM `model_info` parameter semantics

These fields in `config.yaml model_info` are **not** forwarded to the provider API.

| Field | Controls | Needed in config? |
|---|---|---|
| `max_tokens` | Output cap metadata | Only to enforce a lower cap than the model's native output limit |
| `max_input_tokens` | Input size guard (pre-call check) | Only for models not in LiteLLM's registry (e.g. `internal`/llama-server) |
| `context_window` | Router metadata for routing/cost | Only for models not in LiteLLM's registry |

For `anthropic/claude-*` models: LiteLLM auto-resolves all three from its internal
`model_prices_and_context_window.json` registry. No `model_info` block needed.

For `internal` (Gemma / llama-server): explicit `model_info` is required because
llama-server is not in LiteLLM's registry. Cline reads `max_input_tokens` from
`/proxy/v1/model/info` to set its context-window display.

`enable_pre_call_checks: true` (set in `router_settings` in `config.yaml`) activates
input-size enforcement using these values.

---

## 8. Future work

Basic tool-use support for the local-model path (gemma-4-e4b and Ollama cloud models) is now implemented
and smoke-validated for all 5 targets (see `docs/TODO.md` IMPLEMENTED banner). The following deferred items remain:

- **Full Claude Code tool catalogue + parallel tool calls** — the 2-turn single-tool smoke is validated; multi-tool
  parallel invocation (multiple `tool_use` blocks in one response) and the full claude-orchestra tool catalogue
  (Read/Write/Bash/Glob/Grep/Edit/TodoWrite/Agent) remain unvalidated. See `docs/TODO.md`.
- Gemma 4 E4B broader workload reliability (smoke PASS; real claude-orchestra sessions not yet run)
- Image-block conversion live validation (waiting for vision-capable model)
- Grammar-constrained generation fallback (conditional on Gemma reliability findings; likely unnecessary)
- `disable_parallel_tool_use` semantic parity across providers
- `is_error: true` semantic parity across providers
- Helper extraction to a dedicated module (trigger: when a second consumer appears)

See `docs/TODO.md` for full detail on each deferred item.

---

## 9. Implementation reference

- **`main.py`**: `_ANTHROPIC_API_BASE`, `_PROXY_EXPOSED_MODELS`, `_resolve_proxy_model()`,
  `_build_proxy_model_entry()`, `proxy_model_info()`, `proxy_models()`,
  `proxy_chat_completions()`, `anthropic_messages()` (with `proxy.blocked_models` check),
  `_anthropic_messages_forward()`, `_anthropic_messages_litellm()`,
  `_anthropic_stop_reason()`, `_TOOL_ID_VALID`, `_sanitize_tool_use_id()`,
  `_claude_code_alias_for()`, `_claude_code_alias_to_public()`, `_display_name_for()`,
  `_LITELLM_ROUTED`, `anthropic_count_tokens()` (token counting endpoint)
- **`router.py`**: `SmartRouter` — used by Cline path and LiteLLM local path;
  3-step timeout backoff for `ollama-cloud/*`; `enable_pre_call_checks` and
  `context_window_fallbacks` wired from `config.yaml`
- **`config.yaml`**: `model_list` entries; `proxy.blocked_models`;
  `router_settings.enable_pre_call_checks: true`; `litellm_settings.context_window_fallbacks`
- **`utils/alias_bijection_test.py`**: validates `_claude_code_alias_for()` ↔ `_claude_code_alias_to_public()` bijection
- **`docs/claude-orchestra-handoff.md`**: deployment runbook, tier recommendations, and integration checklist for claude-orchestra
- **`docs/TODO.md`**: deferred work — local-model tool-use implementation plan
