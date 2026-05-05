---
title: "SoHoAI Proxy — Cline and Claude Code integration"
created_at: 2026-05-04--14-50
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Florian (manual)
updated_at: 2026-05-05--12-10
context: >
  SoHoAI exposes two stateless pass-through paths built on the same LiteLLM Router.
  One is OpenAI-compatible for Cline VSCode plugin. The other is Anthropic-compatible
  for Claude Code (ANTHROPIC_BASE_URL), with model-aware routing: Anthropic models use
  a transparent HTTP forward (full fidelity including tools and caching); local models
  (gemma-4-e4b) use the LiteLLM conversion path (text-only, no tool use currently).
  Future tool-use work for local models is tracked in docs/TODO.md.
---

# SoHoAI proxy — Cline and Claude Code integration

Both proxy paths are **stateless**: no Redis, no SQLite, no RAG, no KV cache, no rolling
summarization. The caller manages its own conversation history.

---

## 1. Cline VSCode plugin — OpenAI-compatible path

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/proxy/v1/models` | OpenAI-compatible model list |
| GET | `/proxy/v1/model/info` | LiteLLM-compatible model info (`max_input_tokens`, `context_window`) — Cline reads this to set its context-window display |
| POST | `/proxy/v1/chat/completions` | Stateless OpenAI chat completions, streaming supported |

### Model mapping (`_PROXY_EXPOSED_MODELS` in `main.py`)

| Public name (caller sends) | Internal router alias | Backend | Context window |
|---|---|---|---|
| `gemma-4-e4b` | `internal` | llama-server, Server 2, Gemma 4 E4B Q8_0 | 110,024 |
| `claude-haiku-4-5` | `claude-haiku-4-5` | Anthropic API | 200,000 |
| `claude-sonnet-4-6` | `external` | Anthropic API (with Gemma fallback) | 200,000 |
| `claude-opus-4-7` | `claude-opus-4-7` | Anthropic API | 200,000 |

`claude-sonnet-4-6` maps to `external` to preserve the automatic Gemma fallback
if Anthropic is unreachable. The other Anthropic models use direct aliases and
hard-fail if Anthropic is down.

`_resolve_proxy_model()` accepts bare names, `anthropic/`-prefixed, or `openai/`-prefixed forms.

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
  ├─ model resolves to "internal" (gemma-4-e4b)
  │     → _anthropic_messages_litellm()
  │     → Anthropic→OpenAI format conversion (text blocks only)
  │     → LiteLLM Router → llama-server (Server 2)
  │     → LIMITATION: tools stripped (see §2.3 and docs/TODO.md)
  │
  └─ model resolves to anything else (claude-*, or unresolved)
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

### §2.2 LiteLLM local path (gemma-4-e4b)

`_anthropic_messages_litellm()` in `main.py` converts the Anthropic-format request to
OpenAI format and routes through LiteLLM Router to llama-server on Server 2.

**Current limitation — tool use is not supported on this path.**

The conversion loop:
```python
if isinstance(content, list):
    content = "".join(
        b.get("text", "") for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )
```
extracts only `text` blocks and silently drops all other content block types.

**What gets dropped and why it matters:**

| Dropped | What the model loses | Practical consequence |
|---|---|---|
| `tools` array | No knowledge of Read/Write/Bash tools | Model responds with text only; cannot request file reads |
| `tool_use` blocks (assistant) | Loses its own prior tool-call requests | Cannot reason about what it already tried |
| `tool_result` blocks (user) | Loses all file contents and command outputs | Cannot use information from previous tool calls |
| `cache_control` markers | No prompt-cache benefit | Irrelevant for local inference (no API cost), but noted |

**Important**: this is a limitation of *our* conversion code, not of LiteLLM.
LiteLLM fully supports OpenAI-format tool calls and correctly translates them to the
provider's native format. If our conversion preserved tools (Anthropic→OpenAI format
translation, ~70 lines), tool use would work — subject to Gemma 4's reliability.
See `docs/TODO.md` for the full implementation plan and format-conversion spec.

**Appropriate use cases for the local path:**
- Simple text generation where tool use is not needed
- Summarization tasks (already used by SoHoAI's `maybe_summarize()`)
- Offline/background drafting when Anthropic is unreachable
- Cost-zero tasks where quality is secondary to API cost

**Inappropriate use cases:**
- Full Claude Code sessions (tools broken → files injected inline → expensive + slow)
- Any sub-agent that needs to read files, write code, or run bash commands
- Tasks requiring multi-turn tool call chains

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

### Local sub-agent (gemma-4-e4b) — current limitations

A sub-agent with `model: gemma-4-e4b` routes via the LiteLLM local path.
**Tool use is not currently supported on this path** (see §2.2 and `docs/TODO.md`).

Such an agent can:
- Generate text responses
- Perform summarization
- Draft content without needing to read/write files

Such an agent CANNOT:
- Read files (Read tool stripped)
- Write or edit files (Write/Edit tools stripped)
- Run bash commands (Bash tool stripped)
- Use sub-agents (Agent tool stripped)

Until tool support is implemented, use `claude-haiku-4-5` for Actor-tier tasks that
require file access. Cost is ~$0.01/session — negligible.

---

## 4. Alternative - Anthropic passthrough -- and comparison 


The problem: LiteLLM's standard proxy endpoint (`/v1/messages`) strips/transforms certain Anthropic-specific elements (cache_control markers, tools array, anthropic-beta headers) when it converts requests. 

This causes tool calls to be stripped—forcing Claude Code to fall back on inline file injection—and strips cache_control markers, eliminating prompt caching and incurring full costs every turn.


The general recommendation from practitioners is: Only use LiteLLM as a fallback router or when wrapping an OpenAI-compatible local endpoint that needs format translation.

For the Brain/Planner/Reviewer tiers in claude-orchestra that use Anthropic models, pointing directly at `api.anthropic.com` or LiteLLM's `/anthropic` passthrough is the correct path. 


LiteLLM's passthrough URL approach uses `ANTHROPIC_BASE_URL=http://localhost:8000/anthropic` to route Claude Code directly to LiteLLM's built-in passthrough endpoint, which forwards requests transparently without any transformation. SoHoAI's solution implements the same transparent relay by creating a custom HTTP endpoint that forwards the exact request body bytes without modification.

The real distinction is that SoHoAI operates at the server level with model-aware routing—Anthropic requests get forwarded transparently while local models get converted through LiteLLM—whereas the LiteLLM passthrough is purely Anthropic-focused with no branching logic, where the client -- e.g. claude code -- would use a custom URL, with no ability for the LiteLLM to do mode-aware routing 


### 4.1 In-depth comparison 

Both are two implementations of the same fix. In more detail: 

The root cause in both cases is identical: LiteLLM's standard `/v1/messages` endpoint **transforms** requests — it parses, modifies, and re-serialises them. In doing so it strips `cache_control` markers, drops the `tools` array, and does not forward `anthropic-beta` headers. Claude Code, receiving responses from this path, loses the SSE event types it needs for compact tool rendering and falls back to injecting file contents as scrolling inline text.

**LiteLLM's `/anthropic` passthrough** fixes this by telling LiteLLM: skip transformation entirely, relay bytes to `api.anthropic.com` unchanged.

**SoHoAI's `_anthropic_messages_forward()`** fixes this by implementing the same relay directly inside the FastAPI app — exact body bytes forwarded, `x-api-key` / `anthropic-version` / `anthropic-beta` headers preserved, raw SSE bytes streamed back.

###  4.2 Why SoHoAI's approach is strictly more capable

The critical difference is the **model-aware branching** at the `POST /v1/messages` handler:

```
model = "claude-*"  →  _anthropic_messages_forward()   (transparent relay)
model = "gemma-4-e4b"  →  _anthropic_messages_litellm()  (LiteLLM conversion)
```

LiteLLM's `/anthropic` passthrough endpoint has no such branching — it can only forward to Anthropic. It cannot route local models. SoHoAI's implementation handles both paths behind the same `ANTHROPIC_BASE_URL`, which is what allows `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`, and `gemma-4-e4b` to all be valid `model:` values in agent frontmatter while sharing one endpoint configuration in `settings.json`.

### 4.3 Observation regarding caching  The one finding worth noting

Section §5 below contains an important empirical result: prompt caching is **not currently active** on your account — `cache_creation_input_tokens: 0` and `inference_geo: not_available` on both proxy and direct Anthropic calls. 

The transparent forward is correctly preserving `cache_control` markers; the limitation is account-tier or region on Anthropic's side. 

This means the cost tables in §5 are the target state once caching activates, not the current state. The transparent forward implementation is correct and future-proof — no changes needed when caching becomes available.



---

## 5. Prompt caching and cost implications

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

## 6. LiteLLM `model_info` parameter semantics

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

## 7. Future work

Tool-use support for the local-model path (gemma-4-e4b and any future local LLMs)
is tracked with full implementation detail in **`docs/TODO.md`**.

---

## 8. Implementation reference

- **`main.py`**: `_ANTHROPIC_API_BASE`, `_PROXY_EXPOSED_MODELS`, `_resolve_proxy_model()`,
  `_build_proxy_model_entry()`, `proxy_model_info()`, `proxy_models()`,
  `proxy_chat_completions()`, `anthropic_messages()`, `_anthropic_messages_forward()`,
  `_anthropic_messages_litellm()`, `_anthropic_stop_reason()`
- **`router.py`**: `SmartRouter` — used by Cline path and LiteLLM local path;
  `enable_pre_call_checks` and `context_window_fallbacks` wired from `config.yaml`
- **`config.yaml`**: `model_list` entries; `router_settings.enable_pre_call_checks: true`;
  `litellm_settings.context_window_fallbacks`
- **`docs/TODO.md`**: deferred work — local-model tool-use implementation plan
