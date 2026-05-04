---
title: "SoHoAI Proxy — Cline and Claude Code integration"
created_at: 2026-05-04--14-50
created_by: Claude Code (Claude Sonnet 4.6)
context: >
  SoHoAI exposes two stateless pass-through paths built on the same LiteLLM Router
  used by the main orchestrator. One path is OpenAI-compatible (for Cline VSCode plugin
  and any other OpenAI-API client). The other is Anthropic-compatible (for Claude Code
  using ANTHROPIC_BASE_URL). Both are implemented in main.py and share the same
  _PROXY_EXPOSED_MODELS mapping and SmartRouter instance.
---

# SoHoAI proxy — Cline and Claude Code integration

Both paths are **stateless**: no Redis, no SQLite, no RAG, no KV cache, no rolling
summarization. The caller manages its own history. Prompt caching via Anthropic
`cache_control` still applies on the external path (same `SmartRouter` instance).

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

`claude-sonnet-4-6` maps to `external` (not the direct alias) to preserve the
automatic Gemma fallback in case Anthropic is unreachable. The other two Anthropic
models use direct aliases and will hard-fail if Anthropic is down.

`_resolve_proxy_model()` accepts bare names, `anthropic/`-prefixed names, or `openai/`-prefixed names.

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

**Symptom**: if a chat looks corrupted after summarization, check whether Cline was
active simultaneously.

---

## 3. LiteLLM `model_info` parameter semantics

These three fields can appear in a `model_list` entry's `model_info` block in
`config.yaml`. None of them is ever forwarded to the provider API.

| Field | Controls | Forwarded to API? | Needed in config? |
|---|---|---|---|
| `max_tokens` | Maximum output tokens the model can generate per response | No | Only if you want LiteLLM to enforce a lower cap than the model's native limit |
| `max_input_tokens` | Input size guard — LiteLLM rejects requests that exceed this before sending | No | Only if you want a guard stricter than the model's native context window |
| `context_window` | Total context window metadata (input + output); used by router pre-call checks and cost tracking | No | Only for models not in LiteLLM's registry |

### When each is actually needed

**For well-known Anthropic models** (`anthropic/claude-opus-4-7`, `anthropic/claude-sonnet-4-6`,
`anthropic/claude-haiku-4-5`): LiteLLM auto-resolves all three from its internal
`model_prices_and_context_window.json` registry at startup. No `model_info` block is
needed — and none is present in `config.yaml` for those entries.

**For `internal` (Gemma / llama-server)**: llama-server is not in LiteLLM's registry.
`model_info` with explicit `max_tokens`, `max_input_tokens`, and `context_window` is
required so that:
- Cline reads the correct `max_input_tokens` from `/proxy/v1/model/info`
  (`refreshLiteLlmModels.ts:37`) to set its context-window display
- `enable_pre_call_checks` can enforce the 110,024-token per-slot limit

**For `external` (Sonnet via the main routing alias)**: same as above — explicit values
needed for the `/proxy/v1/model/info` response and pre-call checks.

### Who reads `max_tokens` on a request

Claude Code and Cline both send their own `max_tokens` in every request.
LiteLLM passes that value through transparently to the provider — it does **not**
override it with the `model_info.max_tokens` value unless `enable_pre_call_checks: true`
is set *and* the caller's value exceeds the configured limit. In practice the caller's
value always wins.

### `enable_pre_call_checks` in this setup

`router_settings.enable_pre_call_checks: true` is set in `config.yaml`.
This activates input-size enforcement: if a request's estimated input token count
exceeds `max_input_tokens` (from `model_info` or LiteLLM's registry), the router
rejects it before hitting the provider and attempts any configured
`context_window_fallbacks`. For the three direct Anthropic models this is a no-op in
practice (200K limit is rarely approached); for `internal`/`gemma-4-e4b` it prevents
oversized requests from reaching the 110,024-token slot limit.

---

## 2. Claude Code — Anthropic Messages API path

### Endpoint

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/messages` | Native Anthropic Messages API format; enables `ANTHROPIC_BASE_URL` |

Implements the Anthropic Messages API wire format: accepts `POST /v1/messages` with
`anthropic-version` headers and `system` / `messages` / `max_tokens` / `stream` fields.
Converts to OpenAI format internally, calls `SmartRouter.complete()`, converts response
back to Anthropic format (including proper SSE streaming: `message_start` →
`content_block_start` → `content_block_delta` → `content_block_stop` → `message_delta`
→ `message_stop`).

The same `_PROXY_EXPOSED_MODELS` mapping applies. Anthropic-prefixed model names
(`anthropic/claude-opus-4-7`) are accepted and stripped before lookup.

### Claude Code configuration

Set in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.1.93:8000"
  }
}
```

Claude Code will call `POST http://192.168.1.93:8000/v1/messages` for all model
requests. The proxy routes each model through the appropriate backend:

| Model | Routed via | Notes |
|---|---|---|
| `claude-opus-4-7` | Anthropic API (direct) | Brain / research tier |
| `claude-sonnet-4-6` | Anthropic API via `external` | Planner/Reviewer; Gemma fallback if cloud down |
| `claude-haiku-4-5` | Anthropic API (direct) | Actor tier, cheapest/fastest |
| `gemma-4-e4b` | llama-server, Server 2 | Local-only, no cloud cost |

### Claude Code sub-agent frontmatter (alternative to ANTHROPIC_BASE_URL)

If you only want specific sub-agents to route through SoHoAI (rather than all Claude
Code traffic), use `api_base_url` in the agent frontmatter instead:

```yaml
# In agent definition file
model: claude-opus-4-7
api_base_url: http://192.168.1.93:8000/proxy
api_key: sohoai-local
```

This uses the OpenAI-compatible `/proxy/v1/chat/completions` path, not `/v1/messages`.
Choose this when you want per-agent control; use `ANTHROPIC_BASE_URL` when you want
all Claude Code traffic to go through SoHoAI.

### Smoke-test

```bash
# Non-streaming
curl -s -X POST http://192.168.1.93:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: sohoai-local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"claude-haiku-4-5","max_tokens":32,"messages":[{"role":"user","content":"say hi"}]}' \
  | python3 -m json.tool

# Streaming
curl -s -X POST http://192.168.1.93:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'x-api-key: sohoai-local' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"claude-haiku-4-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"count to 3"}]}'
```

---

## Implementation notes

- **`main.py`**: `_PROXY_EXPOSED_MODELS`, `_resolve_proxy_model()`, `_build_proxy_model_entry()`, `proxy_model_info()`, `proxy_models()`, `proxy_chat_completions()`, `anthropic_messages()`, `_anthropic_stop_reason()`
- **`router.py`**: `SmartRouter` — both paths use the same instance; `enable_pre_call_checks` and `context_window_fallbacks` wired from `config.yaml router_settings` / `litellm_settings`
- **`config.yaml`**: `model_list` entries for `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5` (no `model_info` — LiteLLM registry covers these); `router_settings.enable_pre_call_checks: true`; `litellm_settings.context_window_fallbacks`
