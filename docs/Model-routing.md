---
title: "SoHoAI Model routing — Cline and OpenCode integration"
created_at: 2026-05-04--14-50
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Haiku 5.5)
updated_at: 2026-11-12--17-03
context: >
  Describes the stateless model-routing surface of the SoHoAI gateway on Server 1
  (192.168.1.93) as it stands after the 2026-11-12 cleanup: the OpenAI-compatible
  /proxy/v1 endpoints used by Cline and OpenCode, and the /upstream transparent
  forwarder used for the audio engines on Server 2. Claude Code no longer goes
  through SoHoAI (it calls Anthropic directly since 2026-11-05). Earlier versions
  of this document covered the Anthropic Messages passthrough (/v1/messages),
  KV-cache slots and usage telemetry; those are removed and recorded in
  docs/design-history.md. Claims were re-checked against main.py, router.py and
  SoHoAI-config.yaml on 2026-11-12; the config was read from the working tree,
  which had uncommitted edits at that time.
---

# SoHoAI model routing — /proxy/v1 and /upstream

## Scope and removals

This document covers the gateway's model-routing surface only:

- `/proxy/v1/*` — stateless OpenAI-compatible endpoints for Cline and OpenCode (§1–§9).
- `/upstream/{model}/{path}` — byte forwarder to llama-swap for the audio engines (§10).

The gateway also serves `/health`, `/`, `/v1/models/health` and `/v1/rag/search`. Those are
covered in `CLAUDE.md` (RAG) and in §1 below (health).

**Removed 2026-11-12** (history in `docs/design-history.md`):

- `POST /v1/messages`, `/v1/messages/beta`, `/v1/messages/count_tokens` — the Claude Code `ANTHROPIC_BASE_URL` passthrough, both the transparent Anthropic forward and the LiteLLM Anthropic↔OpenAI conversion path.
- Claude Code configuration (`ANTHROPIC_BASE_URL`, `api_base_url` in sub-agent frontmatter) and the `/duo` and `/brain` tier cost tables that depended on the passthrough.
- LiteLLM tool-use conversion notes (`_convert_tools`, `_convert_tool_choice`, `_convert_assistant_message`, `_convert_user_message`, `_sanitize_tool_use_id`) and the XML tool-use sentinel.
- Stateful `POST /v1/chat/completions` and `GET/DELETE /v1/chats*` — Redis, SQLite, rolling summarization.
- KV-cache slot targeting and the native llama-server `/completion` local path.
- `proxy.blocked_models` (already unenforced since 2026-10-09).
- `GET /v1/usage/stats`, `UsageTracker` and `usage_events` cost recording.
- `litellm.register_model()` pricing overrides (the Fable 5.1 cache rate and the Haiku 5.5 >100K tier). Prices now come from LiteLLM's own cost map (§8).
- The `/v1/messages` validation harness `utils/tool_use_smoke_test.py`, and the historical per-model smoke results that it produced.

## Contents

1. Endpoints
2. Model list, aliases and name resolution
3. Routing, fallbacks and failure behaviour
4. Client setup (Cline, OpenCode) and request parameters
5. Reasoning effort
6. Ollama Cloud reliability and timeouts
7. Anthropic prompt caching on /proxy/v1
8. Prices
9. Implementation reference and `model_info` semantics
10. Audio engines — transparent forwarder

---

## 1. Endpoints

| Method | Path | Purpose | Handler (`main.py`) |
|---|---|---|---|
| GET | `/proxy/v1/models` | OpenAI-style model list: every key of `_PROXY_EXPOSED_MODELS` | `proxy_models()` |
| GET | `/proxy/v1/model/info` | LiteLLM-style entries (`model_name`, `litellm_params`, `model_info`) | `proxy_model_info()` |
| POST | `/proxy/v1/chat/completions` | Stateless OpenAI chat completions; SSE when `stream: true` | `proxy_chat_completions()` |
| GET, POST | `/upstream/{model}/{path}` | Byte forwarder to llama-swap, audio engines only | `upstream_audio_forwarder()` |
| GET | `/v1/models/health` | Sends a 1-token completion to **every** `model_list` entry (see warning below) | `SmartRouter.health_check()` |
| GET | `/health` | `{"status": "ok", "models": [...]}` — the router's `available_models` (no Redis key) | `health()` |

> **Warning — `/v1/models/health` is not free.** `SmartRouter.health_check()` loops over every
> `model_list` entry, including the paid `anthropic/*` and `ollama-cloud/*` entries, and sends
> each one a real 1-token request. It also counts as a chat request for llama-swap, so calling it
> evicts the audio engines and the resident 9B (see §10). Do not poll it during meetings.

---

## 2. Model list, aliases and name resolution

### 2.1 Models exposed on `/proxy/v1`

The public ID is what a client sends. The internal alias is the `model_list` `model_name` the
router uses. Context figures come from the `model_info` of the entry that
`_build_proxy_model_entry()` reads.

| Public ID | Internal alias | Backend | Key | Context in `/proxy/v1/model/info` |
|---|---|---|---|---|
| `local/qwen3.5-9b-q4` | same | llama-swap on 192.168.1.95:8000 → Qwen3.5-9B Q4, `--reasoning off` | `dummy-key` | 131,072 |
| `local/qwen3.5-9b-agentic` | same | same llama-swap process; sends `chat_template_kwargs.enable_thinking: true` via `extra_body` | `dummy-key` | 131,072 |
| `anthropic/claude-haiku-5-5` | `claude-haiku-5-5` | Anthropic API | `ANTHROPIC_API_KEY` (bare entry) | **none** (see §2.3) |
| `anthropic/claude-sonnet-5-5` | `claude-sonnet-5-5` | Anthropic API | `ANTHROPIC_API_KEY` (bare entry) | **none** (see §2.3) |
| `anthropic/claude-opus-5-5` | `claude-opus-5-5` | Anthropic API | `ANTHROPIC_API_KEY` (bare entry) | 1,000,000 |
| `anthropic/claude-fable-5` | `claude-fable-5` | Anthropic API | `ANTHROPIC_API_KEY` (bare entry) | **none** (see §2.3) |
| `anthropic/claude-fable-5-1` | `claude-fable-5-1` | Anthropic API | `ANTHROPIC_API_KEY` (bare entry) | **none** (see §2.3) |
| `ollama-cloud/deepseek-v4.1-flash` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |
| `ollama-cloud/deepseek-v4-pro` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |
| `ollama-cloud/minimax-m3` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |
| `ollama-cloud/kimi-k2.7-code` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 256,000 |
| `ollama-cloud/kimi-k3` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |
| `ollama-cloud/glm-5.3` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |
| `ollama-cloud/glm-5.3-flash` | same | https://ollama.com/v1 | `OLLAMA_API_KEY` | 1,000,000 |

Local and Ollama entries use `openai/<name>` as the LiteLLM provider with a custom `api_base`.
Their `allowed_openai_params` setting is covered in §5.

### 2.2 The `anthropic/*` entries in `model_list`

`SoHoAI-config.yaml` holds two kinds of Anthropic entry:

- `anthropic/<name>` — carries no `api_key`. `_build_proxy_tables()` maps the public name
  `anthropic/<name>` to the bare internal name `<name>`, so proxy traffic never resolves to these
  entries. They are still loaded by `SmartRouter` and pinged by `/v1/models/health`.
- `<name>` (bare) — carries `api_key: os.environ/ANTHROPIC_API_KEY`. This is the entry that serves
  proxy requests.

### 2.3 Known gap: `model_info` missing on most bare Anthropic entries

`_build_proxy_model_entry()` copies `model_info` from the **bare** entry, not from the
`anthropic/*` entry. As of 2026-11-12 only `claude-opus-5-5` has `model_info` among the bare
Anthropic entries. For `claude-haiku-5-5`, `claude-sonnet-5-5`, `claude-fable-5` and
`claude-fable-5-1`, `/proxy/v1/model/info` returns `model_info: {}`. Clients that read
`max_input_tokens` or `context_window` from there (Cline does, for its context display) fall back
to their own defaults. Fix by adding `model_info` to those four bare entries in
`SoHoAI-config.yaml`. This document does not make that change.

### 2.4 Name resolution — `_resolve_proxy_model()`

A client may send any of the forms below. The lookup order is the code's order:

1. Meta-alias from `proxy.aliases` — currently only `local` → `local/qwen3.5-9b-q4`. Validated against `model_list` at import by `_build_proxy_aliases()`.
2. Exact public name in `_PROXY_EXPOSED_MODELS` (built by `_build_proxy_tables()`).
3. Legacy bare name in `_LEGACY_ALIASES` — the `local/`, `ollama-cloud/` and `anthropic/` prefixes stripped, and bare `model_list` names mapped to themselves.
4. The name with any provider prefix (`openai/`, `anthropic/`, `ollama-cloud/`, `local/`) stripped, retried against steps 2–3.
5. The raw internal alias, passed through.
6. The name with a trailing `[…]` annotation and a trailing `-YYYYMMDD` date removed, then retried from step 1.

Anything else returns **HTTP 400** with the accepted names listed.

Results of calling `_resolve_proxy_model()` directly on the current config (2026-11-12):

| Sent | Resolves to |
|---|---|
| `local` | `local/qwen3.5-9b-q4` |
| `qwen3.5-9b-q4` | `local/qwen3.5-9b-q4` |
| `openai/qwen3.5-9b-q4` | `local/qwen3.5-9b-q4` |
| `anthropic/claude-sonnet-5-5` | `claude-sonnet-5-5` |
| `claude-sonnet-5-5[1m]` | `claude-sonnet-5-5` |
| `claude-opus-5-5-20251001` | `claude-opus-5-5` |
| `kimi-k3` | `ollama-cloud/kimi-k3` |
| `claude-sonnet-5` | **None → HTTP 400** (renamed to `claude-sonnet-5-5`) |
| `local/qwen3-asr` | **None → HTTP 400** (audio models are not chat models; use §10) |

Renamed or retired names:

- `claude-sonnet-5` is now `claude-sonnet-5-5`. The bare old name no longer resolves.
- `ollama-cloud/qwen3-coder-next` was retired by Ollama Cloud on 2026-07-15 (`410 Gone`) and removed from config.
- `ollama-cloud/kimi-k2.7` (plain tag) returned `404` on 2026-07-30. Ollama split it into `kimi-k2.7-code` and `kimi-k3`.

---

## 3. Routing, fallbacks and failure behaviour

### 3.1 Which model a request uses

`proxy_chat_completions()` resolves the public name (§2.4) and always passes it to
`SmartRouter.complete(model=...)`. In `SmartRouter.select_model()` an explicit `model` wins
over everything else.

The rest of `select_model()` (`routing.default_model`, `routing.cloud_model`,
`routing.complexity_threshold_tokens`) runs only when `model` is `None`. No remaining caller
does that: the two `/proxy/v1` handlers pass a resolved model, and the RAG
`variant_llm_fn` in `main.py` passes `rag.multi_query.variant_model`. These routing keys are
therefore inert for current traffic. `routing.default_model` is still required at startup
(`SmartRouter.__init__` raises if it is missing). `routing.summarization_model` is not read
by any code.

### 3.2 LiteLLM fallbacks — `router_settings.fallbacks`

LiteLLM retries a failed deployment (`Router(num_retries=2, timeout=60)`), then tries the
fallback for that name. Current rules:

| Requested | Falls back to |
|---|---|
| `ollama-cloud/minimax-m3` | `anthropic/claude-opus-5-5` |
| `ollama-cloud/kimi-k3` | `anthropic/claude-sonnet-5-5` |
| `ollama-cloud/kimi-k2.7-code` | `anthropic/claude-sonnet-5-5` |
| `ollama-cloud/glm-5.3` | `anthropic/claude-sonnet-5-5` |
| `ollama-cloud/glm-5.3-flash` | `anthropic/claude-sonnet-5-5` |

`ollama-cloud/deepseek-v4-pro` and `ollama-cloud/deepseek-v4.1-flash` have **no** fallback, so
they fail outright.

> **Silent cost switch.** A failed call to one of the five names above is answered by an
> Anthropic model and billed to SoHoAI's `ANTHROPIC_API_KEY`. The client is not told. The
> non-streaming handler overwrites the response's `model` with the name the client sent
> (`resp_dict["model"] = reported_model`), and the streaming handler does the same for each
> chunk. Nothing else in the gateway records which model answered. Check the Anthropic usage
> dashboard if an Ollama model's calls look expensive.

This is a per-request fallback, so a model that recovers on the next turn answers again. That
can mix models within one agent task.

### 3.3 Retries and timeouts (`SmartRouter.complete()`)

- **Non-Anthropic targets** (`ollama-cloud/*`, `local/*`) get three attempts with increasing
  request timeouts, 60 s → 90 s → 120 s. Only `litellm.Timeout` triggers a retry. Auth errors
  and 4xx responses propagate at once. Worst case is 270 s in total. After the third timeout,
  `complete()` logs `all 3 attempts timed out` and re-raises.
- **Anthropic targets** get one call with no extra timeout, so LiteLLM's `timeout=60` from the
  `Router` constructor applies.

Errors reach the client as follows:

| Path | After the final failure |
|---|---|
| Non-streaming | Exception propagates. `main.py` defines no exception handler, so FastAPI returns its default HTTP 500. |
| Streaming (`stream: true`) | HTTP 200 is already sent. The handler emits `data: {"error": {"message", "type"}}` followed by `data: [DONE]`. |

**Idle-stream watchdog.** Every streaming response from `complete()` is wrapped in
`_with_idle_timeout()`. If no chunk arrives for `IDLE_CHUNK_TIMEOUT_S` (60 s), it raises
`StreamStalledError`. The streaming handler reports that as the SSE error event above. This
guards against an upstream that opens a stream and then goes silent.

---

## 4. Client setup and request parameters

### 4.1 Cline (VSCode plugin)

Choose **LiteLLM** as the provider, not "OpenAI Compatible":

```
Base URL : http://192.168.1.93:8000/proxy/v1
API Key  : sohoai-local     (any non-empty string)
Model    : local/qwen3.5-9b-q4 (131K ctx)   or   claude-sonnet-5-5 (1M ctx)
```

- The Base URL **must end in `/v1`**. Cline appends `/chat/completions`, and only
  `/proxy/v1/chat/completions` is served. `http://192.168.1.93:8000/proxy` alone returns 404 on
  chat (verified 2026-10-10).
- The API key must be non-empty, because Cline rejects an empty string on the client side.
- Cline's model dropdown is built from `/proxy/v1/models`. This was observed earlier and not
  re-checked on 2026-11-12.
- Cline was previously observed to read context limits from `/proxy/v1/model/info`. Entries
  without `model_info` (§2.3) give it no limit. Not re-checked on 2026-11-12.

To check model info by hand:

```bash
curl http://192.168.1.93:8000/proxy/v1/model/info | python3 -m json.tool | grep -E "model_name|max_input_tokens"
```

### 4.2 OpenCode

OpenCode reads a provider from its own config, `~/.config/opencode/opencode.json` on each
OpenCode host. That file is outside this repository and is deployed from oconona. The relevant
block as of 2026-11-12:

- Provider `sohoai`, npm package `@ai-sdk/openai-compatible`.
- `options.baseURL`: `http://192.168.1.93:8000/proxy/v1`, `apiKey`: `sohoai-local`,
  `chunkTimeout: 45000`, `headerTimeout: 20000`.
- Models map to `ollama-cloud/deepseek-v4.1-flash`, `ollama-cloud/deepseek-v4-pro`,
  `ollama-cloud/minimax-m3`, `ollama-cloud/kimi-k3`, `ollama-cloud/glm-5.3`,
  `ollama-cloud/glm-5.3-flash`, `local/qwen3.5-9b-q4`, and `anthropic/claude-haiku-5-5`.

A separate provider named `anthropic` in the same file calls Anthropic directly and does not
pass through SoHoAI.

### 4.3 Request parameters

`proxy_chat_completions()` consumes `model`, `messages` and `stream`, and drops the
SoHoAI-internal keys `force_cloud`, `chat_id` and `user_id`. Every other OpenAI-standard
parameter goes unchanged to LiteLLM through `**body`. That covers `tools`, `tool_choice`,
`max_tokens`, `temperature`, `reasoning_effort`, and `user`.

Whether a model acts on `tools` depends on the provider. This document makes no claim for
any model. Earlier smoke records came from `utils/tool_use_smoke_test.py`, which was removed on
2026-11-12.

Ollama Cloud reasoning models (DeepSeek V4, Kimi K3 / K2.7-Code, GLM-5.x, MiniMax M3) spend
tokens on hidden reasoning before visible output. Use `max_tokens ≥ 500`. At very low limits
such as 20, the reply can come back with empty visible content (observed on the Anthropic-format
path before 2026-11-12; not re-checked on the OpenAI-format path).

---

## 5. Reasoning effort

Cline's reasoning setting is sent as the OpenAI `reasoning_effort` field and passes through
unchanged. Who honours it (probed directly, 2026-10-10):

| Model family | Honours `reasoning_effort`? | Notes |
|---|---|---|
| `ollama-cloud/*` | Yes | Ollama maps it to its native think level. `none` disables thinking. Reasoning text comes back in `reasoning`. Only `minimax-m3` was compared across levels; `glm-5.3*` returned no reasoning text at `low`. |
| `anthropic/*` bare entries | Yes | LiteLLM ≥ 1.104 translates it to adaptive thinking with an effort setting. The repo pins `litellm==1.104.2`. Older LiteLLM sent a legacy `thinking.type=enabled` that Anthropic rejects for these models. |
| `local/qwen3.5-9b-q4`, `local/qwen3.5-9b-agentic` | **No — accepted and ignored** | llama-server takes the field without error, and any level behaves the same. Thinking is on/off only: off for `local/qwen3.5-9b-q4` (`--reasoning off`), on for `local/qwen3.5-9b-agentic` (`extra_body.chat_template_kwargs.enable_thinking`). |

**Why `allowed_openai_params` is set.** Non-Anthropic models are routed as `openai/<name>`
with a custom `api_base`. LiteLLM allows `reasoning_effort` only for OpenAI's own reasoning
models, so any other `openai/*` deployment raised `UnsupportedParamsError`. Every `local/*` and
`ollama-cloud/*` entry in `SoHoAI-config.yaml` therefore carries
`allowed_openai_params: ["reasoning_effort"]`. A global `drop_params: true` was rejected because
it would also silently drop the reasoning level the user asked for. **Any new `openai/*` entry
in `model_list` needs the same line.**

---

## 6. Ollama Cloud reliability and timeouts

These are observations, not guarantees. Ollama shared inference carries no SLA.

- **`ollama-cloud/deepseek-v4-pro`** had a ~70% HTTP 500/503 failure rate on Ollama's shared
  tier, reported in April 2026 (Ollama GitHub issues #15832, #15934). A live run on 2026-05-11
  failed with a 503. It has no LiteLLM fallback (§3.2).
- **`ollama-cloud/kimi-k3`** returned `402 Payment Required` on 2026-07-30 ("extra usage"
  balance empty). A later check, recorded in the operator's notes on 2026-09-18, reported it
  live. Re-probe before relying on it. While it fails, its fallback bills Sonnet 5.5 (§3.2).
- Retry and timeout behaviour for non-Anthropic targets is in §3.3. The timeouts were raised
  from 30/60/90 s to 60/90/120 s on 2026-05-15, after logs showed `kimi-k2.7` routinely needed
  30–60 s on complex coding requests.

---

## 7. Anthropic prompt caching on /proxy/v1

`SmartRouter.complete()` adds Anthropic `cache_control: {type: ephemeral}` markers only when the
target's `litellm_params.model` starts with `anthropic/` (`_uses_anthropic()`). The helper
`_apply_cache_control()` places two markers:

1. On the system message, when it is a plain string. This is the long-lived anchor.
2. On `messages[-2]`, when there are at least three messages. This is the rolling prefix anchor.

Messages whose content is already a list of blocks are left alone. The helper returns a new list
and does not mutate the caller's messages. Ollama and `local/*` targets receive the messages
unchanged.

A caching test on 2026-05-04 showed `cache_creation_input_tokens: 0` on the account, which
suggested caching was not active. That result is dated and was not re-run on 2026-11-12.

---

## 8. Prices

SoHoAI records no cost any more (see the removals list). These figures are LiteLLM's native cost
map for the installed `litellm==1.104.2`, read on 2026-11-12. They have not been checked against
Anthropic's price page. Units are USD per million tokens.

| Model (bare name) | Input | Cached read | Output |
|---|---|---|---|
| `claude-opus-5-5` | $4.00 | $0.20 | $20.00 |
| `claude-sonnet-5-5` | $2.00 | $0.10 | $10.00 |
| `claude-haiku-5-5` | $0.10 | $0.01 | $0.50 |
| `claude-fable-5` | $10.00 | $1.00 | $50.00 |
| `claude-fable-5-1` | $10.00 | $0.25 | $50.00 |

- `local/*` runs on the operator's own hardware and costs $0 per token.
- Ollama Cloud prices are not held in this repo. Usage is billed against the Ollama account.

---

## 9. Implementation reference and `model_info` semantics

### 9.1 Code map

- **`main.py`**
  - Proxy tables: `_build_proxy_tables()` (→ `_PROXY_EXPOSED_MODELS`, `_LEGACY_ALIASES`), `_build_proxy_aliases()` (→ `_PROXY_ALIASES`).
  - Resolution and listing: `_resolve_proxy_model()`, `_build_proxy_model_entry()`, `proxy_models()`, `proxy_model_info()`.
  - Chat: `proxy_chat_completions()`, which also builds the `metadata` passed to LiteLLM.
  - Session tagging: `inject_orchestra_session_id` (middleware) sets `source: "orchestra"` in that metadata when an orchestra `.lck` file is live. Nothing in SoHoAI reads that metadata now that UsageTracker is gone.
  - Audio: `_build_audio_table()` (→ `_AUDIO_MODELS`, `_AUDIO_ACCEPTED_NAMES`), `upstream_audio_forwarder()`.
- **`router.py`** — `SmartRouter`: `select_model()`, `complete()`, `_with_idle_timeout()` (`IDLE_CHUNK_TIMEOUT_S`), `_uses_anthropic()`, `_apply_cache_control()`, `health_check()`, `available_models`.
- **`SoHoAI-config.yaml`**
  - `model_list`, `proxy.aliases`, `routing`, `router_settings.fallbacks`, `audio_models`.
  - `router_settings` sets only `routing_strategy` and `fallbacks`.
  - `litellm_settings` is `{}`.

### 9.2 `model_info` — what it does

`model_info` is never sent to a provider. It is read only by `_build_proxy_model_entry()`, which
copies it into `/proxy/v1/model/info`. Its fields are:

| Field | Used by |
|---|---|
| `max_input_tokens`, `context_window` | Client context-window display (Cline). |
| `max_tokens` | Output cap metadata. |

Not configured, so not active:

- `enable_pre_call_checks` is absent from `router_settings`, so `SmartRouter` passes `False`.
- `litellm_settings.context_window_fallbacks` is absent.

---

## 10. Audio engines (ASR + diarization) — transparent forwarder

Added 2026-10-07. Qwen3-ASR and Nemotron-3-Diarization run under llama-swap on Server 2 (see
`docs/design-history.md` 2026-10-03 and 2026-10-04). They are **not chat models**, so they are
kept out of `model_list` and out of `/proxy/v1`.

### 10.1 Endpoint

`ANY /upstream/{model}/{path}` on the orchestrator mirrors llama-swap's own
`/upstream/<model>/<path>` form. A client moves from llama-swap to SoHoAI by changing only the
host and port (`192.168.1.95:8000` → `192.168.1.93:8000`).

| Engine | SoHoAI URL | Forwarded to (llama-swap) |
|---|---|---|
| Diarization | `POST /upstream/nemotron-3-diarization/v1/audio/diarizations` (multipart `file`) | `/upstream/nemotron-3-diarization/v1/audio/diarizations` |
| ASR | `POST /upstream/qwen3-asr/v1/chat/completions` (`input_audio` content part) | `/upstream/qwen3-asr-1.7b-q8/v1/chat/completions` |

- **Names.** SoHoAI's short name for the ASR engine is `qwen3-asr`. llama-swap's key is
  `qwen3-asr-1.7b-q8`. The mapping is applied to the URL path only. The request body, including
  any `model` field, is never read or rewritten.
- **`local/` prefix.** `local/qwen3-asr` and `local/nemotron-3-diarization` are also accepted,
  as `/upstream/local/qwen3-asr/...`. `upstream_audio_forwarder()` takes the first path segment
  after `local` as the engine name.
- **Why ASR cannot use `/proxy/v1/chat/completions`.** That endpoint resolves names only through
  `model_list` (§2.4), so `local/qwen3-asr` returns HTTP 400. The ASR client must use the
  `/upstream/qwen3-asr/...` form.
- **Allowlist.** Only the two engines are forwarded. Any other name, including
  `qwen3.5-9b-q4` and `bge-reranker-v2-m3`, returns **404** listing the accepted names.

### 10.2 Behaviour

- **Pure byte forward.** There is no LiteLLM, no `SmartRouter`, no retry, no fallback to cloud
  models, and no usage record (the calls are not token-billed). Request headers pass through
  except `host`, `content-length` and hop-by-hop headers. The response streams back with the
  upstream status and headers, through `aiter_raw()`, so `content-encoding` stays consistent.
- **Cold swap is transparent.** llama-swap holds the request until the model is healthy
  (`healthCheckTimeout: 180`, a 30–60 s swap in practice) and returns 502 if the start fails.
  SoHoAI passes that 502 through. SoHoAI's own upstream timeout is
  `audio_models.timeout_seconds: 300`, which is longer than llama-swap's load window. Mapped
  errors: connect failure → 502, read timeout → 504.
- **No early reply.** llama-swap sends nothing while loading, so SoHoAI cannot either. The client
  must tolerate a silent wait of about 200 s, or warm the engine first.
- **GPU side effect (llama-swap, not SoHoAI).** Both engines are in the exclusive `asr` group, so
  a request evicts the resident 9B. The next chat request evicts them again, and the 9B takes
  about 48–64 s to cold-load.
- **`GET /v1/models/health` counts as a chat request.** It pings every `model_list` entry with
  a 1-token completion, including the 9B aliases, so calling it unloads Qwen3-ASR and the
  diarizer (observed 2026-10-07). It also pings the paid Anthropic and Ollama entries (§1). Do
  not poll it during meetings. For a liveness check that does not evict anything, use
  `curl http://192.168.1.95:8000/v1/models`.

### 10.3 Configuration

`SoHoAI-config.yaml`, top-level `audio_models:`:

- `upstream_base` — `http://${server2_ip}:8000`, with the placeholder substituted at import.
- `timeout_seconds` — must be a positive integer.
- `models` — keyed by short name, each with `public_name`, `upstream_key`, `kind` and
  `description`. `upstream_key` and `public_name` are required.

A malformed section raises at import time.

**Why a separate section and not `model_list`?** Every `model_list` entry is loaded by
`SmartRouter`, pinged by `/v1/models/health`, and exposed on `/proxy/v1/models` by
`_build_proxy_tables()`. For `qwen3-asr`, a ping would load the `asr` group and evict the 9B. A
separate section avoids all three.

### 10.4 Verification status

- Route behaviour (multipart and 3 MB bodies, name and `local/` mapping, 404 allowlist, 502
  pass-through, gzip, connection refused) was tested against a local stub upstream on
  2026-10-07.
- A live smoke test against llama-swap ran on 2026-10-07. The diarizer and ASR both responded
  correctly through SoHoAI, and an unsupported model failed fast with a 404. The results are
  not recorded here in detail.
- **Not yet tested:** cold swap-in latency through the forwarder, and a full ~100 MB
  diarization upload. The request body is buffered in memory and then re-sent upstream, so
  expect roughly twice the body size in RAM for that request.
- Not yet reported: the `/v1/models/health` and `/proxy/v1/models` regression, and the
  post-swap chat completion.
