---
title: "SoHoAI Usage and Billing Telemetry Pipeline"
created_at: 2026-05-06--00-00
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (claude-code-kimi-k2.6)
updated_at: 2026-05-16--08-41
context: >
  Cross-project design document for SoHoAI Stage 1 telemetry implementation.
  Goal: Add a complete usage and billing telemetry pipeline to SoHoAI so that
  ALL traffic routed through the SoHoAI proxy (Claude Code native, orchestra
  /brain and /duo, Cline, CLI chat) is tracked with token counts, cost estimates,
  and source attribution. This document serves as the specification for Stage 1
  (SoHoAI implementation in /brain session 20260506T154456Z-1605029) and the
  handoff checklist for Stage 2 (future claude-orchestra branch migration to
  query SoHoAI's telemetry API instead of parsing JSONL transcripts).
---

## §1 Purpose and scope

### Stage 1 (SoHoAI — this session)

SoHoAI builds an independent usage_events pipeline to log and report on ALL
inbound API traffic:

- Track every completion request, response token count, and cost
- Attribute traffic source (orchestra, Claude Code native, Cline, CLI chat)
- Tag orchestra sessions via X-Orchestra-Session-ID header for future cross-project accounting
- Expose `/v1/usage/stats` endpoint for querying aggregated usage by user, model, source, session, or time window
- Use LiteLLM's built-in cost calculator as single source of truth (provider-agnostic, self-updating)

Phases:
- **Phase 0** ✅: Design document (this file)
- **Phase 1** ✅: Add usage_events table to telemetry.db; wire LiteLLM success_callback
- **Phase 2** ✅: Source attribution via endpoint + X-Orchestra-Session-ID header
- **Phase 3** ✅: GET /v1/usage/stats endpoint

**Post-review fix 1 (2026-05-06):** `_anthropic_messages_forward()` (raw httpx path, bypasses
LiteLLM) initially omitted cache token costs. Fixed in commit `1a15abe`:
versioned model IDs have the date suffix stripped before `get_model_info()` lookup
(e.g. `claude-sonnet-4-6-20250219` → `claude-sonnet-4-6`), then
`cache_creation_input_token_cost` and `cache_read_input_token_cost` rates are fetched
and applied to the extracted cache token counts. Without this, heavy-caching orchestra
sessions (typical: ~1.5M cache_read tokens) would undercount cost by ~$0.45/session
on the forward path.

**Post-release fixes (2026-05-11, session fix-LiteLLM-telemetry):** Three bugs prevented
the `sohoai_api` cost source from returning data for orchestra sessions:

1. **Session attribution broken** — `X-Orchestra-Session-ID` header injection was never
   wired up in Claude Code (`apiHeaders` silently ignored; env-based injection doesn't work
   for subagents because they inherit the environment from before the session ID exists).
   Fixed by adding `inject_orchestra_session_id` FastAPI middleware that reads
   `~/.claude/active-sessions/<SESSION_ID>.lck` files (already written by `brain.md`'s setup
   block) on every incoming request and sets `request.state.orchestra_session_id`. All four
   call sites in `proxy_chat_completions`, `_anthropic_messages_litellm`, and
   `_anthropic_messages_forward` updated to read from `request.state` (with header fallback).

2. **Streaming requests not recorded** — `_anthropic_messages_forward()` streaming path
   piped bytes unchanged with no usage recording, so all streaming Anthropic requests (Brain
   parent Opus 4.7, Sonnet/Haiku subagents) produced zero records in `usage_events`. Fixed
   by side-parsing SSE `message_start` and `message_delta` events into a usage accumulator
   while forwarding bytes unchanged, then recording the event in the generator's `finally`
   block after stream completion (or interruption).

3. **LiteLLM has wrong Opus 4.7 rates** — LiteLLM's model registry carries approximately
   1/3 of the correct rates for `claude-opus-4-7` (e.g. `cache_creation_input_token_cost`
   = 6.25e-6 vs. correct 18.75e-6). Fixed in `main.py` via `litellm.register_model()` at
   module level, overriding the wrong rates globally. Also fixed in `telemetry-summarize.py`
   (claude-orchestra) where `query_litellm_cost()` now accepts `pricing_data` and always
   prefers pricing.yaml cache rates for known models over LiteLLM's registry.

**Post-review fix 2 (2026-05-06):** `litellm.completion_cost()` was called with
`prompt_tokens=` and `completion_tokens=` kwargs that were removed in litellm 1.82.6.
The call threw `TypeError` at runtime; the `except` block silently swallowed it,
dropping all usage events on the forward path. Fixed by replacing the broken two-step
pattern (broken `completion_cost()` for base tokens + separate `get_model_info()` for
cache tokens) with a single `get_model_info()` call covering all four token types:

```python
_info = litellm.get_model_info(_normalized_model)
cost = (
    input_tokens  * (_info.get("input_cost_per_token") or 0.0)
    + output_tokens * (_info.get("output_cost_per_token") or 0.0)
    + cache_creation_tokens * (_info.get("cache_creation_input_token_cost") or 0.0)
    + cache_read_tokens     * (_info.get("cache_read_input_token_cost") or 0.0)
)
```

### Stage 2 (claude-orchestra — future session, out of scope here)

Once SoHoAI telemetry stabilises, claude-orchestra's T2 pipeline will:

1. Migrate from JSONL parsing to querying SoHoAI's `/v1/usage/stats` API
2. Transition pricing calculations from static pricing.yaml to `litellm.completion_cost()`
3. Tag /brain and /duo calls with X-Orchestra-Session-ID header so SoHoAI can attribute them

See §6 Stage 2 Checklist for full handoff requirements.

---

## §2 usage_events table schema

Stored in `telemetry.db` alongside existing tables. All columns present from initial
creation; fields are nullable where indicated.

```sql
CREATE TABLE usage_events (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id           TEXT NOT NULL UNIQUE,        -- UUID per API call
  created_at           TEXT NOT NULL,               -- ISO-8601 timestamp
  source               TEXT NOT NULL,               -- orchestra|claude_code_native|cline|cli_chat
  user_id              TEXT,                        -- nullable; from request user field
  chat_id              TEXT,                        -- nullable; SoHoAI chat_id if applicable
  orchestra_session_id TEXT,                        -- nullable; from X-Orchestra-Session-ID header
  model                TEXT NOT NULL,               -- normalized model id (e.g. claude-sonnet-4-6)
  input_tokens         INTEGER NOT NULL DEFAULT 0,
  output_tokens        INTEGER NOT NULL DEFAULT 0,
  cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
  cache_read_tokens    INTEGER NOT NULL DEFAULT 0,
  cost_usd             REAL NOT NULL DEFAULT 0.0,   -- from litellm.completion_cost()
  provider             TEXT                         -- nullable; anthropic|openai|google|local
);

-- 5 indexes for query performance
CREATE INDEX idx_usage_events_created_at ON usage_events(created_at);
CREATE INDEX idx_usage_events_user_created ON usage_events(user_id, created_at);
CREATE INDEX idx_usage_events_source_created ON usage_events(source, created_at);
CREATE INDEX idx_usage_events_session_id ON usage_events(orchestra_session_id);
CREATE INDEX idx_usage_events_model_created ON usage_events(model, created_at);
```

### Column semantics

- **request_id**: UUID generated per completion request (e.g., from FastAPI request context or LiteLLM callback)
- **created_at**: ISO-8601 timestamp (UTC) when response was logged
- **source**: Request origin — inferred from endpoint (§3)
  - `orchestra`: X-Orchestra-Session-ID header present
  - `claude_code_native`: /v1/messages endpoint, no orchestration header
  - `cline`: /proxy/v1/chat/completions with appropriate context
  - `cli_chat`: Local CLI invocation
- **user_id**: User identifier from request `user` field (optional; nullable for stateless proxy calls)
- **chat_id**: SoHoAI's internal chat ID, if conversation is stored (NULL for /v1/messages passthrough)
- **orchestra_session_id**: Session directory basename from X-Orchestra-Session-ID header (e.g., `20260506T154456Z-1605029`); NULL if not an orchestra request
- **model**: Normalized model identifier (requested model normalized to canonical form, e.g., claude-sonnet-4-6)
- **input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens**: From LiteLLM response usage object
- **cost_usd**: Floating-point cost in USD, calculated via `litellm.completion_cost(completion_response=response)` (§4)
- **provider**: Provider enum (anthropic, openai, google, local) — allows filtering by vendor

---

## §3 Source attribution rules

All completion requests entering SoHoAI are attributed to a source via endpoint
and request headers. This enables filtering usage by origin (e.g., cost breakdown by
orchestra vs. Claude Code vs. Cline).

### Endpoint-based source inference

| Endpoint | Source | Header | Notes |
|----------|--------|--------|-------|
| `/v1/messages` | claude_code_native | (none) | Default; Claude Code Anthropic API passthrough |
| `/v1/messages` | orchestra | X-Orchestra-Session-ID present | /brain or /duo request tagged with session ID |
| `/proxy/v1/chat/completions` | cline | (depends on context) | Cline VSCode plugin via LiteLLM OpenAI endpoint |
| `/proxy/v1/chat/completions` | claude_code_native | (if Claude Code uses this path) | Not currently used; documented for completeness |
| `/v1/chat/completions` | cli_chat | (none) | Local CLI chat client (utils/cli_chat.py) |

### Orchestra session ID — middleware (current) vs. header (original design)

**Current (2026-05-11+)**: SoHoAI reads `~/.claude/active-sessions/<SESSION_ID>.lck`
on every request via `inject_orchestra_session_id` middleware. No client-side header
injection required. The `orchestra_session_id` column in `usage_events` is populated
for the duration of any active `/brain` or `/duo` session.

**X-Orchestra-Session-ID header**: Still accepted as fallback for any future
client-side injection. The middleware takes precedence.

**Session ID format**: SESSION_DIR basename, e.g., `20260510T204552Z-2645364`.

### Source enum

Always one of: `orchestra`, `claude_code_native`, `cline`, `cli_chat`

Rationale for separate enums (not endpoint-based filtering):
- Enables future cross-project cost reports ("What did Cline cost this week?")
- Survives endpoint refactoring (if e.g. Cline path changes, source is stable)
- Clearer semantics than raw HTTP metadata

---

## §4 Pricing: LiteLLM as single source of truth

All cost calculations use `litellm.completion_cost(completion_response=response)`.

### Mechanics

```python
from litellm import completion_cost

# After LLM call completes (or in success_callback):
cost_usd = completion_cost(completion_response=response)
# response: CompletionResponse object from LiteLLM (has model, usage fields)

# Insert into usage_events:
db.execute(
  "INSERT INTO usage_events (..., cost_usd, ...) VALUES (..., ?, ...)",
  (cost_usd,)
)
```

LiteLLM internally queries its own pricing database (mirrors Anthropic, OpenAI,
Google, etc.) and returns total cost in USD.

### Local models

For local models (Qwen3.5 / "internal"), `litellm.completion_cost()` returns 0.0
(correct — no API charges). To shadow-cost local inference for internal
accounting, add an optional `local_model_pricing` block to SoHoAI-config.yaml:

```yaml
local_model_pricing:
  qwen3-4b:
    input_per_1m_tokens: 0.0
    output_per_1m_tokens: 0.0
```

(Not required for Stage 1; included for future extensibility.)

### Cloud models and provider detection

LiteLLM identifies the provider (anthropic, openai, google) from model name and
API key routing. The `provider` column in usage_events is populated from
`response.model` or inferred from cost lookup result.

### Warning for zero costs on cloud models

Log a WARNING-level message whenever `completion_cost()` returns 0.0 for a model
that should not be free:

```python
if cost_usd == 0.0 and response.model not in ['internal', 'qwen3-4b']:
  logger.warning(f"Zero cost returned for cloud model {response.model}; "
                 f"pricing may be outdated or model unknown to LiteLLM")
```

This catches gaps when LiteLLM's pricing DB lags behind new model releases.

### Consistency with claude-orchestra pricing.yaml

In Stage 2, orchestra's T2 cost calculation will migrate from its static
`pricing.yaml` to `litellm.completion_cost()`. Until then:

- **SoHoAI** (Stage 1, now): Uses LiteLLM
- **claude-orchestra** (main): Uses pricing.yaml (unchanged until Stage 2 branch)

Both ultimately ground-truth in the same upstream Anthropic/OpenAI pricing data,
just accessed differently. Once Stage 2 completes, both pipelines will use
LiteLLM, with pricing.yaml as a fallback for unknown models only.

---

## §5 `/v1/usage/stats` endpoint contract

REST API for querying aggregated usage and cost data. Implements all filtering
and grouping logic server-side; returns JSON.

### Request

```
GET /v1/usage/stats
```

#### Query parameters (all optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | string | (none) | Filter by user_id |
| `since` | ISO-8601 | 7 days ago | Start of time window (inclusive) |
| `until` | ISO-8601 | now | End of time window (inclusive) |
| `model` | string | (none) | Filter by model (exact match or comma-separated list) |
| `source` | string | (none) | Filter by source enum (comma-separated: orchestra,claude_code_native,cline,cli_chat) |
| `session_id` | string | (none) | Filter by orchestra_session_id (Stage 2 T2 primary use case) |
| `group_by` | enum | (none) | Grouping strategy: `day`, `model`, `source` (optional) |

#### Examples

```
# Last 7 days, all traffic
GET /v1/usage/stats

# User "florian", last 30 days, grouped by model
GET /v1/usage/stats?user=florian&since=2026-04-06&group_by=model

# Orchestra session costs
GET /v1/usage/stats?session_id=20260506T154456Z-1605029

# Cost breakdown by source for a specific model
GET /v1/usage/stats?model=claude-sonnet-4-6&group_by=source
```

### Response

```json
{
  "window": {
    "since": "2026-04-29T00:00:00Z",
    "until": "2026-05-06T23:59:59Z"
  },
  "totals": {
    "requests": 127,
    "input_tokens": 450280,
    "output_tokens": 89432,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 0,
    "cost_usd": 1.23,
    "cache_hit_rate": 0.0
  },
  "by_model": [
    {"model": "claude-sonnet-4-6", "requests": 100, "input_tokens": 400000, "output_tokens": 80000, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 1.00},
    {"model": "qwen3-4b", "requests": 27, "input_tokens": 50280, "output_tokens": 9432, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.0}
  ],
  "by_source": [
    {"source": "claude_code_native", "requests": 80, "input_tokens": 300000, "output_tokens": 60000, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.75},
    {"source": "cli_chat", "requests": 47, "input_tokens": 150280, "output_tokens": 29432, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.48}
  ],
  "by_day": [
    {"date": "2026-04-29", "requests": 18, "input_tokens": 64000, "output_tokens": 12800, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.175},
    {"date": "2026-04-30", "requests": 20, "input_tokens": 71000, "output_tokens": 14200, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.194},
    {"date": "2026-05-06", "requests": 89, "input_tokens": 315280, "output_tokens": 62432, "cache_creation_tokens": 0, "cache_read_tokens": 0, "cost_usd": 0.861}
  ]
}
```

#### Response semantics

- **window**: Query time range applied (defaults filled in)
- **totals**: Aggregated metrics across entire window
  - **cache_hit_rate**: `cache_read_tokens / (cache_creation_tokens + cache_read_tokens)` (0.0 if denominator is 0)
  - **cost_usd**: Sum of all cost_usd values in filtered result set
- **by_model**: Array sorted by cost descending; present if any grouping applied or top-k extraction
- **by_source**: Array sorted by cost descending
- **by_day**: Array in chronological order; **only present if `group_by=day`** was specified in request

All numeric fields are JSON numbers (not strings).

---

## §6 Stage 2 checklist for claude-orchestra

These items are **OUT OF SCOPE** for the SoHoAI Stage 1 session but are the
direct responsibility of the Stage 2 claude-orchestra branch. This checklist is
provided so the future implementer has a clear spec.

### 1. Session attribution — orchestra session ID tagging

**STATUS: DONE (2026-05-11) — implemented via filesystem middleware, not header injection.**

Header injection (`X-Orchestra-Session-ID`) was the original design but proved unreliable:
Claude Code silently ignores `apiHeaders` in agent definitions, and environment-based
injection (`ANTHROPIC_CUSTOM_HEADERS`) fails for subagents because they inherit the
environment from before the session ID was created.

**Actual implementation**: `brain.md`'s setup block already writes
`~/.claude/active-sessions/<SESSION_ID>.lck` (e.g. `20260510T204552Z-2645364.lck`).
SoHoAI's `inject_orchestra_session_id` FastAPI middleware reads these files on every
request, identifies the active orchestra session, and sets `request.state.orchestra_session_id`.
All proxy/messages endpoints consume `request.state` (with `X-Orchestra-Session-ID`
header as fallback for any future client-side injection).

**Verification**: Start a `/brain` session; confirm `~/.claude/active-sessions/<ID>.lck`
exists; call `GET /v1/usage/stats?session_id=<ID>` — `totals.cost_usd` should be non-zero.

### 2. Migrate T2 cost calculation from pricing.yaml to litellm.completion_cost()

**Current state**: claude-orchestra/telemetry-summarize.py parses JSONL
transcripts, looks up model rates from pricing.yaml, calculates cost manually.

**New state**: Use LiteLLM's cost calculator instead.

**Steps**:
1. Add `litellm` to orchestra dependencies (if not already present)
2. In telemetry-summarize.py, replace:
   ```python
   # OLD:
   cost = pricing_yaml[model]['input'] * input_tokens / 1_000_000 + ...
   ```
   with:
   ```python
   # NEW:
   from litellm import completion_cost
   usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, ...}
   cost = completion_cost(model=model, completion_response=response_with_usage)
   ```
3. Keep pricing.yaml as fallback for models LiteLLM doesn't recognize (wrap in try/except)
4. **Test parity**: Run T2 report on 1 week of historical JSONL with both old and new cost calculation; verify costs agree within 5%

**Rationale**: LiteLLM is provider-agnostic, self-updating, and unifies pricing
logic across projects.

### 3. Query SoHoAI API as primary cost source (after stabilisation)

**Timeline**: After 4+ weeks of SoHoAI telemetry data (suggests: early June 2026).

**Verification**: Run one full T2 report cycle (6 sessions) confirming cost parity
with LiteLLM-calculated costs before cutting over.

**Steps**:
1. Modify telemetry-summarize.py to query `GET /v1/usage/stats?session_id=<ID>`
   from SoHoAI instead of parsing JSONL
2. Add fallback: if SoHoAI is unreachable/unavailable, fall back to JSONL parsing
3. Log which sessions used SoHoAI vs. fallback in telemetry-report.sh output (see item 4)
4. Once fallback is never triggered for N consecutive T2 report cycles (suggest N=10):
   remove JSONL parsing path entirely (~300 lines of code)

**Benefit**: Unified cost source, eliminates code duplication, enables real-time
cost visibility across projects.

### 4. Update telemetry-report.sh for transparency

Add a line to telemetry-report.sh output annotating which sessions have
SoHoAI-sourced cost data vs. JSONL-parsed:

```
Session: 20260506T154456Z-1605029
  Duration: 2h 15m
  Requests: 42
  Cost: $3.87 (source: SoHoAI)    ← SoHoAI-sourced
  
Session: 20260415T083012Z-1234567
  Duration: 1h 08m
  Requests: 18
  Cost: $1.42 (source: JSONL parse) ← Fallback during transition
```

This transparency is important during the transition period; once fallback is
removed, the annotation becomes boilerplate and can be simplified to "all costs
from SoHoAI".

---

## §7 Pricing comparison table

Current model rates from orchestra `pricing.yaml` (as of 2026-04-30) and their
LiteLLM equivalents. These should match exactly; discrepancies indicate an
urgent pricing.yaml update is needed.

### Anthropic models

| Model | Input (per 1M) | Output (per 1M) | Cache Create (per 1M) | Cache Read (per 1M) | Source |
|-------|---|---|---|---|---|
| claude-opus-4-7 | $15.00 | $75.00 | $18.75 | $1.50 | pricing.yaml + LiteLLM |
| claude-sonnet-4-6 | $3.00 | $15.00 | $3.75 | $0.30 | pricing.yaml + LiteLLM |
| claude-sonnet-4-5 | $3.00 | $15.00 | $3.75 | $0.30 | pricing.yaml (legacy alias) |
| claude-haiku-4-5 | $1.00 | $5.00 | $1.25 | $0.10 | pricing.yaml + LiteLLM |

### Local models

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| qwen3-4b | $0.00 | $0.00 | Local inference; litellm.completion_cost() returns 0 (correct) |
| internal (alias) | $0.00 | $0.00 | Fallback; treated as local |

### Notes

- **Cache creation** rates in pricing.yaml assume 5-minute TTL (1.25× base input rate per Anthropic docs). Long-cache (1h TTL, 2× input) is not yet tracked separately in SoHoAI.
- **Last verified**: 2026-04-30 against https://docs.anthropic.com/en/docs/about-claude/models/all-models
- **Frequency**: Update pricing.yaml and this table on every Anthropic price change; automate if possible via CI
- **LiteLLM rate discrepancy (2026-05-11)**: LiteLLM's registry carries ~1/3 of correct rates for `claude-opus-4-7` (cache_creation 6.25e-6 vs. correct 18.75e-6). Corrected in `main.py` via `litellm.register_model()` and in `telemetry-summarize.py` via pricing.yaml override in `query_litellm_cost()`. Check other new models when added — LiteLLM's database may lag.

---

## Appendix: Streaming and async considerations

### Token counts at stream end

For streaming responses (SSE /v1/messages or /v1/chat/completions with
stream=true), token counts and cost are only available after the stream
completes. Ensure LiteLLM success_callback or cost logging fires **after**
the final chunk is sent to the client (not at first chunk).

### Async safety

usage_events inserts must be async-safe. Use thread-local SQLite connections
or async wrappers (e.g., aiosqlite if SoHoAI moves to full async) to avoid
contention.

### Callback placement

Register LiteLLM success_callback in router.py's completion wrapper:

```python
import litellm

def log_usage(completion_response, *args, **kwargs):
  # Extract usage, cost, etc. and INSERT into usage_events
  pass

litellm.success_callback = [log_usage]
```

Ensure this fires for all code paths (error handling, streaming, etc.).
