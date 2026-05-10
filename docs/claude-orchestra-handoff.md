---
title: "claude-orchestra integration — handoff brief"
created_at: 2026-05-10--17-52
created_by: Claude Code (Claude Haiku 4.5)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-10--20-51
context: >
  Self-contained brief produced by a SoHoAI /brain run on 2026-05-10. Captures
  the orchestra-side changes needed to consume SoHoAI's new claude-code-*
  alias scheme. This document is the sole interface between the two projects;
  no other code or config in claude-orchestra needs to know about SoHoAI internals.
  Updated 2026-05-10--19-48: /v1/models no longer includes anthropic/* entries —
  only the 5 non-Anthropic models (1 local + 4 ollama-cloud) are returned.
  Anthropic models remain accessible via Claude Code's native built-in list.
---

# claude-orchestra integration — handoff brief

## Implementation status — 2026-05-10

Phases A + B + C have shipped to the claude-orchestra pipeline:

- **Phase A (pricing.yaml + tier annotations)**: Orchestra now references SoHoAI's `pricing.yaml` for LiteLLM-routed model costs; tier annotations (fast/default/heavy) are ready for future per-step usage.
- **Phase B (actor-heavy + planner-long)**: Subagent frontmatter updated to use cost-saving models (Qwen3 Coder for Actor, DeepSeek V4 Pro for Planner; Reviewer remains Sonnet 4.6 per design constraints).
- **Phase C (Reviewer stays Sonnet 4.6)**: Reviewer tier locked to `claude-sonnet-4-6` (Anthropic) for statistical parity with design verification; cost optimization applies only to Actor and Planner.

This document is retained as the alias-contract / caveats reference. See `docs/design-history.md §Amendment 2026-05-10` in claude-orchestra for implementation details.

---

## 1. Purpose and alias scheme contract

SoHoAI's `GET /v1/models` endpoint publishes **non-Anthropic models only** under a stable
`claude-code-*` alias scheme. Anthropic models are intentionally excluded — Claude Code
already knows `claude-haiku-4-5`, `claude-sonnet-4-6`, and `claude-opus-4-7` natively, and
including them caused duplicate entries in the `/model` picker with gateway-synthesized
metadata. They continue to route correctly via `ANTHROPIC_BASE_URL` without needing a
discovery entry.

**Models returned by `GET /v1/models` (non-Anthropic only):**

| Backend | Public ID form (legacy) | claude-code-* alias | Notes |
|---------|---|---|---|
| llama-server | `internal/gemma-4-e4b` | `claude-code-gemma-4-e4b` | LiteLLM-routed |
| Ollama Cloud | `ollama-cloud/deepseek-v4-pro` | `claude-code-deepseek-v4-pro` | LiteLLM-routed; reasoning |
| Ollama Cloud | `ollama-cloud/kimi-k2.6` | `claude-code-kimi-k2.6` | LiteLLM-routed; reasoning |
| Ollama Cloud | `ollama-cloud/glm-5.1` | `claude-code-glm-5.1` | LiteLLM-routed; reasoning |
| Ollama Cloud | `ollama-cloud/qwen3-coder-next` | `claude-code-qwen3-coder-next` | LiteLLM-routed; coding |

Anthropic models (`claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7`) remain fully
available for subagent frontmatter — they come from Claude Code's native built-in list and
route through the gateway transparently.

**Stability invariant**: The `claude-code-{suffix}` scheme is the sole coupling between SoHoAI and claude-orchestra. Subagent frontmatter should use these full IDs (no env-var remapping, no shell aliases) to ensure deterministic model selection across sessions and operator machines.

**Bijection guarantee**: SoHoAI maintains a testable bijection between alias and backend via `utils/alias_bijection_test.py`. The test verifies that each alias round-trips correctly through `_claude_code_alias_for()` and `_claude_code_alias_to_public()` in `main.py`.

---

## 2. Recommended subagent frontmatter changes

The table below maps each claude-orchestra pipeline tier to:
- The default (Anthropic) model
- A recommended cost-saving alternative
- Rationale for the trade-off

**Prerequisites**: All subagent frontmatter examples assume:
- `ANTHROPIC_BASE_URL=http://192.168.1.93:8000` in `~/.claude/settings.json`
- Model names are full `claude-code-*` aliases (not public_id forms)
- Reasoning models require `max_tokens ≥ 500` in the request

### Table: Tier recommendations

| Tier | Pipeline | Default (Anthropic) | Recommended cost-saving alternative | Rationale |
|---|---|---|---|---|
| **Brain** | /brain | `claude-opus-4-7` | Not applicable | Harness constraint: Brain tier must remain Anthropic for statistical parity with claude-orchestra's design verification |
| **Planner** | /brain | `claude-sonnet-4-6` | `claude-code-deepseek-v4-pro` | Sonnet 4.6 depth (planning multi-agent workflows) is comparable to DeepSeek V4 Pro reasoning. Saves ~$0.05–0.15/run. Requires `max_tokens ≥ 500`. |
| **Actor** | /brain, /duo | `claude-haiku-4-5` | `claude-code-qwen3-coder-next` or `claude-code-deepseek-v4-pro` | Haiku covers ~95% of single-file edits and bash execution. Qwen3 Coder is optimized for code tasks (coding > general reasoning). DeepSeek V4 Pro for complex reasoning/debugging. Cost ≈ $0 vs $0.01/session. Requires `max_tokens ≥ 500` for reasoning variant. |
| **Reviewer** | /brain | `claude-sonnet-4-6` | `claude-code-glm-5.1` | Reviewer scans plan + implementation diff; needs reasoning but not creative planning. GLM-5.1 balances cost and coherence. Saves ~$0.03–0.05/run. Requires `max_tokens ≥ 500`. |

### Frontmatter examples

**Example 1: /duo Actor with cost-saving Qwen3 Coder**

```yaml
---
name: actor
description: "Actor — implementation. Fast, code-optimized tier for single-file edits and bash tasks."
model: claude-code-qwen3-coder-next
tools: [Read, Edit, Write, Bash, Glob, Grep, TodoWrite]
---
```

**Example 2: /brain Planner with cost-saving DeepSeek V4 Pro**

```yaml
---
name: planner
description: "Planner — multi-agent workflow design and orchestration."
model: claude-code-deepseek-v4-pro
tools: [Read, Bash]
---
```

**Example 3: /brain Reviewer with cost-saving GLM-5.1**

```yaml
---
name: reviewer
description: "Reviewer — validates plan and implementation against requirements."
model: claude-code-glm-5.1
tools: [Read, Bash]
---
```

---

## 3. Caveats

### Reasoning model token requirements

Reasoning models (deepseek-v4-pro, glm-5.1, kimi-k2.6) allocate tokens to internal reasoning before emitting visible output. If the request's `max_tokens` limit is too low, the model may exhaust tokens during the thinking phase and return empty content with `stop_reason: max_tokens`.

**Minimum requirement**: `max_tokens ≥ 500` on all requests using reasoning models.

If your orchestration tool (e.g. claude-orchestra's Agent tool invocation) has a default `max_tokens`, ensure it is raised to at least 500 when dispatching to reasoning models, or request will fail silently with empty responses.

### Ollama Cloud Pro session and weekly caps

Ollama Cloud models operate under usage limits:
- **Session caps**: Each model has a per-session token budget. Multiturn sessions consume the budget faster.
- **Weekly caps**: Per-API-key limits reset on a weekly cycle (day of week varies by signup date).

Monitor `~/.claude/settings.json` `OLLAMA_API_KEY` usage via the Ollama Cloud dashboard. If a session hits the cap, the proxy returns a 429 error. Recommended recovery: fall back to local Gemma 4 or Anthropic Haiku for that session.

SoHoAI's telemetry (usage_tracker.py) logs model and token count per request, enabling post-hoc analysis of cap exhaustion patterns.

### Self-identification as Claude under claude-orchestra system prompt

When an `ollama-cloud/*` model processes a request that includes claude-code's system prompt ("You are Claude Code..."), the model follows the persona and self-identifies as Claude. The actual model identity is confirmed by:
- The status bar model indicator in the Claude Code UI
- Server logs on Server 1 (`uvicorn` stdout shows `model=claude-code-...`)
- The `model_used` field in the response body (if returned by `/v1/messages`)

Model self-description in generated text is not a reliable model identifier.

### No prompt caching on LiteLLM path

Anthropic prompt caching (reducing recurring token costs by ~90% after turn 1) is available only on the transparent forward path (`anthropic/*` models). The LiteLLM conversion path (`claude-code-internal/*`, `claude-code-ollama-cloud/*`) strips `cache_control` markers because local models do not understand Anthropic's caching directives.

This is not a cost problem — LiteLLM-routed models either have zero per-token cost (local Gemma) or cost a fraction of Anthropic Haiku; the 10x cache discount does not apply. It is a consideration for task design: if a sub-agent must process the same 100K-token context repeatedly, the transparent path (Anthropic models) is cheaper per-turn due to caching, while the LiteLLM path (Qwen/DeepSeek/Gemma) is cheaper per-session because the fixed cost is lower overall.

---

## 4. Per-step tier annotation (deferred)

A proposed future enhancement to claude-orchestra would allow Planner to annotate each step in `PLAN.md` with a recommended tier:

```yaml
# Step 1
task: "..."
tier: fast          # use lightweight/local model (Actor-default or Actor-light)
---

# Step 5
task: "..."
tier: default       # use standard Actor (Haiku / Qwen Coder)
---

# Step 12
task: "..."
tier: heavy         # use costly/capable model (Sonnet / DeepSeek) for complex reasoning
```

Brain would then read `tier:` and dispatch each step's Actor invocation to a correspondingly-tuned model.

**Defer condition**: This annotation schema is postponed until real orchestration telemetry shows FIX-loop costs (repeated step failures requiring re-planning). A heavy-actor tier is only justified if such costs materialize and the heavier model demonstrably reduces failures. See claude-orchestra's own `docs/TODO.md §0` decision-gate framework.

Until telemetry is collected and analyzed, the static frontmatter recommendation (§2) remains the default.

---

## 5. Post-merge deployment runbook

Follow these steps after merging the SoHoAI PR containing the alias scheme into the main worktree.

### 5.1 Restart uvicorn and verify model discovery

On Server 1 (192.168.1.93):

```bash
# Restart uvicorn (if running under systemd, use systemctl; if manual, SIGTERM and restart)
# The FastAPI app reloads automatically if using --reload

# Verify GET /v1/models shape — should contain ONLY claude-code-* aliases (5 models total)
curl -s http://192.168.1.93:8000/v1/models | python3 -m json.tool | grep '"id"'

# Expected output — exactly 5 entries, no bare anthropic IDs:
#   "id": "claude-code-gemma-4-e4b",
#   "id": "claude-code-deepseek-v4-pro",
#   "id": "claude-code-kimi-k2.6",
#   "id": "claude-code-glm-5.1",
#   "id": "claude-code-qwen3-coder-next",
#
# NOTE: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7 are intentionally absent.
# They are served from Claude Code's native built-in list, not from gateway discovery.
# Seeing them here would indicate a regression — the duplicate-picker bug is back.
```

### 5.2 Apply Claude Code settings.json env-var addition (if not already done in-session)

This step ONLY applies if Claude Code settings.json does not yet have the env-var entry. Check:

```bash
cat ~/.claude/settings.json | grep -A 5 'CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY'
```

If the env-var is missing, add it. Requires Claude Code v2.1.129 or later. **WARNING: Live edit — restart Claude Code after modifying settings.json.**

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.1.93:8000",
    "ANTHROPIC_API_KEY": "<your-real-anthropic-api-key>",
    "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY": "1"
  }
}
```

Restart Claude Code (or reload settings):

```bash
# Verify settings are loaded by checking version and env
claude --version    # must be >= 2.1.129
claude env | grep CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
```

### 5.3 Verify picker populates with claude-code-* aliases

In Claude Code, open the model picker (`/model`):

Expected layout — two distinct groups, no duplicates:

- **Native Claude Code entries** (from Claude Code's built-in list, not from gateway):
  `Default (Opus 4.7)`, `Sonnet`, `Sonnet (1M context)`, `Haiku`

- **From gateway** (from `GET /v1/models` — exactly 5 entries):
  `Gemma 4 E4B (local, 110k ctx)`, `Deepseek V4 Pro (Ollama Cloud, ...)`,
  `Kimi K2.6 (Ollama Cloud, ...)`, `Glm 5.1 (Ollama Cloud, ...)`,
  `Qwen3 Coder Next (Ollama Cloud, ...)`

**What to watch for:**
- If `Claude Haiku 4 (Anthropic, ...)`, `Claude Sonnet 4 (Anthropic, ...)`, or
  `Claude Opus 4 (Anthropic, ...)` appear under "From gateway": **regression** —
  `/v1/models` is again returning `anthropic/*` entries. Check `main.py:list_models()`.
- If no gateway entries appear at all: `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` is
  not set or the gateway is unreachable.

```bash
# If CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=0 or unset, models won't populate
# Check with:
claude env | grep CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY
```

### 5.4 Run alias bijection test

On Server 1 or any machine with Python 3.12 + venv access:

```bash
cd ~/Gin-AI/projects/SoHoAI
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
python utils/alias_bijection_test.py
```

Expected output:

```
Testing bijection: _claude_code_alias_for() ↔ _claude_code_alias_to_public()
  ✓ internal/gemma-4-e4b         → claude-code-gemma-4-e4b         → internal/gemma-4-e4b
  ✓ ollama-cloud/deepseek-v4-pro → claude-code-deepseek-v4-pro    → ollama-cloud/deepseek-v4-pro
  ...
All bijections verified.
```

### 5.5 Spot-check token counting endpoint

Claude Code pre-flight token estimation calls `POST /v1/messages/count_tokens` before dispatching requests. Verify it works:

```bash
curl -X POST http://192.168.1.93:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: <any-non-empty-string>" \
  -d '{
    "model": "claude-code-qwen3-coder-next",
    "messages": [
      {"role": "user", "content": "hello world"}
    ]
  }' | python3 -m json.tool
```

Expected response:

```json
{
  "input_tokens": 13,
  "output_tokens": 1024
}
```

For Anthropic models, the endpoint forwards the request to `api.anthropic.com/v1/messages/count_tokens`. For LiteLLM-routed models, it uses `litellm.token_counter()` with fallback to `len(text) // 4` for unknown tokenizers. Both paths should return a valid response with `input_tokens` and `output_tokens` fields.

### 5.6 Attempt /duo dispatch with claude-code-* model in frontmatter

In Claude Code, create a test agent file with a `claude-code-*` model:

```yaml
---
name: test-actor
model: claude-code-qwen3-coder-next
tools: [Read, Bash]
---

You are a test agent. List the current directory.
```

Run `/duo` or dispatch the agent manually. Verify:

1. The agent receives the correct model identifier (`claude-code-qwen3-coder-next`)
2. SoHoAI's proxy routes it to the LiteLLM path (check Server 1 logs: `model=claude-code-qwen3-coder-next` or `model=ollama-cloud/qwen3-coder-next`)
3. The agent successfully executes Bash commands (Bash tool is forwarded and works)
4. Streaming responses are received correctly

If the agent fails, check:
- `ANTHROPIC_BASE_URL` is set and correct
- `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` is set
- Server 1 uvicorn is running
- Ollama API key (`OLLAMA_API_KEY` in SoHoAI `.env`) is valid (for Ollama cloud models)

---

## Appendix: Integration checklist

Use this checklist to track deployment progress:

- [ ] Merge SoHoAI PR to main worktree
- [ ] Restart uvicorn on Server 1
- [ ] Verify `GET /v1/models` returns exactly 5 `claude-code-*` aliases (no `anthropic/*` entries)
- [ ] Update `~/.claude/settings.json` with `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` (if not already done)
- [ ] Restart Claude Code
- [ ] Check model picker populates with aliases
- [ ] Run `python utils/alias_bijection_test.py` — PASS
- [ ] Curl `POST /v1/messages/count_tokens` with a `claude-code-*` model — expect valid response
- [ ] Dispatch test /duo agent with `model: claude-code-qwen3-coder-next` — verify execution succeeds
- [ ] Review logs for `model=` entries confirming correct routing
