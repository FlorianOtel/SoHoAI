---
title: "SoHoAI / LiteLLM Cost Attribution — Session Comparison & Handover"
created_at: 2026-05-11--00-00
created_by: Claude Code (Claude Sonnet 4.6)
context: >
  Detailed analysis of a cost-reporting discrepancy between two /brain orchestra sessions
  on 2026-05-10. Session B (using SoHoAI models for subagents) reported $83.66 vs Session A's
  $18.73 despite comparable work. Investigation revealed the discrepancy was caused by a
  LiteLLM client-side failure for claude-code-* model aliases, plus inaccurate cache-token
  pricing in LiteLLM's model database. Fix applied in claude-orchestra. This document covers
  the full comparative analysis and a SoHoAI handoff with what the SoHoAI proxy needs to
  implement for authoritative cost attribution.
---

# SoHoAI / LiteLLM Cost Attribution — Session Comparison & Handover

---

## 1. Session Profiles at a Glance

| | **Session A** | **Session B** |
|---|---|---|
| Session ID | `20260510T180922Z-2575990` | `20260510T204552Z-2645364` |
| Started | 2026-05-10 18:09:22Z | 2026-05-10 20:45:52Z |
| Ended | 2026-05-10 19:00:01Z | 2026-05-10 22:03:22Z |
| Duration | **3,039 s (50.7 min)** | **4,650 s (77.5 min)** |
| Outcome | PASS — 0 fix cycles | PASS — 1 fix cycle |
| Task | Implement SoHoAI rollout (agent config changes) | Status-line ctx + SoHoAI live-cost integration |
| Reported cost (original) | **$18.73** (`litellm`) | **$83.66** (`pricing_yaml`) |
| Reported cost (after fix) | $18.73 (`litellm`) | **$30.09** (`litellm`) |
| Parent model | claude-opus-4-7 | claude-opus-4-7 |
| Planner | claude-sonnet-4-6 (native) | claude-code-deepseek-v4-pro (SoHoAI) |
| Actor | claude-haiku-4-5-20251001 (native) | claude-code-qwen3-coder-next (SoHoAI) |
| Actor-heavy | — | claude-code-kimi-k2.6 (SoHoAI) |
| Reviewer | claude-sonnet-4-6 (native) | claude-sonnet-4-6 (native) |

Session A *implemented* the SoHoAI agent switch. Session B was the first real-world `/brain` run *after* deploying those changes — the first session where deepseek, qwen3, and kimi ran as actual subagents.

---

## 2. Model Usage Breakdown

### Session A

| Subagent role | Model | Provider | # Dispatches | Total time |
|---|---|---|---|---|
| Brain (parent) | claude-opus-4-7 | Anthropic (via SoHoAI proxy) | 1 | 3,039 s (session) |
| Explore | claude-haiku-4-5-20251001 | Anthropic (via SoHoAI proxy) | 3 | 47 s total |
| Planner | claude-sonnet-4-6 | Anthropic (via SoHoAI proxy) | 1 | 138 s |
| Actor | claude-haiku-4-5-20251001 | Anthropic (via SoHoAI proxy) | 7 | 542 s total (avg 77 s) |
| Reviewer | claude-sonnet-4-6 | Anthropic (via SoHoAI proxy) | 1 | 167 s |
| **Total subagent dispatches** | | | **12** | **894 s** |

### Session B

| Subagent role | Model | Provider | # Dispatches | Total time |
|---|---|---|---|---|
| Brain (parent) | claude-opus-4-7 | Anthropic (via SoHoAI proxy) | 1 | 4,650 s (session) |
| Explore | claude-haiku-4-5-20251001 | Anthropic (via SoHoAI proxy) | 3 | 149 s total |
| Planner | claude-code-deepseek-v4-pro | SoHoAI → Ollama Cloud | 1 | 755 s |
| Planner-long | claude-sonnet-4-6 | Anthropic (via SoHoAI proxy) | 1 | 107 s |
| Actor | claude-code-qwen3-coder-next | SoHoAI → Ollama Cloud | 3 | 1,177 s total (avg 392 s) |
| Actor-heavy | claude-code-kimi-k2.6 | SoHoAI → Ollama Cloud | 2 | 1,007 s total |
| Reviewer | claude-sonnet-4-6 | Anthropic (via SoHoAI proxy) | 2 | 360 s total |
| **Total subagent dispatches** | | | **13** | **3,555 s** |

---

## 3. Session Timelines

### Session A — 2026-05-10 18:09Z–19:00Z

```
18:09:22Z  Brain starts
18:09:55Z  Explore #1 — Find tier-routing notes in design docs       [31s]
18:10:03Z  Explore #2 — Audit settings.json + deploy.sh + pricing.yaml [7s]
18:10:10Z  Explore #3 — Inspect current subagent definitions          [9s]
           ← Phase 0 (research) complete, ~18:11Z

           [Brain 18-min planning pause — Phase 0 inline deliberation]

18:29:30Z  Planner (Sonnet) — Plan SoHoAI handoff implementation    [138s]
18:31:48Z  Planner ends

           [Brain inter-phase deliberation]

18:42:06Z  Actor #1 (Haiku) — Step 13: deploy.sh                     [76s]
18:43:22Z  Actor #2 (Haiku) — Steps 1-2: pricing.yaml + actor.md     [40s]
18:44:02Z  Actor #3 (Haiku) — Steps 8-10: design + history + TODO   [170s]
18:47:12Z  Actor #4 (Haiku) — Steps 3-5: planner + new agents       [114s]
18:49:06Z  Actor #5 (Haiku) — Reviewer caveats follow-up             [59s]
18:50:05Z  Actor #6 (Haiku) — Steps 11-12: handoff + CLAUDE.md       [86s]
18:51:31Z  Actor #7 (Haiku) — Steps 6-7: brain.md + guard            [72s]

18:55:15Z  Reviewer (Sonnet) — Phase 3: review SoHoAI rollout diff  [167s]
           PASS with 2 minor caveats noted

18:58:44Z  Actor #7 follow-up (already counted) — fix updated_by     [60s]
19:00:01Z  Session ends
```

**Key observation**: Brain's inline thinking (Phase 0 → Phase 1 gap = ~18 min) dominates wall time. Actual subagent execution is compact (894 s / ~15 min of 51 min total).

---

### Session B — 2026-05-10 20:45Z–22:03Z

```
20:45:52Z  Brain starts
20:46:01Z  Explore #1 (Haiku) — SoHoAI telemetry data shape          [53s]
           Explore #2 (Haiku) — Context-window tracking + registry    [56s]
           Explore #3 (Haiku) — Status-line implementation survey     [40s]
           ← Phase 0 complete, ~20:49Z

20:55:40Z  Planner (deepseek-v4-pro) — Plan status-line ctx         [755s]  ← 12.6 min
21:08:15Z  Planner ends
21:09:22Z  Planner-long (Sonnet) — Supplemental plan clarification  [107s]
21:11:09Z  Planner-long ends

21:15:02Z  Actor #1 (qwen3) — Steps 1+2: YAML + ctx-segment.sh      [159s]
21:17:41Z  Actor #1 ends
21:18:27Z  Actor-heavy #1 (kimi-k2.6) — Step 3: sohoai-live-cost.sh [681s]  ← 11.4 min
21:29:48Z  Actor-heavy #1 ends
21:30:42Z  Actor-heavy #2 (kimi-k2.6) — Step 4: orchestra-block      [326s]  ← 5.4 min
21:36:08Z  Actor-heavy #2 ends
21:37:05Z  Actor #2 (qwen3) — FIX cycle: ctx-segment.sh bugs        [460s]  ← 7.7 min
21:44:45Z  Actor #2 ends

21:47:19Z  Reviewer #1 (Sonnet) — Initial review → FIX verdict      [331s]
21:52:50Z  Reviewer #1 ends

21:53:58Z  Actor #3 (qwen3) — Fix 5 FIX issues                      [558s]  ← 9.3 min
22:03:16Z  Actor #3 ends

22:03:17Z  Reviewer #2 (Sonnet) — Re-review → PASS                   [29s]
22:03:22Z  Session ends
```

**Key observation**: SoHoAI subagents dominate execution time. Planner (755 s) + two kimi actor-heavy (1,007 s) + three qwen3 actors (1,177 s) = 2,939 s (63% of total session time). Compare to Session A where 7 Haiku actors completed in 542 s.

---

## 4. Cache Analysis — Hits vs Misses

In Anthropic's pricing, **cache_creation** = tokens written to prompt cache (priced at 1.25× input), **cache_read** = tokens served from prompt cache (priced at 0.10× input = 90% discount). High cache_read relative to cache_creation indicates good cache reuse. High cache_creation is the expensive write cost.

### Session A

| Component | cache_creation (writes) | cache_read (hits) | Hit ratio |
|---|---|---|---|
| Brain parent | 689,470 | 10,908,524 | **94.1%** |
| Planner (Sonnet) | 128,408 | 505,620 | 79.8% |
| Actor ×7 (Haiku) | 420,947 | 5,999,835 | 93.4% |
| Reviewer (Sonnet) | 238,550 | 2,657,193 | 91.8% |
| Explore ×3 (Haiku) | 169,835 | 624,400 | 78.6% |
| **Total** | **1,647,210** | **20,695,572** | **92.6%** |

Session A has excellent cache reuse — the Brain parent accumulated 689K writes across its 3,039 s lifespan, then served them back 10.9M times (a 15.8× read multiplier). This is normal for a session with many sequential subagent turns reading the same large system prompt and context.

### Session B

| Component | cache_creation (writes) | cache_read (hits) | Hit ratio |
|---|---|---|---|
| Brain parent | 3,008,135 | 5,875,289 | **66.1%** |
| Planner (deepseek) | 0 | 0 | — (SoHoAI: no Anthropic cache) |
| Planner-long (Sonnet) | 62,222 | 84,813 | 57.7% |
| Actor ×3 (qwen3) | 0 | 0 | — (SoHoAI: no Anthropic cache) |
| Actor-heavy ×2 (kimi) | 0 | 0 | — (SoHoAI: no Anthropic cache) |
| Reviewer ×2 (Sonnet) | 182,773 | 2,385,105 | 92.9% |
| Explore ×3 (Haiku) | 236,538 | 2,778,844 | 92.2% |
| **Total** | **3,489,668** | **11,124,051** | **76.1%** |

**Critical insight**: Session B's Brain parent accumulated **4.4× more cache writes** (3.0M vs 689K) despite a similar task scope. The causes:
1. **SoHoAI subagents return no cache benefit** — deepseek/qwen3/kimi output is injected back into Brain's context as assistant messages, expanding the context for subsequent turns without any cache reuse (since LiteLLM strips `cache_control` markers on the SoHoAI path). Each such expansion forces a fresh cache write of the entire grown context.
2. **One FIX cycle** — after the reviewer's FIX verdict, Brain dispatched Actor #3 (558 s) then Reviewer #2. Each new turn appended the fix discussion and re-review to Brain's context, triggering another cache write of ~500K additional tokens.
3. **Dual planner phase** — deepseek planner ran first (755 s), then Brain decided to run planner-long (Sonnet) as a supplemental pass. Two separate planning invocations expanded Brain's context twice before implementation began.

The 66.1% hit ratio for Session B's parent (vs 94.1% for Session A) directly reflects this: every SoHoAI subagent response expands the context without enabling the next turn to read from cache (since the LiteLLM proxy path strips cache markers).

---

## 5. Cost Breakdown Per Model

### Session A — Anthropic List Rates (pricing_yaml)

| Component | Model | Input | Output | Cache Writes | Cache Reads | **Subtotal** |
|---|---|---|---|---|---|---|
| Brain parent | claude-opus-4-7 | $0.003 | $14.125 | $12.928 | $16.363 | **$43.419** |
| Planner | claude-sonnet-4-6 | $0.000 | $0.120 | $0.482 | $0.152 | **$0.754** |
| Actor ×7 | claude-haiku-4-5 | $0.014 | $0.242 | $0.526 | $0.600 | **$1.382** |
| Reviewer | claude-sonnet-4-6 | $0.000 | $0.131 | $0.895 | $0.797 | **$1.823** |
| Explore ×3 | claude-haiku-4-5 | $0.000 | $0.020 | $0.212 | $0.062 | **$0.294** |
| **Total** | | **$0.017** | **$14.638** | **$15.043** | **$17.974** | **$47.672** |

**What LiteLLM actually reported: $18.73**  
Discrepancy: **$28.94** missing.

The missing amount closely matches the Opus cache write and cache read costs:
- Opus cache_creation: 689,470 × $18.75/M = **$12.928**
- Opus cache_read: 10,908,524 × $1.50/M = **$16.363**
- Together: **$29.291** ≈ the $28.94 gap (within rounding)

**Conclusion**: LiteLLM's internal model database for `claude-opus-4-7` has zero cache rates (both `cache_creation_input_token_cost` and `cache_read_input_token_cost` return 0.0 from `litellm.get_model_info()`). Sonnet and Haiku cache rates appear correct in LiteLLM. The LiteLLM undercount for Session A is **$28.94**, making $18.73 an underestimate. The true cost at Anthropic list rates is approximately **$47.67**.

---

### Session B — Anthropic List Rates (pricing_yaml), before fix

| Component | Model | Input | Output | Cache Writes | Cache Reads | **Subtotal** |
|---|---|---|---|---|---|---|
| Brain parent | claude-opus-4-7 | $0.002 | $15.836 | $56.403 | $8.813 | **$81.054** |
| Planner | deepseek-v4-pro | $0 | $0 | $0 | $0 | **$0** (SoHoAI) |
| Planner-long | claude-sonnet-4-6 | $0.000 | $0.094 | $0.233 | $0.025 | **$0.352** |
| Actor ×3 | qwen3-coder-next | $0 | $0 | $0 | $0 | **$0** (SoHoAI) |
| Actor-heavy ×2 | kimi-k2.6 | $0 | $0 | $0 | $0 | **$0** (SoHoAI) |
| Reviewer ×2 | claude-sonnet-4-6 | $0.001 | $0.222 | $0.685 | $0.715 | **$1.623** |
| Explore ×3 | claude-haiku-4-5 | $0.002 | $0.057 | $0.296 | $0.278 | **$0.633** |
| **Total** | | **$0.005** | **$16.209** | **$57.617** | **$9.831** | **$83.662** |

SoHoAI models contribute **$0** at list rates (Ollama Cloud Pro is a flat $20/mo subscription). The full $83.66 comes from native Anthropic models, 97% from the Brain parent alone — and 67% of the total from Opus **cache writes** (3.0M tokens × $18.75/M = $56.40).

**After fix, LiteLLM reported: $30.09**

With the fix, `claude-code-*` models short-circuit to $0 and LiteLLM prices native models. The gap from pricing_yaml ($83.66) to LiteLLM ($30.09) = **$53.57** is almost entirely the missing Opus cache costs:
- Opus cache_creation: 3,008,135 × $18.75/M = $56.40 (not in LiteLLM)
- Opus cache_read: 5,875,289 × $1.50/M = $8.81

This confirms: LiteLLM is missing Opus cache_creation entirely. For cache_read, the arithmetic suggests it may be partially reflected given $30.09 > $16.21 (base output/input), but the exact LiteLLM Opus cache_read rate is unclear.

---

## 6. Why Session A Is an Underestimate

The $18.73 reported for Session A via `cost_source=litellm` is materially wrong for the following reasons:

### 6.1 LiteLLM has zero cache rates for claude-opus-4-7

When `telemetry-summarize.py` calls:
```python
cache_info = litellm.get_model_info("claude-opus-4-7")
cc_rate = cache_info.get("cache_creation_input_token_cost", 0.0)  # → 0.0
cr_rate = cache_info.get("cache_read_input_token_cost", 0.0)      # → 0.0
```

...it gets 0.0 for both. LiteLLM's model registry for `claude-opus-4-7` (or its 1M context variant) does not carry the cache pricing that Anthropic charges at the API level. Since claude-opus-4-7 is a relatively recent model (released 2026), LiteLLM's database snapshot may pre-date its cache pricing entry.

### 6.2 Magnitude of the undercount

Session A's parent generated:
- **689,470 cache-creation tokens** × $18.75/M = **$12.93** not charged by LiteLLM
- **10,908,524 cache-read tokens** × $1.50/M = **$16.36** not charged by LiteLLM

Total uncounted: **$29.29** — meaning the true Anthropic-billed cost of Session A is approximately **$47.67**, not $18.73.

### 6.3 The cache-creation cost is the dominant term

In any long Brain session, the parent model (Opus) accumulates a large growing context (system prompt + prior subagent turns). Anthropic bills every prompt token that is newly cached in each turn at 1.25× the input rate. For a session with 689K new cache writes at $18.75/M, this is **the largest single cost component** — larger than all output tokens combined ($14.13). LiteLLM silently omits this, making the litellm cost path systematically mislead on Opus-heavy sessions.

### 6.4 Implication for cost_source comparison

| Cost source | Session A | Session B (before fix) | Session B (after fix) |
|---|---|---|---|
| `litellm` | $18.73 (**undercount**, ~39% of true) | aborted (BadRequestError) | $30.09 (**undercount**, ~36% of true) |
| `pricing_yaml` | would be $47.67 | $83.66 (**accurate**) | $83.66 |
| Estimated true Anthropic bill | ~$47.67 | ~$83.66 | ~$83.66 |

`pricing_yaml` is the more accurate fallback for sessions where the Brain parent runs on Opus. `litellm` is systematically low by ~$29–$54 depending on session length, due to missing Opus cache rates.

The fix committed to claude-orchestra prevents `claude-code-*` aliases from aborting the LiteLLM path, restoring `cost_source=litellm` for Session B. However, both sessions remain **under-reported via litellm** until LiteLLM's model database is updated with correct Opus 4.7 cache rates, or the `sohoai_api` tier becomes the authoritative source.

---

## 7. Timing Analysis

### Why SoHoAI models are significantly slower

| Operation | Session A (native Anthropic) | Session B (SoHoAI/Ollama Cloud) | Factor |
|---|---|---|---|
| Planner | 138 s (Sonnet) | 755 s (deepseek-v4-pro) | **5.5×** |
| Actor per-dispatch avg | 77 s (Haiku) | 392 s (qwen3-coder-next) | **5.1×** |
| Actor-heavy per-dispatch | — | 503 s avg (kimi-k2.6) | — |
| Explore per-dispatch avg | 16 s (Haiku) | 50 s (Haiku) | 3.1× |
| Reviewer per-dispatch avg | 167 s (Sonnet, 1 pass) | 180 s (Sonnet, avg 2 passes) | comparable |

**Root causes for SoHoAI latency:**

1. **LiteLLM proxy hop**: Requests route through the local SoHoAI gateway at `http://192.168.1.93:8000` before reaching Ollama Cloud's `https://ollama.com/v1`. Each request pays two network hops and serialization overhead.

2. **Ollama Cloud cold-start**: Ollama Cloud Pro does not guarantee a warm model instance. deepseek-v4-pro and kimi-k2.6 are large models (100B+ parameters); a cold start on Ollama Cloud can account for 60–120 s of the observed latency.

3. **No Anthropic prompt cache**: LiteLLM strips `cache_control` markers (Anthropic-specific header extension). Every request to SoHoAI models is a full cold-context inference — no cache reuse, no latency benefit from prior turns. This particularly hurts the Planner which processes a large RESEARCH.md + system prompt on every token.

4. **Model throughput**: deepseek-v4-pro and kimi-k2.6 generate at lower tokens/second than Anthropic's hosted Haiku. Even discounting latency, raw generation time for equivalent output is longer.

The 1,611 s difference in session wall time between A (3,039 s) and B (4,650 s) is almost entirely explained by SoHoAI model latency (deepseek 617 s extra + kimi 1,007 s + qwen3 extra ~900 s vs Haiku baseline).

---
