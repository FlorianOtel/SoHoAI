---
title: "SoHoAI — Memory tiers"
created_at: 2026-05-05--16-38
created_by: Claude Code (Claude Haiku 4.5)
context: >
  Three-tier conversation memory architecture for SoHoAI: Redis short-term cache,
  SQLite long-term persistence, and GPU KV cache for efficient inference on subsequent turns.
  Coordinated by ConversationCache API with rolling summarization on throughput overflow.
  Design decisions covering memory strategy, summarization trade-offs, and KV cache ownership.
---

# SoHoAI — Memory tiers

SoHoAI maintains conversation state across three distinct storage tiers, each optimized for a different time scale and access pattern:

## Three-tier architecture

### Short-term: Redis

- **Storage**: Server 1 (`127.0.0.1:6379`), TTL 24h
- **Format**: Serialized conversation object, keyed by `conv:{chat_id}`
- **Persistence**: Dumped to NAS via Redis AOF (`/mnt/nfs/__Backups/SoHoAI--databases/redis/`)
- **Purpose**: Fast in-memory retrieval during active conversations; supports rapid turn-by-turn access

### Long-term: SQLite

- **Storage**: Single file on NAS (`/mnt/nfs/__Backups/SoHoAI--databases/sqlite/telemetry.db`)
- **Format**: SQL `chats` table + `messages` table (one row per user/assistant message)
- **Columns**: `id`, `user_id`, `chat_id`, `role`, `content`, `created_at`, plus `summary_text` and `summary_covers_through_message_id` (populated by rolling summarization)
- **Purpose**: Permanent audit log, full conversation history, chat list/search, markdown/JSONL export, RL training data extraction
- **Scale**: supports ~10K chats without performance degradation

### GPU KV cache

- **Storage**: llama-server slot state, serialized to NAS (`/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/k-v-caches/`) as `{chat_id}.bin` per conversation
- **Format**: Binary checkpoint from llama-server slot API (`/slots/{id}?action=save`)
- **Purpose**: Avoid re-computing the prompt prefix on subsequent turns (GPU memory efficiency)
- **Used by**: internal (Gemma 4) inference path only; external (Sonnet) path uses Anthropic prompt caching instead

---

## ConversationCache API

All three tiers are coordinated by the `ConversationCache` class in `conversation.py`. Core methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `resume(chat_id)` | `(chat_id) → slot_id, kv_ok` | Restore KV slot from NFS before inference (internal path); no-op for external |
| `park(chat_id)` | `(chat_id) → None` | Save KV slot to NFS + refresh Redis TTL after inference |
| `clear(chat_id)` | `(chat_id) → None` | Wipe Redis + erase KV slot + delete NFS file (full cleanup) |
| `maybe_summarize(chat_id, fn, store=None)` | `(chat_id, summarization_fn, store) → summarized` | If Redis context > ~100K tokens: summarize old turns via llama-server, rebuild Redis, erase stale KV, optionally persist to SQLite |
| `is_cold(chat_id)` | `(chat_id) → bool` | True if Redis has no messages for this chat (cold start needed) |
| `warm_from_store(chat_id, store=None, messages=None)` | `(chat_id, store, messages) → None` | Cold-start Redis reload: reconstruct as `[system?, summary_msg, messages_where_id > covers_through]` if summary exists, else last-N raw messages |

**Key behaviors**:
- `resume()` and `park()` are the mirror operations that bracket every LLM turn
- `maybe_summarize()` is **tolerant of failure** — if summarization raises, the turn proceeds with oversized context (no loss of correctness, just cost)
- `warm_from_store()` with a persisted summary produces zero data duplication (summary replaces old turns, raw log always retained)
- `clear()` is the only way to delete a chat completely (removes from all three tiers)

---

## Rolling summarization

Triggered when Redis context exceeds ~100K tokens (approximately 400K characters). The process:

1. **Erase stale KV**: delete the slot file on NAS — the slot will be assigned fresh on the next `resume()`
2. **Summarize via internal (llama-server)**: extract the oldest turns up to a boundary, generate a summary
3. **Rebuild Redis**: replace old turns with a single synthesized `role=assistant` message containing the summary; keep the last ~20 turns verbatim
4. **Persist to SQLite**: if `store` is passed, write `summary_text` and `summary_covers_through_message_id` to the `chats` row for cold-resume recovery
5. **Continue the turn**: re-enter the LLM loop with the condensed context

**Frequency**: approximately once per ~50-turn chat (empirical, at 100K-token threshold).

**Model choice**: hardcoded to `internal` (Gemma 4) via `routing.summarization_model` in `config.yaml`. Deliberate choice: keeps the inference deterministic and avoids cloud API cost on a background operation.

---

## Design decisions

### Redis for short-term memory

Redis is the working set: fast (microseconds), TTL-based eviction at 24h, and sufficient capacity for conversation history below the summarization threshold (~100K tokens ≈ 50–80 turns). The in-memory structure supports rapid turn-by-turn append and retrieval. NAS AOF persistence guards against process crashes while maintaining Redis's responsiveness (async replication, not synchronous waits on every write).

### SQLite for long-term

A single SQLite file on NAS provides durability without ops complexity. Zero maintenance (no schema migrations, no connection pooling, no background compaction). Single-writer semantics (only the orchestrator writes) eliminates locking contention. Fits the ~10K chats use case without performance degradation. All critical queries (by `chat_id`, by `user_id`, by timestamp) are indexed. Supports full-text search via FTS5 (not currently used, but available for chat list filtering).

### KV cache in ConversationCache

`conversation.py` is the **single owner** of all conversation state — both Redis and KV. This co-location ensures save/restore operations stay synchronized: if the process crashes mid-turn, both Redis and KV slot are left in a consistent state on the next `resume()`. A separate KVCacheManager would require explicit synchronization and increase the surface for state corruption.

### Rolling summarization uses internal (llama-server)

Summarization is a background operation triggered by conversational throughput, not a primary user-facing inference. Pinning it to the local model (Gemma 4) keeps the cost predictable (~0–5 cents per summarization event, depending on parent size). If Anthropic's Sonnet were used for summarization, a single active chat could incur $0.05–0.10 per summary, and families with multiple concurrent chats would accumulate costs quickly. Gemma 4's per-token cost is negligible by comparison.

### Summary persistence to SQLite

Before this design, summaries were held in Redis only. A 24h Redis TTL expiry meant that a chat resuming after 25+ hours would reload with last-N raw messages, losing the summary and re-computing the LLM context from scratch on the next turn (costly, slow). Now `summary_text` and `summary_covers_through_message_id` columns on the `chats` table persist the summary permanently. On cold resume, `warm_from_store()` reconstructs Redis as `[system?, summary_msg, messages_with_id > boundary]` — no re-computation, no duplication.

### Prompt caching vs. summarization trade-off

Anthropic's prompt caching on Sonnet 4.6 maintains a rolling prefix cache (system message + `messages[-2]`) that grows more efficient with each turn (~10× input cost reduction on cache hits). Each `maybe_summarize()` event rewrites Redis (summary replaces old turns), which invalidates the rolling prefix cache on the next turn (the cache_control token resets). At the 100K-token threshold, summarization fires roughly once per ~50-turn chat. The cost spike on the summary turn (~$0.05) is amortized over the ongoing savings from a smaller, faster-to-cache prefix on all subsequent turns. The threshold (100K) is deliberately aligned with the cloud-routing threshold (`complexity_threshold_tokens` in `config.yaml`), so both events fire in lockstep — the summarization never catches the system by surprise.

---

## Code references

- **Main coordinator**: `conversation.py::ConversationCache` class
- **KV slot management**: `kv_cache.py::KVCacheManager` class (slot save/restore, inference with slot_id)
- **SQLite persistence**: `chat_store.py::ChatStore` class (full CRUD for chats and messages)
- **Rolling summarization**: `conversation.py::ConversationCache.maybe_summarize()` method
- **Gemma prompt template**: `kv_cache.py::apply_gemma_template()` function (Gemma 4 `<|turn>` format)
