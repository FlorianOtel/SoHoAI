"""
Short-term conversation memory backed by Redis.

Stores the rolling message history for active conversations.
- TTL-based expiry (default 24h) keeps memory bounded
- Redis AOF persistence survives reboots when pointed at NAS
- Provides the context window that gets sent to the LLM each turn

KV cache integration (optional):
- resume(chat_id) → assigns llama-server slot + restores NFS KV file before inference
- park(chat_id)   → saves KV slot to NFS + refreshes Redis TTL after inference
- clear(chat_id)  → wipes Redis + erases KV slot + deletes NFS file

Rolling summarization:
- maybe_summarize(chat_id, summarize_fn) → if conversation exceeds threshold,
  summarizes older turns via LLM and rebuilds Redis with condensed context.
  Also erases KV cache (prompt changed, cached tokens are stale).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

import redis.asyncio as redis

from schemas import Message, Role

if TYPE_CHECKING:
    from chat_store import ChatStore
    from kv_cache import KVCacheManager

logger = logging.getLogger(__name__)


class ConversationCache:
    """Redis-backed sliding-window conversation store with optional KV cache integration."""

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        default_ttl: int = 86400,
        max_turns: int = 50,
        kv_cache: "KVCacheManager | None" = None,
        summarize_threshold_chars: int = 200_000,
        summarize_keep_turns: int = 20,
    ):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = default_ttl
        self.max_turns = max_turns
        self.kv_cache = kv_cache
        self.summarize_threshold_chars = summarize_threshold_chars
        self.summarize_keep_turns = summarize_keep_turns

    # -- Core Redis operations -------------------------------------------------

    async def append(self, chat_id: str, role: str, content: str) -> int:
        """Add a message to the conversation. Returns new length."""
        key = f"conv:{chat_id}"
        msg = json.dumps({
            "role": role,
            "content": content,
            "ts": time.time(),
        })
        pipe = self.redis.pipeline()
        pipe.rpush(key, msg)
        pipe.expire(key, self.default_ttl)
        results = await pipe.execute()
        return results[0]  # length after push

    async def get_context(
        self, chat_id: str, max_turns: Optional[int] = None
    ) -> list[Message]:
        """
        Retrieve recent conversation history for LLM context.
        Always includes the system message (index 0) if present.
        """
        key = f"conv:{chat_id}"
        limit = max_turns or self.max_turns

        all_msgs = await self.redis.lrange(key, 0, -1)
        if not all_msgs:
            return []

        parsed = [json.loads(m) for m in all_msgs]

        # Preserve system message + last N turns
        if parsed and parsed[0]["role"] == "system":
            system = [parsed[0]]
            recent = parsed[1:][-limit:]
            result = system + recent
        else:
            result = parsed[-limit:]

        return [
            Message(role=Role(m["role"]), content=m["content"])
            for m in result
        ]

    async def set_system_prompt(self, chat_id: str, content: str):
        """Set or replace the system prompt (always first message)."""
        key = f"conv:{chat_id}"
        exists = await self.redis.exists(key)

        if exists:
            first = await self.redis.lindex(key, 0)
            first_parsed = json.loads(first)
            if first_parsed["role"] == "system":
                msg = json.dumps({"role": "system", "content": content, "ts": time.time()})
                await self.redis.lset(key, 0, msg)
                return

        msg = json.dumps({"role": "system", "content": content, "ts": time.time()})
        await self.redis.lpush(key, msg)
        await self.redis.expire(key, self.default_ttl)

    async def is_cold(self, chat_id: str) -> bool:
        """Return True if Redis has no messages for this chat (expired or new)."""
        return await self.redis.llen(f"conv:{chat_id}") == 0

    async def warm_from_store(self, chat_id: str, store: Optional["ChatStore"] = None, messages: Optional[list[Message]] = None) -> None:
        """
        Populate Redis from SQLite message history on cold-start resume.

        Called when Redis TTL has expired but the chat still exists in SQLite.
        
        If store is provided, consults the summary columns. If a summary exists,
        reconstructs Redis as: [system_msg?, summary_msg, messages_after_boundary].
        Otherwise, falls back to loading last max_turns messages.
        
        If store is not provided, uses the messages list directly (backward compat).
        
        Also erases the KV cache — the saved slot was built on the full old
        prompt which no longer matches the reloaded (possibly truncated) history.
        """
        if store is None:
            # Backward compat: messages passed directly
            if messages is None:
                logger.warning(f"warm_from_store called with neither store nor messages")
                return
            to_load = messages[-self.max_turns:] if len(messages) > self.max_turns else messages
        else:
            # Load from store, preferring summary if available
            summary_text, covers_through_id = store.get_summary(chat_id)
            
            if summary_text and covers_through_id is not None:
                # Reconstruction: [system?, summary_msg, raw_tail]
                all_raw = store.get_messages_after(chat_id, None)
                
                # Defensively: if boundary id doesn't exist, fall back to last-N
                tail = store.get_messages_after(chat_id, covers_through_id) if covers_through_id else all_raw
                
                # Build the reconstructed list
                to_load: list[Message] = []
                
                # Include system message if present
                if all_raw and all_raw[0].role == Role.system:
                    to_load.append(all_raw[0])
                
                # Add the summary as a synthetic assistant message
                summary_msg = Message(
                    role=Role.assistant,
                    content=f"[Earlier conversation summary]\n{summary_text}"
                )
                to_load.append(summary_msg)
                
                # Add the raw tail
                to_load.extend(tail)
            else:
                # No summary: just load last max_turns messages
                all_raw = store.get_messages_after(chat_id, None)
                to_load = all_raw[-self.max_turns:] if len(all_raw) > self.max_turns else all_raw

        key = f"conv:{chat_id}"
        pipe = self.redis.pipeline()
        pipe.delete(key)
        for msg in to_load:
            pipe.rpush(key, json.dumps({
                "role": msg.role.value,
                "content": msg.content,
                "ts": time.time(),
            }))
        pipe.expire(key, self.default_ttl)
        await pipe.execute()

        # Stale KV slot won't match the reloaded history — erase it
        if self.kv_cache is not None:
            await self.kv_cache.erase(chat_id)

        logger.info(
            f"Cold resume: warmed Redis for chat {chat_id[:8]} "
            f"with {len(to_load)} messages from SQLite"
        )

    async def clear(self, chat_id: str) -> None:
        """Wipe conversation from Redis and KV cache (slot + NFS file)."""
        await self.redis.delete(f"conv:{chat_id}")
        if self.kv_cache is not None:
            await self.kv_cache.erase(chat_id)

    async def touch(self, chat_id: str):
        """Reset TTL without modifying content."""
        await self.redis.expire(f"conv:{chat_id}", self.default_ttl)

    async def list_active(self) -> list[str]:
        """List all active conversation IDs."""
        keys = []
        async for key in self.redis.scan_iter("conv:*"):
            keys.append(key.removeprefix("conv:"))
        return keys

    # -- KV cache lifecycle (coordinates Redis + llama-server slot) -----------

    async def resume(self, chat_id: str) -> int | None:
        """
        Prepare conversation for inference: assign a llama-server slot and
        restore its KV cache from NFS (if a saved file exists).

        Returns slot_id to pass to kv_cache.inference(), or None if KV
        cache is not configured.
        """
        if self.kv_cache is None:
            return None
        return await self.kv_cache.restore(chat_id)

    async def park(self, chat_id: str) -> None:
        """
        Park conversation after inference: save KV slot to NFS and refresh
        the Redis TTL. Call this after every successful LLM response.
        """
        if self.kv_cache is not None:
            await self.kv_cache.save(chat_id)
        await self.touch(chat_id)

    # -- Rolling summarization ------------------------------------------------

    async def maybe_summarize(
        self,
        chat_id: str,
        summarize_fn: Callable[[str], Awaitable[str]],
        store: Optional["ChatStore"] = None,
    ) -> bool:
        """
        If total raw size of the Redis conversation exceeds summarize_threshold_chars,
        summarize older turns via LLM and rebuild Redis with condensed context.

        Rebuilt as: [system_msg] + [summary_msg] + [recent_turns]

        Also erases the KV cache — the prompt has changed so cached tokens
        are stale. The next inference rebuilds the KV from scratch.

        If store is provided, persists the summary to SQLite with the boundary id
        of the last message included in the summary.

        Returns True if summarization ran, False if threshold not reached.
        """
        key = f"conv:{chat_id}"
        all_msgs_raw = await self.redis.lrange(key, 0, -1)
        if not all_msgs_raw:
            return False

        total_chars = sum(len(r) for r in all_msgs_raw)
        if total_chars <= self.summarize_threshold_chars:
            return False

        parsed = [json.loads(r) for r in all_msgs_raw]

        # Separate system message
        if parsed and parsed[0]["role"] == "system":
            system_msg = parsed[0]
            body = parsed[1:]
        else:
            system_msg = None
            body = parsed

        # Split: old turns to summarize vs recent turns to keep verbatim
        keep = self.summarize_keep_turns
        old_turns = body[:-keep] if len(body) > keep else []
        recent_turns = body[-keep:] if len(body) >= keep else body

        if not old_turns:
            return False

        # Ask LLM to summarize the old turns
        old_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in old_turns
        )
        try:
            summary_text = await summarize_fn(old_text)
        except Exception as e:
            # If summarization fails (e.g., Gemma down), log but don't fail the turn.
            # The context will remain oversized but the conversation continues.
            logger.warning(
                f"Summarization failed for chat {chat_id[:8]}: {e}. Skipping, turn proceeds."
            )
            return False

        summary_msg = {
            "role": "assistant",
            "content": f"[Earlier conversation summary]\n{summary_text}",
            "ts": time.time(),
        }

        # Atomically rebuild the Redis list
        new_msgs: list[dict] = []
        if system_msg:
            new_msgs.append(system_msg)
        new_msgs.append(summary_msg)
        new_msgs.extend(recent_turns)

        pipe = self.redis.pipeline()
        pipe.delete(key)
        for msg in new_msgs:
            pipe.rpush(key, json.dumps(msg))
        pipe.expire(key, self.default_ttl)
        await pipe.execute()

        # KV cache is now stale — erase it so the next request rebuilds fresh
        if self.kv_cache is not None:
            await self.kv_cache.erase(chat_id)

        # Persist summary to SQLite if store is available.
        # Compute the boundary id: the id of the last message included in old_turns.
        if store:
            try:
                boundary_id = store.get_summary_boundary_id(chat_id, keep)
                if boundary_id is not None:
                    store.update_summary(chat_id, summary_text, boundary_id)
                else:
                    logger.warning(
                        f"Could not compute summary boundary for chat {chat_id[:8]}, skipping SQLite write"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to persist summary to SQLite for chat {chat_id[:8]}: {e}"
                )

        logger.info(
            f"Summarized chat {chat_id[:8]}: {len(old_turns)} old turns condensed, "
            f"{len(recent_turns)} recent turns kept"
        )
        return True

    # -- Lifecycle -------------------------------------------------------------

    async def ping(self) -> bool:
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def close(self):
        await self.redis.aclose()
