"""
KV Cache Manager for llama-server slot save/restore.

Maps chat_id → slot_id using LRU eviction.
Saves/restores per-conversation KV caches to NFS via the llama-server slot API.
Uses the native /completion endpoint (not OpenAI-compat) to control slot_id.

llama-server must be started with:
  --slot-save-path <nfs-dir>   (base dir for all slot .bin files)
  --parallel N                 (N = num_slots; default 1)

Slot API:
  POST /slots/{id}?action=save     {"filename": "{chat_id}.bin"}
  POST /slots/{id}?action=restore  {"filename": "{chat_id}.bin"}
  POST /slots/{id}?action=erase    {}
  GET  /slots                       → list of slot states
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Chat template
# -----------------------------------------------------------------------------

def apply_mistral_template(messages: list[dict]) -> str:
    """
    Format a messages list using the Mistral instruction template.

    System message content is prepended to the first user turn.
    Format: [INST] {user} [/INST] {assistant}</s>[INST] {user} [/INST]
    """
    prompt = ""
    pending_system: Optional[str] = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            pending_system = content

        elif role == "user":
            if pending_system:
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            prompt += f"[INST] {content} [/INST]"

        elif role == "assistant":
            prompt += f" {content}</s>"

    return prompt


# -----------------------------------------------------------------------------
# KVCacheManager
# -----------------------------------------------------------------------------

class KVCacheManager:
    """
    Manages llama-server slot assignments and NFS-persisted KV caches.

    Slot assignment is LRU: when all slots are occupied and a new chat arrives,
    the least-recently-used slot is reclaimed.

    Granular API (used by ConversationCache.resume / park):
        slot_id = await kv.restore(chat_id)   # assign slot + load from NFS
        await kv.save(chat_id)                 # write slot back to NFS
        await kv.erase(chat_id)                # delete slot state + NFS file
        result = await kv.inference(slot_id, prompt, ...)

    Convenience wrapper (all-in-one):
        result = await kv.complete(chat_id, messages, ...)
    """

    def __init__(
        self,
        base_url: str,
        slot_save_path: str,
        num_slots: int = 1,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.slot_save_path = Path(slot_save_path)
        self.num_slots = num_slots
        self.timeout = timeout

        # LRU map: chat_id → slot_id (OrderedDict, last = most recent)
        self._lru: OrderedDict[str, int] = OrderedDict()
        self._lock = asyncio.Lock()  # protects _lru only

        logger.info(
            f"KVCacheManager ready: {base_url}, {num_slots} slot(s), "
            f"save path: {slot_save_path}"
        )

    # -- Internal helpers ------------------------------------------------------

    def _filename(self, chat_id: str) -> str:
        return f"{chat_id}.bin"

    def _cache_exists(self, chat_id: str) -> bool:
        return (self.slot_save_path / self._filename(chat_id)).exists()

    async def _assign_slot(self, chat_id: str) -> int:
        """Return slot_id for chat_id, assigning one via LRU if needed."""
        async with self._lock:
            if chat_id in self._lru:
                self._lru.move_to_end(chat_id)
                return self._lru[chat_id]

            used = set(self._lru.values())
            free = next((i for i in range(self.num_slots) if i not in used), None)

            if free is not None:
                slot_id = free
            else:
                evicted_chat, slot_id = self._lru.popitem(last=False)
                logger.info(f"KV slot {slot_id} evicted from chat {evicted_chat[:8]}")

            self._lru[chat_id] = slot_id
            return slot_id

    async def _slot_action(
        self,
        client: httpx.AsyncClient,
        slot_id: int,
        action: str,
        chat_id: str,
    ) -> None:
        """POST /slots/{id}?action=save|restore|erase"""
        url = f"{self.base_url}/slots/{slot_id}"
        body = {"filename": self._filename(chat_id)} if action != "erase" else {}
        resp = await client.post(url, params={"action": action}, json=body)
        resp.raise_for_status()

    # -- Granular public API ---------------------------------------------------

    async def restore(self, chat_id: str) -> int:
        """
        Assign a slot to chat_id and restore its KV cache from NFS (if exists).

        Returns the slot_id for use in inference().
        """
        slot_id = await self._assign_slot(chat_id)

        if self._cache_exists(chat_id):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                try:
                    await self._slot_action(client, slot_id, "restore", chat_id)
                    logger.info(f"KV restored: chat {chat_id[:8]} → slot {slot_id}")
                except Exception as e:
                    logger.warning(f"KV restore failed for {chat_id[:8]}: {e} — continuing cold")

        return slot_id

    async def save(self, chat_id: str) -> None:
        """Save the current slot KV state to NFS."""
        async with self._lock:
            slot_id = self._lru.get(chat_id)
        if slot_id is None:
            logger.warning(f"KV save: no slot assigned for {chat_id[:8]}, skipping")
            return

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                await self._slot_action(client, slot_id, "save", chat_id)
                logger.info(f"KV saved: chat {chat_id[:8]} → slot {slot_id}")
            except Exception as e:
                logger.warning(f"KV save failed for {chat_id[:8]}: {e}")

    async def erase(self, chat_id: str) -> None:
        """
        Erase the KV cache for a chat:
          - Remove from LRU map
          - Erase the slot state on llama-server
          - Delete the .bin file from NFS
        """
        async with self._lock:
            if chat_id not in self._lru:
                # No slot assigned — just clean up any stale file
                (self.slot_save_path / self._filename(chat_id)).unlink(missing_ok=True)
                return
            slot_id = self._lru.pop(chat_id)

        async with httpx.AsyncClient(timeout=10) as client:
            try:
                await self._slot_action(client, slot_id, "erase", chat_id)
            except Exception as e:
                logger.warning(f"KV slot erase failed for {chat_id[:8]}: {e}")

        (self.slot_save_path / self._filename(chat_id)).unlink(missing_ok=True)
        logger.info(f"KV erased: chat {chat_id[:8]}")

    async def inference(
        self,
        slot_id: int,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> dict:
        """
        POST /completion to the assigned slot.

        Returns llama-server result dict:
          content, tokens_evaluated, tokens_predicted, stop_reason
        """
        payload = {
            "prompt": prompt,
            "slot_id": slot_id,
            "cache_prompt": True,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or ["</s>", "[INST]"],
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/completion", json=payload)
            resp.raise_for_status()
            return resp.json()

    # -- Convenience wrapper ---------------------------------------------------

    async def complete(
        self,
        chat_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> dict:
        """All-in-one: restore slot → inference → save."""
        slot_id = await self.restore(chat_id)
        prompt = apply_mistral_template(messages)
        result = await self.inference(slot_id, prompt, temperature=temperature,
                                      max_tokens=max_tokens, stop=stop)
        await self.save(chat_id)
        return result
