"""
Embedding helpers — shared by ingestion (ingest.py) and search (search.py).

Callers pass model and ollama_url explicitly from config["rag"] — no global
state here. Both functions are async; ingestion calls embed_batch() from an
asyncio context; search calls embed_text() for a single query.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embeddings"
_DEFAULT_MODEL = "bge-m3"
_BATCH_CONCURRENCY = 5      # max parallel Ollama requests (no native batch endpoint)
_PROGRESS_INTERVAL = 50     # call progress_cb every this many completed embeddings


async def embed_text(
    text: str,
    model: str = _DEFAULT_MODEL,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
) -> list[float]:
    """Generate an embedding for a single text string via Ollama."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            ollama_url,
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]


async def embed_batch(
    texts: list[str],
    model: str = _DEFAULT_MODEL,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Runs up to _BATCH_CONCURRENCY requests concurrently — Ollama has no
    native batch endpoint. Order is preserved (asyncio.gather guarantee).

    Args:
        texts:       Texts to embed.
        model:       Ollama model name.
        ollama_url:  Ollama embeddings API URL.
        progress_cb: Optional callback invoked every _PROGRESS_INTERVAL completions
                     and on the final chunk. Signature: (done: int, total: int).
                     asyncio is single-threaded so the callback is safe to call
                     without locks.
    """
    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)
    total = len(texts)
    _done = [0]  # mutable counter; no lock needed in asyncio single-thread model

    async def _one(t: str) -> list[float]:
        async with sem:
            result = await embed_text(t, model=model, ollama_url=ollama_url)
        _done[0] += 1
        d = _done[0]
        if progress_cb is not None and (d % _PROGRESS_INTERVAL == 0 or d == total):
            progress_cb(d, total)
        return result

    return list(await asyncio.gather(*[_one(t) for t in texts]))
