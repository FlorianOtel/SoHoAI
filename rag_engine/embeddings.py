"""
Embedding helpers — shared by ingestion (ingest.py) and search (search.py).

Callers pass model and ollama_url explicitly from config["rag"] — no global
state here. Both functions are async; ingestion calls embed_batch() from an
asyncio context; search calls embed_text() for a single query.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embeddings"
_DEFAULT_MODEL = "mxbai-embed-large"
_BATCH_CONCURRENCY = 8      # max parallel Ollama requests (no native batch endpoint)


async def embed_text(
    text: str,
    model: str = _DEFAULT_MODEL,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
) -> list[float]:
    """Generate an embedding for a single text string via Ollama."""
    async with httpx.AsyncClient(timeout=30.0) as client:
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
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Runs up to _BATCH_CONCURRENCY requests concurrently — Ollama has no
    native batch endpoint. Order is preserved (asyncio.gather guarantee).
    """
    sem = asyncio.Semaphore(_BATCH_CONCURRENCY)

    async def _one(t: str) -> list[float]:
        async with sem:
            return await embed_text(t, model=model, ollama_url=ollama_url)

    return list(await asyncio.gather(*[_one(t) for t in texts]))
