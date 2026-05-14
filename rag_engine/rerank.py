"""
Cross-encoder reranking — semantic relevance refinement over retrieved candidates.

Wire format (confirmed via probe 2026-05-14):
  POST http://192.168.1.95:8001/v1/rerank
  Request:  {"model": "bge-reranker-v2-m3", "query": <str>, "documents": [<str>, ...]}
  Response: {"model": ..., "object": "list", "usage": {...}, "results": [
              {"index": <int>, "relevance_score": <float>}, ...
            ]}

The reranker server is launched with `-c 768 -b 768 --reranking --pooling rank`,
ensuring no context limit issues. Client does not truncate — the server's
consistent batch size handles all typical queries and child chunks.

Candidates are ranked by relevance_score (higher = more relevant); results are
sorted desc and remapped back to original index. A rerank_score field is attached
to each candidate (alongside existing score field — not replacing it).

If the reranker becomes unavailable or any exception occurs during HTTP request
or parsing, a single WARN is logged and candidates are returned unchanged
(graceful fallback to Qdrant order).
"""

from __future__ import annotations

import logging

import httpx

from .schema import FIELD_TEXT

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    candidates: list[dict],
    rerank_cfg: dict,
) -> list[dict]:
    """
    Rerank candidates using a cross-encoder model (bge-reranker-v2-m3).

    Candidates must include FIELD_TEXT in their payload for the reranker to
    score them (child chunk content, not parent_text). If any candidate lacks
    FIELD_TEXT, reranking is skipped entirely and candidates are returned
    unchanged.

    Args:
        query:        User's query string (sent as-is to the reranker).
        candidates:   List of dicts, each with at least:
                      - FIELD_TEXT: child chunk content to score
                      - score:      original Qdrant cosine score (preserved)
                      Other keys (content, source_path, etc.) are passed through.
        rerank_cfg:   config["rag"]["rerank"] — must have:
                      - server_url:             HTTP endpoint
                      - model:                  model name
                      - timeout_seconds:        per-request timeout

    Returns:
        Reranked candidates list, sorted by rerank_score desc.
        Each dict gains a rerank_score field.
        On any exception, candidates are returned unchanged with a WARN logged.
    """
    # -- 0. Check for FIELD_TEXT in all candidates ----
    if not candidates:
        return candidates

    for cand in candidates:
        if FIELD_TEXT not in cand:
            logger.warning(
                "rerank: candidate missing %s field — skipping reranker entirely",
                FIELD_TEXT,
            )
            return candidates

    # -- 1. Extract raw text from candidates ----
    documents = [cand[FIELD_TEXT] for cand in candidates]

    # -- 2. Call reranker ----
    try:
        timeout = rerank_cfg.get("timeout_seconds", 10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                rerank_cfg["server_url"],
                json={
                    "model": rerank_cfg["model"],
                    "query": query,
                    "documents": documents,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning(
            "reranker failed: %s — falling back to Qdrant order",
            exc,
        )
        return candidates

    # -- 3. Parse response and remap by index ----
    try:
        results = data.get("results", [])
        scored = {}
        for item in results:
            idx = item.get("index")
            score = item.get("relevance_score")
            if idx is not None and score is not None:
                scored[idx] = score
    except Exception as exc:
        logger.warning(
            "reranker response parse failed: %s — falling back to Qdrant order",
            exc,
        )
        return candidates

    # -- 4. Attach rerank_score to candidates and sort desc ----
    for i, cand in enumerate(candidates):
        cand["rerank_score"] = scored.get(i, 0.0)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    logger.debug(
        "rerank: %d candidates → reranked by cross-encoder (top score: %.4f)",
        len(candidates),
        reranked[0]["rerank_score"] if reranked else 0.0,
    )
    return reranked
