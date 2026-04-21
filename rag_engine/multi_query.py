"""Multi-query retrieval with MMR reranking.

Pipeline:
    expand_query()        -> [original, v1, v2, ..., vN]
    parallel search_rag   -> list[list[Candidate]]
    union()               -> dedup by point_id, keep max score
    mmr_rerank()          -> top-k reranked by λ-weighted relevance + diversity

Design notes:
  * Vectors are fetched from Qdrant (with_vectors=True) during search; no
    extra embedding round-trip for reranking.
  * Original query embedding is computed once and reused for both variant 0
    (search) and MMR relevance scoring.
  * Variant generation uses the specialist model by default (fast, free);
    config key allows routing to external for higher quality.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

# Assume these modules exist and contain necessary definitions from RAG-strategy.md
from .collection import DOCUMENTS_COLLECTION, ensure_collection
from .embeddings import embed_text
from .schema import (
    FIELD_FILE_NAME, FIELD_FILE_TYPE, FIELD_OWNER,
    FIELD_PARENT_TEXT, FIELD_SOURCE_PATH,
)

logger = logging.getLogger(__name__)

# --- variant generation ---------------------------------------------------

_EXPANSION_PROMPT = """You are generating search-query variants to improve document
retrieval. Given a user question, produce exactly {n} alternative phrasings that
someone might use when searching for the same information. Vary the vocabulary,
specificity, and perspective. Keep each variant short (under 15 words).

Return ONLY the variants, one per line, no numbering, no commentary.

User question: {query}
"""

async def expand_query(query: str, n_variants: int, llm_fn) -> list[str]:
    """Generate n_variants alternative phrasings plus the original.

    Args:
        llm_fn: async callable(prompt: str) -> str. Supplied by caller so this
                module stays agnostic of specialist/external routing.
    Returns:
        [original_query, variant_1, ..., variant_N]. If generation fails or
        returns fewer lines than requested, falls back to what was returned
        (never below just the original).
    """
    try:
        raw = await llm_fn(_EXPANSION_PROMPT.format(query=query, n=n_variants))
    except Exception as e:
        logger.warning("Query expansion failed: %s — falling back to original only", e)
        return [query]
    variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    variants = [v for v in variants if v.lower() != query.lower()][:n_variants]
    return [query] + variants

# --- retrieval + union ----------------------------------------------------

async def _search_one(
    vector: list[float], user_id: str | None, limit: int,
    qdrant_client: QdrantClient,
) -> list[dict]:
    """Single Qdrant query. Returns raw hits with vectors for MMR."""
    query_filter = Filter(
        must=[FieldCondition(
            key=FIELD_OWNER,
            match=MatchAny(any=[user_id, "la-familia"]),
        )]
    ) if user_id else None
    
    try:
        result = qdrant_client.query_points(
            collection_name=DOCUMENTS_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,   # needed for MMR
        )
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

    return [
        {
            "point_id":    hit.id,
            "vector":      hit.vector,                      # list[float]
            "score":       hit.score,
            "content":     (hit.payload or {}).get(FIELD_PARENT_TEXT, ""),
            "source_path": (hit.payload or {}).get(FIELD_SOURCE_PATH, ""),
            "file_name":   (hit.payload or {}).get(FIELD_FILE_NAME, ""),
            "file_type":   (hit.payload or {}).get(FIELD_FILE_TYPE, ""),
        }
        for hit in result.points
    ]

def _union_by_point_id(result_lists: list[list[dict]]) -> list[dict]:
    """Union multiple result lists, dedup by point_id, keep max score."""
    merged: dict[Any, dict] = {}
    for results in result_lists:
        for r in results:
            pid = r["point_id"]
            if pid not in merged or r["score"] > merged[pid]["score"]:
                merged[pid] = r
    return list(merged.values())

# --- MMR reranking --------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)

def mmr_rerank(
    candidates: list[dict],
    query_vector: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict]:
    """MMR: iterative selection balancing relevance with diversity.

    Uses 'vector' and 'score' fields on each candidate. Returns a new list
    of the top_k picks in selection order (most-relevant-and-diverse first).
    """
    if not candidates:
        return []
    if top_k >= len(candidates):
        # Fallback: just return top K by standard score if k is too large
        return sorted(candidates, key=lambda c: -c["score"])

    q = np.array(query_vector, dtype=np.float32)
    
    # Map point_id to its vector and precompute relevance score (sim(q, d))
    vecs = {c["point_id"]: np.array(c["vector"], dtype=np.float32) for c in candidates}
    rel = {pid: _cosine(q, v) for pid, v in vecs.items()}

    selected: list[dict] = []
    remaining = {c["point_id"]: c for c in candidates}

    # First pick: highest relevance (no diversity term to evaluate yet)
    first_pid = max(remaining, key=lambda pid: rel[pid])
    selected.append(remaining.pop(first_pid))

    while len(selected) < top_k and remaining:
        best_pid = None
        best_mmr = -float("inf")
        
        for pid, cand in remaining.items():
            # Calculate diversity (max similarity to already selected items S)
            div = max(_cosine(vecs[pid], vecs[s["point_id"]]) for s in selected)
            
            # Calculate MMR score
            mmr = lambda_ * rel[pid] - (1.0 - lambda_) * div
            
            if mmr > best_mmr:
                best_mmr = mmr
                best_pid = pid
        
        if best_pid is None:
            # Should only happen if remaining is empty, but as a safeguard
            break
        
        selected.append(remaining.pop(best_pid))

    # Attach the MMR-relevance score (for visibility/testing)
    for c in selected:
        c["mmr_relevance"] = rel[c["point_id"]]
    return selected

# --- orchestrator ---------------------------------------------------------

async def multi_query_search(
    query: str,
    user_id: str | None,
    limit: int,
    qdrant_client: QdrantClient,
    rag_cfg: dict,
    llm_fn,
) -> list[dict]:
    """Drop-in enhanced replacement for search_rag() when multi_query is enabled.

    Shape of return value is identical to search_rag() except each dict also
    has 'point_id' and optionally 'mmr_relevance'. Existing callers (tool_use,
    _build_rag_prompt, rag_smoke_test) work unchanged — they only read
    'content', 'source_path', 'file_name', 'file_type', 'score'.
    """
    mq_cfg      = rag_cfg.get("multi_query", {})
    n_variants  = int(mq_cfg.get("n_variants", 3))
    lambda_     = float(mq_cfg.get("lambda", 0.5))
    pool_mult   = int(mq_cfg.get("pool_multiplier", 4))
    per_query_k = max(limit, pool_mult * limit // (n_variants + 1))

    # -- 1. Generate variants ------------------------------------------------
    queries = await expand_query(query, n_variants=n_variants, llm_fn=llm_fn)
    logger.info(f"multi_query: {len(queries)} queries (1 original + {len(queries) - 1} variants)")

    # -- 2. Embed the user's ORIGINAL query once (for MMR relevance) --------
    model      = rag_cfg.get("embedding_model", "bge-m3")
    ollama_url = rag_cfg.get("ollama_url", "http://192.168.1.93:11434/api/embeddings")
    
    # Embed the original query to get the reference vector for MMR
    original_vector = await embed_text(query, model=model, ollama_url=ollama_url)

    # -- 3. Embed variants + parallel Qdrant search --------------------------
    async def _embed_and_search(q: str) -> list[dict]:
        # Use original vector for the first query, and re-embed for variants
        vec = original_vector if q == query else await embed_text(q, model=model, ollama_url=ollama_url)
        return await _search_one(vec, user_id, per_query_k, qdrant_client)

    ensure_collection(qdrant_client)
    # Run all searches concurrently
    result_lists = await asyncio.gather(*(_embed_and_search(q) for q in queries))

    # -- 4. Union + MMR rerank -----------------------------------------------
    pool = _union_by_point_id(result_lists)
    logger.info(f"multi_query: pool size = {len(pool)} (before MMR top_k={limit}, λ={lambda_:.2f})")
    return mmr_rerank(pool, original_vector, top_k=limit, lambda_=lambda_)