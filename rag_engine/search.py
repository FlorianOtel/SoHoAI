"""
RAG search — query → embed → Qdrant query_points → parent_text + provenance.

search_rag() is the single function exported by rag_engine/__init__.py.
Ownership filtering is applied internally; the caller only provides
query, user_id, and limit.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    MatchAny,
    Prefetch,
    SparseVector,
)

from .collection import DOCUMENTS_COLLECTION, collection_has_sparse, ensure_collection
from .embeddings import embed_sparse, embed_text
from .rerank import rerank as _rerank_candidates
from .schema import (
    FIELD_FILE_NAME,
    FIELD_FILE_TYPE,
    FIELD_OWNER,
    FIELD_PARENT_TEXT,
    FIELD_SESSION_TITLE,
    FIELD_SOURCE_PATH,
    FIELD_TEXT,
    SPARSE_VECTOR_NAME,
)

logger = logging.getLogger(__name__)

async def search_rag(
    query: str,
    user_id: str | None,
    limit: int,
    qdrant_client: QdrantClient,
    rag_cfg: dict,
    file_types: list[str] | None = None,
    score_threshold: float = 0.0,
    rerank: bool | None = None,
    rerank_cfg: dict | None = None,
    hybrid: bool | None = None,
) -> list[dict]:
    """
    Semantic search over the user's documents + shared (la-familia) content.

    Steps:
      1. Embed the query via Ollama (bge-m3); optionally compute sparse vectors (BM25).
      2. Build a Qdrant filter scoped to user_id + "la-familia".
      3. query_points() → ranked ScoredPoint list (hybrid RRF or dense-only).
      4. Optionally rerank via cross-encoder.
      5. Return parent_text (richer context for the LLM) + provenance metadata.

    If user_id is None (pre-auth / dev mode), no ownership filter is applied
    and all documents are searched.

    Args:
        query:           User's question or search string.
        user_id:         Owner string (e.g. "florian") from the authenticated session,
                         or None to search without an ownership filter.
        limit:           Max number of results to return.
        qdrant_client:   Shared Qdrant client (created once at app startup).
        rag_cfg:         config["rag"] dict — embedding_model, ollama_url, rerank, hybrid.
        file_types:      Optional list of file types to filter by.
        score_threshold: Minimum cosine score; 0.0 = no filter (only for dense-only path).
        rerank:          Enable cross-encoder reranking; None = read from rag_cfg default.
        rerank_cfg:      Rerank config dict; None = read from rag_cfg default.
        hybrid:          Enable hybrid dense+sparse search; None = read from rag_cfg default.

    Returns:
        List of dicts with keys:
          content         — parent_text injected into the LLM prompt
          source_path     — full NFS path returned to the user as provenance
          score           — cosine similarity (0–1, higher = more relevant) or RRF rank
          rerank_score    — (optional) cross-encoder relevance score
          file_name       — filename for display
          file_type       — pdf | md | ipynb | etc.
    """
    model      = rag_cfg.get("embedding_model", "bge-m3")
    ollama_url = rag_cfg.get("ollama_url", "http://192.168.1.93:11434/api/embeddings")

    # -- 0b. Resolve effective rerank settings ----
    _rerank_section = rag_cfg.get("rerank", {})
    effective_rerank = rerank if rerank is not None else _rerank_section.get("enabled", False)
    effective_rerank_cfg = rerank_cfg or _rerank_section

    # -- 0c. Resolve effective hybrid settings ----
    _hybrid_section = rag_cfg.get("hybrid", {})
    effective_hybrid = hybrid if hybrid is not None else _hybrid_section.get("enabled", False)

    # -- 1. Embed the query ------------------------------------------------
    vector = await embed_text(query, model=model, ollama_url=ollama_url)

    # -- 2. Build filter (ownership + optional file_types) ------------------
    must: list = []
    if user_id:
        must.append(
            FieldCondition(
                key=FIELD_OWNER,
                match=MatchAny(any=[user_id, "la-familia"]),
            )
        )
    if file_types:
        must.append(
            FieldCondition(
                key=FIELD_FILE_TYPE,
                match=MatchAny(any=file_types),
            )
        )
    query_filter: Filter | None = Filter(must=must) if must else None

    # -- 3. Search ---------------------------------------------------------
    ensure_collection(qdrant_client)

    # Determine fetch limit based on reranking
    if effective_rerank:
        fetch_multiplier = effective_rerank_cfg.get("fetch_multiplier", 6)
        fetch_cap = effective_rerank_cfg.get("fetch_cap", 30)
        _fetch_limit = min(limit * fetch_multiplier, fetch_cap)
    else:
        _fetch_limit = min(limit * 3, 50)

    # Hybrid or dense-only query
    if effective_hybrid:
        # Check if collection supports sparse vectors; fall back to dense if not
        if not collection_has_sparse(qdrant_client):
            logger.info(
                "hybrid query requested but collection lacks sparse_text; using dense-only"
            )
            result = qdrant_client.query_points(
                collection_name=DOCUMENTS_COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=_fetch_limit,
                score_threshold=score_threshold if score_threshold > 0.0 else None,
                with_payload=True,
            )
        else:
            # Compute sparse vectors for the query
            sparse_indices, sparse_values = embed_sparse(query)

            # Read hybrid config parameters
            dense_fetch = _hybrid_section.get("dense_fetch", 30)
            sparse_fetch = _hybrid_section.get("sparse_fetch", 30)

            # Compute outer limit for RRF fusion
            if effective_rerank:
                fetch_cap = effective_rerank_cfg.get("fetch_cap", 30)
                outer_limit = min(dense_fetch + sparse_fetch, fetch_cap)
            else:
                outer_limit = min(dense_fetch + sparse_fetch, 50)

            # Execute hybrid query with RRF fusion
            result = qdrant_client.query_points(
                collection_name=DOCUMENTS_COLLECTION,
                prefetch=[
                    Prefetch(
                        query=vector,
                        using="",
                        limit=dense_fetch,
                        query_filter=query_filter,
                    ),
                    Prefetch(
                        query=SparseVector(indices=sparse_indices, values=sparse_values),
                        using=SPARSE_VECTOR_NAME,
                        limit=sparse_fetch,
                        query_filter=query_filter,
                    ),
                ],
                query=Fusion.RRF,
                limit=outer_limit,
                with_payload=True,
            )
    else:
        # Dense-only query (existing behavior)
        result = qdrant_client.query_points(
            collection_name=DOCUMENTS_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=_fetch_limit,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
            with_payload=True,
        )

    # -- 4. Unpack hits and build full results (with FIELD_TEXT for reranker) ----
    results_full = [
        {
            "content":       (hit.payload or {}).get(FIELD_PARENT_TEXT, ""),
            "source_path":   (hit.payload or {}).get(FIELD_SOURCE_PATH, ""),
            "score":         hit.score,
            "file_name":     (hit.payload or {}).get(FIELD_FILE_NAME, ""),
            "file_type":     (hit.payload or {}).get(FIELD_FILE_TYPE, ""),
            "session_title": (hit.payload or {}).get(FIELD_SESSION_TITLE, ""),
            FIELD_TEXT:      (hit.payload or {}).get(FIELD_TEXT, ""),
        }
        for hit in result.points
    ]

    # -- 5. Optionally rerank ----
    if effective_rerank and results_full:
        results_full = await _rerank_candidates(query, results_full, effective_rerank_cfg)

    # -- 6. Slice to limit ----
    results = results_full[:limit]

    logger.debug(
        "search_rag: user_id=%s  query=%r  hybrid=%s  rerank=%s → %d result(s)",
        user_id, query[:60], effective_hybrid, effective_rerank, len(results),
    )
    return results
