"""
RAG search — query → embed → Qdrant query_points → parent_text + provenance.

search_rag() is the single function exported by rag_engine/__init__.py.
Ownership filtering is applied internally; the caller only provides
query, user_id, and limit.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from .collection import DOCUMENTS_COLLECTION, ensure_collection
from .embeddings import embed_text
from .schema import (
    FIELD_FILE_NAME,
    FIELD_FILE_TYPE,
    FIELD_OWNER,
    FIELD_PARENT_TEXT,
    FIELD_SOURCE_PATH,
)

logger = logging.getLogger(__name__)


async def search_rag(
    query: str,
    user_id: str | None,
    limit: int,
    qdrant_client: QdrantClient,
    rag_cfg: dict,
) -> list[dict]:
    """
    Semantic search over the user's documents + shared (la-familia) content.

    Steps:
      1. Embed the query via Ollama (mxbai-embed-large).
      2. Build a Qdrant filter scoped to user_id + "la-familia".
      3. query_points() → ranked ScoredPoint list.
      4. Return parent_text (richer context for the LLM) + provenance metadata.

    If user_id is None (pre-auth / dev mode), no ownership filter is applied
    and all documents are searched.

    Args:
        query:         User's question or search string.
        user_id:       Owner string (e.g. "florian") from the authenticated session,
                       or None to search without an ownership filter.
        limit:         Max number of results to return.
        qdrant_client: Shared Qdrant client (created once at app startup).
        rag_cfg:       config["rag"] dict — embedding_model, ollama_url.

    Returns:
        List of dicts with keys:
          content      — parent_text injected into the LLM prompt
          source_path  — full NFS path returned to the user as provenance
          score        — cosine similarity (0–1, higher = more relevant)
          file_name    — filename for display
          file_type    — pdf | md | ipynb | etc.
    """
    model      = rag_cfg.get("embedding_model", "mxbai-embed-large")
    ollama_url = rag_cfg.get("ollama_url", "http://localhost:11434/api/embeddings")

    # -- 1. Embed the query ------------------------------------------------
    vector = await embed_text(query, model=model, ollama_url=ollama_url)

    # -- 2. Ownership filter -----------------------------------------------
    # user_id=None → no filter (dev mode / Phase 2 before OAuth2 is wired)
    query_filter: Filter | None = None
    if user_id:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=FIELD_OWNER,
                    match=MatchAny(any=[user_id, "la-familia"]),
                )
            ]
        )

    # -- 3. Search ---------------------------------------------------------
    # ensure_collection so an empty / not-yet-created DB returns []
    # instead of raising. No-op when the collection already exists.
    ensure_collection(qdrant_client)
    result = qdrant_client.query_points(
        collection_name=DOCUMENTS_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    # -- 4. Unpack hits ----------------------------------------------------
    # Return parent_text (not child text) — richer context for the LLM.
    results = [
        {
            "content":     (hit.payload or {}).get(FIELD_PARENT_TEXT, ""),
            "source_path": (hit.payload or {}).get(FIELD_SOURCE_PATH, ""),
            "score":       hit.score,
            "file_name":   (hit.payload or {}).get(FIELD_FILE_NAME, ""),
            "file_type":   (hit.payload or {}).get(FIELD_FILE_TYPE, ""),
        }
        for hit in result.points
    ]

    logger.debug(
        "search_rag: user_id=%s  query=%r  → %d result(s)",
        user_id, query[:60], len(results),
    )
    return results
