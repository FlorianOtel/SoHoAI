"""
Qdrant collection configuration — single source of truth.

All code that creates, opens, or queries the documents collection imports
constants and helpers from here. No collection names, vector dimensions,
or distance metrics are hardcoded anywhere else.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, PayloadSchemaType, SparseVectorParams, VectorParams

from .schema import FIELD_SOURCE_PATH, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection constants
# ---------------------------------------------------------------------------

DOCUMENTS_COLLECTION = "documents"
VECTOR_SIZE = 1024          # bge-m3 output dimension (same as mxbai-embed-large)
DISTANCE = Distance.COSINE

# Module-level cache: once we've confirmed/created the source_path payload index
# in this process, skip the extra get_collection() round-trip on every subsequent
# ensure_collection() call (mirrors the _sparse_ready cache pattern in ingest.py).
_payload_index_ready: bool | None = None


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(url: str, timeout: int = 60) -> QdrantClient:
    """Connect to a running Qdrant server.

    Default timeout is 60 seconds to handle index optimization on large batches.
    During heavy ingestion (e.g., 70K+ points), Qdrant may take >5 seconds to
    respond to delete/upsert requests while it optimizes indexes. Default httpx
    timeout (~5s) is too short; 60s covers normal operations. Pass a higher
    value for bulk-delete operations that may trigger longer re-indexing passes.
    """
    return QdrantClient(url=url, timeout=timeout)


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    """
    Create the documents collection if it does not already exist.

    Safe to call on every startup — no-op when the collection is already present.
    Handles both real collections and Qdrant aliases (the alias is created by
    rag_sparse_migrate.py --swap after migration).

    On fresh creation: configures both dense (unnamed, 1024-dim) and sparse
    (SPARSE_VECTOR_NAME, BM25-style with Qdrant IDF modifier).

    Qdrant 1.17 does NOT support adding a sparse vector space to an existing
    collection that was created without one; use rag_sparse_migrate.py to
    migrate the corpus to a new collection with the correct schema.

    Also ensures the source_path payload index exists (see ensure_payload_indexes()),
    regardless of whether the collection was just created or already present.
    """
    real_collections = {c.name for c in client.get_collections().collections}
    aliases = {a.alias_name for a in client.get_aliases().aliases}
    if DOCUMENTS_COLLECTION not in real_collections and DOCUMENTS_COLLECTION not in aliases:
        client.create_collection(
            collection_name=DOCUMENTS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            },
        )
    ensure_payload_indexes(client)


def ensure_payload_indexes(client: QdrantClient) -> None:
    """
    Ensure a keyword index exists on the source_path payload field.

    Every ingest_file() call (rag_engine/ingest.py step 0) and every
    rag_sync_nfs.py deleted-file cleanup deletes points via
    `Filter(must=[FieldCondition(key="source_path", match=MatchValue(...))])`.
    Without a payload index, Qdrant must linearly scan every point's payload
    in the collection to satisfy that filter. At the corpus sizes this
    collection reaches (millions of points), that scan routinely exceeds the
    QdrantClient timeout (60s — see get_client()) and ingestion fails with
    "ResponseHandlingException: timed out" — see RAG-troubleshoot.md
    (2026-09-13 investigation) for the incident that identified this.

    Idempotent and cheap after the first call in a process: checks a
    module-level cache before hitting Qdrant, then checks the collection's
    live payload_schema before issuing create_payload_index (itself also
    idempotent — Qdrant no-ops if the index already exists). Building the
    index over an existing large collection happens in the background on
    the Qdrant server; this call returns as soon as the index is registered.
    """
    global _payload_index_ready
    if _payload_index_ready:
        return
    try:
        info = client.get_collection(DOCUMENTS_COLLECTION)
        existing = info.payload_schema or {}
    except Exception:
        return  # collection not created yet; nothing to index
    if FIELD_SOURCE_PATH in existing:
        _payload_index_ready = True
        return
    logger.info(
        "Creating payload index on '%s' (one-time; Qdrant builds it in the background)",
        FIELD_SOURCE_PATH,
    )
    client.create_payload_index(
        collection_name=DOCUMENTS_COLLECTION,
        field_name=FIELD_SOURCE_PATH,
        field_schema=PayloadSchemaType.KEYWORD,
    )
    _payload_index_ready = True


def collection_has_sparse(client: QdrantClient) -> bool:
    """Return True if the documents collection has the sparse_text vector space.

    Used by ingest.py to decide whether to compute and include sparse vectors.
    False before rag_sparse_migrate.py runs; True after migration + swap.
    """
    try:
        info = client.get_collection(DOCUMENTS_COLLECTION)
        sparse = getattr(info.config.params, "sparse_vectors", None)  # qdrant-client uses "sparse_vectors", not "sparse_vectors_config"
        return sparse is not None and SPARSE_VECTOR_NAME in sparse
    except Exception:
        return False
