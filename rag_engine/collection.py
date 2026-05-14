"""
Qdrant collection configuration — single source of truth.

All code that creates, opens, or queries the documents collection imports
constants and helpers from here. No collection names, vector dimensions,
or distance metrics are hardcoded anywhere else.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Modifier, SparseVectorParams, VectorParams

from .schema import SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection constants
# ---------------------------------------------------------------------------

DOCUMENTS_COLLECTION = "documents"
VECTOR_SIZE = 1024          # bge-m3 output dimension (same as mxbai-embed-large)
DISTANCE = Distance.COSINE


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
    """
    real_collections = {c.name for c in client.get_collections().collections}
    aliases = {a.alias_name for a in client.get_aliases().aliases}
    if DOCUMENTS_COLLECTION in real_collections or DOCUMENTS_COLLECTION in aliases:
        return   # already present (real collection or alias pointing to migrated collection)
    client.create_collection(
        collection_name=DOCUMENTS_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )


def collection_has_sparse(client: QdrantClient) -> bool:
    """Return True if the documents collection has the sparse_text vector space.

    Used by ingest.py to decide whether to compute and include sparse vectors.
    False before rag_sparse_migrate.py runs; True after migration + swap.
    """
    try:
        info = client.get_collection(DOCUMENTS_COLLECTION)
        sparse = getattr(info.config.params, "sparse_vectors_config", None)
        return sparse is not None and SPARSE_VECTOR_NAME in sparse
    except Exception:
        return False
