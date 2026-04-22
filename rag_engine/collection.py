"""
Qdrant collection configuration — single source of truth.

All code that creates, opens, or queries the documents collection imports
constants and helpers from here. No collection names, vector dimensions,
or distance metrics are hardcoded anywhere else.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ---------------------------------------------------------------------------
# Collection constants
# ---------------------------------------------------------------------------

DOCUMENTS_COLLECTION = "documents"
VECTOR_SIZE = 1024          # bge-m3 output dimension (same as mxbai-embed-large)
DISTANCE = Distance.COSINE


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client(url: str) -> QdrantClient:
    """Connect to a running Qdrant server.
    
    Timeout set to 60 seconds to handle index optimization on large batches.
    During heavy ingestion (e.g., 70K+ points), Qdrant may take >5 seconds to
    respond to delete/upsert requests while it optimizes indexes. Default httpx
    timeout (~5s) is too short; 60s allows adequate time.
    
    See: /TROUBLESHOOTING.md — "Qdrant HTTP Timeouts"
    """
    return QdrantClient(url=url, timeout=60)


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    """
    Create the documents collection if it does not already exist.

    Safe to call on every startup — no-op when the collection is already present.
    """
    existing = {c.name for c in client.get_collections().collections}
    if DOCUMENTS_COLLECTION not in existing:
        client.create_collection(
            collection_name=DOCUMENTS_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=DISTANCE),
        )
