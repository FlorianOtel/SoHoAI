"""
Document parsing, chunking, and Qdrant upsert for the RAG pipeline.

Implements the 7-step atomic worker loop from RAG-strategy.md §4.4.
Called by the ingestion daemon (utils/rag_ingest_daemon.py) and the
FastAPI ingest endpoints in main.py.

Chunking strategy (RAG-strategy.md §3.4):
  - Parent-child (PDF, DOCX, IPYNB, long MD): child ~250 tok embedded;
    parent ~1000 tok stored as context returned to the LLM.
  - Flat 512-tok (PPTX, YAML, CSV, short docs): parent == child.

Parsing:
  - Structured formats (PDF, DOCX, PPTX): docling → markdown export.
  - IPYNB: dedicated cell extractor (JSON parse → markdown + code cells).
  - Text formats (TXT, MD, YAML, CSV): direct UTF-8 read.
  - docling failures fall back to raw UTF-8 read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

import tiktoken
from docling.document_converter import DocumentConverter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
)

from .collection import DOCUMENTS_COLLECTION, ensure_collection
from .embeddings import embed_batch
from .schema import (
    FIELD_CHUNK_INDEX,
    FIELD_FILE_NAME,
    FIELD_FILE_TYPE,
    FIELD_OWNER,
    FIELD_PAGE,
    FIELD_PARENT_TEXT,
    FIELD_SOURCE_PATH,
    FIELD_TAG,
    FIELD_TEXT,
)
from .state import StateDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking parameters (RAG-strategy.md §3.4)
# ---------------------------------------------------------------------------

_PARENT_CHUNK_SIZE    = 1000   # tokens — midpoint of 800–1200 range
_PARENT_CHUNK_OVERLAP = 100
_CHILD_CHUNK_SIZE     = 250    # tokens — midpoint of 200–300 range; safe under bge-m3's 8192-token context
_CHILD_CHUNK_OVERLAP  = 20
_FLAT_CHUNK_SIZE      = 512
_FLAT_CHUNK_OVERLAP   = 50

# Always use flat chunking for these types (compact; parent-child overhead not worth it)
_ALWAYS_FLAT_TYPES = {"pptx", "yaml", "yml", "csv"}

# If total token count is below this threshold, use flat even for PDF/IPYNB/MD
_SHORT_DOC_TOKENS = 600

# Structured formats that go through docling; everything else is direct text read
_DOCLING_TYPES = {"pdf", "docx", "pptx"}   # ipynb handled by _parse_ipynb()

# tiktoken cl100k_base — consistent encoding across ingest and search
_enc = tiktoken.get_encoding("cl100k_base")

# DocumentConverter — instantiated once at module load; designed for reuse
_converter = DocumentConverter()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_ipynb(path: Path) -> str:
    """
    Extract clean text from a Jupyter notebook.

    Concatenates markdown and code cells in order, separated by blank lines.
    Skips raw cells and empty cells. Cell outputs are ignored — source only.
    Falls back to raw UTF-8 read if the file is not valid JSON.
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")

    parts: list[str] = []
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        text = "".join(source).strip() if isinstance(source, list) else str(source).strip()
        if not text:
            continue
        if cell_type == "markdown":
            parts.append(text)
        elif cell_type == "code":
            parts.append(f"```python\n{text}\n```")
    return "\n\n".join(parts)


def _parse_to_text(file_path: str, file_type: str) -> str:
    """
    Extract full text from a document.

    Structured formats (pdf, docx, pptx) go through docling → Markdown.
    IPYNB: dedicated cell extractor (avoids docling which doesn't support it).
    Text formats (txt, md, yaml, csv): direct UTF-8 read.
    docling failures fall back to raw UTF-8 read.

    Synchronous and potentially slow for large PDFs; callers use asyncio.to_thread().
    """
    path = Path(file_path)
    if file_type == "ipynb":
        return _parse_ipynb(path)
    if file_type not in _DOCLING_TYPES:
        return path.read_text(encoding="utf-8", errors="replace")

    try:
        result = _converter.convert(source=str(path))
        return result.document.export_to_markdown()
    except Exception as exc:
        logger.warning(
            "docling failed for %s (%s) — falling back to raw read: %s",
            path.name, file_type, exc,
        )
        return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Token-based chunking
# ---------------------------------------------------------------------------

def _split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap. Preserves order."""
    tokens = _enc.encode(text)
    if not tokens:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def _chunk_parent_child(text: str) -> list[tuple[str, str]]:
    """
    Split text using the parent-child strategy.

    Each parent (~1000 tok) is split into smaller children (~250 tok).
    The child is what gets embedded (precise similarity); the parent is
    stored in the Qdrant payload and returned to the LLM for richer context.

    Returns: list of (child_text, parent_text) pairs.
    """
    pairs: list[tuple[str, str]] = []
    for parent in _split_tokens(text, _PARENT_CHUNK_SIZE, _PARENT_CHUNK_OVERLAP):
        for child in _split_tokens(parent, _CHILD_CHUNK_SIZE, _CHILD_CHUNK_OVERLAP):
            pairs.append((child, parent))
    return pairs


def _chunk_flat(text: str) -> list[tuple[str, str]]:
    """
    Flat chunking for compact files (PPTX, short TXT, YAML, CSV).

    Returns (chunk, chunk) pairs — parent == child (no extra context layer).
    """
    return [(c, c) for c in _split_tokens(text, _FLAT_CHUNK_SIZE, _FLAT_CHUNK_OVERLAP)]


def _select_chunks(text: str, file_type: str) -> tuple[list[tuple[str, str]], str]:
    """
    Choose chunking strategy and return (pairs, strategy_label).

    Flat when:
      - file_type is in _ALWAYS_FLAT_TYPES (pptx, yaml, csv), OR
      - document is short (< _SHORT_DOC_TOKENS tokens).
    Parent-child for everything else.
    """
    if file_type in _ALWAYS_FLAT_TYPES:
        return _chunk_flat(text), "flat"
    if len(_enc.encode(text)) < _SHORT_DOC_TOKENS:
        return _chunk_flat(text), "flat(short)"
    return _chunk_parent_child(text), "parent-child"


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

async def ingest_file(
    file_path: str,
    owner: str,
    rag_cfg: dict,
    state_db: StateDB,
    qdrant_client: QdrantClient,
    tag: str = "",
) -> None:
    """
    Atomically parse, chunk, embed, and upsert one document into Qdrant.

    Implements the 7-step atomic worker loop from RAG-strategy.md §4.4:
      0. Delete stale Qdrant points for this source_path (idempotency).
      1. Mark file as 'processing' in StateDB.
      2. Parse document → full text.
      3. Chunk text → (child, parent) pairs.
      4. Embed child texts via Ollama.
      5. Build PointStruct list with full payload.
      6. Single atomic upsert to Qdrant.
      7. Mark file as 'completed' in StateDB.

    On any failure: marks the file as failed (with auto-retry logic in
    StateDB) and re-raises so the daemon loop can move to the next file.

    Args:
        file_path:     Absolute NFS path to the file.
        owner:         Owner string (e.g. "florian") — from StateDB row.
        rag_cfg:       config["rag"] dict — embedding_model, ollama_url.
        state_db:      StateDB instance for ingestion status tracking.
        qdrant_client: Qdrant client in local persistent mode (NAS).
        tag:           Optional tag derived by the NFS scanner (e.g. "certifications").
    """
    model      = rag_cfg.get("embedding_model", "bge-m3")
    ollama_url = rag_cfg.get("ollama_url", "http://localhost:11434/api/embeddings")

    file_name = Path(file_path).name
    file_type = Path(file_path).suffix.lstrip(".").lower()

    try:
        # ------------------------------------------------------------------
        # Step 0: delete stale Qdrant points (delete-before-insert)
        #
        # Ensures re-ingestion of a modified file never leaves duplicate points.
        # Even if the collection is brand-new, deleting 0 matching points is valid.
        # ------------------------------------------------------------------
        ensure_collection(qdrant_client)
        qdrant_client.delete(
            collection_name=DOCUMENTS_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key=FIELD_SOURCE_PATH,
                            match=MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )

        # ------------------------------------------------------------------
        # Step 1: mark processing
        # ------------------------------------------------------------------
        state_db.mark_processing(file_path)

        # ------------------------------------------------------------------
        # Step 2: parse document
        #
        # _parse_to_text is blocking (docling can take minutes on large PDFs).
        # asyncio.to_thread keeps the event loop responsive for other tasks.
        # ------------------------------------------------------------------
        state_db.set_progress(file_path, "parsing")
        text = await asyncio.to_thread(_parse_to_text, file_path, file_type)

        if not text.strip():
            logger.warning("Empty document, skipping: %s", file_path)
            state_db.mark_completed(file_path)
            return

        # ------------------------------------------------------------------
        # Step 3: chunk
        # ------------------------------------------------------------------
        pairs, strategy = _select_chunks(text, file_type)
        n = len(pairs)
        state_db.set_progress(file_path, f"chunking ({n} chunks)")
        logger.info(
            "Chunked %s → %d chunks  strategy=%s  file_type=%s",
            file_name, n, strategy, file_type,
        )

        if not pairs:
            logger.warning("No chunks produced, skipping: %s", file_path)
            state_db.mark_completed(file_path)
            return

        # ------------------------------------------------------------------
        # Step 4: embed child texts via Ollama (8-concurrent semaphore)
        # ------------------------------------------------------------------
        child_texts = [child for child, _ in pairs]
        state_db.set_progress(file_path, f"embedding 0/{n}")
        vectors = await embed_batch(child_texts, model=model, ollama_url=ollama_url)
        state_db.set_progress(file_path, f"embedding {n}/{n}")

        # ------------------------------------------------------------------
        # Step 5: build PointStructs
        #
        # Random UUIDs — step 0 handles cleanup before re-insert, so there
        # are no orphan-ID concerns when chunking produces a different count.
        # ------------------------------------------------------------------
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    FIELD_TEXT:        child_text,
                    FIELD_PARENT_TEXT: parent_text,
                    FIELD_OWNER:       owner,
                    FIELD_SOURCE_PATH: file_path,
                    FIELD_FILE_NAME:   file_name,
                    FIELD_FILE_TYPE:   file_type,
                    FIELD_PAGE:        0,   # page-level tracking: future enhancement
                    FIELD_CHUNK_INDEX: chunk_idx,
                    FIELD_TAG:         tag,
                },
            )
            for chunk_idx, ((child_text, parent_text), vector) in enumerate(zip(pairs, vectors))
        ]

        # ------------------------------------------------------------------
        # Step 6: single atomic upsert
        # ------------------------------------------------------------------
        state_db.set_progress(file_path, "upserting")
        qdrant_client.upsert(
            collection_name=DOCUMENTS_COLLECTION,
            points=points,
        )

        # ------------------------------------------------------------------
        # Step 7: mark completed
        # ------------------------------------------------------------------
        state_db.mark_completed(file_path)
        logger.info(
            "Ingested %s: %d points  owner=%s  tag=%s",
            file_name, len(points), owner, tag or "—",
        )

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", file_path, exc, exc_info=True)
        state_db.mark_failed(file_path, str(exc))
        raise
