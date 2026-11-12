"""Pydantic schemas for the SoHoAI API.

Chat request/response/management models were removed 2026-11-12 along with the
stateful chat endpoints (Redis / SQLite / KV-cache). Only the RAG models remain.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# -- RAG (Phase 2) ------------------------------------------------------------

class IngestRequest(BaseModel):
    file_path: str
    collection: str = "documents"


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None  # owner filter — set by auth middleware
    collection: str = "documents"
    top_k: int = 5


class SearchResult(BaseModel):
    content: str
    source: str
    score: float
    metadata: dict = {}
