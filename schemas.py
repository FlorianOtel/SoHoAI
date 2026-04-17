"""Pydantic schemas for the HomeAI-Lab API."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# -- Chat Request/Response -----------------------------------------------------

class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    role: Role
    content: str
    timestamp: Optional[datetime] = None

    def to_llm_dict(self) -> dict:
        """Strip metadata for LLM API call."""
        return {"role": self.role.value, "content": self.content}


class ChatRequest(BaseModel):
    """Incoming chat request — OpenAI-compatible with extensions."""
    model: Optional[str] = None  # None = use routing policy
    messages: list[Message]
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stream: bool = False
    # HomeAI-Lab extensions
    user_id: Optional[str] = None  # Google OAuth owner (e.g. "florian"); set by auth middleware
    use_rag: bool = False
    force_cloud: bool = False


class ChatResponse(BaseModel):
    chat_id: str
    model_used: str
    message: Message
    rag_sources: Optional[list[str]] = None


# -- Chat Management -----------------------------------------------------------

class ChatSummary(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int
    model_used: str


class ChatExport(BaseModel):
    chat_id: str
    title: str
    messages: list[Message]
    metadata: dict


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
