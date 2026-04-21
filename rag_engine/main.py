"""Main application entry point and router."""

import asyncio
import logging
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Third-party imports (assuming pydantic/fastapi/httpx/etc. are installed)
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
import httpx
from redis.asyncio import Redis
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import yaml # Added yaml import

# Local imports
from schemas import ChatRequest, ChatResponse, Role, RagMode, IngestRequest, SearchRequest, SearchResult, ChatSummary, ChatExport
from rag_engine.tool_use import build_tool_spec, parse_tool_call, format_tool_result
from prompts.rag_system_prompts import build_system_prompt
from rag_engine.embeddings import embed_text # Placeholder for now, will implement
from rag_engine.search import search_rag # Placeholder for now, will implement
from rag_engine.collection import ensure_collection, get_qdrant_client # Placeholder for now
from rag_engine.state import get_db_engine, get_rag_state_db_path # Placeholder for now
from utils.rag_sync_nfs import scan_nfs_roots # Placeholder for now
from utils.rag_ingest_daemon import start_ingest_daemon # Placeholder for now
from utils.rag_reset import rag_reset # Placeholder for now

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State Management ----------------------------------------------------

app = FastAPI(title="HomeAI-Lab RAG Orchestrator")
app.state = {}

# Cache for conversation history (Redis)
redis_client: Optional[Redis] = None

# Qdrant Client (Global state)
qdrant_client: Optional[Any] = None

# RAG Configuration (Global state)
rag_cfg: Dict[str, Any] = {}

# LLM Inference components (Global state)
target_model: Any = None
kv_cache_manager: Any = None
variant_llm_fn: Any = None

# --- Startup/Shutdown Hooks ------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global redis_client, qdrant_client, rag_cfg, target_model, kv_cache_manager, variant_llm_fn

    # 1. Load configuration
    try:
        with open("config.yaml", 'r') as f:
            global rag_cfg
            rag_cfg = yaml.safe_load(f)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError:
        logger.error("config.yaml not found. RAG features will be disabled.")
        rag_cfg = {}

    # 2. Initialize external services
    try:
        redis_client = Redis.from_url(f"redis://{rag_cfg['redis']['host']}:{rag_cfg['redis']['port']}")
        await redis_client.ping()
        logger.info("Redis connection established.")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}")
        redis_client = None

    # 3. Initialize Qdrant Client
    try:
        global qdrant_client
        qdrant_url = rag_cfg.get('rag', {}).get('qdrant_url')
        if qdrant_url:
            # Use the correct IP for Ollama embeddings server 1
            qdrant_client = await httpx.AsyncClient(timeout=30) # Use async client for API calls
            logger.info("Qdrant client initialized (placeholder).")
        else:
            logger.warning("Qdrant URL missing in config. RAG features disabled.")
    except Exception as e:
        logger.error(f"Error initializing Qdrant client: {e}")
        qdrant_client = None

    # 4. Initialize LLM/KV Cache (Placeholder logic from RAG-strategy.md §3.1)
    # This is where the complex llama-server/Ollama logic would live.
    logger.info("LLM and KV cache system initialized (placeholder).")
    
    # 5. Setup RAG dependencies
    if qdrant_client:
        ensure_collection(qdrant_client)
        logger.info("RAG document collection checked/ensured.")

async def shutdown_event():
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis client closed.")

app.add_event_handler("shutdown", shutdown_event)

# --- Core RAG Pipeline Functions (Implementation target for §8.2 & §8.3) -----------------

async def _retrieve(query: str, user_id: Optional[str], state: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Retrieves documents based on query.
    This function is the hook where §8.3 (Multi-query) or §8.2 (Single-query) is selected.
    """
    global rag_cfg, qdrant_client
    
    if not qdrant_client:
        logger.warning("Qdrant client unavailable. Returning empty results.")
        return []

    # Check for multi-query enablement (Section 8.3)
    mq_cfg = rag_cfg.get("multi_query", {})
    mq_enabled = mq_cfg.get("enabled", False)
    
    if mq_enabled:
        logger.info("Executing Multi-Query Search (MMR).")
        # Placeholder: Replace this with call to rag_engine/multi_query.py
        # return await multi_query_search(
        #     query=query, user_id=user_id, limit=rag_cfg.get("top_k", 5),
        #     qdrant_client=qdrant_client, rag_cfg=rag_cfg,
        #     llm_fn=state.variant_llm_fn,
        # )
        
        # Mock return for now
        return [{"content": "Mock MMR result 1", "source_path": "/mock/path/1", "score": 0.9}]
    else:
        logger.info("Executing Standard Search Query.")
        # Placeholder: Replace this with call to rag_engine/search.py
        # return await search_rag(
        #     query=query, user_id=user_id, limit=rag_cfg.get("top_k", 5),
        #     qdrant_client=qdrant_client, rag_cfg=rag_cfg,
        # )
        
        # Mock return for now
        return [{"content": "Mock Standard result 1", "source_path": "/mock/path/2", "score": 0.8}]


async def _run_inference(
    target_model_name: str, 
    messages: list[Dict[str, str]], 
    cache: Dict[str, Any], 
    slot_id: int, 
    req: Request, 
    router: Any
) -> tuple[str, str]:
    """
    Simulates the core LLM inference call, handling both specialist (KV-cache) and external paths.
    (Placeholder for complex llama-server/LiteLLM integration)
    """
    logger.info(f"Running inference on model: {target_model_name}")
    
    # Mock response: In a real scenario, this would be the actual API call result
    mock_response = "This is a mock assistant response. I am ready to use the search tool."
    return mock_response, "specialist"

# --- API Endpoints --------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    global qdrant_client, rag_cfg
    
    rag_mode = request.rag_mode
    
    # 1. Resolve effective mode (Section 8.1)
    # Effective mode is request.rag_mode, unless it is None, then it comes from config.
    effective_mode: RagMode = rag_mode if rag_mode != RagMode.on else rag_cfg.get("rag", {}).get("default_mode", "on")
    
    logger.info(f"Incoming chat request. Effective RAG Mode: {effective_mode.value}")

    # 2. Tool-use loop initialization (Section 8.2)
    tool_spec = build_tool_spec() if effective_mode != RagMode.off else None
    system_prompt = build_system_prompt(effective_mode, tool_spec)

    # Initialize conversation state/history (Mocked)
    messages: list[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ] + [m.to_llm_dict() for m in request.messages]
    
    rag_sources: list[str] | None = None
    rag_chunks_used: list[Dict[str, Any]] = []
    
    # 3. Run Tool-use loop (Section 8.2)
    max_iter = rag_cfg.get("rag", {}).get("tool_use", {}).get("max_iterations", 2)
    assistant_content = ""
    
    for iteration in range(max_iter + 1):
        # Step 5a: Run inference
        assistant_text, model_used = await _run_inference(
            rag_cfg.get("routing", {}).get("default_model", "specialist"), 
            messages, 
            {}, 
            0, 
            request, 
            None # router
        )

        # Step 5b: tool-call parse
        tool_call = parse_tool_call(assistant_text) if effective_mode != RagMode.off else None
        
        if tool_call is None or iteration == max_iter:
            # No tool call, or we've hit the cap — this is the final answer.
            assistant_content = assistant_text
            break

        # Step 5c: dispatch the tool
        if tool_call["name"] != "search_documents":
            logger.warning("Unknown tool call: %s", tool_call["name"])
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "tool", "content": f"Unknown tool: {tool_call['name']}"})
            continue

        query = tool_call["arguments"].get("query", "").strip()
        if not query:
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "tool", "content": "Empty query"})
            continue

        # Step 5d: retrieve and process
        chunks = await _retrieve(query, request.user_id, app.state)
        rag_chunks_used.extend(chunks)
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "content": format_tool_result(chunks)})

    # Step 6: build rag_sources from *actual* tool results, dedup preserving order.
    if rag_chunks_used:
        seen: set[str] = set()
        rag_sources: list[str] = []
        for c in rag_chunks_used:
            if c["source_path"] not in seen:
                seen.add(c["source_path"])
                rag_sources.append(c["source_path"])
    
    # 4. Build Response
    response_message = Message(role=Role.assistant, content=assistant_content)
    response = ChatResponse(
        chat_id=request.chat_id,
        model_used=rag_cfg.get("routing", {}).get("default_model", "specialist"),
        message=response_message,
        rag_sources=rag_sources,
        rag_mode_used=effective_mode
    )
    
    return JSONResponse(content=response.model_dump())


@app.post("/v1/rag/ingest/sync")
async def trigger_sync_nfs():
    global qdrant_client
    
    if not qdrant_client:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant service is unavailable.")
    
    logger.info("Triggering NFS synchronization and Qdrant cleanup.")
    try:
        # Placeholder: Replace with actual call to utils/rag_sync_nfs.py
        # await scan_nfs_roots() 
        return {"status": "success", "message": "NFS scan and Qdrant sync initiated (mocked)."}
    except Exception as e:
        logger.error(f"NFS sync failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"NFS sync failed: {e}")


@app.post("/v1/rag/ingest/start")
async def start_ingestion_daemon():
    global qdrant_client
    if not qdrant_client:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Qdrant service is unavailable.")
    
    logger.info("Starting ingestion daemon.")
    # Placeholder: Replace with actual call to utils/rag_ingest_daemon.py
    # start_ingest_daemon()
    return {"status": "success", "message": "Ingestion daemon started (mocked)."}


@app.post("/v1/rag/ingest/stop")
async def stop_ingestion_daemon():
    logger.info("Stopping ingestion daemon.")
    # Placeholder: Implement graceful stop logic
    return {"status": "success", "message": "Ingestion daemon stopped (mocked)."}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "HomeAI-Lab"}