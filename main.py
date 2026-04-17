"""
HomeAI-Lab — API Gateway & Orchestrator

This is the central nervous system running on Server 1.
It wires together:
  - Smart model routing (LiteLLM → local GPU / local CPU / cloud)
  - Short-term memory (Redis conversation cache)
  - Long-term memory (SQLite chat persistence on NAS)
  - RAG pipeline (Phase 2)
  - MCP tool gateway (Phase 3)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from schemas import (
    ChatExport,
    ChatRequest,
    ChatResponse,
    ChatSummary,
    Message,
    Role,
)
from chat_store import ChatStore
from conversation import ConversationCache
from rag_engine import search_rag
from rag_engine.collection import ensure_collection, get_client
from rag_engine.ingest import ingest_file
from rag_engine.scanner import scan_nfs_roots
from rag_engine.state import StateDB
from router import SmartRouter
from kv_cache import KVCacheManager, apply_mistral_template

load_dotenv()

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("HomeAI-Lab")

# -- Load config ---------------------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

_db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI-lab--databases")


# -- App lifecycle -------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting HomeAI-Lab orchestrator...")

    chat_cfg = config.get("chat", {})
    llama_cfg = config.get("llama_server")
    kv_cache_instance: KVCacheManager | None = None
    if llama_cfg:
        base_url = llama_cfg["base_url"]
        if "${server2_ip}" in base_url:
            base_url = base_url.replace("${server2_ip}", config.get("server2_ip", "192.168.1.95"))

        kv_cache_instance = KVCacheManager(
            base_url=base_url,
            slot_save_path=llama_cfg["slot_save_path"],
            num_slots=llama_cfg.get("num_slots", 1),
            timeout=llama_cfg.get("timeout_seconds", 120),
        )
        logger.info(f"KV cache manager: {kv_cache_instance.num_slots} slot(s) ✓")

    redis_cfg = config.get("redis", {})
    redis_url = f"redis://{redis_cfg.get('host', '127.0.0.1')}:{redis_cfg.get('port', 6379)}/{redis_cfg.get('db', 0)}"
    app.state.cache = ConversationCache(
        redis_url=redis_url,
        default_ttl=redis_cfg.get("default_ttl_seconds", 86400),
        max_turns=chat_cfg.get("max_turns_in_context", 50),
        kv_cache=kv_cache_instance,
        summarize_threshold_chars=chat_cfg.get("summarize_threshold_chars", 200_000),
        summarize_keep_turns=chat_cfg.get("summarize_keep_turns", 20),
    )

    if await app.state.cache.ping():
        logger.info("Redis connected ✓")
    else:
        logger.warning("Redis not available — conversation cache disabled")

    app.state.store = ChatStore(
        db_path=f"{_db_base}/sqlite/chats.db"
    )
    logger.info("Chat store (SQLite) ✓")

    app.state.router = SmartRouter(config_path="config.yaml")
    logger.info(f"Router: {app.state.router.available_models}")

    async def summarize_fn(text: str) -> str:
        prompt = (
            "Summarize the following conversation excerpt concisely, "
            "preserving all key facts, decisions, and context needed to continue:\n\n"
            + text
        )
        resp = await app.state.router.complete(
            messages=[{"role": "user", "content": prompt}],
            model="specialist",
            force_cloud=False,
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    app.state.summarize_fn = summarize_fn
    app.state.rag_cfg = config.get("rag", {})
    qdrant_path = f"{_db_base}/qdrant"
    try:
        app.state.qdrant_client = get_client(qdrant_path)
        ensure_collection(app.state.qdrant_client)
        logger.info("Qdrant connected and collection ready ✓")
    except Exception as exc:
        logger.warning("Qdrant unavailable — RAG disabled: %s", exc)
        app.state.qdrant_client = None

    rag_state_db_path = f"{_db_base}/sqlite/rag_state.db"
    app.state.state_db = StateDB(rag_state_db_path)
    app.state.ingest_task: asyncio.Task | None = None
    app.state.ingest_stop = asyncio.Event()
    logger.info("RAG state DB ✓")

    yield

    await app.state.cache.close()
    logger.info("Orchestrator shut down cleanly.")


# -- FastAPI app ---------------------------------------------------------------

app = FastAPI(
    title="HomeAI-Lab",
    description="Distributed two-server AI orchestrator",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
#  CHAT ENDPOINT
# =============================================================================

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat endpoint. Orchestrator owns all state."""
    return await _server_managed_completion(req, app.state.router)


# =============================================================================
#  Chat pipeline
#  Orchestrator owns all state: Redis (short-term), SQLite (long-term),
#  GPU KV cache slot (prefix acceleration).
#  Client sends only the latest message each turn.
# =============================================================================

def _build_rag_prompt(user_query: str, chunks: list[dict]) -> str:
    """Inject retrieved context chunks into the user query for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_path", "unknown")
        text = chunk.get("content", "")
        context_parts.append(f"[{i}] (Source: {source})\n{text}")
    context_block = "\n\n".join(context_parts)
    return (
        "Use the following context to answer the user's question. "
        "If the context doesn't contain relevant information, say so.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {user_query}"
    )


async def _server_managed_completion(req: ChatRequest, router: SmartRouter):
    chat_id = req.chat_id
    cache: ConversationCache = app.state.cache
    store: ChatStore = app.state.store

    # -- 0. Resume KV slot (assign + restore from NFS) ------------------------
    slot_id: int | None = await cache.resume(chat_id)

    # -- 0b. Cold-start: Redis expired but chat exists in SQLite --------------
    if await cache.is_cold(chat_id):
        existing = store.get_chat(chat_id)
        if existing and existing.messages:
            await cache.warm_from_store(chat_id, existing.messages)
            slot_id = await cache.resume(chat_id)
            logger.info(f"Cold resume: chat {chat_id[:8]} reloaded from SQLite")

    # -- 1. Persist user message ----------------------------------------------
    user_msg = req.messages[-1]
    await cache.append(chat_id, user_msg.role.value, user_msg.content)
    store.save_message(chat_id, user_msg.role.value, user_msg.content)
    store.auto_title(chat_id)

    # -- 2. Rolling summarization ---------------------------------------------
    summarized = await cache.maybe_summarize(chat_id, app.state.summarize_fn)
    if summarized:
        # maybe_summarize() erased the KV slot (prompt changed); re-assign.
        slot_id = await cache.resume(chat_id)
        logger.info(f"Context summarized for chat {chat_id[:8]}")

    # -- 3. Build context from Redis ------------------------------------------
    history = await cache.get_context(chat_id)
    messages = [m.to_llm_dict() for m in history]

    # -- 4. RAG augmentation (Phase 2) ----------------------------------------
    rag_sources = None
    if req.use_rag and app.state.qdrant_client is not None:
        try:
            chunks = await search_rag(
                query=user_msg.content,
                user_id=req.user_id,
                limit=app.state.rag_cfg.get("top_k", 5),
                qdrant_client=app.state.qdrant_client,
                rag_cfg=app.state.rag_cfg,
            )
            if chunks:
                messages[-1]["content"] = _build_rag_prompt(user_msg.content, chunks)
                rag_sources = [c.get("source_path", "") for c in chunks]
        except Exception as exc:
            logger.warning("RAG search failed, proceeding without context: %s", exc)

    # -- 5. Route and infer ---------------------------------------------------
    target_model = router.select_model(messages, req.model, req.force_cloud)
    assistant_content: str
    model_used: str

    inference_ok = False
    used_specialist = (
        target_model == "specialist"
        and cache.kv_cache is not None
        and slot_id is not None
    )

    try:
        if used_specialist:
            prompt = apply_mistral_template(messages)
            result = await cache.kv_cache.inference(slot_id=slot_id, prompt=prompt)
            assistant_content = result["content"].strip()
            model_used = "specialist"
        else:
            response = await router.complete(
                messages=messages,
                model=req.model,
                force_cloud=req.force_cloud,
                stream=False,
            )
            assistant_content = response.choices[0].message.content
            model_used = response.get("model", req.model or "unknown")

        await cache.append(chat_id, "assistant", assistant_content)
        store.save_message(
            chat_id, "assistant", assistant_content,
            model_used=model_used,
            token_count=0,
        )
        inference_ok = True

    except Exception as e:
        logger.error(f"Inference failed (server-managed, chat {chat_id[:8]}): {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    finally:
        if inference_ok and used_specialist:
            await cache.park(chat_id)      # save KV slot + refresh Redis TTL
        elif not used_specialist:
            await cache.touch(chat_id)     # refresh Redis TTL (no KV slot used)
        else:
            # Specialist inference failed — discard undefined slot state
            if cache.kv_cache is not None:
                await cache.kv_cache.erase(chat_id)

    # -- 6. Return ------------------------------------------------------------
    return ChatResponse(
        chat_id=chat_id,
        model_used=model_used,
        message=Message(role=Role.assistant, content=assistant_content),
        rag_sources=rag_sources,
    )


# =============================================================================
#  CHAT MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/v1/chats", response_model=list[ChatSummary])
async def list_chats(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List saved chats (server-managed history only), most recent first."""
    return app.state.store.list_chats(limit=limit, offset=offset)


@app.get("/v1/chats/{chat_id}", response_model=ChatExport)
async def get_chat(chat_id: str):
    """Retrieve full chat history."""
    chat = app.state.store.get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return chat


@app.delete("/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat from Redis, KV cache (slot + NFS file), and SQLite."""
    await app.state.cache.clear(chat_id)
    app.state.store.delete_chat(chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@app.get("/v1/chats/{chat_id}/export/markdown", response_class=PlainTextResponse)
async def export_markdown(chat_id: str):
    """Export chat as Markdown."""
    md = app.state.store.export_markdown(chat_id)
    if not md:
        raise HTTPException(404, "Chat not found")
    return PlainTextResponse(md, media_type="text/markdown")


@app.post("/v1/chats/{chat_id}/export/save")
async def save_markdown_to_disk(chat_id: str):
    """Save chat as Markdown file to NAS."""
    md = app.state.store.export_markdown(chat_id)
    if not md:
        raise HTTPException(404, "Chat not found")

    export_dir = Path(config.get("nas_mount", "/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab")) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    chat = app.state.store.get_chat(chat_id)
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in chat.title)[:60]
    filename = f"{safe_title}_{chat_id[:8]}.md"
    filepath = export_dir / filename

    filepath.write_text(md, encoding="utf-8")
    return {"status": "saved", "path": str(filepath)}


@app.get("/v1/chats/{chat_id}/export/rl", response_class=PlainTextResponse)
async def export_rl_data(chat_id: str):
    """Export chat as JSONL for RL training."""
    jsonl = app.state.store.export_rl_jsonl(chat_id)
    if not jsonl:
        raise HTTPException(404, "No feedback data found for this chat")
    return PlainTextResponse(jsonl, media_type="application/jsonl")


@app.post("/v1/chats/{chat_id}/feedback")
async def submit_feedback(chat_id: str, message_index: int, signal: str, detail: str = ""):
    """
    Submit feedback on a specific message (for RL data collection).
    Signals: thumbs_up, thumbs_down, edited, regenerated
    """
    app.state.store.save_feedback(chat_id, message_index, signal, detail)
    return {"status": "recorded"}


# =============================================================================
#  RAG INGESTION ENDPOINTS  (Phase 2)
# =============================================================================

async def _ingest_worker(
    state_db: StateDB,
    qdrant_client,
    rag_cfg: dict,
    stop_event: asyncio.Event,
) -> None:
    """Background ingestion worker — runs as an asyncio Task."""
    state_db.crash_recovery()
    processed = failed = 0
    while not stop_event.is_set():
        rows = state_db.fetch_pending_full(limit=5)
        if not rows:
            break
        for row in rows:
            if stop_event.is_set():
                break
            try:
                await ingest_file(
                    file_path=row["file_path"],
                    owner=row["owner"],
                    rag_cfg=rag_cfg,
                    state_db=state_db,
                    qdrant_client=qdrant_client,
                )
                processed += 1
            except Exception:
                failed += 1
    logger.info("Ingest worker finished: %d processed, %d failed", processed, failed)


@app.post("/v1/rag/ingest/sync")
async def rag_ingest_sync(user: str | None = Query(None)):
    """
    Scan configured NFS roots and populate the ingestion queue.

    Runs the NFS scanner in a thread (blocking I/O). New and modified files
    are queued as 'pending'; deleted files are removed.

    Query params:
        user — scan only this owner's roots (e.g. ?user=florian)
    """
    result = await asyncio.to_thread(
        scan_nfs_roots,
        app.state.state_db,
        config,
        user,
    )
    counts = app.state.state_db.get_counts()
    return {**result, "queue": counts}


@app.post("/v1/rag/ingest/start")
async def rag_ingest_start():
    """
    Start the ingestion daemon as an asyncio background task.

    Returns immediately; the worker processes pending files in the background.
    Call GET /v1/rag/ingest/status to monitor progress.
    Call POST /v1/rag/ingest/stop to halt gracefully.
    """
    if app.state.qdrant_client is None:
        raise HTTPException(503, "Qdrant unavailable — RAG is disabled")

    task: asyncio.Task | None = app.state.ingest_task
    if task is not None and not task.done():
        return {"status": "already_running"}

    app.state.ingest_stop.clear()
    app.state.ingest_task = asyncio.create_task(
        _ingest_worker(
            app.state.state_db,
            app.state.qdrant_client,
            app.state.rag_cfg,
            app.state.ingest_stop,
        )
    )
    return {"status": "started"}


@app.post("/v1/rag/ingest/stop")
async def rag_ingest_stop():
    """Signal the background ingestion worker to stop after the current file."""
    app.state.ingest_stop.set()
    task: asyncio.Task | None = app.state.ingest_task
    running = task is not None and not task.done()
    return {"status": "stop_requested", "was_running": running}


@app.get("/v1/rag/ingest/status")
async def rag_ingest_status(user: str | None = Query(None)):
    """
    Return ingestion queue metrics and Qdrant point count.

    Query params:
        user — filter SQLite counts to this owner (e.g. ?user=florian)
    """
    state_db: StateDB = app.state.state_db
    counts = state_db.get_counts()

    if user:
        cur = state_db._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_queue "
            "WHERE owner = ? GROUP BY status",
            (user,),
        )
        raw = {row["status"]: row["n"] for row in cur.fetchall()}
        counts = {
            "pending":    raw.get("pending", 0),
            "processing": raw.get("processing", 0),
            "completed":  raw.get("completed", 0),
            "failed":     raw.get("failed", 0),
            "total":      sum(raw.values()),
        }

    qdrant_points: int | None = None
    if app.state.qdrant_client is not None:
        try:
            from rag_engine.collection import DOCUMENTS_COLLECTION
            qdrant_points = app.state.qdrant_client.count(
                DOCUMENTS_COLLECTION, exact=True
            ).count
        except Exception:
            pass

    task: asyncio.Task | None = app.state.ingest_task
    worker_running = task is not None and not task.done()

    progress_pct = (
        round(counts["completed"] / counts["total"] * 100, 1)
        if counts["total"] > 0 else 0.0
    )

    return {
        "queue":          counts,
        "progress_pct":   progress_pct,
        "qdrant_points":  qdrant_points,
        "worker_running": worker_running,
        "scope":          f"owner={user}" if user else "all",
    }


# =============================================================================
#  SYSTEM ENDPOINTS
# =============================================================================

@app.get("/health")
async def health():
    """System health check."""
    cache_ok = await app.state.cache.ping()
    return {
        "status": "ok",
        "redis": cache_ok,
        "models": app.state.router.available_models,
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {"models": app.state.router.available_models}


@app.get("/v1/models/health")
async def model_health():
    """Check which model endpoints are reachable."""
    return await app.state.router.health_check()


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
