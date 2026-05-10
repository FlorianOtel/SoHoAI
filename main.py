"""
SoHoAI — API Gateway & Orchestrator

This is the central nervous system running on Server 1.
It wires together:
  - Smart model routing (LiteLLM → local GPU / local CPU / cloud)
  - Short-term memory (Redis conversation cache)
  - Long-term memory (SQLite chat persistence on NAS)
  - RAG pipeline (Phase 2 + §8 advanced features)
  - MCP tool gateway (Phase 3)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import litellm

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from schemas import (
    ChatExport,
    ChatRequest,
    ChatResponse,
    ChatSummary,
    Message,
    RagMode,
    Role,
)
from chat_store import ChatStore
from conversation import ConversationCache
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_engine import search_rag, multi_query_search
from rag_engine.collection import DOCUMENTS_COLLECTION, ensure_collection, get_client
from rag_engine.ingest import ingest_file
from rag_engine.scanner import scan_nfs_roots, scan_claude_chats
from rag_engine.schema import FIELD_SOURCE_PATH
from rag_engine.state import StateDB
from rag_engine.tool_use import build_tool_spec, format_tool_result, parse_tool_call
from prompts.rag_system_prompts import build_system_prompt
from router import SmartRouter
from kv_cache import KVCacheManager, apply_gemma_template

load_dotenv()

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("SoHoAI")

_ANTHROPIC_API_BASE = "https://api.anthropic.com"

# -- Load config ---------------------------------------------------------------
with open("config.yaml") as f:
    config = yaml.safe_load(f)

_db_base = config.get("db_base_path", "/mnt/nfs/__Backups/SoHoAI--databases")


# -- App lifecycle -------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting SoHoAI orchestrator...")

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

    app.state.router = SmartRouter(config_path="config.yaml", store=app.state.store)
    logger.info(f"Router: {app.state.router.available_models}")

    async def summarize_fn(text: str) -> str:
        # Note: intermediate tool-call / tool-result messages are not persisted
        # to Redis (only the final assistant answer), so this prompt sees only
        # user / assistant text — no tool-handling directive needed.
        prompt = (
            "Summarize the following conversation excerpt concisely, "
            "preserving all key facts, decisions, and context needed to continue:\n\n"
            + text
        )
        summarization_model = config.get("routing", {}).get("summarization_model", "internal/gemma-4-e4b")
        resp = await app.state.router.complete(
            messages=[{"role": "user", "content": prompt}],
            model=summarization_model,
            force_cloud=False,
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    app.state.summarize_fn = summarize_fn
    app.state.rag_cfg = config.get("rag", {})

    try:
        qdrant_url = app.state.rag_cfg.get("qdrant_url", "http://192.168.1.93:6333")
        app.state.qdrant_client = get_client(qdrant_url)
        ensure_collection(app.state.qdrant_client)
        logger.info("Qdrant connected and collection ready ✓")
    except Exception as exc:
        logger.warning("Qdrant unavailable — RAG disabled: %s", exc)
        app.state.qdrant_client = None

    # Variant LLM function for multi-query expansion (§8.3).
    # Uses the internal model via LiteLLM's OpenAI-compat endpoint — no KV slot involved.
    _variant_model = app.state.rag_cfg.get("multi_query", {}).get("variant_model", "internal/gemma-4-e4b")

    async def variant_llm_fn(prompt: str) -> str:
        resp = await app.state.router.complete(
            messages=[{"role": "user", "content": prompt}],
            model=_variant_model,
            force_cloud=(_variant_model == "anthropic/claude-sonnet-4-6"),
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    app.state.variant_llm_fn = variant_llm_fn

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
    title="SoHoAI",
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
#  RAG HELPERS
# =============================================================================

def _apply_system_prompt(messages: list[dict], system_prompt: str) -> list[dict]:
    """Prepend a system message or replace an existing one."""
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": system_prompt}] + messages[1:]
    return [{"role": "system", "content": system_prompt}] + messages


def _fold_tool_messages(messages: list[dict]) -> list[dict]:
    """Fold role=tool messages into role=user for models without native tool support.

    Both the internal (Gemma chat template) and the external path (plain text
    conversation via LiteLLM) use this transformation so tool results are always
    readable by the model.
    """
    folded = []
    for m in messages:
        if m["role"] == "tool":
            folded.append({"role": "user", "content": f"Tool result:\n\n{m['content']}"})
        else:
            folded.append(m)
    return folded


async def _retrieve(query: str, user_id: str | None, file_types: list[str] | None = None) -> list[dict]:
    """Dispatch to multi-query+MMR or standard search based on config."""
    rag_cfg = app.state.rag_cfg
    limit = rag_cfg.get("top_k", 5)
    mq_enabled = rag_cfg.get("multi_query", {}).get("enabled", False)

    if mq_enabled:
        logger.info("RAG retrieve: multi-query+MMR  query=%r", query[:60])
        return await multi_query_search(
            query=query,
            user_id=user_id,
            limit=limit,
            qdrant_client=app.state.qdrant_client,
            rag_cfg=rag_cfg,
            llm_fn=app.state.variant_llm_fn,
        )

    logger.info("RAG retrieve: standard search  query=%r", query[:60])
    return await search_rag(
        query=query,
        user_id=user_id,
        limit=limit,
        qdrant_client=app.state.qdrant_client,
        rag_cfg=rag_cfg,
        file_types=file_types,
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
            await cache.warm_from_store(chat_id, store=store)
            slot_id = await cache.resume(chat_id)
            logger.info(f"Cold resume: chat {chat_id[:8]} reloaded from SQLite")

    # -- 1. Persist user message ----------------------------------------------
    user_msg = req.messages[-1]
    await cache.append(chat_id, user_msg.role.value, user_msg.content)
    store.save_message(chat_id, user_msg.role.value, user_msg.content)
    store.auto_title(chat_id)

    # -- 2. Rolling summarization ---------------------------------------------
    summarized = await cache.maybe_summarize(chat_id, app.state.summarize_fn, store=store)
    if summarized:
        slot_id = await cache.resume(chat_id)
        logger.info(f"Context summarized for chat {chat_id[:8]}")

    # -- 3. Build context from Redis ------------------------------------------
    history = await cache.get_context(chat_id)
    messages = [m.to_llm_dict() for m in history]

    # -- 4. RAG mode + system prompt (§8.1 / §8.2) ---------------------------
    # If the client sent rag_mode explicitly, honour it. Otherwise fall back to
    # the server-side default from config.yaml (rag.default_mode).
    if "rag_mode" in req.model_fields_set:
        rag_mode: RagMode = req.rag_mode
    else:
        rag_mode = RagMode(app.state.rag_cfg.get("default_mode", "off"))
    if rag_mode != RagMode.off and app.state.qdrant_client is None:
        logger.warning("RAG mode=%s requested but Qdrant unavailable; forcing off", rag_mode)
        rag_mode = RagMode.off

    tool_spec = build_tool_spec() if rag_mode != RagMode.off else None
    system_prompt = build_system_prompt(rag_mode, tool_spec)
    messages = _apply_system_prompt(messages, system_prompt)

    # -- 5. Tool-use loop (§8.2 / §8.3) -------------------------------------
    rag_cfg = app.state.rag_cfg
    max_iter = rag_cfg.get("tool_use", {}).get("max_iterations", 2)
    strip_on_final = rag_cfg.get("tool_use", {}).get("strip_on_final", True)

    rag_sources: list[str] | None = None
    rag_chunks_used: list[dict] = []
    assistant_content = ""
    model_used = "internal/gemma-4-e4b"
    used_internal = False
    inference_ok = False

    try:
        for iteration in range(max_iter + 1):
            # 5a. Select model and run inference
            target_model = router.select_model(messages, req.model, req.force_cloud)
            used_internal = (
                target_model.startswith("internal/")
                and cache.kv_cache is not None
                and slot_id is not None
            )

            llm_messages = _fold_tool_messages(messages)

            if used_internal:
                prompt = apply_gemma_template(llm_messages)
                result = await cache.kv_cache.inference(slot_id=slot_id, prompt=prompt)
                raw_text = result["content"].strip()
                model_used = target_model
            else:
                response = await router.complete(
                    messages=llm_messages,
                    model=req.model,
                    force_cloud=req.force_cloud,
                    stream=False,
                    metadata={"source": "cli_chat", "user_id": req.user_id, "chat_id": chat_id, "orchestra_session_id": None},
                )
                raw_text = response.choices[0].message.content
                model_used = response.get("model", req.model or "unknown")

            inference_ok = True

            # 5b. Parse tool call (only when RAG is active)
            tool_call = parse_tool_call(raw_text) if rag_mode != RagMode.off else None

            if tool_call is None or iteration == max_iter:
                # Final answer — optionally strip any stray tool-call tags
                if strip_on_final and tool_call is not None:
                    raw_text = re.sub(
                        r"<tool_call>.*?</tool_call>", "", raw_text, flags=re.DOTALL
                    ).strip()
                assistant_content = raw_text
                break

            # 5c. Dispatch known tool
            if tool_call["name"] != "search_documents":
                logger.warning("Unknown tool call: %s", tool_call["name"])
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "tool", "content": f"Unknown tool: {tool_call['name']}"})
                continue

            query = tool_call["arguments"].get("query", "").strip()
            file_types = tool_call["arguments"].get("file_types") or None
            if not query:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "tool", "content": "Empty query — please provide a search term."})
                continue

            # 5d. Retrieve and feed result back
            chunks = await _retrieve(query, req.user_id, file_types)
            rag_chunks_used.extend(chunks)
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "tool", "content": format_tool_result(chunks)})

    except Exception as e:
        logger.error(f"Inference failed (chat {chat_id[:8]}): {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    finally:
        if inference_ok and used_internal:
            await cache.park(chat_id)      # save KV slot + refresh Redis TTL
        elif inference_ok and not used_internal:
            await cache.touch(chat_id)     # refresh Redis TTL (no KV slot used)
        elif used_internal:
            # Inference failed — discard undefined slot state
            if cache.kv_cache is not None:
                await cache.kv_cache.erase(chat_id)

    # -- 6. Build rag_sources from actual tool results (dedup, order preserved)
    if rag_chunks_used:
        seen: set[str] = set()
        rag_sources = []
        for c in rag_chunks_used:
            path = c.get("source_path", "")
            if path and path not in seen:
                seen.add(path)
                rag_sources.append(path)

    # -- 7. Persist assistant message ----------------------------------------
    await cache.append(chat_id, "assistant", assistant_content)
    # Calculate token_count from response usage when using cloud model
    token_count = 0
    if not used_internal and 'response' in locals() and hasattr(response, 'usage') and response.usage:
        token_count = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
    store.save_message(
        chat_id, "assistant", assistant_content,
        model_used=model_used,
        token_count=token_count,
    )

    # -- 8. Return ------------------------------------------------------------
    return ChatResponse(
        chat_id=chat_id,
        model_used=model_used,
        message=Message(role=Role.assistant, content=assistant_content),
        rag_sources=rag_sources,
        rag_mode_used=rag_mode,
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

    export_dir = Path(config.get("nas_mount", "/mnt/nfs/Florian/Gin-AI/projects/SoHoAI")) / "exports"
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
    """Submit feedback on a specific message (for RL data collection)."""
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
    """Scan configured NFS roots and claude chat directories; populate the ingestion queue."""
    result_nfs = await asyncio.to_thread(
        scan_nfs_roots,
        app.state.state_db,
        config,
        user,
    )
    result_chats = await asyncio.to_thread(
        scan_claude_chats,
        app.state.state_db,
        config,
        user,
    )

    all_existing = result_nfs["existing_paths"] | result_chats["existing_paths"]
    deleted_paths, stale_paths = app.state.state_db.find_deleted(all_existing)

    # Qdrant cleanup BEFORE SQLite deletion — if killed mid-loop, the SQLite rows
    # survive intact and the next sync retries automatically (no orphaned Qdrant points).
    if deleted_paths and app.state.qdrant_client is not None:
        for path in deleted_paths:
            app.state.qdrant_client.delete(
                collection_name=DOCUMENTS_COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key=FIELD_SOURCE_PATH,
                                match=MatchValue(value=path),
                            )
                        ]
                    )
                ),
            )
        logger.info("Deleted Qdrant points for %d removed file(s)", len(deleted_paths))
    if stale_paths:
        app.state.state_db.purge_deleted(stale_paths)

    counts = app.state.state_db.get_counts()
    return {
        "nfs_scanned": result_nfs["scanned"],
        "chats_scanned": result_chats["scanned"],
        "deleted": len(stale_paths),
        "queue": counts,
    }


@app.post("/v1/rag/ingest/start")
async def rag_ingest_start():
    """Start the ingestion daemon as an asyncio background task."""
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
    """Return ingestion queue metrics and Qdrant point count."""
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


@app.get("/v1/rag/search")
async def rag_search(
    q: str,
    user: str | None = Query(None),
    top_k: int = Query(5, ge=1, le=20),
    file_types: list[str] | None = Query(None),
):
    """Retrieve RAG document hits without invoking any LLM.

    Intended for external LLM clients (Claude Code, Cline) that manage their own
    reasoning loop.  The caller decides when to search and how to use the results.
    """
    if app.state.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available")

    results = await search_rag(
        query=q,
        user_id=user,
        limit=top_k,
        qdrant_client=app.state.qdrant_client,
        rag_cfg=app.state.rag_cfg,
        file_types=file_types,
    )

    logger.info("RAG search: q=%r user=%s top_k=%d → %d result(s)", q, user, top_k, len(results))
    return {"query": q, "user": user, "results": results}


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
    """List available models in Anthropic format."""
    models = [
        {
            "type": "model",
            "id": public_id,
            "display_name": public_id,
            "created_at": "2025-01-01T00:00:00Z",
        }
        for public_id in _PROXY_EXPOSED_MODELS
    ]
    return {
        "data": models,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
        "has_more": False,
    }


@app.get("/v1/models/health")
async def model_health():
    """Check which model endpoints are reachable."""
    return await app.state.router.health_check()


@app.get("/v1/usage/stats")
async def usage_stats(
    user: str | None = Query(None),
    since: str | None = Query(None),
    until: str | None = Query(None),
    model: str | None = Query(None),
    source: str | None = Query(None),
    session_id: str | None = Query(None),
    group_by: str | None = Query(None),
) -> JSONResponse:
    """Query usage statistics across completions.

    Supported query parameters:
      - user: filter by user_id
      - since: ISO-8601 start time (default: 7 days ago UTC)
      - until: ISO-8601 end time (default: now UTC)
      - model: filter by model name
      - source: filter by source (orchestra, claude_code_native, cline, cli_chat)
      - session_id: filter by orchestra_session_id
      - group_by: aggregate by day, model, or source (default: no grouping)

    Returns:
      - window: {since, until}
      - totals: {requests, input_tokens, output_tokens, cache_tokens, cost_usd, cache_hit_rate}
      - by_model: model → totals (if not grouped by model)
      - by_source: source → totals (if not grouped by source)
      - by_day: date → totals (if grouped by day)
    """
    # Validate source
    valid_sources = {"orchestra", "claude_code_native", "cline", "cli_chat"}
    if source and source not in valid_sources:
        return JSONResponse(
            {"error": "invalid source", "valid": sorted(valid_sources)},
            status_code=400
        )

    # Validate group_by
    valid_group_by = {None, "day", "model", "source"}
    if group_by and group_by not in valid_group_by:
        return JSONResponse(
            {"error": "invalid group_by", "valid": sorted(g for g in valid_group_by if g)},
            status_code=400
        )

    # Default time window: 7 days ago to now (UTC)
    since_dt = None
    until_dt = None

    if since:
        try:
            since_dt = datetime.datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            return JSONResponse(
                {"error": "invalid since format", "expected": "ISO-8601"},
                status_code=400
            )
    else:
        since_dt = datetime.datetime.utcnow() - datetime.timedelta(days=7)

    if until:
        try:
            until_dt = datetime.datetime.fromisoformat(until.replace("Z", "+00:00"))
        except ValueError:
            return JSONResponse(
                {"error": "invalid until format", "expected": "ISO-8601"},
                status_code=400
            )
    else:
        until_dt = datetime.datetime.utcnow()

    # Convert back to ISO-8601 strings
    since_str = since_dt.isoformat() + "Z" if since_dt and not since_dt.isoformat().endswith("Z") else (since_dt.isoformat() if since_dt else None)
    until_str = until_dt.isoformat() + "Z" if until_dt and not until_dt.isoformat().endswith("Z") else (until_dt.isoformat() if until_dt else None)

    # Query usage stats
    result = app.state.store.query_usage_stats(
        since=since_str,
        until=until_str,
        user=user,
        model=model,
        source=source,
        session_id=session_id,
        group_by=group_by,
    )

    return JSONResponse(result)


# =============================================================================
#  Proxy pass-through — LiteLLM-compatible stateless endpoints (2026-04-22)
#
#  These endpoints let external clients (Cline VSCode plugin, Claude Code sub-agents, and any
#  OpenAI-compatible client) hit SoHoAI's LiteLLM Router directly,
#  bypassing the orchestrator's stateful machinery (Redis, SQLite, KV cache,
#  RAG tool-use loop, rolling summarization). Callers manage their own history.
#
#  Mimics the LiteLLM proxy response shapes so LiteLLM-compatible clients read
#  `model_info.max_input_tokens` correctly (v3.79.0 bug: defaults to 8K/128K
#  if the endpoint returns nothing or an empty API-key is set in the client).
#
#  Proxy base URL (Cline, Claude Code, etc.): http://192.168.1.93:8000/proxy
#  See CLAUDE.md §Design decisions and memory/project_shared_llama_server.md.
# =============================================================================

# Public-facing model name → internal router alias.
# Both paths reach the same LiteLLM Router, so prompt caching on external
# applies here too; Gemma routing still lands on the shared llama-server.
_PROXY_EXPOSED_MODELS: dict[str, str] = {
    # Local inference
    "internal/gemma-4-e4b": "internal/gemma-4-e4b",
    # Anthropic cloud
    "anthropic/claude-haiku-4-5": "claude-haiku-4-5",
    "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-7": "claude-opus-4-7",
    # Ollama cloud
    "ollama-cloud/deepseek-v4-pro": "ollama-cloud/deepseek-v4-pro",
    "ollama-cloud/kimi-k2.6": "ollama-cloud/kimi-k2.6",
    "ollama-cloud/glm-5.1": "ollama-cloud/glm-5.1",
    "ollama-cloud/qwen3-coder-next": "ollama-cloud/qwen3-coder-next",
}


def _build_proxy_model_entry(config: dict, public_name: str, internal_name: str) -> dict | None:
    """Build one entry for /proxy/v1/model/info from the config.yaml model_list."""
    internal = next(
        (m for m in config.get("model_list", []) if m.get("model_name") == internal_name),
        None,
    )
    if internal is None:
        return None
    # From the client's perspective, BOTH models are reached via the same
    # OpenAI-compatible /proxy/v1/chat/completions endpoint. Cline's LiteLLM
    # provider filters the model dropdown to entries with an `openai/` prefix,
    # so advertise both models that way — the proxy itself handles the internal
    # routing (Gemma → llama-server, Sonnet → Anthropic).
    litellm_params = {
        "model": f"openai/{public_name}",
        # api_base points at this proxy; any tool that uses it will loop back
        # here, which is what we want (single entry point for external clients).
        "api_base": f"http://192.168.1.93:8000/proxy/v1",
    }
    return {
        "model_name": public_name,
        "litellm_params": litellm_params,
        "model_info": internal.get("model_info", {}),
    }


_LEGACY_ALIASES: dict[str, str] = {
    "gemma-4-e4b":       "internal/gemma-4-e4b",
    "claude-haiku-4-5":  "claude-haiku-4-5",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-opus-4-7":   "claude-opus-4-7",
}


def _resolve_proxy_model(name: str | None) -> str | None:
    """Resolve a proxy client's model name to the internal router alias.

    Accepts any of:
      - bare public name:   "gemma-4-e4b", "ollama-cloud/deepseek-v4-pro"
      - provider-prefixed:  "openai/gemma-4-e4b", "anthropic/claude-sonnet-4-6", "ollama-cloud/deepseek-v4-pro"
      - legacy bare names:  "claude-sonnet-4-6" (backward compat)
      - internal aliases:   "internal", "external"  (for direct testing)
    Returns the internal router alias or None if unknown.
    """
    if not name:
        return None
    # Direct match on public name
    if name in _PROXY_EXPOSED_MODELS:
        return _PROXY_EXPOSED_MODELS[name]
    # Legacy aliases (bare names without prefix)
    if name in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[name]
    # Strip a provider prefix like "openai/...", "anthropic/...", "ollama-cloud/..."
    bare = name.split("/", 1)[-1] if "/" in name else name
    if bare in _PROXY_EXPOSED_MODELS:
        return _PROXY_EXPOSED_MODELS[bare]
    if bare in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[bare]
    # Allow the raw internal alias to pass through
    all_aliases = set(_PROXY_EXPOSED_MODELS.values()) | set(_LEGACY_ALIASES.values())
    if name in all_aliases:
        return name
    return None


@app.get("/proxy/v1/model/info")
async def proxy_model_info():
    """LiteLLM-compatible model info for proxy clients."""
    config = app.state.router.config
    data = []
    for public_name, internal_name in _PROXY_EXPOSED_MODELS.items():
        entry = _build_proxy_model_entry(config, public_name, internal_name)
        if entry is not None:
            data.append(entry)
    return {"data": data}


@app.get("/proxy/v1/models")
async def proxy_models():
    """OpenAI-compatible /v1/models list for proxy clients."""
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": created, "owned_by": "sohoai"}
            for name in _PROXY_EXPOSED_MODELS
        ],
    }


@app.post("/proxy/v1/chat/completions")
async def proxy_chat_completions(req: Request):
    """Stateless OpenAI-compatible chat completions.

    Maps the public model name to the internal router alias, then passes
    straight to `SmartRouter.complete()`. No chat_id, no Redis, no RAG loop.
    Supports streaming (SSE) when `stream: true` is set.
    """
    body = await req.json()
    messages = body.pop("messages", [])
    public_model = body.pop("model", None)
    stream = bool(body.pop("stream", False))

    internal_model = _resolve_proxy_model(public_model)
    if internal_model is None:
        logger.warning(
            "Proxy rejected unknown model %r (accepted: %s or prefixed forms)",
            public_model, sorted(_PROXY_EXPOSED_MODELS),
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{public_model}' not exposed by this proxy. "
                f"Accepted: {sorted(_PROXY_EXPOSED_MODELS)} (with or without provider prefix)."
            ),
        )
    # Report public name as received (preserves provider-prefixed form if sent).
    reported_model = public_model

    # force_cloud/max_tokens and any other OpenAI-standard params flow through
    # via **body into router.complete → LiteLLM. Drop anything SoHoAI-internal.
    body.pop("force_cloud", None)
    body.pop("chat_id", None)
    body.pop("user_id", None)
    body.pop("rag_mode", None)

    router: SmartRouter = app.state.router

    if stream:
        async def event_stream():
            try:
                # Determine source from X-Orchestra-Session-ID header
                orchestra_session_id = req.headers.get("x-orchestra-session-id")
                source = "orchestra" if orchestra_session_id else "cline"
                # Extract user_id from body if present
                user_id = body.get("user")

                response_gen = await router.complete(
                    messages=messages,
                    model=internal_model,
                    stream=True,
                    metadata={"source": source, "user_id": user_id, "chat_id": None, "orchestra_session_id": orchestra_session_id},
                    **body,
                )
                async for chunk in response_gen:
                    if hasattr(chunk, "model_dump"):
                        chunk_dict = chunk.model_dump()
                    elif hasattr(chunk, "dict"):
                        chunk_dict = chunk.dict()
                    else:
                        chunk_dict = dict(chunk)
                    # Overwrite model field with public name the caller sent
                    chunk_dict["model"] = reported_model
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Proxy stream failed: {e}")
                err = {"error": {"message": str(e), "type": type(e).__name__}}
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming
    # Determine source from X-Orchestra-Session-ID header
    orchestra_session_id = req.headers.get("x-orchestra-session-id")
    source = "orchestra" if orchestra_session_id else "cline"
    # Extract user_id from body if present
    user_id = body.get("user")

    response = await router.complete(
        messages=messages,
        model=internal_model,
        stream=False,
        metadata={"source": source, "user_id": user_id, "chat_id": None, "orchestra_session_id": orchestra_session_id},
        **body,
    )
    if hasattr(response, "model_dump"):
        resp_dict = response.model_dump()
    elif hasattr(response, "dict"):
        resp_dict = response.dict()
    else:
        resp_dict = dict(response)
    resp_dict["model"] = reported_model
    return JSONResponse(resp_dict)


# =============================================================================
#  Anthropic Messages API passthrough  (enables ANTHROPIC_BASE_URL)
# =============================================================================
#
#  Claude Code sets ANTHROPIC_BASE_URL and calls POST /v1/messages in native
#  Anthropic format. This endpoint translates that to an internal router call
#  and returns the response in Anthropic format.
#
#  Set in ~/.claude/settings.json:
#    { "env": { "ANTHROPIC_BASE_URL": "http://192.168.1.93:8000" } }
# =============================================================================

def _anthropic_stop_reason(openai_finish_reason: str | None) -> str | None:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    if not openai_finish_reason:
        return None
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(openai_finish_reason, "end_turn")


@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    """Anthropic Messages API — model-aware routing.

    Local model (gemma-4-e4b → internal alias) and Ollama cloud models:
    routes via LiteLLM with Anthropic→OpenAI format conversion.
    Current limitation: the conversion strips tools, tool_use, tool_result,
    and cache_control — see docs/TODO.md for the planned fix (~70 lines of
    format-conversion code).

    Anthropic models (claude-*) or unresolved models: transparent HTTP forward
    to api.anthropic.com. Preserves tools, tool_use/tool_result history,
    cache_control, anthropic-beta headers, and native SSE streaming exactly.
    Full Claude Code tool loop and prompt caching work correctly on this path.
    """
    body_bytes = await req.body()
    body = json.loads(body_bytes)

    public_model = body.get("model")
    resolved = _resolve_proxy_model(public_model)

    _LITELLM_ROUTED = frozenset(
        {v for v in _PROXY_EXPOSED_MODELS.values()
         if v.startswith("internal/") or v.startswith("ollama-cloud/")}
    )

    if resolved in _LITELLM_ROUTED:
        logger.info("anthropic_messages: %r → LiteLLM path (resolved: %s)", public_model, resolved)
        body = {**body, "model": resolved, "_public_model": public_model}
        return await _anthropic_messages_litellm(body, req)

    # Strip provider prefix (e.g. "anthropic/claude-sonnet-4-6" → "claude-sonnet-4-6")
    # before forwarding — the Anthropic API only accepts bare model names.
    if public_model and "/" in public_model:
        bare_model = public_model.split("/", 1)[-1]
        body = {**body, "model": bare_model}
        body_bytes = json.dumps(body).encode()

    logger.info("anthropic_messages: %r → transparent forward", public_model)
    return await _anthropic_messages_forward(req, body_bytes, body)


async def _anthropic_messages_forward(
    req: Request, body_bytes: bytes, body: dict
) -> StreamingResponse | JSONResponse:
    """Transparent HTTP forward to api.anthropic.com/v1/messages.

    Forwards the exact request body and Anthropic-specific headers.
    x-api-key is taken from the incoming request first, then ANTHROPIC_API_KEY env.
    """
    api_key = req.headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY", "")
    forward_headers: dict[str, str] = {
        "content-type": "application/json",
        "x-api-key": api_key,
    }
    for h in ("anthropic-version", "anthropic-beta"):
        if req.headers.get(h):
            forward_headers[h] = req.headers[h]

    stream = bool(body.get("stream", False))

    if stream:
        async def _forward_stream():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{_ANTHROPIC_API_BASE}/v1/messages",
                    content=body_bytes,
                    headers=forward_headers,
                ) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            _forward_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{_ANTHROPIC_API_BASE}/v1/messages",
            content=body_bytes,
            headers=forward_headers,
        )

    # Manually record usage event for the transparent forward path (bypasses LiteLLM)
    try:
        resp_body = r.json()
        model = body.get("model", "unknown")
        usage = resp_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens = usage.get("cache_read_input_tokens", 0)

        # Determine source and orchestra_session_id
        orchestra_session_id = req.headers.get("x-orchestra-session-id")
        source = "orchestra" if orchestra_session_id else "claude_code_native"

        # Calculate cost via get_model_info() — covers base + cache token rates.
        # Versioned model IDs (e.g. claude-sonnet-4-6-20250219) are not in
        # litellm's model map, so strip the trailing date suffix first.
        import re as _re
        _normalized_model = _re.sub(r"-\d{8}$", "", model)
        try:
            _info = litellm.get_model_info(_normalized_model)
            cost = (
                input_tokens * (_info.get("input_cost_per_token") or 0.0)
                + output_tokens * (_info.get("output_cost_per_token") or 0.0)
                + cache_creation_tokens * (_info.get("cache_creation_input_token_cost") or 0.0)
                + cache_read_tokens * (_info.get("cache_read_input_token_cost") or 0.0)
            )
        except Exception:
            cost = 0.0  # model not in litellm map; WARNING logged below
        if cost == 0.0 and source != "local":
            logger.warning(
                "completion_cost returned 0 for model=%s (forward path) — pricing gap", model
            )

        # Record usage event
        request_id = str(uuid.uuid4())
        created_at = datetime.datetime.utcnow().isoformat() + "Z"

        app.state.store.record_usage_event(
            request_id=request_id,
            created_at=created_at,
            source=source,
            user_id=None,
            chat_id=None,
            orchestra_session_id=orchestra_session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=cost,
            provider="anthropic",
        )
    except Exception as e:
        # Log error but do not raise — we don't want usage tracking to break the passthrough
        logger.error(f"Error recording usage for forward path: {e}", exc_info=True)

    return JSONResponse(r.json(), status_code=r.status_code)


async def _anthropic_messages_litellm(body: dict, req: Request) -> StreamingResponse | JSONResponse:
    """LiteLLM conversion path for local models (gemma-4-e4b → internal) and Ollama cloud.

    CURRENT LIMITATION — tools are stripped by our Anthropic→OpenAI conversion:
    - `tools` array: dropped (never forwarded to LiteLLM or the model)
    - `tool_use` blocks in assistant messages: dropped (model loses call history)
    - `tool_result` blocks in user messages: dropped (model loses file-read results)
    - `cache_control` markers: dropped (no Anthropic prompt caching on local path)

    Note: LiteLLM itself fully supports tool use. The stripping is in *our*
    conversion code, not in LiteLLM. Fix is ~70 lines — see docs/TODO.md.

    Use this path only for simple text generation (summarization, offline drafts).
    Do NOT use for Claude Code sessions that require Read/Write/Bash tool calls.
    """
    internal_model = body.get("model")
    public_model = body.get("_public_model", internal_model)  # fallback to internal if not provided
    stream = bool(body.get("stream", False))
    system_raw = body.get("system")
    max_tokens = body.get("max_tokens", 1024)
    ant_messages = body.get("messages", [])

    system_text: str | None = None
    if isinstance(system_raw, list):
        system_text = " ".join(
            b.get("text", "") for b in system_raw
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif isinstance(system_raw, str):
        system_text = system_raw

    oai_messages: list[dict] = []
    if system_text:
        oai_messages.append({"role": "system", "content": system_text})
    for msg in ant_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        oai_messages.append({"role": role, "content": content})

    extra = {k: body[k] for k in ("temperature", "top_p", "top_k") if k in body}
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    router: SmartRouter = app.state.router

    # Determine source and session_id from request headers
    orchestra_session_id = req.headers.get("x-orchestra-session-id")
    source = "orchestra" if orchestra_session_id else "claude_code_native"

    if stream:
        async def anthropic_event_stream():
            try:
                yield (
                    f"event: message_start\n"
                    f"data: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': public_model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                )
                yield (
                    f"event: content_block_start\n"
                    f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                )
                yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                response_gen = await router.complete(
                    messages=oai_messages,
                    model=internal_model,
                    stream=True,
                    max_tokens=max_tokens,
                    metadata={"source": source, "user_id": None, "chat_id": None, "orchestra_session_id": orchestra_session_id},
                    **extra,
                )
                finish_reason = None
                output_tokens = 0
                async for chunk in response_gen:
                    c = chunk.model_dump() if hasattr(chunk, "model_dump") else (chunk.dict() if hasattr(chunk, "dict") else dict(chunk))
                    for choice in (c.get("choices") or []):
                        text = (choice.get("delta") or {}).get("content") or ""
                        if text:
                            output_tokens += max(1, len(text) // 4)
                            yield (
                                f"event: content_block_delta\n"
                                f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                            )
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                yield (
                    f"event: message_delta\n"
                    f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': _anthropic_stop_reason(finish_reason), 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
                )
                yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            except Exception as e:
                logger.error("Anthropic messages LiteLLM stream failed: %s", e)
                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"

        return StreamingResponse(
            anthropic_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    response = await router.complete(
        messages=oai_messages,
        model=internal_model,
        stream=False,
        max_tokens=max_tokens,
        metadata={"source": source, "user_id": None, "chat_id": None, "orchestra_session_id": orchestra_session_id},
        **extra,
    )
    r = response.model_dump() if hasattr(response, "model_dump") else (response.dict() if hasattr(response, "dict") else dict(response))
    choices = r.get("choices") or []
    text = ""
    finish_reason = None
    if choices:
        text = (choices[0].get("message") or {}).get("content") or ""
        finish_reason = choices[0].get("finish_reason")
    usage = r.get("usage") or {}
    return JSONResponse({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": public_model,
        "stop_reason": _anthropic_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    })


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
