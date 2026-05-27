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
from rag_engine.scanner import scan_nfs_roots, scan_claude_chats, scan_opencode_sessions
from rag_engine.schema import FIELD_SOURCE_PATH
from rag_engine.state import StateDB
from rag_engine.tool_use import build_tool_spec, format_tool_result, parse_tool_call
from prompts.rag_system_prompts import build_system_prompt
from router import SmartRouter
from kv_cache import KVCacheManager, apply_qwen3_template

load_dotenv()

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("SoHoAI")

_ANTHROPIC_API_BASE = "https://api.anthropic.com"

# -- Orchestra session detection -----------------------------------------------
# brain.md writes ~/.claude/active-sessions/<SESSION_ID>.lck (e.g. 20260510T204552Z-12345.lck)
# at session start and removes it on cleanup/abandon. Native CC sessions use the
# native-<UUID>.lck naming. We scan for the non-native pattern to tag all proxy
# requests with the active orchestra session ID without relying on client-side headers.

_ACTIVE_SESSIONS_DIR = Path.home() / ".claude/active-sessions"
_ORCHESTRA_LCK_RE = re.compile(r"^(\d{8}T\d{6}Z-\d+)\.lck$")


def _read_active_orchestra_session_id() -> str | None:
    """Return the active orchestra session ID from ~/.claude/active-sessions/, or None.

    Scans for non-native .lck files (pattern: YYYYMMDDTHHMMSSZ-<pid>.lck).
    Verifies the CC process in the file is still alive to skip stale files
    from crashed/abandoned sessions that didn't clean up their .lck.
    """
    try:
        for lck in _ACTIVE_SESSIONS_DIR.iterdir():
            m = _ORCHESTRA_LCK_RE.match(lck.name)
            if not m:
                continue
            try:
                content = lck.read_text()
            except OSError:
                continue
            pid_m = re.search(r"cc_pid=(\d+)", content)
            if pid_m:
                try:
                    os.kill(int(pid_m.group(1)), 0)  # signal 0 = alive check only
                    return m.group(1)
                except OSError:
                    continue  # process dead — stale lck
    except Exception:
        pass
    return None


# -- LiteLLM model rate overrides -----------------------------------------------
# LiteLLM's model registry has zero cache rates for claude-opus-4-7 (not yet in its
# database as of 2025-08). Register correct Anthropic list rates so completion_cost()
# and get_model_info() return accurate values for both the forward path and the
# litellm fallback cost source in telemetry-summarize.py.
litellm.register_model({
    "claude-opus-4-7": {
        "input_cost_per_token": 0.000015,
        "output_cost_per_token": 0.000075,
        "cache_creation_input_token_cost": 0.00001875,
        "cache_read_input_token_cost": 0.0000015,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "max_tokens": 32000,
        "max_input_tokens": 200000,
    }
})

# -- Load config ---------------------------------------------------------------
with open("SoHoAI-config.yaml") as f:
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
        db_path=f"{_db_base}/sqlite/telemetry.db"
    )
    logger.info("Chat store (SQLite) ✓")

    app.state.router = SmartRouter(config_path="SoHoAI-config.yaml", store=app.state.store)
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
        summarization_model = config.get("routing", {}).get("summarization_model", "local/qwen3-9b-q4")
        resp = await app.state.router.complete(
            messages=[{"role": "user", "content": prompt}],
            model=summarization_model,
            force_cloud=False,
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    app.state.summarize_fn = summarize_fn
    app.state.rag_cfg = config.get("rag", {})
    _oc_cfg = config.get("opencode", {})
    app.state.rag_cfg["opencode_api_url"] = _oc_cfg.get("api_url", "http://localhost:4096")

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
    _variant_model = app.state.rag_cfg.get("multi_query", {}).get("variant_model", "local/qwen3-4b-q6")

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


@app.middleware("http")
async def inject_orchestra_session_id(request: Request, call_next):
    """Tag every request with the active orchestra session ID (if any).

    Reads ~/.claude/active-sessions/ for a live brain.md .lck file and stores
    the session ID in request.state.orchestra_session_id. All proxy/messages
    endpoints consume this instead of the X-Orchestra-Session-ID header, which
    Claude Code never reliably injects.
    """
    request.state.orchestra_session_id = _read_active_orchestra_session_id()
    return await call_next(request)


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

    Both the internal (Qwen3.5 ChatML path) and the external path (plain text
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
    # the server-side default from SoHoAI-config.yaml (rag.default_mode).
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
    model_used = "local/qwen3-4b-q6"
    used_internal = False
    inference_ok = False

    try:
        for iteration in range(max_iter + 1):
            # 5a. Select model and run inference
            target_model = router.select_model(messages, req.model, req.force_cloud)
            used_internal = (
                target_model.startswith("local/")
                and cache.kv_cache is not None
                and slot_id is not None
            )

            llm_messages = _fold_tool_messages(messages)

            if used_internal:
                prompt = apply_qwen3_template(llm_messages)
                result = await cache.kv_cache.inference(slot_id=slot_id, prompt=prompt, chat_id=chat_id)
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
    result_opencode = await asyncio.to_thread(
        scan_opencode_sessions,
        app.state.state_db,
        config,
        user,
    )

    all_existing = result_nfs["existing_paths"] | result_chats["existing_paths"]
    if result_opencode["existing_paths"] is not None:
        all_existing |= result_opencode["existing_paths"]
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
        "opencode_scanned": result_opencode["scanned"],
        "opencode_reachable": result_opencode["existing_paths"] is not None,
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
    score_threshold: float = Query(0.0, ge=0.0, le=1.0,
                                   description="Minimum cosine score; 0=no filter"),
    multi_query: bool = Query(False,
                              description="Enable multi-query expansion + MMR reranking"),
    rerank: bool | None = Query(None,
                                description="Enable cross-encoder reranking; None=config default"),
):
    """Retrieve RAG document hits without invoking any LLM.

    Intended for external LLM clients (Claude Code, Cline) that manage their own
    reasoning loop.  The caller decides when to search and how to use the results.
    """
    if app.state.qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available")

    if multi_query:
        results = await multi_query_search(
            query=q,
            user_id=user,
            limit=top_k,
            qdrant_client=app.state.qdrant_client,
            rag_cfg=app.state.rag_cfg,
            llm_fn=app.state.variant_llm_fn,
        )
    else:
        results = await search_rag(
            query=q,
            user_id=user,
            limit=top_k,
            qdrant_client=app.state.qdrant_client,
            rag_cfg=app.state.rag_cfg,
            file_types=file_types,
            score_threshold=score_threshold,
            rerank=rerank,
        )

    logger.info("RAG search: q=%r user=%s top_k=%d score_threshold=%.2f multi_query=%s rerank=%s → %d result(s)", q, user, top_k, score_threshold, multi_query, rerank, len(results))
    return {"query": q, "user": user, "results": results}


# =============================================================================
#  SYSTEM ENDPOINTS
# =============================================================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Root liveness probe — CC uses HEAD / to check if the gateway is reachable."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    """System health check."""
    cache_ok = await app.state.cache.ping()
    return {
        "status": "ok",
        "redis": cache_ok,
        "models": app.state.router.available_models,
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

def _build_proxy_tables(cfg: dict) -> tuple[dict[str, str], frozenset[str], dict[str, str]]:
    """Derive proxy routing tables from model_list in SoHoAI-config.yaml.

    Returns (exposed_models, litellm_routed, legacy_aliases).

    Rules per model_name prefix:
      local/X        → exposed identity, litellm_routed, bare legacy alias X→local/X
      anthropic/X    → exposed anthropic/X→X (bare has api_key), NOT litellm_routed
      ollama-cloud/X → exposed identity, litellm_routed, bare legacy alias X→ollama-cloud/X
      bare name      → legacy identity alias only (backward compat for clients that omit prefix)
    """
    exposed: dict[str, str] = {}
    litellm_routed: set[str] = set()
    legacy: dict[str, str] = {}

    for entry in cfg.get("model_list", []):
        name: str = entry.get("model_name", "")
        if name.startswith("local/"):
            exposed[name] = name
            litellm_routed.add(name)
            legacy[name[len("local/"):]] = name
        elif name.startswith("anthropic/"):
            # Public anthropic/X → internal bare X, which carries ANTHROPIC_API_KEY
            exposed[name] = name[len("anthropic/"):]
        elif name.startswith("ollama-cloud/"):
            exposed[name] = name
            litellm_routed.add(name)
            legacy[name[len("ollama-cloud/"):]] = name
        else:
            # Bare name (e.g. claude-sonnet-4-6) — internal alias; backward-compat legacy entry
            legacy[name] = name

    # "qwen3-4b" is a short alias that cannot be derived from the config model_name;
    # keep it hardcoded so existing Cline configs that use this short form still work.
    if "local/qwen3-4b-q6" in exposed:
        legacy["qwen3-4b"] = "local/qwen3-4b-q6"

    return exposed, frozenset(litellm_routed), legacy


_PROXY_EXPOSED_MODELS, _LITELLM_ROUTED, _LEGACY_ALIASES = _build_proxy_tables(config)


def _build_proxy_model_entry(config: dict, public_name: str, internal_name: str) -> dict | None:
    """Build one entry for /proxy/v1/model/info from the SoHoAI-config.yaml model_list."""
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
    # routing (Qwen3.5 → llama-server, Sonnet → Anthropic).
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


def _resolve_proxy_model(name: str | None) -> str | None:
    """Resolve a proxy client's model name to the internal router alias.

    Accepts any of:
      - bare public name:   "qwen3-4b", "ollama-cloud/deepseek-v4-pro"
      - provider-prefixed:  "openai/qwen3-4b", "anthropic/claude-sonnet-4-6", "ollama-cloud/deepseek-v4-pro"
      - legacy bare names:  "claude-sonnet-4-6" (backward compat)
      - local aliases:      "local", "auto"  (for direct testing)
    Returns the internal router alias or None if unknown.
    """
    if not name:
        return None
    # Special aliases for direct testing
    if name == "auto":
        return None
    if name == "local":
        return "local/qwen3-4b-q6"
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
    if bare == "local":
        return "local/qwen3-4b-q6"
    # Allow the raw local alias to pass through
    all_aliases = set(_PROXY_EXPOSED_MODELS.values()) | set(_LEGACY_ALIASES.values())
    if name in all_aliases:
        return name
    # Normalize CC context-window annotations ("[1m]") and Anthropic date suffixes
    # ("-20251001"), then retry. This handles:
    #   "claude-sonnet-4-6[1m]"      → "claude-sonnet-4-6"
    #   "claude-haiku-4-5-20251001"  → "claude-haiku-4-5"
    normalized = re.sub(r"\[.*?\]$", "", name)
    normalized = re.sub(r"-\d{8}$", "", normalized)
    if normalized != name:
        return _resolve_proxy_model(normalized)
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
                # Prefer middleware-injected session ID; fall back to header for future
                # client-side injection support.
                orchestra_session_id = (
                    getattr(req.state, "orchestra_session_id", None)
                    or req.headers.get("x-orchestra-session-id")
                )
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
    # Prefer middleware-injected session ID; fall back to header for future
    # client-side injection support.
    orchestra_session_id = (
        getattr(req.state, "orchestra_session_id", None)
        or req.headers.get("x-orchestra-session-id")
    )
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

_TOOL_ID_VALID = re.compile(r'^[a-zA-Z0-9_-]+$')


def _sanitize_tool_use_id(raw_id: str | None) -> str:
    """Ensure a tool_use id satisfies Anthropic's '^[a-zA-Z0-9_-]+$' pattern.

    Some non-Anthropic models (e.g. Ollama kimi-k2.6) return IDs like
    'functions.Bash:38' containing '.' and ':'.  Replace invalid chars with '_'
    so the ID stays recognisable and round-trips deterministically through CC's
    conversation history.  Falls back to a fresh UUID-based ID if raw_id is
    absent or would sanitize to an empty string.
    """
    if raw_id and _TOOL_ID_VALID.match(raw_id):
        return raw_id
    if raw_id:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)
        if sanitized:
            return sanitized
    return f"toolu_{uuid.uuid4().hex[:24]}"


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


@app.post("/v1/messages/beta")
async def anthropic_messages_beta(req: Request):
    """Claude Code extended-thinking discovery endpoint.

    CC calls POST /v1/messages/beta to check if the server supports interleaved
    thinking (auto-mode). We forward identically to /v1/messages — the gateway
    transparently relays to api.anthropic.com which handles the beta path.
    """
    return await anthropic_messages(req)


@app.post("/v1/messages")
async def anthropic_messages(req: Request):
    """Anthropic Messages API — model-aware routing.

    Local model (qwen3-4b-q6 → local alias) and Ollama cloud models:
    routes via LiteLLM with Anthropic→OpenAI format conversion.
    Tool use is now forwarded on the LiteLLM path (implemented 2026-05-10).
    cache_control markers are also forwarded but have no effect on non-Anthropic
    providers. See docs/TODO.md §[2026-05-04] for implementation notes and
    outstanding items (model reliability, image-block live validation, etc.).

    Anthropic models (claude-*) or unresolved models: transparent HTTP forward
    to api.anthropic.com. Preserves tools, tool_use/tool_result history,
    cache_control, anthropic-beta headers, and native SSE streaming exactly.
    Full Claude Code tool loop and prompt caching work correctly on this path.
    For ollama-cloud models, tool use now also works on the LiteLLM path, but
    local/qwen3-4b-q6 tool reliability remains unvalidated.
    """
    body_bytes = await req.body()
    body = json.loads(body_bytes)

    public_model = body.get("model")


    resolved = _resolve_proxy_model(public_model)

    if resolved in _LITELLM_ROUTED:
        logger.info("anthropic_messages: %r → LiteLLM path (resolved: %s)", public_model, resolved)
        body = {**body, "model": resolved, "_public_model": public_model}
        if not resolved.startswith("ollama-cloud/"):
            return await _anthropic_messages_litellm(body, req)
        try:
            return await _anthropic_messages_litellm(body, req)
        except Exception as e:
            logger.error(
                "anthropic_messages: ollama-cloud LiteLLM path failed for %r: %s",
                public_model, e,
            )
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": (
                            f"Model '{public_model}' is temporarily unavailable "
                            f"(Ollama cloud overloaded or timed out). "
                            f"Retry or switch models with /model. Detail: {str(e)[:200]}"
                        ),
                    },
                },
                status_code=529,
            )

    # Strip provider prefix and CC context-window annotations before forwarding.
    # Anthropic API only accepts bare model names like "claude-sonnet-4-6".
    # CC appends e.g. "[1m]" to distinguish 1M-context variants in its UI — that
    # annotation is not a real model ID and causes a 404 from api.anthropic.com.
    api_model = public_model or ""
    if "/" in api_model:
        api_model = api_model.split("/", 1)[-1]
    api_model = re.sub(r"\[.*?\]$", "", api_model)
    if api_model != body.get("model"):
        body = {**body, "model": api_model}
        body_bytes = json.dumps(body).encode()

    logger.info("anthropic_messages: %r → transparent forward", public_model)
    return await _anthropic_messages_forward(req, body_bytes, body)


async def _anthropic_messages_forward(
    req: Request, body_bytes: bytes, body: dict
) -> StreamingResponse | JSONResponse:
    """Transparent HTTP forward to api.anthropic.com/v1/messages.

    Supports both auth modes:
      - API key:  CC sends x-api-key → forwarded as-is (env fallback if absent)
      - OAuth:    CC sends Authorization: Bearer <token> → forwarded as-is
    """
    forward_headers: dict[str, str] = {"content-type": "application/json"}
    if req.headers.get("authorization"):           # OAuth mode: Bearer token
        forward_headers["authorization"] = req.headers["authorization"]
    else:                                           # API key mode
        api_key = req.headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            forward_headers["x-api-key"] = api_key
    for h in ("anthropic-version", "anthropic-beta"):
        if req.headers.get(h):
            forward_headers[h] = req.headers[h]

    stream = bool(body.get("stream", False))

    if stream:
        # Accumulate SSE usage tokens while forwarding bytes unchanged.
        # Anthropic SSE: message_start carries input+cache tokens; message_delta
        # carries the final output_tokens count.
        _usage_agg = {"input_tokens": 0, "output_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        _streamed_model = [body.get("model", "unknown")]
        _line_buf = [""]

        def _parse_sse_chunk(raw: bytes) -> None:
            _line_buf[0] += raw.decode("utf-8", errors="ignore")
            while "\n" in _line_buf[0]:
                line, _line_buf[0] = _line_buf[0].split("\n", 1)
                if not line.startswith("data: "):
                    continue
                try:
                    ev = json.loads(line[6:])
                    ev_type = ev.get("type")
                    if ev_type == "message_start":
                        msg = ev.get("message", {})
                        u = msg.get("usage", {})
                        for k in ("input_tokens", "cache_creation_input_tokens",
                                  "cache_read_input_tokens"):
                            _usage_agg[k] += u.get(k, 0)
                        if msg.get("model"):
                            _streamed_model[0] = msg["model"]
                    elif ev_type == "message_delta":
                        _usage_agg["output_tokens"] += ev.get("usage", {}).get("output_tokens", 0)
                except (json.JSONDecodeError, KeyError):
                    pass

        async def _forward_stream():
            try:
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream(
                        "POST",
                        f"{_ANTHROPIC_API_BASE}/v1/messages",
                        content=body_bytes,
                        headers=forward_headers,
                    ) as r:
                        async for chunk in r.aiter_bytes():
                            _parse_sse_chunk(chunk)
                            yield chunk
            finally:
                # Record usage event after stream completes or is interrupted.
                if sum(_usage_agg.values()) > 0:
                    try:
                        _fwd_orch_id = (
                            getattr(req.state, "orchestra_session_id", None)
                            or req.headers.get("x-orchestra-session-id")
                        )
                        _fwd_src = "orchestra" if _fwd_orch_id else "claude_code_native"
                        _fwd_model_norm = re.sub(r"-\d{8}$", "", _streamed_model[0])
                        try:
                            _fwd_info = litellm.get_model_info(_fwd_model_norm) or {}
                            _fwd_cost = (
                                _usage_agg["input_tokens"] * (_fwd_info.get("input_cost_per_token") or 0.0)
                                + _usage_agg["output_tokens"] * (_fwd_info.get("output_cost_per_token") or 0.0)
                                + _usage_agg["cache_creation_input_tokens"] * (_fwd_info.get("cache_creation_input_token_cost") or 0.0)
                                + _usage_agg["cache_read_input_tokens"] * (_fwd_info.get("cache_read_input_token_cost") or 0.0)
                            )
                        except Exception:
                            _fwd_cost = 0.0
                        app.state.store.record_usage_event(
                            request_id=str(uuid.uuid4()),
                            created_at=datetime.datetime.utcnow().isoformat() + "Z",
                            source=_fwd_src,
                            user_id=None,
                            chat_id=None,
                            orchestra_session_id=_fwd_orch_id,
                            model=_streamed_model[0],
                            input_tokens=_usage_agg["input_tokens"],
                            output_tokens=_usage_agg["output_tokens"],
                            cache_creation_tokens=_usage_agg["cache_creation_input_tokens"],
                            cache_read_tokens=_usage_agg["cache_read_input_tokens"],
                            cost_usd=_fwd_cost,
                            provider="anthropic",
                        )
                    except Exception as _e:
                        logger.error("Error recording streaming usage: %s", _e, exc_info=True)

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

        # Prefer middleware-injected session ID; fall back to header for future
        # client-side injection support.
        orchestra_session_id = (
            getattr(req.state, "orchestra_session_id", None)
            or req.headers.get("x-orchestra-session-id")
        )
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


# =============================================================================
#  Anthropic ↔ OpenAI format conversion helpers (for LiteLLM path)
# =============================================================================

def _convert_tools(ant_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI format.

    Anthropic tools have `input_schema` (JSON Schema); OpenAI tools have
    `parameters` (JSON Schema). Wrap each tool in OpenAI's standard function-call
    envelope: `{"type":"function","function":{...}}`.
    """
    if not ant_tools:
        return []
    oai_tools = []
    for tool in ant_tools:
        oai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
        }
        oai_tools.append(oai_tool)
    return oai_tools


def _convert_tool_choice(ant_tc: dict | str | None) -> dict | str | None:
    """Convert Anthropic tool_choice to OpenAI format.

    Mappings:
    - {"type":"auto"} → "auto"
    - {"type":"any"} → "required"
    - {"type":"tool","name":"X"} → {"type":"function","function":{"name":"X"}}
    - None / absent → None
    """
    if ant_tc is None:
        return None
    if isinstance(ant_tc, str):
        return ant_tc
    if not isinstance(ant_tc, dict):
        return None

    tc_type = ant_tc.get("type")
    if tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        return "required"
    elif tc_type == "tool":
        tool_name = ant_tc.get("name")
        return {
            "type": "function",
            "function": {"name": tool_name}
        }
    return None


def _convert_assistant_message(msg: dict) -> dict:
    """Convert Anthropic assistant message to OpenAI format.

    Anthropic: mixed `content` list with text, tool_use blocks.
    OpenAI: single message with `content` (text parts joined) and `tool_calls` (function-call list).
    """
    content_blocks = msg.get("content") or []
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })

    return {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
        "tool_calls": tool_calls if tool_calls else None,
    }


def _convert_user_message(msg: dict) -> list[dict]:
    """Convert Anthropic user message to OpenAI format (may emit multiple messages).

    Anthropic user messages can contain text, image, and tool_result blocks.
    OpenAI distinguishes:
    - role: "user" for text/image
    - role: "tool" for tool results (with tool_call_id)

    Text + image blocks → single user message with multi-part content.
    Text adjacent to tool_result blocks → emitted as separate user message before tool results.
    Tool results → emitted as role:"tool" messages.

    Returns a list of messages to emit.
    """
    content_blocks = msg.get("content") or []
    if isinstance(content_blocks, str):
        # Simple text message
        return [{"role": "user", "content": content_blocks}]

    text_parts = []
    image_parts = []
    tool_results = []
    result_messages = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            # Convert Anthropic image format to OpenAI image_url format
            source = block.get("source", {})
            source_type = source.get("type")
            if source_type == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"}
                })
            elif source_type == "url":
                url = source.get("url", "")
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
        elif block_type == "tool_result":
            # Emit any accumulated text/image blocks as a user message first
            if text_parts or image_parts:
                text_content = "".join(text_parts)
                if len(image_parts) == 0 and text_content:
                    result_messages.append({"role": "user", "content": text_content})
                elif image_parts:
                    content_list = []
                    if text_content:
                        content_list.append({"type": "text", "text": text_content})
                    content_list.extend(image_parts)
                    result_messages.append({"role": "user", "content": content_list})
                text_parts = []
                image_parts = []

            # Emit tool result message
            tool_result_content = block.get("content")
            if isinstance(tool_result_content, list):
                # Join text blocks
                tool_result_text = "".join(
                    b.get("text", "") for b in tool_result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                tool_result_text = str(tool_result_content) if tool_result_content else ""

            if block.get("is_error"):
                tool_result_text = "[ERROR] " + tool_result_text

            result_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": tool_result_text,
            })

    # Emit any remaining text/image blocks
    if text_parts or image_parts:
        text_content = "".join(text_parts)
        if len(image_parts) == 0 and text_content:
            result_messages.append({"role": "user", "content": text_content})
        elif image_parts:
            content_list = []
            if text_content:
                content_list.append({"type": "text", "text": text_content})
            content_list.extend(image_parts)
            result_messages.append({"role": "user", "content": content_list})

    return result_messages if result_messages else [{"role": "user", "content": ""}]


async def _anthropic_messages_litellm(body: dict, req: Request) -> StreamingResponse | JSONResponse:
    """LiteLLM conversion path for non-Anthropic models (`local/*` and `ollama-cloud/*`).

    Converts Anthropic-format requests to OpenAI format and routes through LiteLLM.
    As of 2026-05-10 the conversion forwards: `tools` array, `tool_use` blocks in
    assistant messages, `tool_result` blocks in user messages, and `image` content
    blocks (Anthropic base64/URL → OpenAI image_url data URL/URL). Streaming
    responses emit Anthropic SSE `content_block_start/delta/stop` events for tool
    calls (input_json_delta deltas). `cache_control` markers are forwarded but
    have no effect on non-Anthropic providers (no API cost on local inference).

    Reliability per provider:
    - `ollama-cloud/qwen3-coder-next` and `ollama-cloud/deepseek-v4-pro`: native
      OpenAI function calling, expected reliable.
    - `ollama-cloud/kimi-k2.6` and `ollama-cloud/glm-5.1`: untested.
    - `local/qwen3-4b-q6`: tool-call JSON reliability at Q6_K_XL is unvalidated;
      may need grammar-constrained generation as a fallback. See docs/TODO.md.

    Validation harness: `utils/tool_use_smoke_test.py`.
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
        if role == "assistant":
            oai_messages.append(_convert_assistant_message(msg))
        else:
            oai_messages.extend(_convert_user_message(msg))

    # Build tool_kwargs for router calls
    oai_tools = _convert_tools(body["tools"]) if body.get("tools") else None
    oai_tool_choice = _convert_tool_choice(body.get("tool_choice"))
    disable_parallel = body.get("disable_parallel_tool_use", False)

    tool_kwargs: dict = {}
    if oai_tools:
        tool_kwargs["tools"] = oai_tools
    if oai_tool_choice is not None:
        tool_kwargs["tool_choice"] = oai_tool_choice
    if disable_parallel:
        tool_kwargs["parallel_tool_calls"] = False

    extra = {k: body[k] for k in ("temperature", "top_p", "top_k") if k in body}
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    router: SmartRouter = app.state.router

    # Prefer middleware-injected session ID; fall back to header for future
    # client-side injection support.
    orchestra_session_id = (
        getattr(req.state, "orchestra_session_id", None)
        or req.headers.get("x-orchestra-session-id")
    )
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
                    **tool_kwargs,
                    **extra,
                )
                finish_reason = None
                output_tokens = 0
                text_block_open = True
                tool_call_state: dict[int, dict] = {}

                async for chunk in response_gen:
                    c = chunk.model_dump() if hasattr(chunk, "model_dump") else (chunk.dict() if hasattr(chunk, "dict") else dict(chunk))
                    for choice in (c.get("choices") or []):
                        delta = choice.get("delta") or {}

                        # Handle text deltas
                        text = delta.get("content") or ""
                        if text:
                            output_tokens += max(1, len(text) // 4)
                            yield (
                                f"event: content_block_delta\n"
                                f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
                            )

                        # Handle tool_calls deltas
                        tool_calls = delta.get("tool_calls") or []
                        for tc in tool_calls:
                            tc_index = tc.get("index")
                            if tc_index not in tool_call_state:
                                # New tool call — close text block if still open
                                if text_block_open:
                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                    text_block_open = False

                                # Open new tool_use block
                                tc_id = _sanitize_tool_use_id(tc.get("id"))
                                tc_name = (tc.get("function") or {}).get("name", "")
                                tool_call_state[tc_index] = {"id": tc_id, "name": tc_name}
                                sse_index = 1 + tc_index
                                yield (
                                    f"event: content_block_start\n"
                                    f"data: {json.dumps({'type': 'content_block_start', 'index': sse_index, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_name, 'input': {}}})}\n\n"
                                )

                            # Emit input_json_delta if function.arguments present
                            func_args = (tc.get("function") or {}).get("arguments") or ""
                            if func_args:
                                sse_index = 1 + tc_index
                                yield (
                                    f"event: content_block_delta\n"
                                    f"data: {json.dumps({'type': 'content_block_delta', 'index': sse_index, 'delta': {'type': 'input_json_delta', 'partial_json': func_args}})}\n\n"
                                )

                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                # Close any open blocks
                if text_block_open:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                for tc_idx in tool_call_state:
                    sse_index = 1 + tc_idx
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': sse_index})}\n\n"

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
        **tool_kwargs,
        **extra,
    )
    r = response.model_dump() if hasattr(response, "model_dump") else (response.dict() if hasattr(response, "dict") else dict(response))

    # Build content_blocks from text and tool_calls
    choices = r.get("choices") or []
    content_blocks: list[dict] = []
    finish_reason = None
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason")
        if text:
            content_blocks.append({"type": "text", "text": text})
        for tc in (message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_input = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_input = {"_raw": raw_args}
            content_blocks.append({
                "type": "tool_use",
                "id": _sanitize_tool_use_id(tc.get("id")),
                "name": fn.get("name", ""),
                "input": parsed_input,
            })
    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    usage = r.get("usage") or {}
    return JSONResponse({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": public_model,
        "stop_reason": _anthropic_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    })


def _extract_text_from_blocks(blocks: list | str | None) -> str:
    """Extract text content from Anthropic message blocks.

    Handles both simple string content and complex block lists.
    For tool_use and tool_result blocks, extracts text representation.
    """
    if not blocks:
        return ""

    if isinstance(blocks, str):
        return blocks

    text_parts = []
    for block in (blocks if isinstance(blocks, list) else []):
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            # Include tool name and args as text for token estimation
            name = block.get("name", "")
            input_dict = block.get("input", {})
            text_parts.append(f"{name}({json.dumps(input_dict)})")
        elif block_type == "tool_result":
            # Include tool result text
            content = block.get("content")
            if isinstance(content, list):
                for cb in content:
                    if isinstance(cb, dict) and cb.get("type") == "text":
                        text_parts.append(cb.get("text", ""))
            elif isinstance(content, str):
                text_parts.append(content)

    return "".join(text_parts)


@app.post("/v1/messages/count_tokens")
async def count_tokens_endpoint(req: Request):
    """Count tokens for Anthropic Messages API requests.

    Routes based on model type:
    - Anthropic-native models: forward to api.anthropic.com/v1/messages/count_tokens
    - LiteLLM-path models (local/*, ollama-cloud/*): convert to OpenAI format
      and use litellm.token_counter()
    - Unresolved models: return HTTP 400
    """
    body_bytes = await req.body()
    body = json.loads(body_bytes)

    public_model = body.get("model")
    resolved = _resolve_proxy_model(public_model)

    if resolved is None:
        logger.warning(
            "count_tokens rejected unknown model %r (accepted: %s)",
            public_model, sorted(_PROXY_EXPOSED_MODELS),
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{public_model}' not exposed by this proxy. "
                f"Accepted: {sorted(_PROXY_EXPOSED_MODELS)} (with or without provider prefix)."
            ),
        )

    # Anthropic-native models: forward to Anthropic API
    if resolved not in _LITELLM_ROUTED:
        # Strip provider prefix and CC context-window annotations (e.g. "[1m]")
        # before forwarding — Anthropic accepts bare/date-versioned IDs only.
        stripped_model = public_model.split("/", 1)[-1] if public_model and "/" in public_model else public_model
        stripped_model = re.sub(r"\[.*?\]$", "", stripped_model)
        forward_body = {**body, "model": stripped_model}
        forward_body_bytes = json.dumps(forward_body).encode()

        forward_headers: dict[str, str] = {"content-type": "application/json"}
        if req.headers.get("authorization"):           # OAuth mode: Bearer token
            forward_headers["authorization"] = req.headers["authorization"]
        else:                                           # API key mode
            api_key = req.headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key:
                forward_headers["x-api-key"] = api_key
        for h in ("anthropic-version", "anthropic-beta"):
            if req.headers.get(h):
                forward_headers[h] = req.headers[h]

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{_ANTHROPIC_API_BASE}/v1/messages/count_tokens",
                content=forward_body_bytes,
                headers=forward_headers,
            )
        return JSONResponse(r.json(), status_code=r.status_code)

    # LiteLLM-path models: convert to OpenAI format and use litellm.token_counter()
    logger.info("count_tokens: %r → LiteLLM path (resolved: %s)", public_model, resolved)

    # Extract system text
    system_raw = body.get("system")
    system_text: str | None = None
    if isinstance(system_raw, list):
        system_text = " ".join(
            b.get("text", "") for b in system_raw
            if isinstance(b, dict) and b.get("type") == "text"
        )
    elif isinstance(system_raw, str):
        system_text = system_raw

    # Convert Anthropic messages to OpenAI format
    oai_messages: list[dict] = []
    if system_text:
        oai_messages.append({"role": "system", "content": system_text})

    ant_messages = body.get("messages", [])
    for msg in ant_messages:
        role = msg.get("role", "user")
        if role == "assistant":
            oai_msg = _convert_assistant_message(msg)
            oai_messages.append(oai_msg)
        else:
            oai_msgs = _convert_user_message(msg)
            oai_messages.extend(oai_msgs)

    # Count tokens using litellm
    try:
        token_count = litellm.token_counter(model=resolved, messages=oai_messages)
        return JSONResponse({"input_tokens": token_count})
    except Exception as e:
        logger.warning(
            "litellm.token_counter failed for model=%s: %s — falling back to text-based estimate",
            resolved, e
        )
        # Fallback: rough estimate based on text length
        total_text = system_text or ""
        for msg in oai_messages:
            if isinstance(msg.get("content"), str):
                total_text += msg["content"]
            elif isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_text += block.get("text", "")
        estimated_tokens = len(total_text) // 4
        return JSONResponse({"input_tokens": estimated_tokens})


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
