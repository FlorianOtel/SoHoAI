"""
SoHoAI — API Gateway (Server 1, 192.168.1.93)

This process is a stateless gateway. It wires together:
  - Smart model routing (LiteLLM -> local GPU / cloud) for the /proxy/v1/* clients
  - Audio forwarder to llama-swap on Server 2 (/upstream/*)
  - RAG retrieval over Qdrant (/v1/rag/search)

Stateful chat (Redis / SQLite / KV cache), usage telemetry and the native
/v1/messages Anthropic passthrough were removed 2026-11-12.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from rag_engine import search_rag, multi_query_search
from rag_engine.collection import ensure_collection, get_client
from router import SmartRouter

load_dotenv()

# -- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("SoHoAI")


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


# -- Load config ---------------------------------------------------------------
with open("SoHoAI-config.yaml") as f:
    config = yaml.safe_load(f)

# -- App lifecycle -------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Starting SoHoAI orchestrator...")

    app.state.router = SmartRouter(config_path="SoHoAI-config.yaml")
    logger.info(f"Router: {app.state.router.available_models}")

    app.state.rag_cfg = config.get("rag", {})
    _oc_cfg = config.get("opencode", {})
    app.state.rag_cfg["opencode_api_url"] = _oc_cfg.get("api_url", "http://localhost:4096")
    # Read only when an opencode:// item is parsed, so a host without the file
    # (e.g. Server 1) starts normally and fails only that item, loudly.
    app.state.rag_cfg["opencode_password_env_file"] = _oc_cfg.get("password_env_file")

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
    _multi_query_cfg = app.state.rag_cfg.get("multi_query", {})

    async def variant_llm_fn(prompt: str) -> str:
        variant_model = _multi_query_cfg.get("variant_model")
        if not variant_model:
            raise RuntimeError(
                "rag.multi_query.variant_model is not set in SoHoAI-config.yaml — "
                "cannot run multi-query expansion without an explicit variant model."
            )
        resp = await app.state.router.complete(
            messages=[{"role": "user", "content": prompt}],
            model=variant_model,
            force_cloud=(variant_model == "anthropic/claude-sonnet-5"),
            stream=False,
        )
        return resp.choices[0].message.content.strip()

    app.state.variant_llm_fn = variant_llm_fn

    yield

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
#  RAG SEARCH ENDPOINT
# =============================================================================
# RAG ingestion is NOT served here. The /v1/rag/ingest/{sync,start,stop,status}
# endpoints were removed 2026-11-04: ingestion runs on Server 2
# (rag-ingest.timer -> scripts/rag-ingest-run.sh), and the RAG state DB
# (SQLite, WAL mode, on NFS) must only be opened from ONE host. A connection
# from this orchestrator on Server 1, opened by a uvicorn reload during an
# ingest run, corrupted that run's view of the DB ("database disk image is
# malformed"). Equivalents, run on Server 2: utils/rag_sync_nfs.py,
# scripts/rag-ingest-run.sh (or utils/rag_ingest_daemon.py), utils/rag_status.py.
# Search below reads Qdrant only.

@app.get("/v1/rag/search")
async def rag_search(
    q: str,
    user: str | None = Query(None),
    top_k: int = Query(5, ge=1, le=100),
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
    """System health check: gateway liveness and the configured model aliases."""
    return {
        "status": "ok",
        "models": app.state.router.available_models,
    }


@app.get("/v1/models/health")
async def model_health():
    """Check which model endpoints are reachable."""
    return await app.state.router.health_check()


# =============================================================================
#  Proxy pass-through — LiteLLM-compatible stateless endpoints (2026-04-22)
#
#  These endpoints let external clients (Cline VSCode plugin, Claude Code sub-agents, and any
#  OpenAI-compatible client) hit SoHoAI's LiteLLM Router directly,
#  stateless: no conversation memory, KV cache, or RAG tool-use loop.
#  Callers manage their own history.
#
#  Mimics the LiteLLM proxy response shapes so LiteLLM-compatible clients read
#  `model_info.max_input_tokens` correctly (v3.79.0 bug: defaults to 8K/128K
#  if the endpoint returns nothing or an empty API-key is set in the client).
#
#  Proxy base URL (Cline, Claude Code, etc.): http://192.168.1.93:8000/proxy/v1
#  (Cline appends /chat/completions to its Base URL, so it must include /v1.)
#  See CLAUDE.md §Design decisions and memory/project_shared_llama_server.md.
# =============================================================================

def _build_proxy_tables(cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Derive proxy routing tables from model_list in SoHoAI-config.yaml.

    Returns (exposed_models, legacy_aliases).

    Rules per model_name prefix:
      local/X        → exposed identity, bare legacy alias X→local/X
      anthropic/X    → exposed anthropic/X→X (bare has api_key)
      ollama-cloud/X → exposed identity, bare legacy alias X→ollama-cloud/X
      bare name      → legacy identity alias only (backward compat for clients that omit prefix)
    """
    exposed: dict[str, str] = {}
    legacy: dict[str, str] = {}

    for entry in cfg.get("model_list", []):
        name: str = entry.get("model_name", "")
        if name.startswith("local/"):
            exposed[name] = name
            legacy[name[len("local/"):]] = name
        elif name.startswith("anthropic/"):
            # Public anthropic/X → internal bare X, which carries ANTHROPIC_API_KEY
            exposed[name] = name[len("anthropic/"):]
        elif name.startswith("ollama-cloud/"):
            exposed[name] = name
            legacy[name[len("ollama-cloud/"):]] = name
        else:
            # Bare name (e.g. claude-sonnet-5) — internal alias; backward-compat legacy entry
            legacy[name] = name

    return exposed, legacy


_PROXY_EXPOSED_MODELS, _LEGACY_ALIASES = _build_proxy_tables(config)


def _build_proxy_aliases(cfg: dict) -> dict[str, str]:
    """Build meta-aliases (e.g. "local") from proxy.aliases in SoHoAI-config.yaml.

    Unlike _LEGACY_ALIASES — which is derived generically from model_list prefixes —
    these are hand-authored meta-aliases that do not map to one specific model_name by
    convention. Validated against model_list at import time so a stale alias fails loudly
    instead of silently resolving to a model that no longer exists.
    """
    known_model_names = {m.get("model_name") for m in cfg.get("model_list", [])}
    aliases: dict[str, str] = {}
    for alias_name, target in cfg.get("proxy", {}).get("aliases", {}).items():
        if target not in known_model_names:
            raise RuntimeError(
                f"SoHoAI-config.yaml: proxy.aliases.{alias_name} = {target!r} does not "
                f"match any model_list entry's model_name. "
                f"Known model names: {sorted(n for n in known_model_names if n)}"
            )
        aliases[alias_name] = target
    return aliases


_PROXY_ALIASES = _build_proxy_aliases(config)


def _build_audio_table(cfg: dict) -> tuple[str, int, dict[str, str], set[str]]:
    """Build audio model mapping from audio_models section in SoHoAI-config.yaml.

    Returns (upstream_base_url, timeout_seconds, short_name_to_upstream_key_map, accepted_public_names).

    Raises RuntimeError at import time if the audio_models section is malformed.
    The accepted_public_names set includes both short names (e.g. "qwen3-asr") and
    local/-prefixed names (e.g. "local/qwen3-asr") for convenience.
    """
    audio_cfg = cfg.get("audio_models", {})
    if not audio_cfg:
        # No audio_models section — return empty defaults
        return "", 300, {}, set()

    # Extract and validate base URL
    upstream_base = audio_cfg.get("upstream_base", "").strip()
    if not upstream_base:
        raise RuntimeError("SoHoAI-config.yaml: audio_models.upstream_base is required and must not be empty")
    # Replace ${server2_ip} placeholder
    if "${server2_ip}" in upstream_base:
        upstream_base = upstream_base.replace("${server2_ip}", cfg.get("server2_ip", "192.168.1.95"))

    # Extract timeout
    timeout_seconds = audio_cfg.get("timeout_seconds", 300)
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise RuntimeError("SoHoAI-config.yaml: audio_models.timeout_seconds must be a positive integer")

    # Build short_name → upstream_key map and collect accepted public names
    short_to_key: dict[str, str] = {}
    accepted_names: set[str] = set()

    models = audio_cfg.get("models", {})
    if not isinstance(models, dict):
        raise RuntimeError("SoHoAI-config.yaml: audio_models.models must be a dict")

    for short_name, model_info in models.items():
        if not isinstance(model_info, dict):
            raise RuntimeError(
                f"SoHoAI-config.yaml: audio_models.models.{short_name} must be a dict, got {type(model_info).__name__}"
            )

        upstream_key = model_info.get("upstream_key", "").strip()
        if not upstream_key:
            raise RuntimeError(
                f"SoHoAI-config.yaml: audio_models.models.{short_name}.upstream_key is required"
            )

        public_name = model_info.get("public_name", "").strip()
        if not public_name:
            raise RuntimeError(
                f"SoHoAI-config.yaml: audio_models.models.{short_name}.public_name is required"
            )

        short_to_key[short_name] = upstream_key
        # Register both short form and local/-prefixed form as accepted names
        accepted_names.add(short_name)
        if public_name.startswith("local/"):
            accepted_names.add(public_name)
            # Also accept the bare name without "local/" prefix
            bare_name = public_name[len("local/"):]
            accepted_names.add(bare_name)

    return upstream_base, timeout_seconds, short_to_key, accepted_names


_AUDIO_BASE_URL, _AUDIO_TIMEOUT, _AUDIO_MODELS, _AUDIO_ACCEPTED_NAMES = _build_audio_table(config)


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
      - bare public name:   "qwen3.5-9b-q4", "ollama-cloud/deepseek-v4-pro"
      - provider-prefixed:  "openai/qwen3.5-9b-q4", "anthropic/claude-sonnet-5", "ollama-cloud/deepseek-v4-pro"
      - legacy bare names:  "claude-sonnet-5" (backward compat)
      - config-driven meta-alias: "local" (proxy.aliases in SoHoAI-config.yaml)
    Returns the internal router alias or None if unknown.
    """
    if not name:
        return None
    # Config-driven meta-aliases
    if name in _PROXY_ALIASES:
        return _PROXY_ALIASES[name]
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
    if bare in _PROXY_ALIASES:
        return _PROXY_ALIASES[bare]
    # Allow the raw local alias to pass through
    all_aliases = set(_PROXY_EXPOSED_MODELS.values()) | set(_LEGACY_ALIASES.values())
    if name in all_aliases:
        return name
    # Normalize CC context-window annotations ("[1m]") and Anthropic date suffixes
    # ("-20251001"), then retry. This handles:
    #   "claude-sonnet-5[1m]"        → "claude-sonnet-5"
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


@app.api_route("/upstream/{model}/{path:path}", methods=["GET", "POST"])
async def upstream_audio_forwarder(req: Request, model: str, path: str):
    """Transparent forwarder for audio models (ASR + diarization) to llama-swap.

    URL mapping:
      /upstream/{short_name}/{path} → llama-swap /upstream/{upstream_key}/{path}?{query}
      /upstream/local/{short_name}/{path} → same (if routed via /upstream/local/...)

    Models: qwen3-asr, nemotron-3-diarization (and their local/ prefixed forms).
    Streams bytes unchanged; forwards all headers except host and hop-by-hop.
    Request timeout: 300 seconds (to accommodate model swap-in + inference).
    No retry, no LiteLLM, no usage tracking (not token-billed).
    """
    _start = time.time()

    # Handle /upstream/local/{model_name}/{path} parsing:
    # FastAPI parses /upstream/local/qwen3-asr/v1/chat/completions as model="local", path="qwen3-asr/v1/chat/completions"
    # Extract the model name from the path if model == "local"
    model_name = model
    if model_name == "local":
        # Split path to get the actual model name (first segment)
        model_name, _, path = path.partition("/")

    # Resolve model name to upstream key
    if model_name not in _AUDIO_MODELS:
        accepted = sorted(_AUDIO_ACCEPTED_NAMES)
        logger.warning(
            "upstream_audio_forwarder: unknown model %r; accepted: %s",
            model_name, accepted
        )
        return JSONResponse(
            {
                "error": f"unknown audio model {model_name!r}",
                "accepted_models": accepted,
            },
            status_code=404,
        )

    upstream_key = _AUDIO_MODELS[model_name]
    upstream_url = f"{_AUDIO_BASE_URL}/upstream/{upstream_key}/{path}"

    # Preserve query string
    if req.url.query:
        upstream_url = f"{upstream_url}?{req.url.query}"

    # Read request body
    try:
        body_bytes = await req.body()
    except ClientDisconnect:
        logger.warning("upstream_audio_forwarder: client disconnected before body read (model=%s)", model_name)
        return Response(status_code=499)

    # Prepare headers to forward: skip host, content-length, and hop-by-hop.
    # httpx sets content-length itself when forwarding the request.
    forward_headers: dict[str, str] = {}
    hop_by_hop = {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade"
    }
    for key, value in req.headers.items():
        key_lower = key.lower()
        if key_lower in ("host", "content-length"):
            continue
        if key_lower in hop_by_hop:
            continue
        forward_headers[key_lower] = value

    # Forward request to llama-swap with extended timeout.
    # Use build_request + send with stream=True so the async generator owns the client lifetime.
    client: httpx.AsyncClient | None = None
    upstream_response: httpx.Response | None = None

    try:
        client = httpx.AsyncClient(timeout=_AUDIO_TIMEOUT)
        req_up = client.build_request(
            req.method.upper(),
            upstream_url,
            content=body_bytes if body_bytes else None,
            headers=forward_headers,
        )
        upstream_response = await client.send(req_up, stream=True)

        # Collect response headers, removing hop-by-hop
        response_headers = dict(upstream_response.headers)
        for h in hop_by_hop:
            response_headers.pop(h, None)

        async def _stream_response():
            """Stream upstream response body; close client+response in finally."""
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                if upstream_response:
                    await upstream_response.aclose()
                if client:
                    await client.aclose()

        _elapsed = time.time() - _start
        logger.info(
            "upstream_audio_forwarder: model=%s path=%s status=%d elapsed=%.2fs",
            model_name, path, upstream_response.status_code, _elapsed
        )

        return StreamingResponse(
            _stream_response(),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    except httpx.ConnectError as e:
        if client:
            await client.aclose()
        _elapsed = time.time() - _start
        logger.error(
            "upstream_audio_forwarder: connect error for model=%s: %s (elapsed=%.2fs)",
            model_name, e, _elapsed
        )
        return JSONResponse({"error": "upstream service unavailable"}, status_code=502)

    except httpx.ReadTimeout as e:
        if client:
            await client.aclose()
        _elapsed = time.time() - _start
        logger.error(
            "upstream_audio_forwarder: read timeout for model=%s: %s (elapsed=%.2fs)",
            model_name, e, _elapsed
        )
        return JSONResponse({"error": "upstream service timeout"}, status_code=504)

    except Exception as e:
        if client:
            await client.aclose()
        _elapsed = time.time() - _start
        logger.error(
            "upstream_audio_forwarder: unexpected error for model=%s: %s (elapsed=%.2fs)",
            model_name, e, _elapsed, exc_info=True
        )
        return JSONResponse({"error": "internal gateway error"}, status_code=500)



# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
