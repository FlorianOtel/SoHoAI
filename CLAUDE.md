---
title: "SoHoAI — Project Context & Design Reference"
created_at: 20260407-000000
created_by: Florian Otel / Cline (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 20260504-000000
context: >
  SoHoAI project (https://github.com/FlorianOtel/SoHoAI);
  Project instructions and design decisions for Claude Code;
  Infrastructure, architecture, API, implementation phases, RAG strategy,
  multi-tenancy (Google OAuth2, per-user NFS roots, Qdrant owner filtering)
---

# CLAUDE.md — SoHoAI Project Context

**Repository**: https://github.com/FlorianOtel/SoHoAI

## What is this project?

SoHoAI is a distributed, two-server home-office AI system. It provides a unified
gateway for LLM inference (local + cloud), with conversation memory, document RAG,
and an MCP server for tooling integration. The long-term goal includes image search
for family photos and RL training data collection from chat interactions.

## Infrastructure

### Servers

| Server | IP | Role | Hardware |
|--------|------|------|----------|
| Server 1 | 192.168.1.93 | Orchestrator, front-end, Redis cache, RAG, MCP server | 32GB RAM, AMD iGPU Radeon 680M |
| Server 2 | 192.168.1.95 | LLM inference engine | 16GB RAM, Nvidia RTX 5070 12GB, llama-server |
| NAS | NFS-mounted | Persistent storage for everything | 27TB |

### Storage paths

> SQLite and Redis paths are derived from `db_base_path` in `config.yaml`. Qdrant active storage is local-only (see below).

- Chat DB: `/mnt/nfs/__Backups/SoHoAI--databases/sqlite/chats.db` (SQLite, NAS)
- RAG state DB: `/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db` (SQLite, NAS)
- Vector store (active): `/var/lib/qdrant/storage` (local NVMe, Server 1 only — **not NFS**)
- Vector store (snapshots/DR): `/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/` (NAS)
  - **Note**: `/mnt/nfs/__Backups/SoHoAI--databases/qdrant/` is intentionally empty — it is NOT used.
    Qdrant server uses RocksDB which is incompatible with NFS file locking. Active data must stay local.
    Snapshots (passive files) are safe on NFS and taken daily at 03:00 via cron.
- Redis persistence: `/mnt/nfs/__Backups/SoHoAI--databases/redis/`
- KV cache slots: `/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/k-v-caches/` (one `.bin` per chat_id)
- LLM model cache: `/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/`
- Documents for RAG: `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/documents/`
- RL training exports: `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/rl-data/`
- Chat markdown exports: `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/exports/`

## Architecture

```
                         Anthropic API (cloud)
                         ┌───────────────────────────────┐
                ─primary─→│ Claude Sonnet 4.6            │
                         │ (prompt caching: system +     │
                         │  rolling prefix breakpoints)  │
                         └───────────────────────────────┘
                                    │
Server 1 (192.168.1.93)             │          Server 2 (192.168.1.95)
┌──────────────────────────┐  fallback/       ┌──────────────────────────┐
│ FastAPI orchestrator:8000│──summarize/────→ │ llama-server :8000       │
│  SmartRouter (LiteLLM)   │  variants →      │  Gemma 4 E4B 7.52B Q8_0  │
│  ConversationCache       │─── /slots ─────→ │  2×110024 ctx (2 slots)  │
│    Redis (short-term)    │                  │  KV slot save/restore    │
│    KV cache mgr          │                  │  CLIP (Phase 4)          │
│    rolling summarization │                  └──────────────────────────┘
│ SQLite (long-term)       │                             │
│  + summary_text +        │                             │
│    covers_through_msg_id │                             │
│ MCP server :3001 (HTTP)  │                             │
│ Qdrant server :6333      │                             │
│  active storage: NVMe    │                             │
│  snapshots → NAS daily   │                             │
└──────────────────────────┘                             │
           │                                             │
           └──── both mount ──→ NAS (27TB NFS) ──────────┘
                               (Redis AOF, SQLite, KV .bin files,
                                model cache, Qdrant snapshots, exports)
```

### LLM routing (via LiteLLM)

Two model tiers with automatic fallback (2026-04-22 flip: primary is now external):

1. **external** — Claude Sonnet 4.6 via Anthropic API (primary, cloud, interactive chat default)
2. **internal** — Gemma 4 E4B 7.52B Q8_0 on RTX 5070 via llama-server (fallback if Anthropic unreachable, summarization, offline tasks)

Routing logic is in `router.py`. Default is external (Sonnet 4.6); falls back to local Gemma if cloud unreachable.
Rolling summarization uses internal (Gemma 4) and persists summaries to SQLite for cold-resume recovery.
Prompt caching on Sonnet 4.6 reduces cost by ~90% on cache hits.

**External (Sonnet 4.6) path** goes through LiteLLM with optional prompt caching enabled 
(cache_control markers on system message and rolling prefix breakpoints).

**Specialist (Gemma 4) path** bypasses LiteLLM and calls llama-server's native
`/completion` endpoint directly, enabling KV cache slot targeting (`slot_id`).
This is used only on fallback (Anthropic down), summarization operations,
and background/offline tasks. Rolling summarization erases the KV slot before calling, and both
summarization and the subsequent main inference start cold on the same slot sequentially.

### Memory — three tiers

- **Short-term**: Redis on Server 1, keyed by `conv:{chat_id}`. 24h TTL.
  Persisted to NAS via Redis AOF.
- **Long-term**: SQLite on NAS. Every message written here permanently.
  Supports chat list/search, markdown export, RL data export.
- **GPU KV cache**: Per-conversation llama-server slot state, saved as
  `{chat_id}.bin` on NAS after every turn. Restored before the next turn
  so the GPU skips recomputing the prompt prefix. Survives server restarts.

All three tiers are coordinated by `ConversationCache` in `conversation.py`:
- `resume(chat_id)` — restore KV slot from NFS before inference
- `park(chat_id)` — save KV slot to NFS + refresh Redis TTL after inference
- `clear(chat_id)` — wipe Redis + erase KV slot + delete NFS file
- `maybe_summarize(chat_id, fn, store=None)` — if Redis context exceeds ~100K tokens,
  summarize old turns via internal (llama-server), rebuild Redis as
  `[system?, summary_msg, recent_turns]`, erase stale KV cache, and (if `store` is
  passed) persist the summary + boundary `message_id` to SQLite `chats` columns.
  Summarization failure is tolerated: the turn proceeds with oversized context.
- `is_cold(chat_id)` — True if Redis has no messages for this chat
- `warm_from_store(chat_id, store=None, messages=None)` — cold-start Redis reload.
  If `store` is provided and a persisted summary exists, reconstructs Redis as
  `[system?, summary_msg, messages_with_id_greater_than_covers_through]` — no data
  duplication since raw messages below the boundary are represented only by the
  summary. Falls back to last-N raw messages if no summary is stored. Also erases
  stale KV cache.

Code: `conversation.py`, `kv_cache.py`, `chat_store.py`

### MCP server ✅ (completed, working)

Subdirectory: `./NFS-files--MCP-server/`

Files:
- `nfs_files_mcp_server.py` — MCP server implementation (22.9 KB)
- `setup_mcp.sh` — install + validation script
- `claude_desktop_config.json` — config snippet for Claude Desktop
- `server_config.json` — server configuration

Exposes all files under `/mnt/nfs/Florian/Gin-AI` to any MCP client.
Project files are under `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI`.

MCP server name (as seen by clients): **`nfs-files`**

Tools: `list_directory`, `read_file`, `write_file`, `edit_file`, `delete_file`, `search_files`, `get_file_info`

Resources: `file:///nfs_files/config`, `file://structure`

Transport: streamable HTTP on port 3001 (default) or stdio

Path safety: all paths resolved against ALLOWED_ROOTS; traversal blocked.
Root is hardcoded to `/mnt/nfs/Florian` in `nfs_files_mcp_server.py` (ALLOWED_ROOTS).

Listing supports depth 0-20 (0 = unlimited recursive). Search uses rglob
(always recursive) with up to 1000 result lines.

**Note**: relative paths resolve from the Gin-AI root, so project files need
the `projects/SoHoAI/` prefix (e.g. `projects/SoHoAI/config.yaml`).

## Project structure

```
SoHoAI/
├── main.py                     # FastAPI app — the central orchestrator
├── config.yaml                 # All configuration (models, Redis, RAG, routing, llama-server)
├── .env                        # ANTHROPIC_API_KEY (not committed)
├── .mcp.json                   # Claude Code auto-discovers this
├── schemas.py                  # Pydantic models (ChatRequest, ChatResponse, etc.)
├── router.py                   # SmartRouter — LiteLLM wrapper with routing logic
├── conversation.py             # ConversationCache — Redis + KV cache coordinator
├── kv_cache.py                 # KVCacheManager — llama-server slot save/restore + inference; apply_gemma_template() (Gemma 4 <|turn> format)
├── chat_store.py               # ChatStore — SQLite long-term persistence
├── mcp_gateway.py              # MCP tool gateway (Phase 3 stub, interface defined)
├── pyproject.toml              # Python dependencies (uv-managed)
├── prompts/                    # System prompt builders (§8.1)
│   └── rag_system_prompts.py   # build_system_prompt(mode, tool_spec) → off/on/only prompts
├── rag_engine/                 # RAG pipeline (Phase 2 ✅ + §8 advanced retrieval ✅)
│   ├── __init__.py             # re-exports search_rag, multi_query_search
│   ├── schema.py               # Qdrant payload field constants + derive_owner(); FIELD_SESSION_ID + FIELD_PROJECT added for claude_chat documents
│   ├── collection.py           # Collection name, vector size, get_client(), ensure_collection()
│   ├── embeddings.py           # embed_text(), embed_batch(progress_cb) via Ollama on Server 1
│   ├── state.py                # StateDB — ingestion queue CRUD + crash recovery; find_deleted()/purge_deleted() are the crash-safe split of the old handle_deleted() — callers must Qdrant-clean before calling purge_deleted() so a kill leaves SQLite intact for retry
│   ├── scanner.py              # NFS + claude chat scanner → populates StateDB; scan_nfs_roots() walks configured NFS roots (filters from config.yaml rag.scanner) and returns {scanned, existing_paths}; scan_claude_chats() walks config.yaml claude_chats.roots for .jsonl sessions and returns {scanned, existing_paths}; callers merge existing_paths sets and call state_db.find_deleted() once before Qdrant cleanup + purge_deleted(); followlinks=True with visited_real_dirs + visited_real_files global dedup sets; exclude_dir_names uses trailing-slash path-suffix matching
│   ├── ingest.py               # docling parse + parent-child chunking + Qdrant upsert; _parse_claude_chat() extracts user/assistant text turns from .jsonl session files and returns (text, {session_id, project}) metadata for Qdrant payload
│   ├── search.py               # query → embed → Qdrant query_points → parent_text + provenance
│   ├── multi_query.py          # §8.3: expand_query() + parallel search + MMR reranking (permanently disabled — no-go 2026-04-22)
│   └── tool_use.py             # §8.2: build_tool_spec(), parse_tool_call(), format_tool_result()
├── utils/
│   ├── cli_chat.py             # Terminal chat client; RAG off by default (opt-in); --user OWNER sends user_id; /rag on|off|only; /rag search <query> inspects retrieval; /user <id> changes owner mid-session
│   ├── rag_sync_nfs.py         # CLI: scan NFS roots + claude chat dirs → queue new/modified files; re-queue failed; skip ignored; purge Qdrant for removed files; calls scan_nfs_roots() + scan_claude_chats(), merges existing_paths, calls find_deleted() once
│   ├── rag_ingest_daemon.py    # CLI: process pending files (parse → chunk → embed → upsert)
│   ├── rag_status.py           # CLI: queue counts + Qdrant point stats; --ignored for detail; --watch LOG_FILE for live ETA monitor; --list-pending [N] to print pending paths (pipeable)
│   ├── rag_search_cli.py       # CLI: retrieval-only — embed query + Qdrant search; prints top-k hits + parent_text preview
│   ├── rag_smoke_test.py       # CLI: end-to-end smoke test — retrieval + /v1/chat/completions with rag_mode=on; --expect SUBSTR assertion; pass/fail exit code
│   ├── rag_mmr_bench.py        # CLI: MMR benchmark harness (evaluated 2026-04-22; no-go verdict — kept for reference)
│   ├── rag_bench_queries.txt   # 12 verified queries used by rag_mmr_bench.py
│   ├── rag_reset.py            # CLI: reset Qdrant collection + ingestion queue
│   ├── notebooklm_auth.py      # NotebookLM browser automation (Playwright + system Chrome)
│   ├── snapshot_codebase.py    # Aggregate project files → codebase_snapshot.md
│   ├── sync_to_notebook.py     # End-of-session sync: snapshot → delete old → upload
│   ├── notebooklm_session.json # Saved Google session cookies (not committed)
│   └── codebase_snapshot.md    # Generated snapshot (not committed)
├── scripts/
│   └── qdrant/
│       ├── qdrant-config.yaml      # Qdrant server config (storage paths, ports)
│       ├── qdrant.service          # systemd unit file (copy to /etc/systemd/system/)
│       └── qdrant-snapshot.sh      # Snapshot + cleanup script (cron: daily 03:00)
└── NFS-files--MCP-server/      # nfs-files MCP server (✅ complete)
    ├── nfs_files_mcp_server.py  # MCP server implementation
    ├── nfs_files_mcp_server.sh  # Launch script (HTTP mode, port 3001)
    ├── setup_mcp.sh             # Install + validation script
    ├── claude_desktop_config.json # Config snippet for Claude Desktop
    └── server_config.json       # Server configuration
```

## Key API endpoints (main.py)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | Main chat — OpenAI-compatible, dual-mode (see below) |
| GET | `/v1/chats` | List saved chats |
| GET | `/v1/chats/{id}` | Get full chat history |
| DELETE | `/v1/chats/{id}` | Delete chat from Redis, KV cache, and SQLite |
| GET | `/v1/chats/{id}/export/markdown` | Export chat as Markdown |
| POST | `/v1/chats/{id}/export/save` | Save Markdown to NAS |
| GET | `/v1/chats/{id}/export/rl` | Export as JSONL for RL training |
| POST | `/v1/chats/{id}/feedback` | Record thumbs up/down (for RL) |
| GET | `/health` | Health check (Redis, models) |
| GET | `/v1/models` | List models — OpenAI-compatible format |
| GET | `/v1/models/health` | Check each model endpoint |
| GET | `/proxy/v1/model/info` | LiteLLM-compatible model info for proxy clients (Cline, Claude Code) — max_input_tokens, context_window |
| GET | `/proxy/v1/models` | OpenAI model list for proxy clients (Cline, Claude Code) |
| POST | `/proxy/v1/chat/completions` | Stateless OpenAI-compatible pass-through for proxy clients (Cline, Claude Code) — no Redis/RAG/summarization |
| POST | `/v1/messages` | Anthropic Messages API passthrough — enables `ANTHROPIC_BASE_URL` for Claude Code |
| POST | `/v1/rag/ingest/sync` | Scan NFS roots → populate ingestion queue |
| POST | `/v1/rag/ingest/start` | Start background ingestion worker |
| POST | `/v1/rag/ingest/stop` | Stop ingestion worker gracefully |
| GET | `/v1/rag/ingest/status` | Queue metrics + Qdrant point count |

Responses use a custom `ChatResponse` model (`chat_id`, `model_used`, `message`, `rag_sources`, `rag_mode_used`).
**Not** OpenAI-compatible format — the CLI reads `data["message"]["content"]`.
OpenAI-compatible response format is a Phase 3 requirement for Open WebUI integration.

## Implementation phases

### Phase 1 — Core loop ✅ (complete)
- FastAPI orchestrator on Server 1
- LiteLLM routing with fallback chain: external (Sonnet 4.6, cloud) → internal (Gemma 4, local) on external failure (reversed 2026-04-22)
- Anthropic prompt caching on the external path (`SmartRouter._apply_cache_control()`) ✅ — ephemeral breakpoints on system + `messages[-2]` rolling prefix
- llama-server on Server 2 GPU: Gemma 4 E4B 7.52B Q8_0, 2 slots × 110024 ctx (`-c 220048 --parallel 2`) — fallback + summarization role ✅
- Per-conversation KV cache persisted to NAS via llama-server slot API ✅ (used on internal path only)
- Rolling summarization at ~100K token threshold (via internal/llama-server) ✅ — summary + boundary `message_id` persisted to SQLite `chats` row
- Redis conversation cache with NAS AOF persistence
- SQLite chat store with full CRUD + `summary_text`/`summary_covers_through_message_id` columns
- Markdown and JSONL export
- CLI chat client (`utils/cli_chat.py`)
- Feedback collection for RL
- MCP server for Gin-AI filesystem (`NFS-files--MCP-server/`) ✅

### Per-request flow (§8 tool-use loop)

Both external (primary, Sonnet 4.6) and internal (fallback, Gemma 4) share the same loop structure in `_server_managed_completion()`. The only branch is the inference step at [main.py:333-347](main.py#L333-L347).

```
resume(chat_id)           → assign slot + restore KV from NAS (no-op on external path)
append(user_msg)          → Redis + SQLite
maybe_summarize(store=…)  → if >400K chars (~100K tokens): summarize old turns via
                            internal (Gemma), rebuild Redis, erase stale KV,
                            persist summary + boundary id to SQLite
get_context()             → condensed history from Redis
build_system_prompt()     → off/on/only mode → tool spec injected into system message
loop (max 2 iterations):
  # Branch on selected model:
  if target == internal (fallback):
    apply_gemma_template()   → <|turn>role\n… format; stop=["<|turn>"]
    inference(slot_id)       → POST /completion to llama-server GPU slot
  else (primary = external):
    router.complete()        → LiteLLM → Anthropic with cache_control markers
                               (cache_read grows each turn as prefix rolls forward)
  parse_tool_call()       → detect <tool_call>…</tool_call> in output
  if tool_call:
    _retrieve()           → search_rag() (multi-query permanently disabled)
    append tool result    → re-enter loop
  else: final answer → break
append(assistant)         → Redis + SQLite
park(chat_id)             → save KV slot to NAS + refresh Redis TTL
```

### Phase 2 — RAG ✅ (complete, ingested 2026-04-21)

Initial ingestion run produced **2891 files completed, 0 pending, 0 ignored**, yielding
**98,737 Qdrant points** in the `documents` collection (avg ~34 chunks/file). End-to-end
retrieval + chat injection verified via `utils/rag_smoke_test.py`.

- **Multi-tenancy** — per-user document isolation via Google OAuth2 (see design decisions)
  - Each family member has a private NFS root (`/mnt/nfs/{Florian,Eva,Annika,Laura}`)
  - Shared content under `/mnt/nfs/La-Familia` visible to all authenticated users
  - `owner` field in every Qdrant point; search filtered by `MatchAny(any=[user_owner, "la-familia"])`
  - `user_id` field added to `ChatRequest`, `SearchRequest`, and SQLite `chats` table
  - User→NFS root mapping in `config.yaml` (`users:` + `shared:` sections)
- Document ingestion: `docling` (PDF, PPTX, DOCX) + dedicated ipynb cell extractor + `python-pptx` PPTX fallback + direct UTF-8 read (TXT, MD, YAML, CSV) — replaces `unstructured`
  - **ipynb NOT handled by docling** — docling silently rejects `.ipynb` and falls back to raw JSON read, which embeds the notebook's JSON structure as text (garbage for RAG). Fix: `_parse_ipynb()` in `ingest.py` parses the JSON directly, extracts markdown cells as prose and code cells as fenced blocks, skips outputs and empty cells.
  - **PPTX docling format detection failure** — docling can fail to detect `.pptx` format (`format None`) for older `.ppt` binaries renamed to `.pptx` or files with unusual internal structure. Prior fallback was raw UTF-8 read of the binary ZIP container — garbage for RAG. Fix: `_parse_pptx()` in `ingest.py` uses `python-pptx` as a secondary fallback; iterates slides/shapes, extracts all text frame content with `Slide N` headers. Raw UTF-8 read is last resort only if `python-pptx` also fails. 29 affected files force-re-queued (2026-04-19).
  - PST/Outlook archive parsing **de-prioritized** (only 2 `.msg` files found on NAS; `libpst` not needed)
- Chunking — **parent-child strategy** (see design decision below):
  - Child chunks (~250 tokens, 20 overlap) → embedded, stored in Qdrant as search index
  - Parent chunks (~800–1200 tokens, 100 overlap) → raw text only, stored in Qdrant payload (`parent_text` field)
  - Flat 512-token chunks used only for PPTX slides and short TXT/YAML/config files (already compact)
- Embeddings via **bge-m3 served by Ollama** — server is configurable (`ollama_url` in `config.yaml`)
  - 1024 dimensions, 570M params, ~1.2GB — top MTEB scores; 8192-token context window
  - Two modes: CPU (Server 1, 193) ~650ms/chunk; GPU (Server 2, 195, RTX 5070) ~10–20ms/chunk
  - `embed_batch()` runs up to N concurrent requests via asyncio semaphore; N = `--batch` in daemon
  - `embed_batch()` accepts optional `progress_cb(done, total)` — called every 50 chunks (`_PROGRESS_INTERVAL`) + on the final chunk; used by `ingest.py` to log `"Embedding progress: N/M  filename"` lines read by `rag_status.py --watch`
  - HTTP timeout in `embed_text()`: **120s** — must be this high or search times out while ingest daemon is running (CPU mode)
  - Daemon flags: `--workers` (file-level concurrency) and `--batch` (chunk-level Ollama concurrency) are both **required**; see RAG-strategy.md §5.4 for operating points per mode
- Qdrant vector store on Server 1 (`http://192.168.1.93:6333`), active storage on local NVMe — **confirmed right choice** (see design decisions)
- Package: `rag_engine/` — fully implemented (schema, collection, embeddings, state, scanner, ingest, search, multi_query, tool_use); `rag.py` deleted
- Full design details: `RAG-strategy.md`

### Advanced RAG features ✅ (§8.1–§8.3, implemented 2026-04-21)

- **`rag_mode` enum** (`off` / `on` / `only`) — the legacy boolean toggle is gone.
  `ChatRequest.rag_mode` default is `"off"` (opt-in); server falls back to
  `rag.default_mode` from `config.yaml` when caller omits the field.
  `ChatResponse` adds `rag_mode_used` for debuggability.
- **Tool-use via system prompt** — LLM decides when to retrieve; no unconditional top-k injection.
  `<tool_call>{"name":"search_documents","args":{"query":"…"}}</tool_call>` sentinel parsed
  by `rag_engine/tool_use.py`. Tool messages folded to `role=user` for both internal and
  external paths. Hard cap: `rag.tool_use.max_iterations: 2`.
- **Multi-query + MMR** — `rag_engine/multi_query.py`; **permanently disabled** (`rag.multi_query.enabled: false`).
  Evaluated 2026-04-22: no-go verdict. Standard single-query retrieval is sufficient
  for this corpus. Code retained for reference but will not be enabled.
- **`prompts/rag_system_prompts.py`** — `build_system_prompt(mode, tool_spec)` is the single
  source of all three mode prompts; called by `main.py` before each tool-use loop.
- **Service addresses** — all explicit IPs: Ollama `192.168.1.93:11434`, Qdrant `192.168.1.93:6333`.
  Redis remains `127.0.0.1:6379` (loopback required by Redis protected mode).
- **`§8.4 Contextual retrieval`** — documented in `RAG-strategy.md §8.4`, NOT implemented.
  Requires re-embedding all 98,737 points; gated on primary-model decision + benchmark harness.

#### Phase 2 design decisions (confirmed 2026-03-30)

**Embedding model — bge-m3 via Ollama (updated 2026-05-04)**
`bge-m3` (1024-dim, 570M params, ~1.2GB) served by Ollama on **Server 2 GPU** (RTX 5070).
Active config: `ollama_url: http://192.168.1.95:11434/api/embeddings`

GPU embed is stable since 2026-05-04 after setting `OLLAMA_CONTEXT_LENGTH=768` in the Ollama
systemd service on Server 2. This caps the embedding compute buffer at 53 MiB (down from 1168 MiB
at the default ctx=4096), reducing bge-m3's total VRAM footprint from ~1.75 GiB to ~635 MiB.
bge-m3 now coexists reliably with llama-server even when both KV slots are active.
RAG child chunks are ~250 tokens; flat chunks cap at 512 tokens — 768 ctx is sufficient with headroom.
Server 1 CPU (`192.168.1.93:11434`) remains the fallback if Server 2 is unavailable.

Chosen over `mxbai-embed-large` (512 BERT-token hard limit caused constant Ollama 500 errors
for technical content — tiktoken BPE undercounts vs BERT WordPiece tokenizer).
Chosen over `qwen3-embedding:8b` (4.7GB, best MTEB): Server 2 VRAM conflict —
llama-server uses ~9.3 GB; qwen3 Q4 ~5 GB = 14.3 GB > 12 GB RTX 5070. bge-m3 is the practical
optimum: top MTEB quality, 8192-token context, ~10–20ms/chunk on GPU.

Server 2 RTX 5070: model is Gemma 4 E4B Q8_0 (7.52B params, 7.5 GB file → ~4,788 MiB VRAM).
KV cache: new llama.cpp SWA-aware implementation uses 4 global KV layers (110024-ctx, f16: ~1,719 MiB) +
20 SWA layers (512-window, f16: 40 MiB) ≈ 1,760 MiB per slot. 2 slots × 1,760 ≈ 3,520 MiB KV.
Total VRAM: ~9,321 MiB / 12,227 MiB (~2,906 MiB headroom) — estimated from measured 131072-ctx baseline.
With `OLLAMA_CONTEXT_LENGTH=768`, bge-m3 needs ~635 MiB total — fits in the ~2,906 MiB headroom.

**Vector DB — Qdrant confirmed**
Qdrant's payload system stores arbitrary JSON per vector point and returns it with every
search result — this is exactly how provenance (source file references) works. Alternatives
considered and rejected:

- ChromaDB: weaker payload filtering, can't efficiently filter by tag/directory at scale.
  Additionally has a critical process-level singleton bug when used via LangChain: the default
  collection name `'langchain'` is shared across all `Chroma.from_documents()` calls in the same
  Python process — subsequent calls append to the existing collection instead of creating a fresh one,
  silently duplicating documents and corrupting search result ranking. `EphemeralClient` does NOT fix
  this (it is still a process singleton). Qdrant has no such issue: collections are explicitly named,
  isolated on disk, and there is no shared process state. SoHoAI also uses `qdrant-client`
  directly (no LangChain wrapper), so no default-name trap can occur.
- LanceDB: good columnar metadata but less mature; no existing integration
- pgvector: requires Postgres (project uses SQLite)

**Qdrant collections — one per modality**
Different embedding models per modality require separate collections:

- `documents` — bge-m3 via Ollama (1024-dim), text chunks
- `images` — CLIP embeddings (Phase 4)
- `videos` — CLIP or frame-level embeddings (Phase 4)

**Parent-child chunking strategy (confirmed 2026-04-10)**

The dataset is dominated by dense technical content (PDFs, Jupyter notebooks, long Markdown docs)
where flat single-size chunks force a bad trade-off: large chunks dilute embedding precision; small
chunks leave the LLM with insufficient context.

Parent-child splitting resolves this:
- **Child chunks** (~200–300 tokens): precise, focused embeddings → better cosine similarity scores
- **Parent chunks** (~800–1200 tokens): full surrounding context → richer LLM answers

Benefit by file type:
- PDFs (certifications, work docs): **high** — 512-token slices of 100-page docs lose all context
- Jupyter notebooks: **high** — code cell alone is meaningless without surrounding markdown explanation
- Long Markdown: **medium** — benefits multi-section docs; short files not affected
- PPTX / short TXT / YAML / config: **low** — already compact, use flat chunking

The docstore (parent text storage) requires **exact ID lookup only — never similarity search**.
A separate vector DB is not needed. The correct choice is to store `parent_text` directly in the
Qdrant payload of each child point (Option A below was selected over SQLite to avoid a second store):

| Option | Mechanism | Trade-off |
|--------|-----------|-----------|
| **Qdrant payload** (chosen) | `parent_text` field on every child point | Trivial duplication across sibling children; single query, no join |
| SQLite table | `rag_parents(id, text, source_path)` + foreign key in payload | No duplication; two-step retrieval (Qdrant → SQLite) |
| Redis | Key-value lookup | TTL risk; Redis already used for conversation state |

Storage overhead of Option A: 50K child chunks × ~2KB average parent text ≈ ~100MB extra on NAS —
negligible at this scale. Gemma 4's 131,072-token context window handles 800–1200 token parents
with no pressure.

Chunk sizes do not apply uniformly across file types — `ingest_file()` should select strategy by type.

**Qdrant payload schema for documents (provenance)**
Each child chunk point stores:

```python
{
    "text": str,           # child chunk content — what was embedded
    "parent_text": str,    # parent chunk — what gets returned to the LLM for context
    "owner": str,          # user who owns this document — derived from NFS root at ingestion
                           # values: "florian", "eva", "annika", "laura", "la-familia"
    "source_path": str,    # full NFS path — this IS the reference returned to the user
    "file_name": str,
    "file_type": str,      # pdf / docx / pptx / txt / ipynb / md / yaml / claude_chat
    "page": int,           # or slide_number for PPTX; cell_index for notebooks
    "chunk_index": int,    # child chunk index within its parent
    "tag": str,            # e.g. "certifications", "cisco-backup", "family"
    # claude_chat documents only (absent on NFS document points):
    "session_id": str,     # Claude Code session UUID — Qdrant-filterable (find all chunks of a session)
    "project": str,        # derived project name, e.g. "SoHoAI" — Qdrant-filterable (find sessions by project)
}
```

`owner` is derived from the NFS path root at ingestion time (`/mnt/nfs/Eva/... → "eva"`), or from the `.claude` parent dir for chat sessions (`/home/florian/.claude → "florian"`).
For `claude_chat` documents, `session_id` and `project` enable cross-referencing:
`Filter(must=[FieldCondition(key="session_id", match=MatchValue(value="<uuid>"))])` retrieves all chunks of a specific session ordered by `chunk_index`.
Search filters: `MatchAny(any=[user_owner, "la-familia"])` — user sees own + shared docs.
`source_path` is the full NFS path — directly usable without a second lookup.
`main.py:_build_rag_prompt()` uses `chunk["source_path"]`.
On retrieval, `chunk["parent_text"]` (not `chunk["text"]`) is what gets injected into the LLM prompt.

#### NFS content survey (2026-04-08) — `/mnt/nfs/Florian` only

Total: **139,195 files, 151GB** — ~120K are Python virtualenv internals and must be excluded.
Other user directories (`/mnt/nfs/{Eva,Annika,Laura}`) and `/mnt/nfs/La-Familia` not yet surveyed.

**RAG-relevant files (~2,800 documents + ~10K media):**

| Type | Count | Notes |
|------|-------|-------|
| PDF | ~559 | Certifications, work docs |
| PPTX | 44 | Slide decks |
| DOCX | 6 | Word docs |
| XLSX | 10 | Spreadsheets |
| Markdown | ~1,667 | Project docs |
| Notebooks | ~1,072 | `.ipynb` — technical content |
| TXT / CSV / YAML | ~1,500 | Config, data, notes |
| Images | ~9,600 | `.jpg`/`.jpeg`/`.png` — family photos (Phase 4) |
| Videos | ~384 | `.mp4`/`.avi` (Phase 4) |
| MSG | 2 | Email — negligible |

**Exclude at ingestion time (config.yaml `rag.scanner`):**
- `.Gin-AI-python-3.12/` — virtualenv (~46K `.py`, 37K `.pyc`, 15K `.h`/`.hpp`)
- `*.pyc`, `*.so`, `*.mo` — compiled artifacts (via extension whitelist — not in RAG extensions)
- `@synoeastream` — Synology NAS streaming metadata (exclude_file_patterns)
- `.dist-info` — pip package metadata (exclude_dir_suffixes)
- `._` prefix — macOS AppleDouble resource fork sidecars (exclude_file_patterns; binary metadata, not parseable)
- `~$` prefix — Microsoft Office lock/temp files (exclude_file_patterns; binary stubs)
- `Microsoft--flotel/Documents/` — IRM/DRM-encrypted old PPTs, unrecoverable (exclude_dir_names)

**Qdrant size estimate:**
- `documents` collection: ~30K–50K chunks × 1024-dim float32 ≈ ~200MB on NAS
- `images` collection (Phase 4): ~9,600 CLIP vectors × 512-dim ≈ ~19MB on NAS
- HNSW index (~500–800MB) fits entirely in Server 1 RAM (32GB) after first load — NFS latency only affects cold start and segment flushes, not query time

#### Phase 2 implementation — ✅ code complete (2026-04-16)

1. ~~Replace `unstructured` with `docling`~~ — ✅ done
2. ~~Replace `sentence-transformers` with Ollama in `rag.py`~~ — ✅ done 2026-04-08
3. ~~Fix `config.yaml` RAG section (bge-m3, ollama_url)~~ — ✅ done 2026-04-17
4. ~~Add `owner` to Qdrant payload schema; `user_id` to `ChatRequest`/`SearchRequest`~~ — ✅ done 2026-04-16
5. ~~Add multi-user config (`users:` + `shared:` sections) to `config.yaml`~~ — ✅ done 2026-04-16
6. ~~Implement `rag_engine/` package (schema, collection, embeddings, state, scanner, ingest, search)~~ — ✅ done 2026-04-16
7. ~~Wire `rag_engine` into `main.py`; delete `rag.py`~~ — ✅ done 2026-04-16
8. ~~Implement standalone CLI utils (`utils/rag_*.py`)~~ — ✅ done 2026-04-16
9. ~~Add `POST /v1/rag/ingest/*` FastAPI endpoints~~ — ✅ done 2026-04-16
10. ~~Add `db_base_path` global config variable (single place to relocate all databases)~~ — ✅ done 2026-04-16
11. ~~Configure `users:` section in `config.yaml` with real Google emails~~ — ✅ done 2026-04-17 (`florian.otel@gmail.com` active; others commented out until ready)

12. ~~Run initial NFS scan and ingestion~~ — ✅ done 2026-04-21 (2891 files, 98,737 Qdrant points; see `RAG-strategy.md §5` for the runbook)
13. ~~Implement `rag_mode` (off/on/only), system-prompt tool-use loop, `prompts/` module~~ — ✅ done 2026-04-21 (§8.1–§8.2; legacy boolean toggle removed)
14. ~~Implement multi-query + MMR reranking (`rag_engine/multi_query.py`)~~ — ✅ code done 2026-04-21; 🚫 evaluated 2026-04-22, no-go verdict — permanently disabled

**→ NOW**: Phase 3 — Google OAuth2 middleware + OpenAI-compatible response format for Open WebUI (not blocking; RAG works end-to-end today via `--user florian`).

Full design details and worker loop spec: `RAG-strategy.md`.

### Phase 3 — MCP integration + Web UI + Auth
- **Google OAuth2 (OIDC)** authentication middleware — family members authenticate with
  separate Google accounts within the same Google Family Group
- MCP gateway in orchestrator (`mcp_gateway.py` — stub ready)
- Initial MCP tool servers: filesystem (done), web search, calendar
- Web frontend: custom FastAPI + HTMX/React with server-managed history (chat_id, Redis, KV cache)
- OpenAI-compatible response format for Open WebUI integration
- Offline resilience: locally cached session tokens with multi-hour TTL; CLI local API key fallback

### Phase 4 — Image search + RL
- CLIP model (openai/clip-vit-base-patch32) on Server 2 GPU
- Family photo ingestion → CLIP embeddings → separate Qdrant `images` collection (same Qdrant instance as `documents`)
- Text-to-image similarity search
- RL training data pipeline: export conversations with feedback signals
  as DPO-format JSONL for training with TRL framework

## Conventions

### Code style
- Python 3.12+, type hints everywhere
- Pydantic v2 for all models (`model_config`, `field_validator`, `model_dump()`)
- Async everywhere in FastAPI and Redis operations
- SQLite operations are sync (wrapped in ChatStore class methods)
- Logging via stdlib `logging` module

### Naming
- Project name: **SoHoAI**
- Python identifiers: `sohoai_*` (underscored)
- MCP tool names: `sohoai_{action}_{resource}` (e.g. `sohoai_read_file`)
- Env var prefix: `SOHOAI_`
- Config file: `config.yaml` (single file, all settings)

### Dependencies

Managed via `pyproject.toml` (uv). Key packages:
- Core: fastapi, uvicorn, litellm, redis, pyyaml, python-dotenv, pydantic, httpx
- MCP: mcp[cli]
- RAG: qdrant-client, docling, tiktoken (embeddings via Ollama HTTP — no Python dep)
- Phase 4: pillow (already included)

### Running

```bash
# Activate the project virtualenv (required on Server 1)
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate

# Server 2 — llama-server (GPU inference + KV cache)
# Gemma 4 E4B 7.52B Q8_0 — hybrid SWA+global attention (42 layers, 2 KV heads)
# New llama.cpp SWA-aware KV: 4 global KV layers (110024-ctx) + 20 SWA layers (512-window)
# VRAM: ~4.8 GB model weights + 2×1,760 MiB KV (f16, 2 slots × 110024 ctx) + ~1 GB overhead = ~9.3 GB / 12 GB (estimated)
# --cache-ram 0: KV pre-allocated in VRAM (no RAM offload); --parallel 2 matches config.yaml num_slots: 2
# NOTE: existing .bin KV slot files are incompatible when switching quantizations — erase k-v-caches/*.bin first
llama-server \
  -m ~/Gin-AI/LLMs-cache/llama-server/google_gemma-4-E4B-it-Q8_0.gguf \
  -c 220048 -ngl 99 \
  --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --cache-ram 0 \
  --parallel 2 \
  --slot-save-path ~/Gin-AI/LLMs-cache/llama-server/k-v-caches/ \
  --host 0.0.0.0 --port 8000

# Server 1 — Qdrant vector store (enabled at boot; restart after reboot)
sudo systemctl start qdrant          # starts /usr/local/bin/qdrant on port 6333
# Active storage: /var/lib/qdrant/storage (local NVMe — NOT NFS)
# Snapshots (DR):  /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/
# Manual snapshot: bash scripts/qdrant/qdrant-snapshot.sh   (auto-runs daily 03:00 via cron)
# Restore from snapshot: PUT http://192.168.1.93:6333/collections/documents/snapshots/recover

# Server 1 — Redis
redis-server --appendonly yes --dir /mnt/nfs/__Backups/SoHoAI--databases/redis

# Server 1 — orchestrator
cd ~/Gin-AI/projects/SoHoAI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server 1 — MCP server (HTTP mode for remote access)
bash NFS-files--MCP-server/nfs_files_mcp_server.sh

# Cline VSCode plugin and Claude Code proxy — see docs/proxy-functionality.md
# Quick reference:
#   Cline (LiteLLM provider):  Base URL http://192.168.1.93:8000/proxy  API key sohoai-local
#   Claude Code (ANTHROPIC_BASE_URL): http://192.168.1.93:8000
#   Claude Code (sub-agent frontmatter): api_base_url http://192.168.1.93:8000/proxy

# CLI chat — RAG off by default (opt-in); --user florian enables ownership filter (omit for dev mode)
python utils/cli_chat.py --server http://192.168.1.93:8000 --user florian
#   in-session: /rag on | /rag only | /rag status | /rag search <query> | /user <id>

# RAG ingestion (see RAG-strategy.md §5 for full walkthrough)
python utils/rag_sync_nfs.py                   # scan NFS roots + claude chat dirs → queue new/modified files; re-queue failed; skip ignored; purge Qdrant for removed files
python utils/rag_ingest_daemon.py --workers 1 --batch 5 --log-file /tmp/rag-ingestion.log    # CPU embed (Server 1); --log-file required for --watch
python utils/rag_ingest_daemon.py --workers 3 --batch 20 --log-file /tmp/rag-ingestion.log   # GPU embed (Server 2, RTX 5070); see RAG-strategy.md §5.4
python utils/rag_status.py                     # one-shot queue counts + Qdrant stats (ignored count always shown)
python utils/rag_status.py --ignored           # detail listing of ignored files + rationale
python utils/rag_status.py --watch /tmp/rag-ingestion.log   # live monitor: chunk progress bar + ETA (requires --log-file on daemon)
python utils/rag_status.py --list-pending              # print every pending file path (pipeable; combine with --user)

# RAG testing
python utils/rag_search_cli.py --query "certifications" --user florian        # retrieval only
python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS-Certification"  # end-to-end retrieval + chat; pass/fail exit

# NotebookLM — first-time login (requires X display)
DISPLAY=:1 python utils/notebooklm_auth.py --login
# NotebookLM — verify saved session
python utils/notebooklm_auth.py --verify

# End-of-session sync: regenerate snapshot + delete old source + upload fresh
python utils/sync_to_notebook.py
# Options:
#   --no-snapshot  skip regeneration, upload existing codebase_snapshot.md
#   --no-delete    keep old source instead of replacing it
```

### Testing a change

After modifying any orchestrator code:
1. The FastAPI app reloads automatically if run with `--reload`
2. Test via CLI: `python utils/cli_chat.py --user florian`
3. Test via curl: `curl -X POST http://192.168.1.93:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hello"}],"rag_mode":"on"}'`
4. Check health: `curl http://192.168.1.93:8000/health`
5. RAG regression: `python utils/rag_smoke_test.py --query "..." --user florian --expect "<known-source-substring>"` (non-zero exit on failure)

### Performance benchmarks

Tool: `~/Gin-AI/tools/llama-performance-test/llama_perf_test.py`
Results: `~/Gin-AI/tools/llama-performance-test/results_q6k.json` (Q6_K baseline), `results_q8_0.json` (Q8_0 current)

Benchmark methodology: 3 runs averaged per scenario, `/completion` native endpoint, `cache_prompt: false`
(cold prefill every run), `ignore_eos: true`, `n_predict: 200`, `temperature: 0` (greedy).

#### Q6_K → Q8_0 comparison (measured 2026-04-20, RTX 5070 12 GB, 2 slots × 131072 ctx at time of measurement)

| Scenario | Q6_K tok/s | Q8_0 tok/s | Δ |
|---|---|---|---|
| **Decode serial** | 109.4 | 96.2 | **−12%** |
| Decode parallel/slot0 | 101.1 | 89.4 | −12% |
| Decode parallel/slot1 | 101.5 | 89.8 | −12% |
| Decode parallel combined | 202.6 | 179.2 | −12% |
| Prefill short (13 tok) | 440 | 399 | −9% (noisy) |
| **Prefill long (4126 tok)** | 5420 | 6270 | **+16%** |
| Wall clock short serial | 1.86s | 2.12s | +14% |
| Wall clock long serial | 2.68s | 2.82s | +5% |
| Parallel wall clock | 2.03s | 2.29s | +13% |
| Parallel speedup ratio | 0.92× | 0.92× | identical |

**Key findings:**
- Decode is ~12% slower — Q8_0 weights are 27% larger (5.9 → 7.5 GB); decode is memory-bandwidth-bound so larger weights cost proportionally more per step.
- Long-context prefill is +16% faster — Q8_0's uniform INT8 maps cleanly to tensor core INT8 ops; Q6_K's mixed-precision K-quant requires a dequantize step that costs more at high token counts.
- Real-world impact at 200 output tokens: ~260ms extra latency per turn. At 100 tokens: ~130ms. Imperceptible in streaming mode.
- Continuous batching unchanged: 0.92× parallel speedup ratio is identical across both quantizations.
- **Conclusion:** ~12% decode regression is the cost of near-f16 weight quality. Acceptable for interactive use at 96 tok/s serial decode.

### Important design decisions
- **Sonnet 4.6 as interactive primary (flipped 2026-04-22)** — Claude Sonnet 4.6 (`anthropic/claude-sonnet-4-5`) is the default for interactive chat; Gemma 4 is reserved for fallback + summarization + offline use. Driver: at ~50–100 turns/day for a 4-user family, Sonnet with prompt caching costs ~$30–60/mo — tolerable — while delivering substantially better reasoning and tool-use fidelity than a 4B local model. Gemma's role shifted from "default inferencer" to "specialized worker" without removing the infrastructure.
- **Anthropic prompt caching on the external path** — `SmartRouter._apply_cache_control()` injects ephemeral `cache_control` on the system message (long-lived anchor) and on `messages[-2]` (rolling prefix anchor). Verified end-to-end: turn-2 onwards hits with `cache_read_input_tokens ≈ 98% of prompt_tokens` on multi-turn chats. ~10× input-cost reduction on steady-state turns. Specialist path is skipped — llama-server's OpenAI-compat endpoint does not cache.
- **Prompt caching vs. summarization trade-off (documented)** — each `maybe_summarize()` event rewrites Redis (summary replaces old turns), which invalidates Anthropic's rolling prefix cache on the next turn. At 100K-token threshold this fires roughly once per ~50-turn chat; the one-time cost spike (~$0.05 on the summary turn) is dominated by the ongoing savings from a smaller cached prefix on all subsequent turns. The summarization threshold (100K) is deliberately aligned with the cloud-routing threshold (`complexity_threshold_tokens: 100000`) so both triggers fire at the same point; Gemma per-slot hard limit is 110,024 tokens.
- **Summary persistence to SQLite** — summaries are persisted to `chats.summary_text` + `chats.summary_covers_through_message_id`. On cold resume (Redis TTL expiry), `warm_from_store` reconstructs Redis as `[system?, summary_msg, messages WHERE id > boundary]` — no data duplication, full conversation raw log always retained in `messages` table. Closes a pre-existing gap where summaries were lost across 24h TTL.
- **llama-server over vLLM** — native KV slot save/restore API (`/slots/{id}?action=save|restore`); 2 slots × 110024 ctx at ~9,321 MiB VRAM with f16 KV, flash-attn, SWA-aware KV allocation (Gemma 4 E4B 7.52B Q8_0)
- **KV cache in `ConversationCache`** — `conversation.py` is the single owner of all conversation state (Redis + KV). `resume()`/`park()` keep save/restore co-located with Redis ops
- **Internal bypasses LiteLLM** — native `/completion` required to pass `slot_id`. External path goes through LiteLLM; internal path calls llama-server directly. Branch at [main.py:333-347](main.py#L333-L347).
- **Proxy endpoints for external clients (2026-04-22, Option 2)** — Cline VSCode plugin uses `POST /proxy/v1/chat/completions` (OpenAI-compatible); Claude Code uses `POST /v1/messages` (Anthropic Messages API, enables `ANTHROPIC_BASE_URL`). Both are stateless pass-throughs to `SmartRouter.complete()` — no Redis/RAG/summarization. Model-name mapping in `_PROXY_EXPOSED_MODELS` in [main.py](main.py). Prompt caching applies on the external path. Supersedes old Server-2 LiteLLM proxy at `:8001`. Full details: `docs/proxy-functionality.md`.
- **Shared llama-server (2026-04-22)** — same Server 2 llama-server backs SoHoAI's `internal` path and Cline's `gemma-4-e4b` proxy path. Known KV-slot race; accepted (post-flip Gemma hits are rare, self-heals next turn). See `docs/proxy-functionality.md §1`.
- **Rolling summarization uses internal (llama-server)** — `maybe_summarize()` erases the KV slot before calling; both summarization and subsequent inference start cold sequentially on the same slot; triggered at ~100K tokens, keeps last 20 turns verbatim. Model pinned via `routing.summarization_model` in `config.yaml` (default `internal`).
- **LiteLLM stays as the routing + fallback layer** — handles OpenAI/Anthropic API differences and executes the fallback chain `external → internal` (reversed direction of pre-flip implementation)
- **Tool-use — XML sentinel path only (native Anthropic tool-use deferred)** — both external and internal paths emit and parse the custom `<tool_call>...</tool_call>` sentinel. Sonnet handles it reliably in practice. Native Anthropic `tools=[...]` + `tool_use` content blocks remains a deferred follow-up; unifying the two paths isn't blocking current quality.
- **Redis for short-term memory** — fast, TTL-based, LLM context builder reads it every request (server-managed only)
- **SQLite for long-term** — single file on NAS, zero ops overhead, plenty fast for ~10K chats (server-managed only)
- **Qdrant for vectors** — persistent local mode (on NAS), gRPC + REST API; `qdrant-client` Python dep
- **Embeddings via Ollama on Server 2 GPU** — `bge-m3` (1024-dim, 8192-tok context) served by Ollama on Server 2 (RTX 5070, `192.168.1.95:11434`); stable since 2026-05-04 with `OLLAMA_CONTEXT_LENGTH=768` in the Ollama service (reduces compute buffer from 1168 MiB to 53 MiB, total VRAM ~635 MiB — fits alongside llama-server even under load); `sentence-transformers` removed. Server 1 CPU (`192.168.1.93:11434`) is the fallback. `embed_batch()` concurrency controlled by `--batch` in `rag_ingest_daemon.py` (default 5, `_BATCH_CONCURRENCY`); lower `--batch` to reduce `httpx.ReadTimeout` errors, raise it for more throughput. Ollama serializes model computation — when ingest daemon runs, search queries queue behind it and can wait 28–30s; `embed_text()` timeout is 120s to survive this. `embed_batch()` fires a `progress_cb(done, total)` every 50 chunks (`_PROGRESS_INTERVAL`); `ingest.py` wires this to a logger call; `rag_status.py --watch` parses those log lines to compute real-time chunk rate + ETA. SQLite fetch batch size is hardcoded to 10 files per iteration (`_FETCH_BATCH_SIZE` in `rag_ingest_daemon.py`). Qdrant upsert is batched in groups of `_UPSERT_BATCH_SIZE=256` points (`ingest.py`) to avoid HTTP 400 on large files.
- **Qdrant client HTTP timeout** — `rag_engine/collection.py::get_client()` initializes the `QdrantClient` with `timeout=60` (increased from httpx default ~5s in 2026-04-22). During heavy bulk ingestion (e.g., 70K+ points from a single file), Qdrant performs extended index optimization that can block response handling for 10–30s. The 5s timeout was too short, causing regular `httpcore.ReadTimeout` cascades during large ingestion runs. 60s allows Qdrant time to complete index restructuring. See `TROUBLESHOOTING.md` for full analysis.
- **Multi-tenancy via `owner` field + Google OAuth2** — every Qdrant point carries an `owner` derived from NFS path root at ingestion; search applies `MatchAny(any=[user_owner, "la-familia"])` filter; user identity from Google OIDC JWT mapped to `owner` via `config.yaml` `users:` section; data model designed before first ingestion to avoid re-ingesting
- **Qdrant-before-SQLite deletion ordering (2026-04-24, refactored 2026-04-25)** — `scan_nfs_roots()` and `scan_claude_chats()` each return `{'scanned': N, 'existing_paths': set[str]}` without calling `find_deleted()` internally. Both callers (`rag_sync_nfs.py` and `main.py /v1/rag/ingest/sync`) merge the two `existing_paths` sets, then call `state_db.find_deleted(merged)` once to get `(deleted_paths, stale_paths)`. Callers then: (1) delete Qdrant points for each path in `deleted_paths`, (2) call `state_db.purge_deleted(stale_paths)`. If killed mid-Qdrant-loop, the SQLite rows survive intact and the next sync retries automatically. Merging before `find_deleted()` is critical — if only NFS paths were passed, claude chat rows in SQLite would be falsely flagged as deleted. The old `handle_deleted()` committed SQLite atomically before Qdrant cleanup and has been deprecated. Observed orphan incident 2026-04-24: 12 paths required manual Qdrant scroll + targeted delete.
- **MCP server uses path sandboxing** — all operations validated against ALLOWED_ROOTS, no escape possible
