---
title: "HomeAI-Lab — Project Context & Design Reference"
date: 2026-04-07
last_updated: 2026-04-16
created_by: Florian Otel / Cline (Claude Sonnet 4.6)
last_updated_by: Claude Code (Claude Sonnet 4.6)
context: >
  HomeAI-Lab project (https://github.com/FlorianOtel/HomeAI-Lab);
  Project instructions and design decisions for Claude Code;
  Infrastructure, architecture, API, implementation phases, RAG strategy,
  multi-tenancy (Google OAuth2, per-user NFS roots, Qdrant owner filtering)
---

# CLAUDE.md — HomeAI-Lab Project Context

**Repository**: https://github.com/FlorianOtel/HomeAI-Lab

## What is this project?

HomeAI-Lab is a distributed, two-server home-office AI system. It provides a unified
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

- Chat DB: `/mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/chats.db` (SQLite, NAS)
- RAG state DB: `/mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/rag_state.db` (SQLite, NAS)
- Vector store (active): `/var/lib/qdrant/storage` (local NVMe, Server 1 only — **not NFS**)
- Vector store (snapshots/DR): `/mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots/` (NAS)
  - **Note**: `/mnt/nfs/__Backups/HomeAI-lab--databases/qdrant/` is intentionally empty — it is NOT used.
    Qdrant server uses RocksDB which is incompatible with NFS file locking. Active data must stay local.
    Snapshots (passive files) are safe on NFS and taken daily at 03:00 via cron.
- Redis persistence: `/mnt/nfs/__Backups/HomeAI-lab--databases/redis/`
- KV cache slots: `/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/k-v-caches/` (one `.bin` per chat_id)
- LLM model cache: `/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/`
- Documents for RAG: `/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab/documents/`
- RL training exports: `/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab/rl-data/`
- Chat markdown exports: `/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab/exports/`

## Architecture

```
Server 1 (192.168.1.93)                    Server 2 (192.168.1.95)
┌──────────────────────────┐               ┌──────────────────────────┐
│ FastAPI orchestrator:8000│─── /completion→ │ llama-server :8000       │
│  ConversationCache       │─── /slots ────→ │  Gemma 4 E4B 7.52B Q8_0│
│    Redis (short-term)    │               │  2×131072 ctx (2 slots)  │
│    KV cache mgr          │               │  KV slot save/restore    │
│    rolling summarization │               │  CLIP (Phase 4)          │
│ SQLite (long-term)       │               └──────────────────────────┘
│ MCP server :3001 (HTTP)  │                          │
│ Qdrant server :6333      │                          │
│  active storage: NVMe    │                          │
│  snapshots → NAS daily   │                          │
└──────────────────────────┘                          │
           │                                          │
           └──── both mount ──→ NAS (27TB NFS) ───────┘
                               (Redis AOF, SQLite, KV .bin files,
                                model cache, Qdrant snapshots, exports)
```

### LLM routing (via LiteLLM)

Two model tiers with automatic fallback:

1. **specialist** — Gemma 4 E4B 7.52B Q8_0 on RTX 5070 via llama-server (primary, GPU, Server 2)
2. **external** — Claude Sonnet via Anthropic API (fallback if Server 2 unreachable, or >120K token context)

Routing logic is in `router.py`. Default is specialist; falls back to cloud if
unreachable. Rolling summarization keeps conversations well below the 100K threshold
in practice.

**Specialist model path** bypasses LiteLLM and calls llama-server's native
`/completion` endpoint directly, enabling KV cache slot targeting (`slot_id`).
This includes rolling summarization — no slot contention since `maybe_summarize()`
erases the KV slot before calling, and both summarization and the subsequent
main inference start cold on the same slot sequentially.

The external model path goes through LiteLLM as usual.

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
- `maybe_summarize(chat_id, fn)` — if Redis context exceeds ~100K tokens,
  summarize old turns via specialist (llama-server), rebuild Redis, erase stale KV cache
- `is_cold(chat_id)` — True if Redis has no messages for this chat
- `warm_from_store(chat_id, messages)` — reload last N turns from SQLite into Redis
  on cold-start resume; also erases stale KV cache

Code: `conversation.py`, `kv_cache.py`, `chat_store.py`

### MCP server ✅ (completed, working)

Subdirectory: `./NFS-files--MCP-server/`

Files:
- `nfs_files_mcp_server.py` — MCP server implementation (22.9 KB)
- `setup_mcp.sh` — install + validation script
- `claude_desktop_config.json` — config snippet for Claude Desktop
- `server_config.json` — server configuration

Exposes all files under `/mnt/nfs/Florian/Gin-AI` to any MCP client.
Project files are under `/mnt/nfs/Florian/Gin-AI/projects/HomeAI-Lab`.

MCP server name (as seen by clients): **`nfs-files`**

Tools: `list_directory`, `read_file`, `write_file`, `edit_file`, `delete_file`, `search_files`, `get_file_info`

Resources: `file:///nfs_files/config`, `file://structure`

Transport: streamable HTTP on port 3001 (default) or stdio

Path safety: all paths resolved against ALLOWED_ROOTS; traversal blocked.
Env var `HOMEAI_LAB_PROJECT_DIR` controls the root (defaults to `~/Gin-AI`).

Listing supports depth 0-20 (0 = unlimited recursive). Search uses rglob
(always recursive) with up to 1000 result lines.

**Note**: relative paths resolve from the Gin-AI root, so project files need
the `projects/HomeAI-Lab/` prefix (e.g. `projects/HomeAI-Lab/config.yaml`).

## Project structure

```
HomeAI-Lab/
├── main.py                     # FastAPI app — the central orchestrator
├── config.yaml                 # All configuration (models, Redis, RAG, routing, llama-server)
├── .env                        # ANTHROPIC_API_KEY (not committed)
├── .mcp.json                   # Claude Code auto-discovers this
├── schemas.py                  # Pydantic models (ChatRequest, ChatResponse, etc.)
├── router.py                   # SmartRouter — LiteLLM wrapper with routing logic
├── conversation.py             # ConversationCache — Redis + KV cache coordinator
├── kv_cache.py                 # KVCacheManager — llama-server slot save/restore + inference
├── chat_store.py               # ChatStore — SQLite long-term persistence
├── mcp_gateway.py              # MCP tool gateway (Phase 3 stub, interface defined)
├── pyproject.toml              # Python dependencies (uv-managed)
├── rag_engine/                 # RAG pipeline (Phase 2 ✅ complete)
│   ├── __init__.py             # exports search_rag(query, user_id, limit)
│   ├── schema.py               # Qdrant payload field constants + derive_owner()
│   ├── collection.py           # Collection name, vector size, get_client(), ensure_collection()
│   ├── embeddings.py           # embed_text(), embed_batch(progress_cb) via Ollama
│   ├── state.py                # StateDB — ingestion queue CRUD + crash recovery
│   ├── scanner.py              # NFS filesystem scanner → populates StateDB; filters read from config.yaml rag.scanner
│   ├── ingest.py               # docling parse + parent-child chunking + Qdrant upsert
│   └── search.py               # query → embed → Qdrant query_points → parent_text + provenance
├── utils/
│   ├── cli_chat.py             # Terminal chat client; RAG on by default; --user OWNER sends user_id for ownership filter; /rag search <query> inspects retrieval; /user <id> changes owner mid-session
│   ├── rag_sync_nfs.py         # CLI: scan NFS roots → queue new/modified files; re-queue failed; skip ignored; purge Qdrant for removed files
│   ├── rag_ingest_daemon.py    # CLI: process pending files (parse → chunk → embed → upsert)
│   ├── rag_status.py           # CLI: queue counts + Qdrant point stats; --ignored for detail; --watch LOG_FILE for live ETA monitor; --list-pending [N] to print pending paths (pipeable)
│   ├── rag_search_cli.py       # CLI: retrieval-only — embed query + Qdrant search; prints top-k hits + parent_text preview
│   ├── rag_smoke_test.py       # CLI: end-to-end smoke test — retrieval + /v1/chat/completions with use_rag=true; --expect SUBSTR assertion; pass/fail exit code
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
| POST | `/v1/rag/ingest/sync` | Scan NFS roots → populate ingestion queue |
| POST | `/v1/rag/ingest/start` | Start background ingestion worker |
| POST | `/v1/rag/ingest/stop` | Stop ingestion worker gracefully |
| GET | `/v1/rag/ingest/status` | Queue metrics + Qdrant point count |

Responses use a custom `ChatResponse` model (`chat_id`, `model_used`, `message`, `rag_sources`).
**Not** OpenAI-compatible format — the CLI reads `data["message"]["content"]`.
OpenAI-compatible response format is a Phase 3 requirement for Open WebUI integration.

## Implementation phases

### Phase 1 — Core loop ✅ (complete)
- FastAPI orchestrator on Server 1
- LiteLLM routing with fallback chain (specialist → orchestrator → external)
- llama-server on Server 2 GPU: Gemma 4 E4B 7.52B Q8_0, 2 slots × 131072 ctx (`-c 262144 --parallel 2`) ✅
- Per-conversation KV cache persisted to NAS via llama-server slot API ✅
- Rolling summarization at ~100K token threshold (via specialist/llama-server) ✅
- Redis conversation cache with NAS AOF persistence
- SQLite chat store with full CRUD
- Markdown and JSONL export
- CLI chat client (`utils/cli_chat.py`)
- Feedback collection for RL
- MCP server for Gin-AI filesystem (`NFS-files--MCP-server/`) ✅

### Per-request flow (specialist model path)
```
resume(chat_id)      → assign slot + restore KV from NAS (if exists)
append(user_msg)     → Redis + SQLite
maybe_summarize()    → if >200K chars: summarize old turns via specialist,
                       rebuild Redis, erase stale KV, re-assign slot
get_context()        → condensed history from Redis
inference(slot_id)   → POST /completion to llama-server GPU slot
append(assistant)    → Redis + SQLite
park(chat_id)        → save KV slot to NAS + refresh Redis TTL
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
- Embeddings via **bge-m3 served by Ollama on Server 1**
  - 1024 dimensions, 570M params, ~1.2GB — top MTEB scores; 8192-token context window
  - API: `POST http://localhost:11434/api/embeddings`; no batch endpoint — `embed_batch()` runs up to N concurrent requests via asyncio semaphore, where N is `--batch` in `rag_ingest_daemon.py` (default 5)
  - `embed_batch()` accepts optional `progress_cb(done, total)` — called every 50 chunks (`_PROGRESS_INTERVAL`) + on the final chunk; used by `ingest.py` to log `"Embedding progress: N/M  filename"` lines read by `rag_status.py --watch`
  - Config keys: `embedding_model: bge-m3`, `ollama_url: http://localhost:11434/api/embeddings`
  - Query latency: ~650ms/chunk unloaded; **~28–30s under full ingest load** (Ollama serializes, search query queues behind ingest batch)
  - HTTP timeout in `embed_text()`: **120s** — must be this high or search times out while ingest daemon is running
- Qdrant vector store on Server 1, persisted to NAS — **confirmed right choice** (see design decisions)
- RAG-augmented prompts (context injection before LLM call)
- Package: `rag_engine/` — fully implemented (schema, collection, embeddings, state, scanner, ingest, search); `rag.py` deleted
- Full design details: `RAG-strategy.md`

#### Phase 2 design decisions (confirmed 2026-03-30)

**Embedding model — bge-m3 via Ollama (updated 2026-04-17)**
`bge-m3` (1024-dim, 570M params, ~1.2GB) served by Ollama on Server 1 CPU.
Chosen over `mxbai-embed-large` (512 BERT-token hard limit caused constant Ollama 500 errors
for technical content — tiktoken BPE undercounts vs BERT WordPiece tokenizer).
Chosen over `qwen3-embedding:8b` (4.7GB, best MTEB): Server 2 VRAM conflict —
llama-server uses 11.6/12GB, leaving no room; Server 1 CPU inference at 8B params
would add 6–10s to every RAG chat query. bge-m3 is the practical optimum:
top MTEB quality, 8192-token context, 650ms/chunk on CPU, fits alongside all workloads.

Server 2 RTX 5070: model is Gemma 4 E4B Q8_0 (7.52B params, 7.5 GB file → ~4,788 MiB VRAM).
KV cache: new llama.cpp SWA-aware implementation uses 4 global KV layers (131072-ctx, f16: 2,048 MiB) +
20 SWA layers (512-window, f16: 40 MiB) = 2,088 MiB per slot. 2 slots × 2,088 = 4,176 MiB KV.
Total VRAM: 9,977 MiB used / 12,227 MiB (1,799 MiB headroom) — measured 2026-04-20.
No reliable embedding model fits alongside llama-server without VRAM risk under load.

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
  isolated on disk, and there is no shared process state. HomeAI-Lab also uses `qdrant-client`
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
    "file_type": str,      # pdf / docx / pptx / txt / ipynb / md / yaml
    "page": int,           # or slide_number for PPTX; cell_index for notebooks
    "chunk_index": int,    # child chunk index within its parent
    "tag": str,            # e.g. "certifications", "cisco-backup", "family"
}
```

`owner` is derived from the NFS path root at ingestion time (`/mnt/nfs/Eva/... → "eva"`).
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

**Exclude at ingestion time (path filter):**
- `*/Gin-AI/.Gin-AI-python-3.12/**` — virtualenv (~46K `.py`, 37K `.pyc`, 15K `.h`/`.hpp`)
- `**/*.pyc`, `**/*.so`, `**/*.mo` — compiled artifacts
- `**/*@synoeastream` — Synology NAS streaming metadata
- `**/*.dist-info/**` — package metadata

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
- Project name: **HomeAI-Lab** (hyphenated in paths/dirs)
- Python identifiers: `homeai_lab_*` (underscored)
- MCP tool names: `homeai_lab_{action}_{resource}` (e.g. `homeai_lab_read_file`)
- Env var prefix: `HOMEAI_LAB_`
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
# New llama.cpp SWA-aware KV: 4 global KV layers (131072-ctx) + 20 SWA layers (512-window)
# VRAM: ~4.8 GB model weights + 2×2,088 MiB KV (f16, 2 slots × 131072 ctx) + ~1 GB overhead = ~9.97 GB / 12 GB (measured)
# --cache-ram 0: KV pre-allocated in VRAM (no RAM offload); --parallel 2 matches config.yaml num_slots: 2
# NOTE: existing .bin KV slot files are incompatible when switching quantizations — erase k-v-caches/*.bin first
llama-server \
  -m ~/Gin-AI/LLMs-cache/llama-server/google_gemma-4-E4B-it-Q8_0.gguf \
  -c 262144 -ngl 99 \
  --flash-attn on \
  --cache-type-k f16 --cache-type-v f16 \
  --cache-ram 0 \
  --parallel 2 \
  --slot-save-path ~/Gin-AI/LLMs-cache/llama-server/k-v-caches/ \
  --host 0.0.0.0 --port 8000

# Server 1 — Qdrant vector store (enabled at boot; restart after reboot)
sudo systemctl start qdrant          # starts /usr/local/bin/qdrant on port 6333
# Active storage: /var/lib/qdrant/storage (local NVMe — NOT NFS)
# Snapshots (DR):  /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots/
# Manual snapshot: bash scripts/qdrant/qdrant-snapshot.sh   (auto-runs daily 03:00 via cron)
# Restore from snapshot: PUT http://192.168.1.93:6333/collections/documents/snapshots/recover

# Server 1 — Redis
redis-server --appendonly yes --dir /mnt/nfs/__Backups/HomeAI-lab--databases/redis

# Server 1 — orchestrator
cd ~/Gin-AI/projects/HomeAI-Lab
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server 1 — MCP server (HTTP mode for remote access)
bash NFS-files--MCP-server/nfs_files_mcp_server.sh

# CLI chat — RAG on by default; --user florian enables ownership filter (omit for dev mode)
python utils/cli_chat.py --server http://192.168.1.93:8000 --user florian
#   in-session: /rag status | /rag search <query> | /rag off | /user <id>

# RAG ingestion (see RAG-strategy.md §5 for full walkthrough)
python utils/rag_sync_nfs.py                   # scan NFS → queue new/modified files; re-queue failed; skip ignored; purge Qdrant for removed files
python utils/rag_ingest_daemon.py              # process queue (runs for hours); --batch controls Ollama concurrency (default 5)
python utils/rag_status.py                     # one-shot queue counts + Qdrant stats (ignored count always shown)
python utils/rag_status.py --ignored           # detail listing of ignored files + rationale
python utils/rag_status.py --watch /tmp/rag-ingestion.log   # live monitor: chunk progress bar + ETA
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
3. Test via curl: `curl -X POST http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hello"}]}'`
4. Check health: `curl http://localhost:8000/health`
5. RAG regression: `python utils/rag_smoke_test.py --query "..." --user florian --expect "<known-source-substring>"` (non-zero exit on failure)

### Performance benchmarks

Tool: `~/Gin-AI/tools/llama-performance-test/llama_perf_test.py`
Results: `~/Gin-AI/tools/llama-performance-test/results_q6k.json` (Q6_K baseline), `results_q8_0.json` (Q8_0 current)

Benchmark methodology: 3 runs averaged per scenario, `/completion` native endpoint, `cache_prompt: false`
(cold prefill every run), `ignore_eos: true`, `n_predict: 200`, `temperature: 0` (greedy).

#### Q6_K → Q8_0 comparison (measured 2026-04-20, RTX 5070 12 GB, 2 slots × 131072 ctx)

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
- **llama-server over vLLM** — native KV slot save/restore API (`/slots/{id}?action=save|restore`); 2 slots × 131072 ctx at 9,977 MiB VRAM with f16 KV, flash-attn, SWA-aware KV allocation (Gemma 4 E4B 7.52B Q8_0)
- **KV cache in `ConversationCache`** — `conversation.py` is the single owner of all conversation state (Redis + KV). `resume()`/`park()` keep save/restore co-located with Redis ops
- **Specialist bypasses LiteLLM** — native `/completion` required to pass `slot_id`. LiteLLM handles external fallback unchanged
- **Rolling summarization uses specialist (llama-server)** — `maybe_summarize()` erases the KV slot before calling; both summarization and subsequent inference start cold sequentially on the same slot; triggered at ~100K tokens, keeps last 20 turns verbatim
- **LiteLLM stays as the routing layer** — handles OpenAI/Anthropic API differences and fallback/retry for the external (cloud) model
- **Redis for short-term memory** — fast, TTL-based, LLM context builder reads it every request (server-managed only)
- **SQLite for long-term** — single file on NAS, zero ops overhead, plenty fast for ~10K chats (server-managed only)
- **Qdrant for vectors** — persistent local mode (on NAS), gRPC + REST API; `qdrant-client` Python dep
- **Embeddings via Ollama on Server 1** — `bge-m3` (1024-dim, 8192-tok context) served by Ollama; Server 2 GPU uses ~9 GB with 2 parallel slots at full context with Gemma 4 (leaving ~3.2 GB headroom, not enough for a reliable embedding model under load); `sentence-transformers` removed. `embed_batch()` concurrency controlled by `--batch` in `rag_ingest_daemon.py` (default 5, `_BATCH_CONCURRENCY`); lower `--batch` to reduce `httpx.ReadTimeout` errors, raise it for more throughput. Ollama serializes model computation — when ingest daemon runs, search queries queue behind it and can wait 28–30s; `embed_text()` timeout is 120s to survive this. `embed_batch()` fires a `progress_cb(done, total)` every 50 chunks (`_PROGRESS_INTERVAL`); `ingest.py` wires this to a logger call; `rag_status.py --watch` parses those log lines to compute real-time chunk rate + ETA. SQLite fetch batch size is hardcoded to 10 files per iteration (`_FETCH_BATCH_SIZE` in `rag_ingest_daemon.py`). Qdrant upsert is batched in groups of `_UPSERT_BATCH_SIZE=256` points (`ingest.py`) to avoid HTTP 400 on large files.
- **Multi-tenancy via `owner` field + Google OAuth2** — every Qdrant point carries an `owner` derived from NFS path root at ingestion; search applies `MatchAny(any=[user_owner, "la-familia"])` filter; user identity from Google OIDC JWT mapped to `owner` via `config.yaml` `users:` section; data model designed before first ingestion to avoid re-ingesting
- **MCP server uses path sandboxing** — all operations validated against ALLOWED_ROOTS, no escape possible
