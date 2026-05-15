---
title: "SoHoAI — Project Context & Design Reference"
created_at: 20260407-000000
created_by: Florian Otel / Cline (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-15--19-54
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
| Server 2 | 192.168.1.95 | LLM inference engine | 16GB RAM, Nvidia RTX 5070 12GB, llama-server (Qwen3.5-4B) |
| NAS | NFS-mounted | Persistent storage for everything | 27TB |

### Storage paths

> SQLite and Redis paths are derived from `db_base_path` in `config.yaml`. Qdrant active storage is local-only (see below).

- Chat DB: `/mnt/nfs/__Backups/SoHoAI--databases/sqlite/telemetry.db` (SQLite, NAS)
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
│  SmartRouter (LiteLLM)   │  variants →      │  Qwen3.5-4B Q6_K_XL     │
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

## Documentation

| Doc | Contents |
|-----|---------|
| [docs/Model-routing.md](docs/Model-routing.md) | LLM routing tiers, proxy paths (Cline + Claude Code), sub-agent mechanics, cost tables |
| [docs/Memory-tiers.md](docs/Memory-tiers.md) | Redis / SQLite / KV cache tiers, ConversationCache API, rolling summarization design |
| [docs/MCP-functionality.md](docs/MCP-functionality.md) | NFS-files MCP server tools, transport, path safety |
| [docs/RAG-strategy.md](docs/RAG-strategy.md) | RAG pipeline design, chunking, embeddings, Qdrant schema, ingestion service, recovery/rollback (§11) |
| [docs/design-history.md](docs/design-history.md) | Implementation milestones, benchmark results, architectural decisions |
| [docs/TODO.md](docs/TODO.md) | Deferred work: Phase 3, Phase 4, local-model tool-use |
| [docs/RAG-troubleshoot.md](docs/RAG-troubleshoot.md) | Qdrant troubleshooting, recovery procedures |
| [docs/Telemetry.md](docs/Telemetry.md) | Usage telemetry design: Stage 1 schema + Stage 2 claude-orchestra migration checklist |

## Project structure

```
SoHoAI/
├── main.py                     # FastAPI app — the central orchestrator
├── config.yaml                 # All configuration (models, Redis, RAG, routing, llama-server)
├── .env                        # ANTHROPIC_API_KEY (not committed)
├── .mcp.json                   # Claude Code auto-discovers this
├── schemas.py                  # Pydantic models (ChatRequest, ChatResponse, etc.)
├── router.py                   # SmartRouter — LiteLLM wrapper with routing logic
├── usage_tracker.py            # UsageTracker — LiteLLM CustomLogger callback; records usage_events to telemetry.db
├── conversation.py             # ConversationCache — Redis + KV cache coordinator
├── kv_cache.py                 # KVCacheManager — llama-server slot save/restore + inference; apply_qwen3_template() (Qwen3.5 ChatML format)
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
│   ├── rag_rerank_bench.py     # CLI: dense vs reranked vs hybrid comparison bench; --mode dense|rerank|both|hybrid|hybrid+rerank
│   ├── rag_bench_queries.txt   # 14 verified queries used by bench harnesses
│   ├── rag_sparse_migrate.py   # CLI: one-time corpus migration to add sparse vectors (migrate / swap / status)
│   ├── rag_purge_corrupt.py    # CLI: scan+delete corrupt/zombie Qdrant vectors; read-only SQLite (mode=ro)
│   ├── qdrant_status.py        # CLI: Qdrant optimizer status, lag, segments; --watch for continuous monitoring
│   ├── rag_reset.py            # CLI: reset Qdrant collection + ingestion queue
│   ├── notebooklm_auth.py      # NotebookLM browser automation (Playwright + system Chrome)
│   ├── snapshot_codebase.py    # Aggregate project files → codebase_snapshot.md
│   ├── sync_to_notebook.py     # End-of-session sync: snapshot → delete old → upload
│   ├── notebooklm_session.json # Saved Google session cookies (not committed)
│   └── codebase_snapshot.md    # Generated snapshot (not committed)
├── scripts/
│   ├── rag-ingest-run.sh           # RAG ingestion wrapper: per-user sync loop + daemon; used by systemd service
│   ├── rag-ingest.service          # systemd oneshot service (install to /etc/systemd/system/)
│   ├── rag-ingest.timer            # fires 01/07/13/19:00 local; Persistent=true for missed slots
│   ├── rag-ingest-logrotate        # logrotate config (daily, 7 days, copytruncate)
│   └── qdrant/
│       ├── qdrant-config.yaml      # Qdrant server config (storage paths, ports)
│       ├── qdrant.service          # systemd unit file (copy to /etc/systemd/system/)
├── sqlite-qdrant-snapshot.sh   # SQLite WAL checkpoint + Qdrant snapshot → NFS (cron: daily 03:00)
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
| GET | `/v1/usage/stats` | Token usage + cost reporting (filter by user, model, source, session_id, time window) |
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

| Phase | Status | Reference |
|-------|--------|-----------|
| Phase 1 — Core loop | ✅ complete | [docs/design-history.md](docs/design-history.md) |
| Phase 2 — RAG | ✅ complete (2026-04-21, 98,737 points) | [docs/RAG-strategy.md](docs/RAG-strategy.md) · [docs/design-history.md](docs/design-history.md) |
| Phase 3 — Web UI + Auth | In progress | [docs/TODO.md](docs/TODO.md) |
| Phase 4 — Image search + RL | Future | [docs/TODO.md](docs/TODO.md) |

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
# Qwen3.5-4B Q6_K_XL — ChatML format, --jinja for template, q8_0 KV
# VRAM: ~3.4 GB model weights + 2×~1,320 MiB KV (q8_0, 2 slots × 110024 ctx) ≈ 6 GB estimated / 12 GB
# NOTE: existing .bin KV slot files are incompatible when switching models — erase k-v-caches/*.bin first
llama-server \
  -m ~/Gin-AI/LLMs-cache/llama-server/Qwen3.5-4B-UD-Q6_K_XL.gguf \
  --jinja --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ngl 99 --ctx-size 220048 --parallel 2 \
  --slot-save-path ~/Gin-AI/LLMs-cache/llama-server/k-v-caches/ \
  --host 0.0.0.0 --port 8000

# Server 1 — Qdrant vector store (enabled at boot; restart after reboot)
sudo systemctl start qdrant          # starts /usr/local/bin/qdrant on port 6333
# Active storage: /var/lib/qdrant/storage (local NVMe — NOT NFS)
# Snapshots (DR):  /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/
# Manual snapshot: bash scripts/sqlite-qdrant-snapshot.sh   (auto-runs daily 03:00 via cron)
# Restore from snapshot: PUT http://192.168.1.93:6333/collections/documents/snapshots/recover

# Server 1 — Redis
redis-server --appendonly yes --dir /mnt/nfs/__Backups/SoHoAI--databases/redis

# Server 1 — orchestrator
cd ~/Gin-AI/projects/SoHoAI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Server 1 — MCP server (HTTP mode for remote access)
bash NFS-files--MCP-server/nfs_files_mcp_server.sh

# Cline VSCode plugin and Claude Code proxy — see docs/Model-routing.md
# Quick reference:
#   Cline (LiteLLM provider):  Base URL http://192.168.1.93:8000/proxy  API key sohoai-local
#   Claude Code (ANTHROPIC_BASE_URL): http://192.168.1.93:8000
#   Claude Code (sub-agent frontmatter): api_base_url http://192.168.1.93:8000/proxy

# CLI chat — RAG off by default (opt-in); --user florian enables ownership filter (omit for dev mode)
python utils/cli_chat.py --server http://192.168.1.93:8000 --user florian
#   in-session: /rag on | /rag only | /rag status | /rag search <query> | /user <id>

# RAG ingestion — automated service (live since 2026-05-05; see RAG-strategy.md §10)
# Fires at 01:00, 07:00, 13:00, 19:00 CEST via systemd timer; log: /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log
systemctl status rag-ingest.timer              # check timer status and next trigger
python utils/rag_status.py --watch /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log  # live monitor for service runs

# RAG ingestion — manual / debug (see RAG-strategy.md §5 for full walkthrough)
bash scripts/rag-ingest-run.sh                 # run the full service wrapper manually (multi-user sync + daemon)
python utils/rag_sync_nfs.py --user florian    # scan only; use --user to avoid path-dedup ambiguity (see RAG-strategy.md §10.5)
python utils/rag_ingest_daemon.py --workers 1 --batch 5 --log-file /tmp/rag-ingestion.log    # CPU embed (Server 1); --log-file required for --watch
python utils/rag_ingest_daemon.py --workers 3 --batch 20 --log-file /tmp/rag-ingestion.log   # GPU embed (Server 2, RTX 5070); see RAG-strategy.md §5.4
python utils/rag_status.py                     # one-shot queue counts + Qdrant stats (ignored count always shown)
python utils/rag_status.py --ignored           # detail listing of ignored files + rationale
python utils/rag_status.py --watch /tmp/rag-ingestion.log   # live monitor for manual daemon runs
python utils/rag_status.py --list-pending              # print every pending file path (pipeable; combine with --user)

# RAG testing
python utils/rag_search_cli.py --query "certifications" --user florian        # retrieval only; add --no-rerank to skip cross-encoder reranking
python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS-Certification"  # end-to-end retrieval + chat; pass/fail exit
python utils/rag_rerank_bench.py --user florian                               # dense-only vs reranked top-5 comparison

# RAG sparse migration — one-time, required to enable hybrid search (see RAG-strategy.md §14)
python utils/rag_sparse_migrate.py status                       # check migration state
python utils/rag_sparse_migrate.py migrate                      # phase 1: migrate data (live)
python utils/rag_sparse_migrate.py swap --confirm               # phase 2: swap (stop uvicorn first)

# Tool-use smoke test — validates LiteLLM path for ollama-cloud/* and internal/qwen3-4b
# Run against worktree port 8001 (or 8000 after merge)
python utils/tool_use_smoke_test.py --server http://192.168.1.93:8001
python utils/tool_use_smoke_test.py --server http://192.168.1.93:8001 --no-stream
python utils/tool_use_smoke_test.py --server http://192.168.1.93:8001 --model ollama-cloud/qwen3-coder-next

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

Gemma 4 E4B replaced by Qwen3.5-4B Q6_K_XL on 2026-05-15. See [docs/design-history.md](docs/design-history.md).
