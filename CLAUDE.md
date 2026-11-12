---
title: "SoHoAI — Project Context & Design Reference"
created_at: 20260407-000000
created_by: Florian Otel / Cline (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Haiku 5.5)
updated_at: 2026-11-12--17-03
context: >
  SoHoAI project (https://github.com/FlorianOtel/SoHoAI);
  Project instructions and design decisions for Claude Code;
  Infrastructure, architecture, API, implementation phases, RAG strategy,
  multi-tenancy (Google OAuth2, per-user NFS roots, Qdrant owner filtering)
---

# CLAUDE.md — SoHoAI Project Context

**Repository**: https://github.com/FlorianOtel/SoHoAI

## What is this project?

SoHoAI is a distributed, two-server home-office AI system. It provides a stateless
model gateway for LLM inference (local + cloud) and document RAG. The gateway keeps no
conversation memory, no chat store, and no usage telemetry; OpenCode is the chat client.
The long-term goal includes image search for family photos (Phase 4, see `docs/TODO.md`).

## Infrastructure

### Servers

| Server | IP | Role | Hardware |
|--------|------|------|----------|
| Server 1 | 192.168.1.93 | SoHoAI model gateway (FastAPI :8000), Qdrant :6333 (RAG) | 32GB RAM, AMD iGPU Radeon 680M |
| Server 2 | 192.168.1.95 | LLM inference engine: llama-swap (Qwen3.5-9B + reranker; or Qwen3-ASR + Nemotron-3-Diarization + reranker, mutually exclusive); Ollama bge-m3 :11434 (embeddings); rag-ingest timer; 03:00 sqlite-qdrant-snapshot cron | 16GB RAM, Nvidia RTX 5070 12GB |
| NAS | NFS-mounted | Persistent storage for everything | 27TB |

### Storage paths

> SQLite paths are derived from `db_base_path` in `SoHoAI-config.yaml`. Qdrant active storage is local-only (see below).

- RAG state DB: `/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db` (SQLite, NAS)
- Vector store (active): `/var/lib/qdrant/storage` (local NVMe, Server 1 only — **not NFS**)
- Vector store (snapshots/DR): `/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/` (NAS)
  - **Note**: `/mnt/nfs/__Backups/SoHoAI--databases/qdrant/` is intentionally empty — it is NOT used.
    Qdrant server uses RocksDB which is incompatible with NFS file locking. Active data must stay local.
    Snapshots (passive files) are safe on NFS and taken daily at 03:00 via cron.
- LLM model cache: `/mnt/nfs/Florian/Gin-AI/LLMs-cache/llama-server/`
- Documents for RAG: `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/documents/`

## Architecture

```
                         Anthropic API (cloud)
                         ┌───────────────────────────────┐
                ─primary─→│ Claude Sonnet 5              │
                         │ (prompt caching: system +     │
                         │  rolling prefix breakpoints)  │
                         └───────────────────────────────┘
                                    │
Server 1 (192.168.1.93)                     Server 2 (192.168.1.95)
┌──────────────────────────┐  fallback/     ┌────────────────────────────────┐
│ FastAPI gateway :8000    │──variants─────→│ llama-swap :8000               │
│  SmartRouter (LiteLLM)   │                │  ┌ resident + reranker ──────┐ │
│  /proxy/v1/* (Cline,     │                │  │ qwen3.5-9b-q4      :8011  │ │
│   OpenCode, sub-agents)  │                │  │   131072 ctx (1 slot)     │ │
│  /upstream/* (audio)     │                │  │   --reasoning off         │ │
│  /v1/rag/search          │── /v1/rerank ─→│  │ bge-reranker-v2-m3 :8012  │ │
│                          │                │  └───────────────────────────┘ │
│ Qdrant server :6333      │                │  ┌ OR asr (evicts the 9B) ───┐ │
│  active storage: NVMe    │                │  │ qwen3-asr-1.7b-q8  :8013  │ │
│  snapshots → NAS daily   │                │  │ nemotron-3-diar.   :8014  │ │
└──────────────────────────┘                │  └───────────────────────────┘ │
           │                                │                                │
           │                                │ Ollama bge-m3 :11434 (embed)   │
           │                                │ CLIP (Phase 4)                 │
           │                                └────────────────────────────────┘
           └──── both mount ──→ NAS (27TB NFS) ──────────┘
                               (RAG state DB, model cache, Qdrant snapshots)
```

## Documentation

| Doc | Contents |
|-----|---------|
| [docs/Model-routing.md](docs/Model-routing.md) | LLM routing tiers, proxy paths (Cline, OpenCode), `/upstream` audio forwarder, cost tables |
| [docs/RAG-strategy.md](docs/RAG-strategy.md) | RAG pipeline design, chunking, embeddings, Qdrant schema, ingestion service, recovery/rollback (§11) |
| [docs/design-history.md](docs/design-history.md) | Implementation milestones, benchmark results, architectural decisions |
| [docs/TODO.md](docs/TODO.md) | Deferred work: Phase 4 (image search + RL); Phase 3 dropped |
| [docs/RAG-troubleshoot.md](docs/RAG-troubleshoot.md) | Qdrant troubleshooting, recovery procedures |

## Project structure

```
SoHoAI/
├── main.py                     # FastAPI app — model gateway (/proxy/v1/*, /upstream/*, /v1/rag/search)
├── SoHoAI-config.yaml                 # All configuration (models, RAG, routing, audio models)
├── .env                        # ANTHROPIC_API_KEY (not committed)
├── schemas.py                  # Pydantic models for the RAG endpoints (IngestRequest, SearchRequest, SearchResult)
├── router.py                   # SmartRouter — LiteLLM wrapper with routing logic
├── pyproject.toml              # Python dependencies (uv-managed)
├── rag_engine/                 # RAG pipeline (Phase 2 ✅ + §8 advanced retrieval ✅)
│   ├── __init__.py             # re-exports search_rag, multi_query_search
│   ├── schema.py               # Qdrant payload field constants + derive_owner(); FIELD_SESSION_ID + FIELD_PROJECT + FIELD_SESSION_TITLE shared by claude_chat and opencode points
│   ├── collection.py           # Collection name, vector size, get_client(), ensure_collection()
│   ├── embeddings.py           # embed_text(), embed_batch(progress_cb) via Ollama on Server 2 (rag.ollama_url)
│   ├── state.py                # StateDB — ingestion queue CRUD + crash recovery; find_deleted()/purge_deleted() are the crash-safe split of the old handle_deleted() — callers must Qdrant-clean before calling purge_deleted() so a kill leaves SQLite intact for retry; paths_with_prefix() lets a caller keep a source's rows when that source could not be scanned. rag_state.db is WAL-mode SQLite on NFS: open it from Server 2 ONLY (docs/RAG-strategy.md §10.6)
│   ├── scanner.py              # NFS + claude chat + opencode scanner → populates StateDB; scan_nfs_roots() walks configured NFS roots (filters from SoHoAI-config.yaml rag.scanner) and returns {scanned, existing_paths}; scan_claude_chats() walks SoHoAI-config.yaml claude_chats.roots for .jsonl sessions and returns {scanned, existing_paths}; scan_opencode_sessions() walks GET /api/session (OpenCode 2.x, Basic auth, cursor-paginated) — returns every opencode session globally regardless of cwd-at-launch; owner is config constant (opencode.owner); returns {scanned, existing_paths: set|None} where None signals the API could not be scanned — the caller then adds the queued opencode:// rows (StateDB.paths_with_prefix) before find_deleted() so their points are preserved; callers merge existing_paths sets and call state_db.find_deleted() once before Qdrant cleanup + purge_deleted(); followlinks=True with visited_real_dirs + visited_real_files global dedup sets; exclude_dir_names uses trailing-slash path-suffix matching
│   ├── ingest.py               # docling parse + parent-child chunking + Qdrant upsert; _parse_claude_chat() extracts user/assistant text turns from .jsonl session files returning {session_id, project, session_title}; _parse_opencode_session() fetches session metadata + messages via the OpenCode 2.x API (/api/session/{id}, /api/session/{id}/message walked by cursor; rag.opencode_api_url) using synthetic opencode://{session_id} keys; errors raise (item failed, retried), returns {session_id, session_title} only (no project — opencode session.directory is cwd-at-launch metadata, not topic classification); the cwd is preserved inside the searchable header body as "Launched from:"; after parsing, file_name is overridden with session_title so renderers get a friendly label without per-type special-casing
│   ├── opencode_client.py      # OpenCode 2.x HTTP client: load_password() reads OPENCODE_SERVER_PASSWORD from opencode.password_env_file; get_json() sends Basic auth and refuses non-JSON (2.x serves HTML with 200 on legacy routes)
│   ├── search.py               # query → embed → Qdrant query_points → parent_text + provenance
│   └── multi_query.py          # §8.3: expand_query() + parallel search + MMR reranking (permanently disabled — no-go 2026-04-22)
├── utils/
│   ├── rag_sync_nfs.py         # CLI: scan NFS roots + claude chat dirs + opencode sessions → queue new/modified files; re-queue failed; skip ignored; purge Qdrant for removed files; calls scan_nfs_roots() + scan_claude_chats() + scan_opencode_sessions(), merges existing_paths (if OpenCode could not be scanned, adds the queued opencode:// rows instead, so their points are preserved), calls find_deleted() once
│   ├── rag_ingest_daemon.py    # CLI: process pending files (parse → chunk → embed → upsert)
│   ├── rag_status.py           # CLI: queue counts + Qdrant point stats; --ignored for detail; --watch LOG_FILE for live ETA monitor; --list-pending [N] to print pending paths (pipeable)
│   ├── rag_search_cli.py       # CLI: retrieval-only — embed query + Qdrant search; prints top-k hits + parent_text preview
│   ├── rag_smoke_test.py       # CLI: smoke test — GET /v1/rag/search retrieval only; --expect SUBSTR assertion; pass/fail exit code
│   ├── rag_mmr_bench.py        # CLI: MMR benchmark harness (evaluated 2026-04-22; no-go verdict — kept for reference)
│   ├── rag_rerank_bench.py     # CLI: dense vs reranked vs hybrid comparison bench; --mode dense|rerank|both|hybrid|hybrid+rerank
│   ├── rag_bench_queries.txt   # 14 verified queries used by bench harnesses
│   ├── rag_sparse_migrate.py   # CLI: one-time corpus migration to add sparse vectors (migrate / swap / status)
│   ├── rag_purge_corrupt.py    # CLI: scan+delete corrupt/zombie Qdrant vectors; read-only SQLite (mode=ro)
│   ├── qdrant_status.py        # CLI: Qdrant optimizer status, lag, segments; --watch for continuous monitoring
│   └── rag_reset.py            # CLI: reset Qdrant collection + ingestion queue
├── scripts/
│   ├── rag-ingest-run.sh           # RAG ingestion wrapper: per-user sync loop + daemon; used by systemd service
│   ├── rag-ingest.service          # systemd oneshot service (install to /etc/systemd/system/)
│   ├── rag-ingest.timer            # fires 01/07/13/19:00 local; Persistent=true for missed slots
│   ├── rag-ingest-logrotate        # logrotate config (daily, 7 days, copytruncate)
│   └── qdrant/
│       ├── qdrant-config.yaml      # Qdrant server config (storage paths, ports)
│       ├── qdrant.service          # systemd unit file (copy to /etc/systemd/system/)
└── sqlite-qdrant-snapshot.sh   # SQLite WAL checkpoint + Qdrant snapshot → NFS (cron: daily 03:00)
```

## Key API endpoints (main.py)

| Method | Path | Purpose |
|--------|------|---------|
| GET, HEAD | `/` | Liveness probe — returns `{"status": "ok"}` |
| GET | `/health` | Gateway liveness and configured model aliases |
| GET | `/v1/models/health` | Check each model endpoint |
| GET | `/proxy/v1/model/info` | LiteLLM-compatible model info for proxy clients (Cline, OpenCode) — max_input_tokens, context_window |
| GET | `/proxy/v1/models` | OpenAI model list for proxy clients (Cline, OpenCode) |
| POST | `/proxy/v1/chat/completions` | Stateless OpenAI-compatible pass-through for proxy clients (Cline, OpenCode) — no RAG, no memory |
| ANY | `/upstream/{model}/{path}` | Transparent byte forwarder to llama-swap for the audio engines only (`qwen3-asr`, `nemotron-3-diarization`); cold swap is held by llama-swap — see `docs/Model-routing.md` §10 |
| GET | `/v1/rag/search` | RAG retrieval (Qdrant only) |

The `/v1/rag/ingest/{sync,start,stop,status}` endpoints were removed 2026-11-04: ingestion runs
on Server 2 only, because the WAL-mode state DB must not be opened from two hosts. CLI
equivalents (on Server 2): `docs/RAG-strategy.md` §4.6.

## Implementation phases

| Phase | Status | Reference |
|-------|--------|-----------|
| Phase 1 — Core loop | ✅ complete | [docs/design-history.md](docs/design-history.md) |
| Phase 2 — RAG | ✅ complete (2026-04-21, 98,737 points) | [docs/RAG-strategy.md](docs/RAG-strategy.md) · [docs/design-history.md](docs/design-history.md) |
| Phase 3 — Web UI + Auth | Dropped (chat UI deprecated in favour of OpenCode) | [docs/TODO.md](docs/TODO.md) |
| Phase 4 — Image search + RL | Future — images not ingested today; self-hosted NotebookLM-style evolution of RAG planned | [docs/TODO.md](docs/TODO.md) |

## Conventions

### Code style
- Python 3.12+, type hints everywhere
- Pydantic v2 for all models (`model_config`, `field_validator`, `model_dump()`)
- Async everywhere in FastAPI operations
- SQLite operations are sync (RAG `StateDB` in `rag_engine/state.py`)
- Logging via stdlib `logging` module

### Naming
- Project name: **SoHoAI**
- Python identifiers: `sohoai_*` (underscored)
- Env var prefix: `SOHOAI_`
- Config file: `SoHoAI-config.yaml` (single file, all settings)

### Dependencies

Managed via `pyproject.toml` (uv). Key packages:
- Core: fastapi, uvicorn, litellm, pyyaml, python-dotenv, pydantic, httpx
- RAG: qdrant-client, docling, tiktoken (embeddings via Ollama HTTP — no Python dep)
- Phase 4: pillow (already included)

### Running

```bash
# Activate the project virtualenv (required on Server 1)
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate

# Server 2 — llama-swap (owns every GPU model; runs in a tmux window named "llama-swap")
# Do NOT start llama-server by hand — llama-swap launches every backend from
# llama-swap--config.yaml (incl. qwen3-asr-1.7b-q8 and nemotron-3-diarization). It listens on :8000 and proxies /v1/chat/completions and
# /v1/rerank, routing by the request's "model" field.
bash ./llama-swap--start.sh          # or: tmux send-keys -t llama-swap:0.0 'source ./llama-swap--start.sh' Enter
#
#   model                backing port   VRAM (measured)              group / policy
#   qwen3.5-9b-q4        8011            8,382 MiB @131072 ctx       resident (exclusive), ttl 0
#   bge-reranker-v2-m3   8012              518-612 MiB               reranker (persistent), ttl 0
#   qwen3-asr-1.7b-q8    8013            4,810 MiB @16384 ctx        asr (exclusive), ttl 0
#   nemotron-3-diarization 8014            638-640 MiB               asr (exclusive), ttl 0
#
# 2026-10-04: Nemotron-3-Diarization added to the asr group (AYA central speaker diarization, AYA
# v1.1-PhaseD2). Backend: NVIDIA NeMo-Speech.cpp v0.2.0 prebuilt linux-x86_64-cuda under
# /usr/src/nemo-speech/v0.2.0 (`nemo-speech serve`, not llama-server), on the PATH as
# /usr/local/bin/nemo-speech -> /usr/src/nemo-speech/current/bin/nemo-speech (symlinks only: its bundled
# libggml 0.25.1 must NOT go into /usr/local/lib, where llama.cpp's 0.25.3 shares the SONAME; the binary's
# RPATH $ORIGIN/../lib finds its own). Install/upgrade/rollback: docs/design-history.md 2026-10-04. Weights
# LLMs-cache/llama-server/Nemotron-3-Diarization.q8_0.gguf. swap: false, so Qwen3-ASR and the
# diarizer run TOGETHER: {Qwen3-ASR + diarizer + reranker} = 6,086 MiB peak, 0 errors in a 10-min
# concurrent soak. Usage (llama-swap has no diarization route, so use the passthrough):
#   POST :8000/upstream/nemotron-3-diarization/v1/audio/diarizations  (multipart "file")
#   -> {"segments":[{"start","end","speaker"}]}  (seconds, speakers 1-based, up to 8)
# NOTE: saving llama-swap--config.yaml restarts ALL models (--watch-config); the 9B cold-reloads.
#
# 2026-10-03: Qwen3-ASR added (AYA central ASR; measured in AYA's v1.1-PhaseC1 bake-off, see
# AYA docs/ASR-engine-benchmarks--PhaseC1.md). Three groups, because llama-swap v214 allows a model
# in only ONE group: the reranker has its own PERSISTENT group (never unloaded), the 9B and
# Qwen3-ASR each an EXCLUSIVE group. So the GPU holds {9B + reranker} OR {Qwen3-ASR + reranker}:
# a request for qwen3-asr-1.7b-q8 UNLOADS the 9B (chat is unavailable until the next chat request
# reloads it, ~48-64 s cold load), and the next chat request unloads Qwen3-ASR. Verified end to end
# 2026-10-03 (reranker kept the same PID through both swaps). Startup preload is still 9B + reranker.
# Qwen3-ASR usage: POST /v1/chat/completions, model "qwen3-asr-1.7b-q8", content type "input_audio"
# (base64 wav/mp3/flac, chunks <= 300 s at 16k ctx); the reply starts "language <X><asr_text>".
#
# 2026-10-03: system llama.cpp upgraded 81df3f7 (build 8855) -> a868c3e (build 11345), source
# /usr/src/llama.cpp, installed to /usr/local (shared libs, CUDA, FA, NCCL). Needed for Qwen3-ASR:
# builds before 2026-05-15 (PR #23073) return fluent nonsense with HTTP 200 for qwen3 audio.
# The new build REJECTS --no-mmap; its equivalent is --load-mode none (both backends switched).
# Rollback: ~/llama.cpp-8855-backup.tar.gz (old bins + libs), ~/llama-swap--config.yaml.pre-upgrade.
#
# 2026-09-17: the 9B became the resident default (routing.default_model) at --ctx-size 131072
# with --reasoning off. At that context it COEXISTS with the reranker and the bge-m3 embedder:
# 9,810 MiB used, 1,965 free, verified by a 10-min concurrent soak (30/30 samples, 0 errors).
# The 262144-ctx mutual exclusion recorded 2026-08-27 applies only at that larger context.
# qwen3.5-4b-q6 was REMOVED from llama-swap on 2026-09-18 — it had been unreachable since
# the 9B became default (gone from SoHoAI model_list and opencode.json). Its GGUF remains in
# LLMs-cache, so restoring it is a config-only change.
# llama-swap v214 has no eviction hook, so eager reload after a TTL expiry is not available.
#
# (!!!) A llama-swap config reload (--watch-config polls every 2 s) restarts the backends and
# returns 502/503 on /v1/rerank for ~20-40 s. Do not edit llama-swap--config.yaml mid-meeting.
# WORSE: two edits within one load window collide — the second reload fires while the first is
# still loading, llama-swap logs "swapState() Unexpected current state starting, expected
# stopped", and it can end up with a WORKING backend it no longer believes it owns. Symptom:
# curl :8012 succeeds but :8000 returns "unable to start process: upstream command exited
# prematurely but successfully", because llama-swap keeps respawning onto an already-bound port
# (ExitError exit status 1). Recovery: kill the desynced child; llama-swap respawns it cleanly.
# Batch edits to this file into ONE write, then wait for the health checks before touching it.

# Server 2 — health checks
curl -sS http://192.168.1.95:8000/v1/models                       # what llama-swap serves
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # what is on the GPU
curl -sS http://192.168.1.95:8000/v1/rerank -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3","query":"test","documents":["foo"]}'    # reranker alive?

# Server 1 — Qdrant vector store (enabled at boot; restart after reboot)
sudo systemctl start qdrant          # starts /usr/local/bin/qdrant on port 6333
# Active storage: /var/lib/qdrant/storage (local NVMe — NOT NFS)
# Snapshots (DR):  /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/
# Manual snapshot: bash scripts/sqlite-qdrant-snapshot.sh   (auto-runs daily 03:00 via cron)
# Restore from snapshot: PUT http://192.168.1.93:6333/collections/documents/snapshots/recover

# Server 1 — orchestrator
cd ~/Gin-AI/projects/SoHoAI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-include "SoHoAI-config.yaml"

# Cline VSCode plugin and OpenCode — see docs/Model-routing.md
# Quick reference:
#   Cline (LiteLLM provider):  Base URL http://192.168.1.93:8000/proxy/v1  API key sohoai-local
#   OpenCode (provider baseURL): http://192.168.1.93:8000/proxy/v1

# RAG ingestion — automated service on SERVER 2 (live since 2026-05-05; see RAG-strategy.md §10)
# rag_state.db (WAL-mode SQLite on NFS) is opened from Server 2 ONLY: run every command below
# that touches it (rag_sync_nfs, daemon, rag_status, rag_reset, rag_purge_corrupt) on Server 2.
# OpenCode sessions need opencode.password_env_file (Server 2: ~/.config/opencode/opencode-server.env).
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
python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS-Certification"  # retrieval only (GET /v1/rag/search); pass/fail exit
python utils/rag_rerank_bench.py --user florian                               # dense-only vs reranked top-5 comparison

# RAG sparse migration — one-time, required to enable hybrid search (see RAG-strategy.md §14)
python utils/rag_sparse_migrate.py status                       # check migration state
python utils/rag_sparse_migrate.py migrate                      # phase 1: migrate data (live)
python utils/rag_sparse_migrate.py swap --confirm               # phase 2: swap (stop uvicorn first)
```

### Testing a change

After modifying any orchestrator code:
1. The FastAPI app reloads automatically if run with `--reload`
2. Test via curl: `curl -X POST http://192.168.1.93:8000/proxy/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"local/qwen3.5-9b-q4","messages":[{"role":"user","content":"hello"}]}'`
3. Check health: `curl http://192.168.1.93:8000/health`
4. RAG regression (retrieval only): `python utils/rag_smoke_test.py --query "..." --user florian --expect "<known-source-substring>"` (non-zero exit on failure)

### Performance benchmarks

Gemma 4 E4B replaced by Qwen3.5-4B Q6_K_XL on 2026-05-15. See [docs/design-history.md](docs/design-history.md).
