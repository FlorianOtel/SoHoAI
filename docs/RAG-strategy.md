---
title: "SoHoAI — RAG Strategy"
created_at: 2026-03-30--00-00
created_by: Florian Otel
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-12--19-47
context: >
  SoHoAI project (https://github.com/FlorianOtel/SoHoAI);
  RAG pipeline design: embedding model, vector DB, chunking strategy,
  NFS corpus survey, Qdrant payload schema, multi-tenancy (Google OAuth2),
  rag_engine/ package layout, fail-safe ingestion (crash recovery, retry,
  delete-before-insert idempotency), incremental sync (additions + deletions),
  Phase 2 implementation plan, ingestion runbook, advanced retrieval and
  generation patterns (rag_mode, system-prompt tool-use, multi-query + MMR,
  contextual retrieval). Updated 2026-04-22: §8.3 multi-query+MMR evaluated
  — no-go verdict, permanently disabled; §8.4 contextual retrieval still frozen.
  Updated 2026-05-05: §10 RAG Ingestion Service — systemd timer + NFS lock +
  multi-user sync wrapper. Updated 2026-05-12: §4.3 incremental sync — Qdrant
  deletions use wait=False (fire-and-forget); get_client() timeout parameter.
---
---

# RAG Strategy — SoHoAI

---

## RAG client modes — server-managed vs. external LLM

Two distinct integration patterns exist for RAG in SoHoAI, serving different callers.

### Server-managed RAG (`POST /v1/chat/completions` with `rag_mode`)

Designed for clients like `cli_chat.py` and Open WebUI where SoHoAI's orchestrator
manages the full conversation loop. The server's two-iteration tool-use loop
(`rag_engine/tool_use.py`) injects a `<tool_call>search_documents</tool_call>` sentinel
into the system prompt; the LLM decides at inference time whether retrieval is needed.

The bge-m3 cosine score distribution for this corpus shows a narrow margin between
signal (0.51–0.55) and noise (0.48–0.51). This margin justified the LLM-decides approach
over always-inject: unconditional top-k injection floods the context with marginally
relevant results on factual queries where no retrieval is needed at all. With tool-use,
the LLM skips the search step entirely when the question can be answered from conversation
history, and fires a targeted query only when genuinely needed. `rag_mode` values:
- `off` — no retrieval, no tool spec in system prompt (default)
- `on` — tool spec injected; LLM decides whether to call `search_documents`
- `only` — tool spec injected; system prompt instructs LLM to answer only from retrieved docs

### External LLM client RAG (`GET /v1/rag/search`)

For Claude Code and similar agents that manage their own context and reasoning loop.
Claude Code IS the LLM deciding when to search; running SoHoAI's server-side tool-use
loop would be redundant — it would fire a second LLM inference (Gemma 4 or Sonnet 4.6)
purely to emit a `<tool_call>` sentinel and then execute the same `search_rag()` call the
client could have made directly.

`GET /v1/rag/search` returns raw document hits as JSON with no LLM invocation on the
server side. Query parameters: `q` (required), `user` (owner filter), `top_k` (1–20,
default 5), `file_types` (list, e.g. `["pdf","md"]`). Each result includes `content`
(parent chunk text), `source_path`, `score`, `file_name`, `file_type`, and `session_title`
(for claude_chat results).

Invoked interactively via the `/user:rag` slash command (`~/.claude/commands/rag.md`):
- `/user:rag search <query>` — one-shot retrieval, displays ranked hits
- `/user:rag on` — automatic pre-search before each Claude Code answer
- `/user:rag only` — answer exclusively from retrieved documents
- `/user:rag off` — disable automatic search

---

## 1. NFS corpus overview

**NAS**: 27TB, NFS-mounted on both servers.

**Per-user NFS roots** (each user has a top-level directory):

| User | NFS root | Content |
|------|----------|---------|
| Florian | `/mnt/nfs/Florian` | Work docs, certifications, projects, Gin-AI |
| Eva | `/mnt/nfs/Eva` | Personal docs |
| Annika | `/mnt/nfs/Annika` | Personal docs |
| Laura | `/mnt/nfs/Laura` | Personal docs |
| (shared) | `/mnt/nfs/La-Familia` | Family-shared content (photos, videos, docs) |

**Total files surveyed (2026-04-08, `/mnt/nfs/Florian` only)**: 139,195 files, 151GB.
~120K of those are Python virtualenv internals and are excluded from RAG.
Other user directories not yet surveyed — ingestion will scan all configured roots.

### 1.1 RAG-relevant files (~2,800 documents + ~10K media)

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
| MSG | 2 | Email — negligible, de-prioritized |

**Content categories**:
- Family related: photos and videos in various formats (likely in per-user dirs + La-Familia)
- Work related: each family member has their own top-level NFS directory
  - For florian (UID 555): multiple directories from working at multiple companies,
    backup files per employer (`Company--login-name`), and a separate
    `certifications--training` directory (may overlap with employer backups)
  - Other users (Eva, Annika, Laura): not yet surveyed — to be scanned at ingestion time

### 1.2 Exclusion filters (updated 2026-04-23)

Exclusion filters are configured in `config.yaml` under `rag.scanner` (required — no
hardcoded fallback). `rag_engine/scanner.py` validates all four keys at startup and
raises `ValueError` immediately if any are missing.

```yaml
rag:
  scanner:
    include_extensions:      # file extension whitelist
      - .pdf
      - .pptx
      - .docx
      - .md
      - .txt
      - .csv
      - .yaml
      - .yml
      - .ipynb

    # All entries MUST have a trailing slash.
    # Single-component entries (e.g. "Library/") match any directory with that exact
    # name at any depth. Multi-component entries (e.g. "Microsoft--flotel/Documents/")
    # match only that exact path-segment sequence.
    # Matching uses endswith("/" + pattern) so "Library/" won't match "PublicLibrary/".
    exclude_dir_names:
      - ".Gin-AI-python-3.12/"  # Python virtualenv
      - "__pycache__/"
      - ".git/"
      - "node_modules/"
      - ".claude/"
      - ".vscode/"
      - ".venv/"
      - "Library/"              # macOS Application Support / Preferences
      - "Applications/"         # macOS app bundles
      - "LLMs-cache/"           # HuggingFace model cache (READMEs/licenses, not RAG content)
      - "Downloads/"            # garbage
      - "Microsoft--flotel/Documents/"  # IRM/DRM-encrypted old PPTs — unrecoverable

    exclude_dir_suffixes:    # directory name suffixes (matched with str.endswith on bare name)
      - .dist-info           # pip package metadata

    exclude_file_patterns:   # filename substrings — any match → file skipped
      - "@synoeastream"      # Synology NAS streaming metadata
      - "._"                 # macOS AppleDouble resource fork sidecars (e.g. ._foo.pptx) —
                             # auto-created by macOS on non-HFS+ (NFS/Synology); binary metadata,
                             # same extension as the real file but not a ZIP — fails all parsers
      - "~$"                 # Microsoft Office lock/temp files (e.g. ~$foo.pptx) —
                             # created while a document is open; tiny binary stubs, not real docs
```

**`exclude_dir_names` matching logic (updated 2026-04-23):** `_is_excluded_dir()` in
`scanner.py` now receives both the parent `dirpath` and the child `dirname`. It forms the
full child path (`os.path.join(dirpath, dirname)`), strips the trailing slash from each
pattern, and checks `full_child.endswith("/" + pattern)`. The leading `/` in the test
enforces an exact path-boundary match — `"Library/"` matches `.../Library` but not
`.../PublicLibrary`. Multi-component patterns like `"Microsoft--flotel/Documents/"` work
naturally with the same logic.

`exclude_dir_suffixes` continues to match against the bare `dirname` only (not the full
path) using `dirname.endswith(suffix)` — kept separate because `.dist-info` is a name
suffix, not a path pattern.

To add or remove an exclusion, edit `config.yaml` and re-run `rag_sync_nfs.py`.
No code change is required. When a directory is newly excluded, `rag_sync_nfs.py`
purges its files from the queue and Qdrant automatically (via `handle_deleted`).

**Claude Code sessions — `.claude/` exclusion + `claude_chats` two-path design (updated 2026-05-04)**

The `.claude/` directory is excluded from the generic NFS scanner to prevent ingestion of Claude Code's internal state (settings, caches, IDE configs, tool metadata) and the `chats/` symlink tree. However, the `.jsonl` session transcripts in `~/.claude/projects/` have high RAG value and must be ingested. A **dedicated scanner function** `scan_claude_chats()` (in `rag_engine/scanner.py`) re-enters the specific subdirectory `~/.claude/projects/` via a separate code path that:
1. Bypasses all generic exclusion rules
2. Scans *only* for `.jsonl` files
3. Uses a dedicated parser `_parse_claude_chat()` (in `rag_engine/ingest.py`) that extracts structured user/assistant text turns from the JSONL format

A raw UTF-8 read of `.jsonl` produces unreadable JSON blobs; the dedicated parser is mandatory for RAG-quality content. Additionally, `_build_title_map()` reads `~/.claude/chats/` as a read-only side channel to derive human-readable session titles — it is never ingested, only consulted during ingest time for metadata enrichment. This two-path architecture solves three constraints: (1) avoid duplicate ingestion via `.chats/` symlinks, (2) ingest high-value sessions at all, (3) process them correctly with structured parsing. Full rationale and code locations documented in `docs/design-history.md` (2026-05-04--12-46).

### 1.3 Symlink handling (updated 2026-04-21)

`rag_engine/scanner.py` uses `os.walk(..., followlinks=True)` so that symlinked
directories are traversed. Two global dedup sets prevent the same content from being
indexed twice regardless of how many symlink aliases exist:

- **`visited_real_dirs`** — tracks resolved real paths of every directory entered.
  If a directory's real path has already been visited (e.g. via another symlink or root),
  `dirnames` is cleared and the subtree is skipped. Handles circular symlinks and the same
  directory reachable via multiple NFS roots.
- **`visited_real_files`** — tracks resolved real paths of every eligible file. If a
  file's real path was already seen (e.g. a file symlink alias), it is skipped before
  `discover_or_update()` is called. The **first** path encountered is the one stored in
  SQLite and Qdrant; traversal order is alphabetical within each directory.

**Known symlink topology of `/mnt/nfs/Florian` (surveyed 2026-04-21):**

| Symlink path | Real path | Impact |
|---|---|---|
| `Florian/Dropbox` | `/mnt/nfs/__Backups/Dropbox` | External — no duplicate |
| `Florian/AWS--fotel` | `/mnt/nfs/__Backups/AWS--MacBookPro/fotel` | External — macOS backup |
| `Florian/Gitlab--fotel` | `/mnt/nfs/__Backups/GitLab--MacBookPro-M3/fotel` | External — macOS backup |
| `Florian/Microsoft--flotel` | `/mnt/nfs/__Backups/Microsoft-flotel--20190617--last/fotel` | External — macOS backup |
| `Florian/Gin-AI/LLMs-cache` | `/mnt/nfs/Temp/Gin-AI--cache/LLMs-cache` | External — excluded by `LLMs-cache` dir name |
| `Florian/Gin-AI/certifications--training` | `Florian/certifications--training/__private/Gin-AI` | **Internal dir alias** — `visited_real_dirs` prevents double-walk ✅ |
| `Florian/Gin-AI/chats/*.md` (14 files) | Various real paths under `certifications--training/` and `Gin-AI/tools/` | **Internal file aliases** — `visited_real_files` prevents double-ingest ✅ |
| Multiple `.venv` symlinks | `Florian/Gin-AI/.Gin-AI-python-3.12` | `exclude_dir_names` blocks descent ✅ |

---

## 2. Requirements

### 2.1 Documents
- Search course and training material; return the source reference (full NFS path)
- Parse multiple file types: `.pdf`, `.docx`, `.pptx`, `.txt`, `.ipynb`, `.md`, `.yaml`
- Outlook/PST archives: **de-prioritized** — only 2 `.msg` files found on NFS;
  `libpst` not needed at this stage

### 2.2 Images (Phase 4)
- Identify time and location from EXIF metadata
- Identify persons (RLHF-trainable)
- Text-to-image similarity search

### 2.3 Videos (Phase 4)
- Find video by text description

### 2.4 Standalone RAG DB updates
- Ability to ingest documents independently of any ongoing LLM conversation
- API endpoint: `POST /v1/rag/ingest`

---

## 3. Architecture decisions

### 3.1 Embedding model — bge-m3 via Ollama (updated 2026-04-22)

**Model**: `bge-m3` — 1024 dimensions, 570M params, ~1.2GB on disk.
Top MTEB retrieval scores; 8192-token context window.

**API**: `POST {ollama_url}/api/embeddings` — no native batch endpoint.
`embed_batch()` runs up to N concurrent requests via asyncio semaphore, where N is the
`--batch` argument of `rag_ingest_daemon.py`. Lower values reduce Ollama queue depth and
prevent `httpx.ReadTimeout` under heavy load; higher values improve throughput when Ollama
has headroom (especially GPU).
`embed_batch()` accepts an optional `progress_cb(done, total)` callback fired every
`_PROGRESS_INTERVAL` (50) completions and on the final chunk. `ingest.py` wires this
to `logger.info("Embedding progress: %d/%d  %s", done, total, file_name)` so that
`rag_status.py --watch` can parse the timestamps and compute a real-time chunk rate
and ETA for the current file.

**Config key** — determines which server Ollama runs on:
```yaml
# CPU embed — Server 1 local (all-local mode)
ollama_url: http://192.168.1.93:11434/api/embeddings

# GPU embed — Server 2 RTX 5070 (split mode; llama-server must not be running or have headroom)
ollama_url: http://192.168.1.95:11434/api/embeddings
```

**Two operating modes** — see §5.4 for daemon parameters:

| Mode | Ollama server | Embed latency | Bottleneck | Daemon flags |
|------|--------------|---------------|------------|--------------|
| All-local (CPU) | Server 1 (193) | ~650ms/chunk | Embedding | `--workers 1 --batch 5` |
| Split (GPU) | Server 2 (195) | ~10ms/chunk | docling parse | `--workers 3 --batch 20` |

**Query latency (CPU mode)**: ~650ms/chunk unloaded; up to ~28–30s under full ingestion load.
Ollama serializes model computation — when the ingest daemon is running, search queries
queue behind the embedding batch. The `embed_text()` HTTP timeout is **120s** to survive
this wait. Do not reduce below 60s while the ingest daemon may be running.
To reduce search latency during ingestion in CPU mode, lower `--batch` (e.g. `--batch 2`).

**Query latency (GPU mode)**: ~10–20ms/chunk. Search queries are not meaningfully blocked
by ingestion since Ollama on Server 2 is not used for search (Server 1 search path uses
Server 1's Ollama). Ensure `ollama_url` in `config.yaml` is set consistently for both
the daemon and the search path — mixing servers will produce wrong cosine distances.

**Why Server 2 VRAM permits bge-m3 when llama-server is idle:**
When llama-server is stopped on Server 2, bge-m3 (~1.2GB) fits easily in the RTX 5070's
12GB. If llama-server is running, it occupies ~9.97GB leaving only ~2.2GB — bge-m3 fits
but inference latency becomes unpredictable under VRAM pressure. Do not run bulk ingestion
with `--workers 3` on GPU while llama-server handles active conversations.
KV cache grows dynamically during active turns (up to ~4.1 GB for a full 53,248-token slot),
so free headroom effectively drops to zero under load. No embedding model fits alongside
llama-server on Server 2. A two-phase approach (bulk ingest on Server 2, incremental on Server 1)
is also unworkable: embedding model must be consistent between ingestion and real-time query
embedding — mixing models corrupts Qdrant search results.

**Why not `mxbai-embed-large`?**
`mxbai-embed-large` uses a BERT WordPiece tokenizer with a hard 512-token limit. Chunk sizes are
measured in tiktoken (GPT BPE), which undercounts relative to BERT — especially for dense
technical content (identifiers, URLs, code). A 250-tiktoken child chunk can easily exceed 512
BERT tokens, causing Ollama to return 500 with `{"error":"the input length exceeds the context
length"}`. Every retry on the same chunk fails identically. bge-m3's 8192-token context
eliminates this class of error entirely.

**Why not `qwen3-embedding:8b`?**
Best MTEB scores and 40K context, but not viable for this setup:
- Server 2 VRAM: llama-server (11.6GB) + qwen3 Q4 (5GB) = 16.6GB > 12GB RTX 5070
- Server 1 CPU inference at 8B params: ~6–10s per embedding vs ~650ms for bge-m3 —
  adds 6–10s to every RAG-enabled chat query, unacceptable for interactive use
- Would require stopping llama-server for bulk ingestion (hours of system downtime)

bge-m3 is the practical optimum: near-top MTEB quality, 8192-token context, fits all workloads.

---

### 3.2 Vector database — Qdrant (confirmed 2026-03-30, updated 2026-04-17)

**Deployment**: Qdrant v1.17.1 native binary, systemd service on Server 1, port 6333.
**Access**: REST API via `qdrant-client` Python library (`QdrantClient(url="http://192.168.1.93:6333")`).
No LangChain wrapper.

Config key: `rag.qdrant_url: "http://192.168.1.93:6333"` in `config.yaml`.

**Why Qdrant?**
Qdrant's payload system stores arbitrary JSON per vector point and returns it with every
search result — this is exactly how provenance (source file path references) works.

**Alternatives considered and rejected**:

| Alternative | Reason rejected |
|-------------|----------------|
| ChromaDB | Weaker payload filtering; can't efficiently filter by tag/directory at scale. Critical process-level singleton bug via LangChain: the default collection name `'langchain'` is shared across all `Chroma.from_documents()` calls in the same Python process — subsequent calls silently append to the existing collection, duplicating documents and corrupting ranking. `EphemeralClient` does NOT fix this. |
| LanceDB | Good columnar metadata but less mature; no existing integration |
| pgvector | Requires Postgres; project uses SQLite |

**Qdrant has none of ChromaDB's issues**: collections are explicitly named, isolated on disk,
no shared process state. SoHoAI uses `qdrant-client` directly — no LangChain default-name trap.

#### Collections — one per modality

Different embedding models per modality require separate collections:

| Collection | Embedding model | Dimensions | Phase |
|------------|----------------|------------|-------|
| `documents` | bge-m3 via Ollama | 1024 | 2 |
| `images` | CLIP (openai/clip-vit-base-patch32) | 512 | 4 |
| `videos` | CLIP or frame-level embeddings | 512 | 4 |

#### Storage size estimates

- `documents` collection: ~30K–50K chunks × 1024-dim float32 ≈ ~200MB vectors + ~500–800MB HNSW index
- `images` collection (Phase 4): ~9,600 CLIP vectors × 512-dim ≈ ~19MB
- HNSW index fits entirely in Server 1 RAM (32GB) after first load

---

### 3.2.1 Qdrant persistence, NFS incompatibility, and snapshot DR

#### Why Qdrant active storage cannot be on NFS

Qdrant's server binary uses **RocksDB** as its storage engine. RocksDB relies on POSIX
`fcntl` advisory locks (`F_SETLK`/`F_SETLKW`) for segment file management and compaction
coordination. NFS does not reliably honor these locks:

- NFS lock requests go through a separate `lockd`/`rpc.statd` daemon pair. Under network
  partitions, lockd may grant the same advisory lock to two processes simultaneously.
- Linux NFS clients do not implement mandatory locking — advisory locks are best-effort.
- The result is silent data corruption: two processes can write to the same RocksDB segment
  concurrently, producing an unrecoverable database.

Qdrant v1.x detects this at startup: it checks the filesystem type of the configured
`storage_path` and refuses to start if it is `nfs` or `nfs4`:

```
ERROR qdrant: Filesystem check failed for storage path ...
Details: NFS may cause data corruption due to inconsistent file locking
```

**Active storage must be on local disk.** Current path: `/var/lib/qdrant/storage` (NVMe on Server 1).

#### Why the previous Python embedded client did not warn

Before 2026-04-17, code used `QdrantClient(path=...)` — the embedded/local mode of the
Python `qdrant-client` library. This mode uses **SQLite** (not RocksDB) as its storage
backend (observable as `collection/documents/storage.sqlite` on disk). SQLite implements its
own coarse-grained WAL locking that tolerates NFS for single-writer scenarios. The Python
client also enforces single-process access at the Python level, so there is never concurrent
RocksDB-style contention.

The consequence: `QdrantClient(path=...)` on NFS worked without error, but it prevents
concurrent access — any second process trying to open the same path gets a Python-level
`already accessed by another instance` exception. This was the original cause of the
`watch -n 30 python utils/rag_status.py` failure.

Switching to server mode (HTTP API) eliminates the concurrency problem and moves the
NFS-incompatibility issue from silent risk to an explicit startup refusal.

**Data format note:** The SQLite-based local-mode storage (`collection/`) is a completely
different format from the RocksDB-based server storage (`collections/`). Existing data
ingested under the Python embedded client cannot be read by the server binary — a full
re-ingest is required when switching modes. The NFS directory
`/mnt/nfs/__Backups/SoHoAI--databases/qdrant/` was cleared of old SQLite data after
the migration and is intentionally left empty.

#### Storage architecture

```
Server 1 (192.168.1.93)
│
├── /var/lib/qdrant/storage/        ← active RocksDB store (local NVMe)
│     Active Qdrant data: vectors, HNSW index, payloads.
│     RocksDB requires local POSIX locking — cannot be NFS.
│
└── /var/lib/qdrant/snapshots/      ← (unused; Qdrant creates this internally)

NAS (NFS-mounted)
│
├── /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/
│     ├── documents/
│     │     ├── documents-<id>-<timestamp>.snapshot      ← snapshot archive
│     │     ├── documents-<id>-<timestamp>.snapshot.checksum
│     │     └── ... (up to 3 kept)
│     └── (future: images/, videos/ as Phase 4 collections are added)
│
└── /mnt/nfs/__Backups/SoHoAI--databases/qdrant/
      (intentionally empty — not used for anything)
```

Snapshots are **passive archive files** (tar-like archives). Qdrant writes them atomically
and includes a SHA256 `.checksum` file alongside each one. No RocksDB locking is involved
during snapshot reads or writes — the NFS risk does not apply.

#### Snapshot mechanism

Snapshots are created via the Qdrant REST API:

```
POST http://192.168.1.93:6333/collections/documents/snapshots
```

Qdrant freezes writes to the collection, serialises the current segment state into a
self-contained archive, and writes it to `snapshots_path/{collection_name}/`. The response
includes the snapshot name and creation timestamp. The entire process is online — no service
downtime required.

The snapshot directory is configured in `scripts/qdrant/qdrant-config.yaml`:

```yaml
storage:
  storage_path: /var/lib/qdrant/storage
  snapshots_path: /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots
```

#### Snapshot frequency and retention

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Schedule | Daily at **03:00** (cron) | Low-activity window; ingestion daemon typically idle |
| Retention | Last **3** snapshots kept | ~3 days of rollback; older ones deleted via API |
| Script | `scripts/qdrant/qdrant-snapshot.sh` | Calls REST API, then prunes via API |
| Log | `/var/log/qdrant-snapshot.log` | Check with `tail -f` or `grep -i error` |

**Maximum data loss exposure**: up to 24 hours of ingested documents if the local NVMe
fails between snapshots. During an active ingestion run (~9 hours for the full NFS corpus),
taking a manual mid-run snapshot is advisable:

```bash
bash scripts/qdrant/qdrant-snapshot.sh
```

Once bulk ingestion is complete, new documents enter only via incremental re-scans
(`rag_sync_nfs.py` detecting mtime changes), so daily snapshots provide adequate protection.

#### Snapshot script logic (`scripts/qdrant/qdrant-snapshot.sh`)

```
1. POST /collections/documents/snapshots  → Qdrant creates archive on NFS
2. GET  /collections/documents/snapshots  → list all snapshots, sort by creation_time
3. DELETE oldest snapshots beyond KEEP=3  → via DELETE /collections/.../snapshots/{name}
```

All API calls use plain `curl` + `python3` (stdlib only, no extra dependencies).
Run manually: `bash scripts/qdrant/qdrant-snapshot.sh [--keep N]`

> **`creation_time: null` quirk (2026-05-06):** Qdrant v1.17.1 returns `"creation_time": null`
> for all snapshots. The sort key uses `s.get("creation_time") or ""` (not `.get(..., "")`)
> so that a present-but-null value is coerced to `""` rather than passed through as `None`.
> Sorting an all-`""` list is a stable no-op; snapshot names contain a `YYYY-MM-DD-HH-MM-SS`
> suffix so filesystem `ls -lt` ordering is always authoritative for human inspection.

#### Recovery procedure

If Server 1 is rebuilt or `/var/lib/qdrant/storage` is lost:

```bash
# 1. Install Qdrant and start the service (empty storage)
sudo systemctl start qdrant

# 2. Identify the latest snapshot on NFS
ls -lt /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/

# 3. Restore via the API (Qdrant reads the file path directly)
curl -X PUT "http://192.168.1.93:6333/collections/documents/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/<snapshot-name>.snapshot"}'

# 4. Verify
curl -s http://192.168.1.93:6333/collections/documents | python3 -m json.tool
python utils/rag_status.py
```

After recovery, only documents ingested after the last snapshot are missing. Run
`rag_sync_nfs.py` (scanner detects mtime-changed files → resets to `pending`) followed by
`rag_ingest_daemon.py` to re-ingest the gap.

---

### 3.3 Document parsing — docling + dedicated parsers (updated 2026-04-19)

**Library**: `docling` — replaces `unstructured`.

Supported file types via docling: PDF, PPTX, DOCX.
Text formats (TXT, MD, YAML, CSV): direct UTF-8 read.
XLSX: not supported by docling — treated as flat text.

**Jupyter notebooks (`.ipynb`) — dedicated parser, NOT docling.**
docling does not support `.ipynb`. When given an ipynb file it logs an ERROR and the fallback
was a raw UTF-8 read of the notebook file — which reads the notebook's raw JSON structure
(`{"cell_type": "code", "source": [...]}`) as plain text. This is garbage for RAG: chunks
contain JSON keys and array syntax rather than the actual content.

Fix: `_parse_ipynb()` in `ingest.py` parses the notebook JSON directly:
- Markdown cells → extracted as prose
- Code cells → extracted as fenced ` ```python ``` ` blocks
- Cell outputs ignored (only source matters for retrieval)
- Empty cells skipped
- Falls back to raw read only if the file is not valid JSON

`ipynb` removed from `_DOCLING_TYPES`; the ERROR log from docling is eliminated entirely.

**PPTX — python-pptx fallback when docling format detection fails (2026-04-19).**
docling can fail to detect `.pptx` format (logs `format None`) for files where the internal
ZIP/OpenXML structure doesn't match what docling's format detector expects — e.g. older `.ppt`
binaries renamed to `.pptx`, or files with non-standard headers. The prior fallback was a raw
UTF-8 read of the binary container, which produced binary garbage for RAG (6053 garbage chunks
observed for one file).

Fix: `_parse_pptx()` in `ingest.py` is now the secondary fallback for PPTX:
- Reads first 4 bytes to detect OLE2 magic (`\xD0\xCF\x11\xE0`)
- **OLE2 path**: runs `libreoffice --headless --convert-to pptx` into a temp dir, then parses
  the converted file with `python-pptx`; temp dir always cleaned up in `finally`
- **OpenXML path** (genuine .pptx that docling rejected): parsed directly with `python-pptx`
- Both paths: iterate slides/shapes, collect text frame content, prefix with `Slide N` headers
- If `_parse_pptx()` raises, the exception propagates — **no raw-read fallback for binary PPTX**.
  The file is marked `failed` in StateDB, not filled with binary garbage chunks.

**IRM/DRM-encrypted .ppt files (2 found: Azure PaaS 20180510 + 20180511):**
Root cause: both are Composite Document File V2 (OLE2) with `EncryptedPackage` stream and
`DRMEncryptedDataSpace`/`DRMEncryptedTransform` — Microsoft IRM enterprise encryption.
Content is inaccessible without the original Azure AD credentials. LibreOffice cannot convert them.
These 2 files will exhaust retries and remain permanently `failed` in StateDB — correct behaviour.

29 completed PPTX files force-reset to `failed` on 2026-04-19 for clean re-ingestion.

---

### 3.4 Chunking — parent-child strategy (confirmed 2026-04-10)

The dataset is dominated by dense technical content (PDFs, Jupyter notebooks, long Markdown docs)
where flat single-size chunks force a bad trade-off: large chunks dilute embedding precision; small
chunks leave the LLM with insufficient context.

**Parent-child splitting resolves this**:
- **Child chunks** (~200–300 tokens, 20-token overlap): precise, focused embeddings →
  better cosine similarity scores; stored in Qdrant as the search index
- **Parent chunks** (~800–1200 tokens, 100-token overlap): full surrounding context →
  richer LLM answers; stored only as `parent_text` in the Qdrant payload of each child point

**Flat chunking** (512-token, no parent): used for PPTX slides and short TXT/YAML/config files
(already compact; parent-child overhead not worth it).

**Benefit by file type**:

| File type | Benefit from parent-child | Reason |
|-----------|--------------------------|--------|
| PDFs (certifications, work docs) | **High** | 512-token slices of 100-page docs lose all context |
| Jupyter notebooks | **High** | Code cell alone is meaningless without surrounding markdown |
| Long Markdown | **Medium** | Benefits multi-section docs; short files not affected |
| PPTX / short TXT / YAML / config | **Low** | Already compact — use flat chunking |

`ingest_file()` selects strategy by file type; chunk sizes are not uniform.

#### Docstore — parent text in Qdrant payload

The docstore (parent text storage) requires **exact ID lookup only — never similarity search**.
A separate vector DB is not needed.

**Decision**: store `parent_text` directly in the Qdrant payload of each child point.

| Option | Mechanism | Trade-off |
|--------|-----------|-----------|
| **Qdrant payload** (chosen) | `parent_text` field on every child point | Trivial duplication across sibling children; single query, no join |
| SQLite table | `rag_parents(id, text, source_path)` + foreign key in payload | No duplication; two-step retrieval (Qdrant → SQLite) |
| Redis | Key-value lookup | TTL risk; Redis already used for conversation state |

Storage overhead: 50K child chunks × ~2KB average parent text ≈ ~100MB extra on NAS —
negligible at this scale. Gemma 4 E4B's 110,024-token per-slot context window handles 800–1200 token
parents with no pressure.

On retrieval, `chunk["parent_text"]` (not `chunk["text"]`) is injected into the LLM prompt.

---

### 3.5 Qdrant payload schema for documents (provenance)

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

`owner` is derived automatically from the NFS path at ingestion time:
`/mnt/nfs/Florian/... → owner="florian"`, `/mnt/nfs/La-Familia/... → owner="la-familia"`.
At search time, a Qdrant filter restricts results to the authenticated user's own documents
plus shared content: `MatchAny(any=[user_owner, "la-familia"])` on the `owner` field.

`source_path` is the full NFS path (not just filename) — directly usable without a second lookup.
The MCP server at port 3001 already exposes these paths.

---

### 3.6 Multi-tenancy & authentication (confirmed 2026-04-16)

SoHoAI serves a family of users, each with private NFS storage and a shared directory.
Authentication is via **Google OAuth2 (OIDC)** — all users are members of the same Google
Family Group but have separate Google accounts.

#### Identity flow

1. **Authentication**: Google OIDC → JWT with `sub` (stable numeric ID) + `email`
2. **User mapping**: `email` → `owner` string via config (e.g. `florian@example.com → "florian"`)
3. **Authorization**: at every RAG search and chat operation, the `owner` value determines what
   the user can access

#### User → NFS root mapping (in `config.yaml`)

```yaml
users:
  florian@example.com:
    owner: "florian"
    nfs_roots: ["/mnt/nfs/Florian"]
  eva@example.com:
    owner: "eva"
    nfs_roots: ["/mnt/nfs/Eva"]
  annika@example.com:
    owner: "annika"
    nfs_roots: ["/mnt/nfs/Annika"]
  laura@example.com:
    owner: "laura"
    nfs_roots: ["/mnt/nfs/Laura"]
shared:
  owner: "la-familia"
  nfs_roots: ["/mnt/nfs/La-Familia"]
```

#### Access control rules

| Resource | Visibility rule |
|----------|----------------|
| Qdrant search results | `owner IN [user_owner, "la-familia"]` — user sees own + shared |
| Chat history (SQLite) | `user_id` column on `chats` table — user sees only own chats |
| MCP file access | Per-user ALLOWED_ROOTS derived from config — user sees own NFS root + La-Familia |
| Redis conversation cache | Keyed by `chat_id` (UUID, unguessable); `chat_id → user_id` enforced at API layer |

#### Ingestion: `owner` derivation

The ingestion worker derives `owner` from the NFS path prefix at ingest time:

```python
def derive_owner(file_path: str, user_config: dict) -> str:
    """Map NFS path → owner. Checks per-user roots, then shared root."""
    for email, cfg in user_config["users"].items():
        for root in cfg["nfs_roots"]:
            if file_path.startswith(root):
                return cfg["owner"]
    for root in user_config["shared"]["nfs_roots"]:
        if file_path.startswith(root):
            return user_config["shared"]["owner"]
    raise ValueError(f"File {file_path} not under any configured NFS root")
```

#### Search: Qdrant filter

```python
from qdrant_client.models import FieldCondition, Filter, MatchAny

def user_filter(user_owner: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="owner", match=MatchAny(any=[user_owner, "la-familia"]))
    ])
```

#### Offline resilience

Google OAuth requires internet. For a home lab this means:
- **Session tokens cached locally** with a multi-hour TTL — survives brief ISP outages
- **CLI fallback** (`cli_chat.py`): local API key or token file for LAN-only access
  when Google is unreachable (Phase 3 concern, not blocking for Phase 2)

---

## 4. Phase 2 implementation plan

### 4.1 Status

| Step | Status |
| ------ | ------ |
| Choose docling over unstructured | ✅ done |
| Replace sentence-transformers with Ollama in rag.py | ✅ done (2026-04-08) |
| Fix config.yaml RAG section (mxbai-embed-large, ollama_url) | ✅ done (2026-04-16) |
| Fix stale collection defaults in rag.py and schemas.py (`"default"` → `"documents"`) | ✅ done (2026-04-16) |
| Add `owner` field to Qdrant payload schema and search filter | ✅ designed (2026-04-16) |
| Add `user_id` to ChatRequest, SearchRequest, SQLite chats table | ✅ designed (2026-04-16) |
| Add multi-user config (`users:` + `shared:` sections in config.yaml) | ✅ done (2026-04-16) |
| Define `rag_engine/` package layout + shared modules (`collection.py`, `schema.py`) | ✅ designed (2026-04-16) |
| Define `ingestion_queue` schema with crash recovery + retry columns | ✅ designed (2026-04-16) |
| Define worker loop with delete-before-insert idempotency | ✅ designed (2026-04-16) |
| Implement `rag_engine/collection.py` + `schema.py` (shared constants) | ✅ done (2026-04-16) |
| Implement `rag_engine/embeddings.py` (extract from `rag.py`) | ✅ done (2026-04-16) |
| Implement `rag_engine/state.py` (SQLite tracker CRUD + crash recovery) | ✅ done (2026-04-16) |
| Implement `rag_engine/scanner.py` (NFS filesystem scanner) | ✅ done (2026-04-16) |
| Implement `rag_engine/ingest.py` (docling parse + chunking + upsert) | ✅ done (2026-04-16) |
| Implement `rag_engine/search.py` + `__init__.py` (`search_rag()` export) | ✅ done (2026-04-16) |
| Wire RAG into main.py — delete `rag.py`, import `rag_engine` | ✅ done (2026-04-16) |
| Implement standalone CLI utils (`utils/rag_*.py`) | ✅ done (2026-04-16) |
| Add `POST /v1/rag/ingest/*` endpoints with user scoping | ✅ done (2026-04-16) |
| Add `db_base_path` global config variable | ✅ done (2026-04-16) |
| Configure `users:` section in `config.yaml` with real Google emails | ✅ done (2026-04-17) — `florian.otel@gmail.com` active; others commented out |
| Move exclusion filters to config.yaml (`rag.scanner` subsection) | ✅ done (2026-04-19) |
| Incremental sync deletes orphaned Qdrant points (not just SQLite rows) | ✅ done (2026-04-19) |
| Run initial NFS scan + data ingestion | ⏳ in progress — see §5 |
| Implement Google OAuth2 middleware (Phase 3, not blocking for RAG) | ⏳ Phase 3 |

### 4.2 Decoupled RAG Pipeline Architecture

To support parallel development without disrupting the main LLM orchestrator, the RAG system is strictly isolated into its own boundary:
* **`rag_engine/` Module:** An independent package containing all ingestion, chunking, and Qdrant DB connection logic.
* **Interface Contract:** The main application (`router.py`/`main.py`) imports only a single function: `search_rag(query, user_id, limit)`. The `user_id` is the `owner` string (e.g. `"florian"`) — the RAG engine applies the Qdrant filter internally. The caller remains completely unaware of `docling`, Ollama batching, or Qdrant internals.

#### Package layout (confirmed 2026-04-16)

```
rag_engine/
├── __init__.py          # exports search_rag(query, user_id, limit) only
├── embeddings.py        # embed_text(), embed_batch() — shared by ingestion + search
├── ingest.py            # docling parse, parent-child chunking, Qdrant upsert
├── search.py            # query → embed → Qdrant query_points → parent_text + provenance
├── collection.py        # Qdrant collection config + creation (single source of truth)
├── schema.py            # payload field name constants + owner derivation
├── scanner.py           # NFS filesystem scanner → populates StateDB (shared by CLI + API)
└── state.py             # SQLite tracker (rag_state.db) — ingestion queue CRUD
```

**Shared definitions** that both ingestion and search must agree on live in two files:
* `collection.py` — collection name (`"documents"`), vector size (1024), distance metric (`COSINE`).
  Called by `ingest.py` (to auto-create on first use) and `search.py` (to query).
* `schema.py` — payload field name constants (`FIELD_OWNER = "owner"`, `FIELD_SOURCE_PATH = "source_path"`,
  `FIELD_PARENT_TEXT = "parent_text"`, etc.) and `derive_owner()`. Both sides import from here; no
  string literals for field names anywhere else.

**Parallel development streams** (once the above two files exist):

| Work stream | Files | Can proceed independently? |
|-------------|-------|---------------------------|
| NFS scanner + SQLite tracker | `state.py`, `utils/rag_sync_nfs.py` | Yes — only needs config.yaml + NFS |
| docling parsing + chunking | `ingest.py` (parse/chunk functions) | Yes — pure functions, no Qdrant |
| Embedding integration | `embeddings.py` | Yes — only needs Ollama on Server 1 |
| Qdrant upsert worker | `ingest.py` (upsert), `utils/rag_ingest_daemon.py` | Yes, after `collection.py` exists |
| Search + prompt injection | `search.py`, main.py wiring | Yes, after `collection.py` + data exists |
| CLI utils (status, search, reset) | `utils/rag_*.py` | Yes — reads SQLite/Qdrant |

`rag.py` has been deleted. `rag_engine/` is the complete implementation.

### 4.3 State Management (The Tracker)

A dedicated SQLite database (`/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db`) guarantees a fail-safe process that can pause and resume seamlessly.

#### `ingestion_queue` table schema

```sql
CREATE TABLE ingestion_queue (
    file_path       TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,           -- derived from NFS root at discovery
    last_modified   REAL NOT NULL,           -- os.path.getmtime() at discovery time
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | completed | ignored
    error_msg       TEXT,                    -- last transient error (cleared on completion)
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 5,
    started_at      TEXT,                    -- ISO timestamp; set when status → processing
    completed_at    TEXT,                    -- ISO timestamp; set when status → completed
    progress_detail TEXT,                    -- e.g. "parsing", "embedding 34/120", "upserting"
    skip_reason     TEXT                     -- set when status → ignored (last error message)
);
```

**Status transitions:**
- `pending → processing` — worker picks up file, sets `started_at`
- `processing → completed` — all 7 steps succeeded, sets `completed_at`
- `processing → pending` — step failed with `retry_count < max_retries`; writes `error_msg`, increments `retry_count`
- `processing → ignored` — step failed with `retry_count >= max_retries`; writes `skip_reason` (last error); never re-queued by `rag_sync_nfs.py` unless file changes on disk
- `completed → pending` — re-discovery detects `last_modified` on disk > `last_modified` in SQLite
- `ignored → pending` — re-discovery detects `last_modified` on disk > `last_modified` in SQLite (file was replaced)

#### Crash recovery

On daemon startup, before entering the worker loop:
1. Query all rows where `status = 'processing'`
2. Reset them to `pending` (daemon was killed mid-file; work is incomplete)
3. Log which files were reset for operator visibility

This prevents files from being permanently stuck in `processing` after a crash, OOM, or NFS timeout.

#### Discovery function — incremental sync handles additions AND deletions

`scan_nfs_roots()` (called by `rag_sync_nfs.py` and `POST /v1/rag/ingest/sync`) performs a
full incremental reconciliation on every run:

- **New files:** inserted as `pending`
- **Modified files:** if `os.path.getmtime()` > stored `last_modified`, reset to `pending`
  (the worker loop handles Qdrant cleanup before re-ingestion — see §4.4 step 0)
- **Deleted or excluded files:** if a previously `completed` file no longer appears in the
  scan results — whether because it was physically deleted from NFS, or because it is now
  matched by an exclusion filter in `config.yaml` — its SQLite row is removed **and** all
  corresponding Qdrant points are deleted, filtered by `source_path`. Each delete uses
  `wait=False` (fire-and-forget) — Qdrant queues it and returns immediately, so the script
  never blocks on index re-optimization. Any exception (Qdrant unreachable) aborts the
  script; SQLite rows survive as retry markers for the next run. Note: the ingestion
  daemon's step-0 delete and step-6 upsert keep `wait=True` — they require synchronous
  confirmation (see §4.4). See [RAG-troubleshoot.md §2026-05-12](RAG-troubleshoot.md)
  for the failure scenario that motivated this design.

The third case covers the config-driven exclusion scenario: adding a new pattern to
`rag.scanner.exclude_dir_names`, `exclude_dir_suffixes`, or `exclude_file_patterns` and
re-running `rag_sync_nfs.py` is sufficient to remove previously ingested content from the
vector store. No manual Qdrant cleanup is needed.

### 4.4 Atomic Document Embedding (Worker Loop)

To prevent the Qdrant database from serving partial context during active building, ingestion occurs via a strict, atomic worker loop:

0. **Delete stale points (re-ingestion safety):** If Qdrant already contains points for this
   `source_path` (i.e., file was previously ingested and is now being re-processed due to
   modification or retry), delete them first:
   `client.delete(collection, Filter(must=[FieldCondition(key="source_path", match=MatchValue(value=file_path))]))`.
   This prevents duplicate points from accumulating across re-ingestion cycles. The brief
   window where the file has zero results is acceptable for a home lab.
1. **Lock State:** Fetch one `pending` file path from SQLite and immediately update to
   `processing`; set `started_at` and `progress_detail = "starting"`.
2. **Extract & Parse:** Pass the file to the appropriate parser: docling (PDF/DOCX/PPTX),
   `_parse_pptx()` python-pptx fallback (PPTX when docling format detection fails),
   `_parse_ipynb()` (ipynb), or direct UTF-8 read (MD/TXT/YAML/CSV).
   Update `progress_detail = "parsing"`.
3. **Generate Chunks:** Execute parent-child split logic (or flat 512-token chunks) entirely
   in memory. Update `progress_detail = "chunking ({n} chunks)"`.
4. **Vectorize:** Send child chunks to `bge-m3` via Ollama using `--batch`-concurrent
   `asyncio` requests. Update `progress_detail = "embedding {i}/{n}"` periodically.
   `embed_batch()` fires `progress_cb` every 50 chunks, which logs
   `"Embedding progress: N/M  filename"` — consumed by `rag_status.py --watch`.
5. **Build Payloads:** Construct Qdrant point objects, binding the vector, UUID, and payload
   (`source_path`, `parent_text`, `owner` from SQLite row). All field names imported from
   `rag_engine/schema.py` constants — no string literals.
6. **Batched Upsert:** Send all points for the document to Qdrant in batches of
   `_UPSERT_BATCH_SIZE` (256) points per `client.upsert()` call. Update `progress_detail = "upserting"`.
   A single upsert with all points can exceed Qdrant's HTTP body limit for large files
   (large PDFs, big CSVs) — Qdrant returns HTTP 400 `"Payload error: JSON payload (N bytes)"`.
   Batching keeps each request well under the limit. If one batch fails mid-file, the retry
   will re-run from step 0 (delete-before-insert), cleaning up any partial set.
7. **Finalize State:** Upon successful upsert, update the SQLite row to `completed` with
   `completed_at` timestamp. If any step fails: increment `retry_count` and set status to
   `pending` if `retry_count < max_retries` (auto-retry), else `ignored` with `skip_reason`
   set to the last error message (permanent skip — `rag_sync_nfs.py` will not re-queue it).

**Idempotency guarantee:** Step 0 (delete-before-insert) ensures that re-processing a file
— whether due to modification, crash recovery, or retry after partial failure — always
produces a clean result. Even if step 6 succeeds but step 7 fails (SQLite write error),
the next retry will delete the stale points before re-inserting.

**Point IDs:** Use random UUIDs (not deterministic hashes). Since step 0 deletes all
existing points for the file before inserting, there are no orphan-ID concerns even when
chunking changes produce a different number of chunks.

### 4.5 Standalone Utilities (`utils/`)

Standalone CLI scripts enable independent progress monitoring and RAG pipeline management without launching the FastAPI server:
* `utils/rag_sync_nfs.py`: Scans all configured NFS roots (per-user + shared), derives `owner` per file, applies filters from `config.yaml`, and populates SQLite with `pending` files. New files → `pending`; modified files (mtime changed) → `pending`; `ignored` files (mtime unchanged) → no-op (permanent skip); `ignored` files (mtime changed, i.e. file replaced) → reset to `pending`. For files removed from NFS or newly excluded by config, removes the SQLite row and deletes the corresponding Qdrant points. Accepts `--user florian` to scan a single user's root only. Always run this before restarting the daemon after failures.
* `utils/rag_ingest_daemon.py`: The worker loop executing the 7-step atomic embedding process (includes `owner` in every Qdrant payload). Both `--workers` and `--batch` are **required** flags — see §5.4 for operating points. `--workers` controls file-level concurrency (how many files parse+embed simultaneously, each in its own OS thread with a thread-local `DocumentConverter`); `--batch` controls chunk-level Ollama concurrency within each file.
* `utils/rag_status.py`: Dashboard querying SQLite/Qdrant to output ingestion metrics (`pending`, `processing`, `completed`, `ignored`). `ignored` count always shown; `--ignored` for full detail listing with retry count and last error. Accepts `--user` to filter by owner. `--watch LOG_FILE` mode: refreshes every 2s, shows per-file chunk progress bar, elapsed time, chunk rate, and ETA (absolute clock + remaining duration) derived from log timestamps. `--list-pending [N]` prints pending file paths one per line (pipeable into `wc -l`, `head`, etc.); combinable with `--user`.
  `--watch` log parser (`parse_log`) tracks all in-flight files simultaneously (updated 2026-04-22). The original single-state machine reset on every `Processing` line and would lose any file whose `Chunked` line had already appeared before a later `Processing` line — i.e. it broke silently with `--workers > 1`. Fix: per-file state dict keyed by full NFS path; `name_to_path` lookup associates `Chunked`/`Embedding progress`/`Ingested` lines (filename only) back to their full-path entry. Display picks the file currently in `embedding` phase with the most recent progress timestamp.
* `utils/rag_search_cli.py`: Query tester returning `parent_text` and cosine similarity scores. Requires `--user` flag to apply the ownership filter (simulates authenticated search).
* `utils/rag_reset.py`: Drops the Qdrant collection and resets SQLite to `pending` for clean re-ingestion. Accepts `--user` to reset a single user's documents only.

### 4.6 APIs for Control & Monitoring

The FastAPI orchestrator exposes the SQLite tracker state for client interfaces.
All endpoints require an authenticated user (Google OAuth2 JWT). Ingestion endpoints
are admin-only (Florian); search is scoped to the authenticated user's `owner` + `"la-familia"`.

* `POST /v1/rag/ingest/sync`: Triggers the NFS scanner across all configured roots. As with `rag_sync_nfs.py`, removes SQLite rows and deletes Qdrant points for any previously ingested files that are no longer present or are now excluded by config.
* `POST /v1/rag/ingest/start`: Spawns the ingestion daemon as an asyncio background task.
* `POST /v1/rag/ingest/stop`: Gracefully halts the ingestion worker.
* `GET /v1/rag/ingest/status`: Returns metrics (`total_files`, `progress_percentage`, etc.) based on SQLite rows. Accepts optional `?user=florian` filter.

---

## 5. Ingestion runbook

This section covers how to populate the Qdrant vector store with documents from the NFS
corpus, and how to keep it up to date. All commands are run on **Server 1 (192.168.1.93)**
from the project root.

### 5.1 Prerequisites

#### Activate the virtualenv

```bash
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
cd ~/Gin-AI/projects/SoHoAI
```

#### Verify Ollama is running with bge-m3

```bash
ollama list | grep bge-m3
```

Expected: `bge-m3:latest`. If missing:

```bash
ollama pull bge-m3
```

#### Verify Qdrant is running

```bash
curl -s http://localhost:6333/collections
# Expected: {"result":{"collections":[...]},"status":"ok",...}
```

If not running: `sudo systemctl start qdrant`

#### Verify database directories exist on NAS

```bash
ls /mnt/nfs/__Backups/SoHoAI--databases/
# Expected: qdrant-snapshots/  sqlite/  redis/
```

All directories must exist and be writable. They are all under `db_base_path` in
`config.yaml` — change that single key to relocate everything (also update `redis.dir`
manually, as it is not auto-derived in Python).

### 5.2 Configure NFS roots in `config.yaml`

`florian.otel@gmail.com` is already active. The `shared:` section (`/mnt/nfs/La-Familia`)
is already active. Other users (Eva, Annika, Laura) remain commented out until ready.

To add another user, uncomment their block and fill in the real Google email:

```yaml
  # eva.otel@gmail.com:
  #   owner: "eva"
  #   nfs_roots: ["/mnt/nfs/Eva"]
```

To adjust exclusion filters (add or remove paths), edit `config.yaml` under `rag.scanner`
(see §1.2). No code change required — the scanner reads these at runtime.

### 5.3 Populate the ingestion queue (NFS scan)

```bash
python utils/rag_sync_nfs.py
```

This walks `/mnt/nfs/Florian` and `/mnt/nfs/La-Familia`, applies all exclusion filters
from `config.yaml` (`rag.scanner`), and inserts discovered files into the SQLite ingestion
queue (`rag_state.db`) as `pending`. Takes 1–3 minutes — filesystem `stat()` calls only,
no embedding yet.

**Incremental behaviour** (safe to re-run at any time):
- New files → inserted as `pending`
- Modified files (mtime changed) → reset to `pending` for re-ingestion
- Previously ingested files no longer found (deleted from NFS, or now matched by an
  exclusion filter) → SQLite row removed **and** Qdrant points deleted immediately

To scan only one user's root:

```bash
python utils/rag_sync_nfs.py --user florian
```

Check the result:

```bash
python utils/rag_status.py
# Expected on first run: ~2800 pending, 0 completed
```

### 5.4 Run the ingestion daemon

Both `--workers` and `--batch` are **required** — the daemon errors immediately if either
is omitted. They are independent knobs for different bottlenecks:

| Flag | Controls | Bottleneck it addresses |
|------|----------|------------------------|
| `--workers N` | Files processed concurrently (file-level) | docling CPU parse time |
| `--batch M` | Ollama embedding requests in-flight per file (chunk-level) | Embedding throughput |

#### Operating mode A — All-local / CPU-bound (Ollama on Server 1)

Use when bge-m3 is served by Ollama on Server 1 (192.168.1.93). Parse and embed share
the same CPU (Ryzen 9 6900HX, 8 physical cores / 16 threads). Within one file, parse and
embed are strictly sequential so they do not compete. `--workers 1` keeps docling's
4 internal threads from crowding out Ollama.

```bash
# config.yaml: ollama_url: http://192.168.1.93:11434/api/embeddings
screen -S rag-ingest
python utils/rag_ingest_daemon.py --workers 1 --batch 5
# Ctrl-A D to detach; screen -r rag-ingest to reattach
```

Timing (CPU embed, ~650ms/chunk):

| File type | Time per file |
|-----------|--------------|
| Short MD / YAML / TXT | ~1–3 s |
| Long Markdown (>50 sections) | ~5–15 s |
| Jupyter notebook | ~5–20 s |
| PDF (10–50 pages) | ~20–90 s |
| PDF (100+ pages) | ~90–300 s |

For ~2,800 files: estimate **6–10 hours**. Ollama serializes inference — when the daemon
runs, live search queries queue behind the embedding batch (up to 28–30s wait). The
`embed_text()` timeout is 120s; do not reduce it while the daemon is running.
To reduce search latency during ingestion, lower `--batch 2`.

#### Operating mode B — Split / GPU-accelerated (Ollama on Server 2)

Use when bge-m3 is served by Ollama on Server 2 (192.168.1.95, RTX 5070). Embedding is
remote GPU (~10–20ms/chunk); docling CPU parse on Server 1 becomes the bottleneck.
`--workers 3` runs 3 files concurrently — while one file's chunks embed on the GPU,
the other two are being parsed by docling on Server 1's CPU threads.

Server 1 hardware: Ryzen 9 6900HX, 8 physical cores / 16 logical threads.
Docling uses 4 CPU threads internally per converter instance (thread-local, one per worker).
3 workers × 4 threads = 12 threads — fits cleanly within the 16 logical threads while
leaving 4 threads for the OS and other light workloads (IDE, Claude Code).
Do not use `--workers 4` — 16 docling threads at the SMT limit gives diminishing returns
for CPU-bound matrix ops and makes the machine noticeably sluggish.

**Prerequisite**: ensure llama-server on Server 2 is idle or stopped before running bulk
ingestion with `--workers 3`. bge-m3 (~1.2GB) plus llama-server (~9.97GB) = ~11.2GB,
leaving only ~1GB headroom in the RTX 5070's 12GB — fine at low concurrency but risky
under a heavy `--batch 20` load.

```bash
# config.yaml: ollama_url: http://192.168.1.95:11434/api/embeddings
screen -S rag-ingest
python utils/rag_ingest_daemon.py --workers 3 --batch 20
# Ctrl-A D to detach; screen -r rag-ingest to reattach
```

Timing (GPU embed, ~10–20ms/chunk):

| File type | Bottleneck | Time per file (effective) |
|-----------|------------|--------------------------|
| Short MD / YAML / TXT | embed | ~0.5–2 s |
| Long Markdown | embed | ~2–8 s |
| Jupyter notebook | parse | ~3–10 s |
| PDF (10–50 pages) | parse (docling) | ~4–40 s |
| PDF (100+ pages) | parse (docling) | ~30–120 s |

For ~2,800 files: estimate **2–4 hours** (3× pipeline overlap hides most of the parse time).
Large plain-text files (tens of thousands of chunks) remain embedding-bound regardless of
GPU speed — this is expected and does not indicate a problem.

#### Per-file steps (both modes)

For each file the daemon:
1. Deletes any stale Qdrant points for that file (idempotency — step 0)
2. Marks the file as `processing` in SQLite
3. Parses with `docling` → full text (or `_parse_pptx()` python-pptx fallback for PPTX; or `_parse_ipynb()` for notebooks)
4. Chunks text (parent-child for PDF/IPYNB/MD; flat 512-tok for PPTX/YAML/CSV)
5. Embeds child chunks via Ollama (`bge-m3`, `--batch` concurrent requests)
6. Upserts all points to Qdrant in batches of 256
7. Marks the file as `completed`

The daemon is **crash-safe**: if killed, restart it and it resumes from where it left off.
Rows stuck in `processing` are automatically reset to `pending` on startup (crash recovery).

### 5.5 Monitor progress

From a separate terminal:

```bash
# One-shot status
python utils/rag_status.py

# Live watch (every 30 s)
watch -n 30 python utils/rag_status.py

# Show ignored files with retry count and last error (rationale)
python utils/rag_status.py --ignored

# Filter to one user
python utils/rag_status.py --user florian
```

Sample output once partially complete:

```
Ingestion queue (all users):
  pending    : 1842
  processing : 1
  completed  : 947
  ignored    : 2
  ─────────────────
  total      : 2792
  progress   : 33.9%

Qdrant 'documents' collection:
  total points    : 12483
```

### 5.6 Test search before full ingestion is complete

You do not need to wait for the full corpus. Once a few hundred files are done, test the
search pipeline:

```bash
python utils/rag_search_cli.py \
  --query "what AWS certifications do I have" \
  --user florian

python utils/rag_search_cli.py \
  --query "docker networking" \
  --user florian \
  --top-k 10
```

Output shows ranked results with cosine similarity scores, source paths, and the
`parent_text` that would be injected into the LLM prompt.

**Note**: if the ingest daemon is running, the search embedding call may take up to 30s
to return (Ollama queues the request behind the batch). This is normal — the 120s timeout
in `embed_text()` covers it.

Search without ownership filter (returns all documents):

```bash
python utils/rag_search_cli.py --query "family trip" --no-filter
```

### 5.7 Enable RAG in the chat client

Once satisfied with search quality, enable RAG in the CLI chat:

```bash
python utils/cli_chat.py --server http://192.168.1.93:8000 --user florian
# In the chat session:
/rag on            # advertise the search_documents tool; LLM decides when to call it
/rag only          # force grounded mode — answers must come from retrieved chunks
```

Or pass `"rag_mode": "on"` (or `"only"`) in the `ChatRequest` JSON. The orchestrator
injects a tool spec into the system prompt; when the LLM emits a `<tool_call>` block,
the orchestrator invokes `search_rag()` (or `multi_query_search()` if `rag.multi_query.enabled: true`),
feeds the retrieved chunks back as a tool message, and the LLM composes the final answer.
`rag_sources` in the response reflects chunks the LLM actually retrieved.

### 5.8 API control (via the orchestrator)

```bash
# Trigger NFS scan (also cleans up Qdrant for deleted/excluded files)
curl -X POST http://192.168.1.93:8000/v1/rag/ingest/sync

# Scan only one user's roots
curl -X POST "http://192.168.1.93:8000/v1/rag/ingest/sync?user=florian"

# Start background ingestion worker
curl -X POST http://192.168.1.93:8000/v1/rag/ingest/start

# Stop ingestion worker (after current file completes)
curl -X POST http://192.168.1.93:8000/v1/rag/ingest/stop

# Status
curl http://192.168.1.93:8000/v1/rag/ingest/status
```

### 5.9 Troubleshooting

#### Search times out while ingestion is running

Ollama serializes model computation. When the ingest daemon is running, search queries
queue behind the embedding batch and can wait 28–30s before being served. The
`embed_text()` timeout is 120s — the query will eventually succeed. This is expected
behaviour during bulk ingestion.

To reduce search wait time, lower the daemon's `--batch` value (e.g. `--batch 2`),
which shrinks the Ollama queue depth. The trade-off is slower ingestion throughput.

#### Embedding timeouts (httpx.ReadTimeout)

If Ollama's queue depth is too high, embedding requests time out after 120s. The daemon
auto-retries each file up to `max_retries` (5) times; after that the file becomes `ignored`
in SQLite with the last error as `skip_reason`.

**Observed in production (2026-04-19):** 78 out of ~3,000 files failed this way during the
initial bulk ingest. All were large CSVs and notebooks from a data-science course directory
that generated many chunks per file. `str(httpx.ReadTimeout()) == ''`, so `skip_reason` in
SQLite is blank for these files — they show up in `rag_status.py --ignored` with no message.

Remedy: lower `--batch` (e.g. `--batch 2`) and re-queue via `rag_sync_nfs.py`. Because
`ignored` files are not automatically re-queued, force-reset them in SQLite first:

```bash
# Force-reset ignored files to pending (run before rag_sync_nfs.py)

# Soft: only re-queue ignored files with no skip_reason (timeout failures — blank error_msg)
# sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
#   "UPDATE ingestion_queue SET status='pending', retry_count=0, skip_reason=NULL \
#    WHERE status='ignored' AND (skip_reason IS NULL OR skip_reason='')"

# Brute-force: re-queue ALL ignored files regardless of skip_reason
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "UPDATE ingestion_queue SET status='pending', retry_count=0, skip_reason=NULL WHERE status='ignored'"

python utils/rag_sync_nfs.py --user florian
python utils/rag_ingest_daemon.py --workers 3 --batch 2
```

#### Qdrant HTTP 400 — oversized upsert payload

Qdrant rejects `upsert` requests whose JSON body exceeds its HTTP limit. Error in SQLite
`error_msg`: `Unexpected Response: 400 … "Payload error: JSON payload (N bytes)"`.

**Observed in production (2026-04-19):** 10 files failed this way — large PDF textbooks
(`Hands_On_Machine_Learning`, `Python for Data Analysis`) and large CSVs
(`house_sales.csv`, `lc_loans.csv`). Step 6 of the worker loop now batches upserts into
groups of `_UPSERT_BATCH_SIZE = 256` points (`ingest.py`), which eliminates this error.

Re-queue after the fix is deployed:

```bash
python utils/rag_sync_nfs.py
python utils/rag_ingest_daemon.py --batch 2
```

#### Ignored files — review and selective retry

After exhausting `max_retries` (5), files become `ignored` with `skip_reason` = last error.
`rag_sync_nfs.py` never re-queues them automatically.

```bash
python utils/rag_status.py --ignored  # list files, retry count, and last error
```

**Key property:** ignored files have **zero Qdrant points** — step 0 of the worker loop
deletes existing points before processing begins, and the upsert step never completed on
any of the failed attempts. No Qdrant cleanup is needed before retrying.

To retry specific files after fixing the underlying issue:

```bash
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "UPDATE ingestion_queue SET status='pending', retry_count=0, skip_reason=NULL WHERE file_path='<path>'"
python utils/rag_ingest_daemon.py --batch 2
```

Do NOT use `rag_reset.py` for retrying — it resets all rows including `completed` ones,
forcing a full re-ingest of thousands of already-processed files.

#### Large PDF / docling parsing failures

`docling` occasionally fails on malformed or encrypted PDFs. The daemon auto-retries up to
`max_retries` (5) times, then marks the file as `ignored` with the last error as `skip_reason`.

```bash
python utils/rag_status.py --ignored
```

After fixing or excluding the file, force-reset it to `pending` in SQLite and re-queue
with `rag_sync_nfs.py` (see "Ignored files — review and selective retry" above).

#### Ollama goes offline mid-ingestion

If `bge-m3` is evicted (e.g. another model loaded), embedding calls fail. The daemon marks
the file as `pending` for auto-retry if `retry_count < max_retries` (5), or `ignored` once
retries are exhausted. Restart Ollama with the model loaded, then re-queue any ignored files
manually (see "Ignored files — review and selective retry" above) and restart the daemon.

#### Qdrant not accessible

If Qdrant is unreachable, the orchestrator logs a warning and sets
`app.state.qdrant_client = None` (RAG is disabled, chat still works). Check:

```bash
sudo systemctl status qdrant
curl -s http://localhost:6333/collections
```

#### Clean restart

To drop everything and start fresh:

```bash
# Full reset — drops Qdrant collection + resets all SQLite rows to pending
python utils/rag_reset.py

# Partial reset — only one user's data
python utils/rag_reset.py --user florian
```

### 5.10 File type coverage

| Type | Strategy | Notes |
|------|----------|-------|
| `.pdf` | Parent-child | High benefit — certifications, work docs |
| `.ipynb` | Parent-child | High benefit — code + markdown context |
| `.md` | Parent-child (if long) | Medium benefit |
| `.docx` | Parent-child | Medium benefit |
| `.pptx` | Flat 512-tok | Slides already compact |
| `.txt` | Flat (short) / Parent-child (long) | Auto-detected by token count |
| `.yaml` / `.yml` / `.csv` | Flat 512-tok | Config files already compact |
| `.xlsx` | Skipped | Not supported by docling |
| `.jpg` / `.png` / `.mp4` | Skipped | Phase 4 (CLIP embeddings) |

Exclusion filters (directories and file patterns) are configured in `config.yaml` under
`rag.scanner` — see §1.2 for the full list and how to add new exclusions.

---

## 6. Phase 4 — Images and videos (future)

- **CLIP model** (`openai/clip-vit-base-patch32`) on Server 2 GPU
- Family photo ingestion → CLIP embeddings → separate Qdrant `images` collection
  (same Qdrant instance as `documents`, different collection)
- `images` and `videos` collections use the same `owner` field as `documents` —
  search filtering works identically (`MatchAny(any=[user_owner, "la-familia"])`)
- Text-to-image similarity search
- EXIF metadata extraction for time/location
- Person identification (RLHF-trainable — see section 2.2)
- Video: CLIP or frame-level embeddings → `videos` collection

---

## 7. Use cases supported

| Use case | Phase | Notes |
|----------|-------|-------|
| Search course/training/certification material | 2 | PDF, PPTX, DOCX → full NFS path reference returned; scoped to authenticated user |
| Search project docs and notebooks | 2 | Markdown, `.ipynb`; user sees own + La-Familia |
| Standalone RAG DB ingestion (no active chat) | 2 | `POST /v1/rag/ingest` endpoints; admin-only |
| Multi-user document isolation | 2 | `owner` field in Qdrant + search filter; Google OAuth2 identity |
| Shared family content search | 2 | La-Familia docs visible to all authenticated users |
| Find family photos by text description | 4 | CLIP embeddings, Qdrant `images`; same `owner` filter |
| Find video by description | 4 | CLIP or frame embeddings, Qdrant `videos`; same `owner` filter |
| RL training from human feedback | ongoing | Thumbs-up/down per turn → JSONL export |
| Train person recognition (images/videos) | 4 | RLHF loop |

---

## 8. Advanced retrieval and generation

**Plan authored:** 2026-04-21  
**§8.1–§8.3 implemented:** 2026-04-21  
**§8.4:** design note only — NOT implemented (see decision gate in §8.4)

**Motivation:** After initial corpus ingestion (2891 files / 98,737 points), user-facing
testing surfaced two symptoms:

- **Irrelevant retrieval on meta questions** — e.g. "what model are you?" returned five
  documents from the ML textbook shelf at scores 0.497–0.508 because top-k retrieval ran
  unconditionally for every user turn whenever RAG was active.
- **Narrow score margins with bge-m3** — genuinely relevant hits score 0.51–0.55; noise
  hits score 0.48–0.51. A single-query, single-score-threshold approach is fragile.

The implementation replaces "always-inject top-k" with a model-agnostic, tool-driven
pipeline in which the LLM decides whether to retrieve and the retriever returns a
relevance-plus-diversity-optimised candidate set.

**Scope (implemented):**
- `schemas.py` — `rag_mode: RagMode` enum (legacy boolean flag removed)
- `main.py` — tool-use loop, `_retrieve()` dispatcher, `_apply_system_prompt()`, `_fold_tool_messages()`, `variant_llm_fn` startup closure
- `rag_engine/multi_query.py` — new (multi-query + MMR)
- `rag_engine/tool_use.py` — new (tool-call protocol parser + formatter)
- `prompts/rag_system_prompts.py` — new (system prompts for off/on/only modes)
- `rag_engine/__init__.py` — simplified to clean re-export of `search_rag` + `multi_query_search`
- `config.yaml` — single merged `rag:` block with `tool_use:` and `multi_query:` sub-sections
- `utils/cli_chat.py` — `/rag on|off|only`, sends `rag_mode` field
- `utils/rag_smoke_test.py` — updated to `rag_mode: "on"` payload

**Not in scope:** re-ingestion (no ingestion schema changes), LangChain runtime dependency
(ruled out), Phase 3 OAuth2/Open WebUI, §8.4 contextual retrieval.

### 8.0 Implementation notes — code review (2026-04-21)

The external worker implementation was reviewed and corrected before being accepted. Key
issues found and fixed:

| File | Issue | Fix |
|---|---|---|
| `main.py` | Entire orchestrator replaced with a mock stub returning static strings | Restored from git; §8 changes applied surgically on top |
| `config.yaml` | Duplicate `rag:` key — second block silently shadowed the first, destroying `qdrant_url`, `ollama_url`, `top_k`, `scanner` | Merged into one `rag:` block |
| `rag_engine/__init__.py` | `def search_rag()` shadowed its own import; `else` branch called itself → infinite recursion | Reduced to two-line re-export; dispatch lives in `main.py:_retrieve()` |
| `rag_engine/__init__.py` | `multi_query_search(..., llm_fn=None)` — crash on first multi-query call | Removed from `__init__`; only `main.py:_retrieve()` calls it with a valid `llm_fn` |
| `rag_engine/multi_query.py` | Missing `from typing import Any` | Added |
| `rag_engine/tool_use.py` | Unused `PointStruct` import | Removed |
| `schemas.py` | `SearchRequest.rag_mode` — irrelevant on retrieval-only endpoint | Removed |
| `prompts/rag_system_prompts.py` | Imported dead `build_tool_spec`; defined its own `RagMode = Literal[...]` | Removed dead import; imports `RagMode` from `schemas` |
| `utils/cli_chat.py` + `rag_smoke_test.py` | Still sent legacy boolean payload | Updated to `rag_mode: str` |
| All Python files | `localhost:11434` (Ollama) and `localhost:6333` (Qdrant) hardcoded | → `192.168.1.93:11434` and `192.168.1.93:6333` throughout (see §8.0.1) |
| `kv_cache.py` | `apply_mistral_template()` used Mistral `[INST]...[/INST]` format with Gemma 4 E4B → hallucinated content after `</tool_call>`, broken multi-turn context, empty answers in `/rag only` mode (root-caused 2026-04-21) | Replaced with `apply_gemma_template()` using Gemma 4 native `<\|turn>role\n` markers; stop tokens `["<\|turn>"]`; Mistral function kept as deprecated stub with warning |

#### 8.0.1 `localhost` disambiguation

All service addresses in the codebase now use explicit IPs:

| Service | Address | Why |
|---|---|---|
| Redis | `127.0.0.1:6379` | **Loopback required** — Redis runs in protected mode on Server 1; orchestrator is co-located. Config comment explains this. Do not change. |
| Ollama | `192.168.1.93:11434` | Ollama runs on Server 1 CPU. Explicit IP for consistency and correctness if any util is ever run from Server 2. |
| Qdrant | `192.168.1.93:6333` | Qdrant runs on Server 1 NVMe. Explicit IP in both `config.yaml` and all Python fallbacks. |
| llama-server | `192.168.1.95:8000` | Server 2 GPU. Always been explicit — unchanged. |

`config.yaml` is the single source of truth for all non-Redis URLs. Python fallback defaults in source files match `config.yaml` exactly and are only reached if the config file fails to load.

### 8.1 Three RAG modes: `off` / `on` / `only` ✅ implemented 2026-04-21

#### Purpose

Give the caller explicit control over how documents participate in generation.

| Mode | Retrieval behaviour | LLM instruction |
|------|--------------------|-----------------|
| `off` | Tool is **not** advertised in the system prompt. No retrieval happens. | "You are a helpful assistant." (no mention of documents.) |
| `on`  | Tool **is** advertised. LLM calls it when it judges the corpus likely to help. | "You may call `search_documents` when the user's personal corpus likely contains the answer. You may also answer from general knowledge." |
| `only` | Tool **is** advertised, **and** the LLM is required to call it before answering; answers must be grounded strictly in tool results. | "Your ONLY source of knowledge is `search_documents`. Call it for every factual question. If the results do not answer the question, reply exactly: 'I don't have information about that in the provided context.' Do not use prior knowledge." |

Note: §8.1 defines the *modes*; the *mechanism* by which the tool is advertised / invoked
is §8.2. The two sections are meant to be implemented in the same change set.

#### Schema changes

File: `schemas.py`.

The legacy boolean toggle has been **removed entirely**. All clients send `rag_mode`
explicitly; server falls back to `rag.default_mode` from `config.yaml` when omitted.

```python
from enum import Enum

class RagMode(str, Enum):
    off  = "off"
    on   = "on"
    only = "only"

class ChatRequest(BaseModel):
    # ... existing fields ...
    rag_mode: RagMode = RagMode.on   # default server-side; see default_mode below
```

`SearchRequest` is unchanged (retrieval-only endpoint has no notion of `only`).

#### Config changes

File: `config.yaml`, under `rag:`:

```yaml
rag:
  default_mode: "on"        # off | on | only — server default when caller omits rag_mode
  # ... existing keys ...
```

`main.py:_server_managed_completion` resolves the effective mode as follows:

```python
if "rag_mode" in req.model_fields_set:
    rag_mode = req.rag_mode                               # client was explicit
else:
    rag_mode = RagMode(app.state.rag_cfg.get("default_mode", "on"))
```

Using `model_fields_set` (Pydantic) distinguishes "client omitted the field" from
"client sent `rag_mode: on` explicitly", so the config `default_mode` only kicks in
when the caller truly didn't specify. The Pydantic field default (`RagMode.on`) is a
belt-and-braces backstop in case a code path bypasses the config read.

#### Prompt module

New file: `prompts/rag_system_prompts.py`. Single source of truth for all three mode
prompts AND the tool-use scaffolding from §8.2. Centralises the strings so changes to
one don't drift away from the others.

```python
# prompts/rag_system_prompts.py
"""System prompts for the three RAG modes. Composed with the tool-use section (§8.2)
at call time by build_system_prompt(mode, tool_spec)."""

_BASE = "You are SoHoAI's assistant. Be concise, accurate, and helpful."

_MODE_OFF = f"""{_BASE}

Answer from general knowledge."""

_MODE_ON = f"""{_BASE}

You have access to the user's personal document corpus via a search tool
(see the TOOLS section below). Call the tool when the question is likely
about the user's documents, projects, certifications, notes, or personal
information. For general questions (greetings, model identity, common
knowledge, code explanations unrelated to the user's corpus), answer
directly without calling the tool."""

_MODE_ONLY = f"""{_BASE}

You have access to the user's personal document corpus via a search tool
(see the TOOLS section below). For every factual question, you MUST call
the tool before answering, and your answer MUST be grounded strictly in
the tool results.

If the tool returns no relevant results, or the results do not answer the
question, you MUST reply EXACTLY with the following sentence and nothing
else:

    I don't have information about that in the provided context.

Do NOT use prior knowledge. Do NOT speculate. Do NOT fill gaps with general
information. This rule applies even if you are confident you know the answer
from training data."""


def build_system_prompt(mode: str, tool_spec: str | None) -> str:
    """Compose the final system prompt.

    Args:
        mode:      "off" | "on" | "only"
        tool_spec: the TOOLS section (output of tool_use.build_tool_spec())
                   or None if mode == "off"
    """
    base = {"off": _MODE_OFF, "on": _MODE_ON, "only": _MODE_ONLY}[mode]
    if mode == "off" or tool_spec is None:
        return base
    return f"{base}\n\n{tool_spec}"
```

#### Pipeline change

File: `main.py`, function `_server_managed_completion`.

The previous top-k auto-inject block is replaced with the tool-use loop from §8.2.
Before entering that loop, resolve the effective mode:

```python
mode: RagMode = req.rag_mode or app.state.rag_cfg.get("default_mode", "on")
if mode != "off" and app.state.qdrant_client is None:
    logger.warning("RAG requested (mode=%s) but Qdrant unavailable; forcing off", mode)
    mode = "off"
```

`rag_sources` is populated from tool results actually returned during this turn
(see §8.2); `rag_mode_used` is added to `ChatResponse` for debuggability.

#### ChatResponse addition

File: `schemas.py`:

```python
class ChatResponse(BaseModel):
    # ... existing fields ...
    rag_mode_used: RagMode | None = None
```

#### CLI changes

File: `utils/cli_chat.py`.

- Default: `self.rag_mode: str = "off"` (callers opt in with `/rag on` or `/rag only`).
- `send_message` payload sends `"rag_mode": self.rag_mode`.
- `/rag` handler subcommands: `off`, `on`, `only`, `status`, `search <query>`
  (pass-through for `status` and `search` is unchanged).
- Preflight banner line includes `mode=<mode>`.
- `rag_sources` display: prefix the block with the mode used by the server, e.g.
  `Sources (mode=only):`.

#### Verification (Acceptance tests)

1. `curl … -d '{"rag_mode":"off", …}'` — no RAG tool advertised; `rag_sources == null`.
2. `curl … -d '{"rag_mode":"on", "messages":[{"role":"user","content":"what model are you?"}]}'`
   — assistant answers directly without calling the tool; `rag_sources == null`.
3. `curl … -d '{"rag_mode":"on", "messages":[{"role":"user","content":"what AWS certifications do I have"}]}'`
   — tool is called; `rag_sources` is populated with paths that include one
   containing `AWS-Certification`.
4. `curl … -d '{"rag_mode":"only", "messages":[{"role":"user","content":"what is the capital of France?"}]}'`
   — tool is called (search returns low-relevance hits), assistant replies with the
   exact decline sentence and nothing else.
5. Config default: `curl … -d '{"messages":[…]}'` (no `rag_mode`) uses
   `rag.default_mode` from `config.yaml`; overriding `default_mode: "only"` and omitting
   `rag_mode` in the request gives `only`-mode behaviour.

### 8.2 Tool-use via system prompt (provider-independent) ✅ implemented 2026-04-21

#### Purpose

Replace unconditional top-k injection with an LLM-driven decision to retrieve. Works
identically for `internal` (Gemma 4 E4B via llama-server `/completion`) and `external`
(Claude Sonnet via LiteLLM/Anthropic). No dependency on native function-calling APIs.

#### Protocol

The assistant is instructed to emit one or zero tool calls per turn, in a fenced JSON
block terminated by a sentinel:

```
<tool_call>
{"name": "search_documents", "arguments": {"query": "…"}}
</tool_call>
```

Rules encoded in the prompt:
- If the assistant emits a tool call, it MUST emit ONLY the tool call block (no prose
  before or after) and stop.
- If the assistant has the information to answer, it emits prose only (no `<tool_call>`).
- One tool call per turn max.

The orchestrator enforces a hard cap of `rag.tool_use.max_iterations` (default 2,
config key) tool-call round trips per user turn. On cap exceeded, the last tool result
is injected as context and the assistant is prompted for a final answer with tools
disabled.

#### Tool specification block

Rendered into the system prompt when `mode in {"on", "only"}`. One tool only in this
phase (`search_documents`). Keep the block schema ready for additional tools later
(Phase 3 MCP tools).

```
## TOOLS

You have access to the following tools. Call a tool by writing a single
<tool_call>…</tool_call> block as your entire reply — no prose before or
after, no additional text.

### search_documents(query: str) -> list[Document]

Searches the user's personal corpus and returns up to N relevant chunks.
Each result has:
  - source_path: full NFS path (cite this in your answer)
  - score:       relevance score (0–1, higher = more relevant)
  - content:     chunk text (use this to answer)

Call this tool when you need information from the user's documents.
DO NOT call it for greetings, identity questions, or general knowledge.

Example invocation:
<tool_call>
{"name": "search_documents", "arguments": {"query": "AWS certifications"}}
</tool_call>
```

#### New module: `rag_engine/tool_use.py`

```python
"""System-prompt-driven tool-use for RAG. Provider-agnostic.

Exported functions:
  build_tool_spec()          → str   # TOOLS section for the system prompt
  parse_tool_call(text)      → dict | None   # extract first <tool_call> block
  format_tool_result(chunks) → str   # render chunks as a tool-role message
"""
import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

def build_tool_spec() -> str:
    """Return the TOOLS section string. Static for now (one tool)."""
    # ... literal string from the protocol section above ...

def parse_tool_call(assistant_text: str) -> dict[str, Any] | None:
    """Find first <tool_call>…</tool_call> block. Return {"name":..., "arguments":...}
    or None if no valid block found. Malformed JSON → None + log warning."""
    m = _TOOL_CALL_RE.search(assistant_text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        logger.warning("tool_call JSON decode failed: %s (text=%r)", e, m.group(1))
        return None
    if not isinstance(call, dict) or "name" not in call:
        return None
    call.setdefault("arguments", {})
    return call

def format_tool_result(chunks: list[dict]) -> str:
    """Render retrieved chunks into a tool-role message body.

    Compact JSON-ish format for the LLM; not the prompt the user sees.
    """
    if not chunks:
        return "search_documents returned no results."
    lines = [f"search_documents returned {len(chunks)} result(s):\n"]
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] score={c['score']:.4f}  source={c['source_path']}\n"
            f"{c['content']}\n"
        )
    return "\n".join(lines)
```

#### Pipeline loop in `main.py`

Replaces the current RAG block in `_server_managed_completion`:

```python
from prompts.rag_system_prompts import build_system_prompt
from rag_engine.tool_use import build_tool_spec, parse_tool_call, format_tool_result

# ... after step 3 (get_context from Redis) ...

# -- 4. System prompt (mode-aware) ----------------------------------------
tool_spec = build_tool_spec() if mode != "off" else None
system_prompt = build_system_prompt(mode, tool_spec)
# Prepend as role=system if not already present in history; otherwise
# replace the first system message with the computed one.
messages = _apply_system_prompt(messages, system_prompt)

# -- 5. Tool-use loop -----------------------------------------------------
max_iter = app.state.rag_cfg.get("tool_use", {}).get("max_iterations", 2)
rag_sources: list[str] | None = None
rag_chunks_used: list[dict] = []

for iteration in range(max_iter + 1):
    # Step 5a: run inference (internal via KV slot, or external via LiteLLM)
    assistant_text, model_used = await _run_inference(
        target_model, messages, cache, slot_id, req, router
    )

    # Step 5b: tool-call parse
    tool_call = parse_tool_call(assistant_text) if mode != "off" else None

    if tool_call is None or iteration == max_iter:
        # No tool call, or we've hit the cap — this is the final answer.
        assistant_content = assistant_text
        break

    # Step 5c: dispatch the tool
    if tool_call["name"] != "search_documents":
        logger.warning("Unknown tool call: %s", tool_call["name"])
        # Feed back as an error tool result so the model can recover.
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "content": f"Unknown tool: {tool_call['name']}"})
        continue

    query = tool_call["arguments"].get("query", "").strip()
    if not query:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "tool", "content": "Empty query"})
        continue

    chunks = await _retrieve(query, req.user_id, app.state)  # §8.3 hooks in here
    rag_chunks_used.extend(chunks)
    messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "tool", "content": format_tool_result(chunks)})

# Step 6: build rag_sources from *actual* tool results, dedup preserving order.
if rag_chunks_used:
    seen: set[str] = set()
    rag_sources = []
    for c in rag_chunks_used:
        if c["source_path"] not in seen:
            seen.add(c["source_path"])
            rag_sources.append(c["source_path"])
```

#### Specialist path considerations

The `internal` path today bypasses LiteLLM and calls llama-server `/completion`
directly with `slot_id` for KV-cache reuse. The tool-use loop must preserve that:

- Each iteration's inference call uses the same `slot_id`. llama-server appends the
  new prompt (which now includes the previous assistant message and tool result) and
  resumes from the cached prefix — no KV re-warm needed for the shared prefix.
- The `apply_gemma_template()` function formats messages using Gemma 4's native
  `<|turn>role\n…` markers (verified against llama-server `/props` chat_template
  2026-04-21). Stop tokens are `["<|turn>"]`. `tool` role messages are folded to
  `user` by `_fold_tool_messages()` before the template is applied; no template
  changes are needed for tool turns. The old `apply_mistral_template` (`[INST]…[/INST]`)
  is kept in `kv_cache.py` as a deprecated stub but must never be called — using
  Mistral format with Gemma 4 causes post-tool-call hallucination and broken multi-turn
  context (root-caused 2026-04-21).
- On KV-slot divergence (rare — e.g. template render difference between turns):
  `kv_cache.py` already handles cache miss by recomputing from scratch. No action needed.

#### External path considerations

LiteLLM passes messages through to Anthropic's Messages API. `role=tool` is mapped to
Anthropic's native `tool_result` content block by LiteLLM when `tools=[…]` is set on
the call — but **we are NOT using native tool-use**. Instead we keep the conversation
as plain user/assistant/tool text messages with the tool-call block in assistant text
and the tool result in a `role=user` message (since Anthropic only has user / assistant
roles without tools enabled). Update `_run_inference` for the external path to fold
`role=tool` → `role=user` with the `"Tool result:\n\n"` preamble, symmetric to the
internal path.

#### Rolling summarization interaction

`ConversationCache.maybe_summarize` triggers at ~100K tokens. The tool-use loop can
add ~2 × top_k × parent_text_size tokens per turn (~4K tokens with defaults).
Summarization already handles arbitrary message roles, but verify:
- Summarizer's input: raw messages from Redis. If a turn contains a `tool_call` block
  (assistant text) and a tool result (user text), summary should include "searched for
  X; found Y" rather than the raw JSON. Update the summarization prompt in
  `conversation.py` to instruct: "If a turn involved a tool call, summarize the query
  and gist of the result — not the raw JSON or source paths."

#### Config additions

```yaml
rag:
  tool_use:
    max_iterations: 2       # hard cap on tool-call round trips per user turn
    strip_on_final: true    # remove any stray <tool_call> tags from final answer
```

#### Verification

1. `rag_mode=off`: system prompt does not contain "TOOLS" section; no tool calls
   parsed; `rag_sources == null`.
2. `rag_mode=on` + meta question: no tool call emitted; `rag_sources == null`;
   latency is a single inference call (no retrieval).
3. `rag_mode=on` + document question: exactly one tool call emitted; retrieval runs;
   second inference consumes tool result; `rag_sources` populated from actual
   retrieval (not pre-fetched top-k).
4. `rag_mode=only` + answerable question: tool call emitted every time; final answer
   grounds in tool result.
5. `rag_mode=only` + unanswerable question: tool called, final reply is the exact
   decline sentence.
6. Malformed tool call (broken JSON) does not crash the pipeline; logged warning, loop
   continues with the assistant text as final answer.
7. Two-iteration cap honoured: if the model emits a second tool call on iteration 2,
   the orchestrator breaks and treats the assistant text as final (strip `<tool_call>`
   block per `strip_on_final: true`).

### 8.3 Multi-query retrieval with MMR reranking 🚫 evaluated 2026-04-22 — no-go, permanently disabled

> **Decision (2026-04-22):** Benchmarked against this corpus. Standard single-query retrieval is sufficient. `rag.multi_query.enabled` is permanently `false`. Code retained for reference only.

#### Purpose

Replace the single-query top-k retrieval in `search_rag()` with:
1. LLM-generated query expansion (N variants + original = N+1 queries).
2. Parallel Qdrant search for each, union by `point_id` (keep highest score).
3. MMR (Maximal Marginal Relevance) reranking over the union to balance relevance
   with diversity before returning the top-k to the LLM.

Directly addresses the narrow-margin bge-m3 observation (top relevant ≈ 0.54, top
noise ≈ 0.51, relevance-to-noise gap ≈ 0.04). Diverse rephrasings recover recall; MMR
removes the redundant near-duplicates that currently occupy 3 of 5 slots on typical
queries (e.g. the "cisco DCNIDS" case where 3/5 slots were the same PDF at different
chunk indices).

#### Algorithm

MMR definition (standard Carbonell & Goldstein 1998):

```
MMR = argmax_{d ∈ C \ S} [ λ · sim(q, d) − (1 − λ) · max_{d' ∈ S} sim(d, d') ]
```

- `q`: original user query vector (not variants — variants expand the candidate pool,
  but relevance is judged against the user's actual question).
- `C`: candidate pool from union of multi-query results.
- `S`: already-selected result set (starts empty).
- `λ ∈ [0, 1]`: relevance/diversity balance. `λ=1` → pure relevance (no MMR),
  `λ=0` → pure diversity. Default **`λ = 0.5`** (config-tunable).
- Similarity metric: cosine (same as Qdrant search).

Iterative selection until `|S| == top_k`. Returns `S` in selection order (the first-
picked chunk is the most relevant + novel; subsequent picks add diversity).

#### New module: `rag_engine/multi_query.py`

```python
"""Multi-query retrieval with MMR reranking.

Pipeline:
    expand_query()        → [original, v1, v2, ..., vN]
    parallel search_rag   → list[list[Candidate]]
    union()               → dedup by point_id, keep max score
    mmr_rerank()          → top-k reranked by λ-weighted relevance + diversity

Design notes:
  * Vectors are fetched from Qdrant (with_vectors=True) during search; no
    extra embedding round-trip for reranking.
  * Original query embedding is computed once and reused for both variant 0
    (search) and MMR relevance scoring.
  * Variant generation uses the internal model by default (fast, free);
    config key allows routing to external for higher quality.
"""
from __future__ import annotations

import asyncio
import logging
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from .collection import DOCUMENTS_COLLECTION, ensure_collection
from .embeddings import embed_text
from .schema import (
    FIELD_FILE_NAME, FIELD_FILE_TYPE, FIELD_OWNER,
    FIELD_PARENT_TEXT, FIELD_SOURCE_PATH,
)

logger = logging.getLogger(__name__)

# --- variant generation ---------------------------------------------------

_EXPANSION_PROMPT = """You are generating search-query variants to improve document
retrieval. Given a user question, produce exactly {n} alternative phrasings that
someone might use when searching for the same information. Vary the vocabulary,
specificity, and perspective. Keep each variant short (under 15 words).

Return ONLY the variants, one per line, no numbering, no commentary.

User question: {query}
"""

async def expand_query(query: str, n_variants: int, llm_fn) -> list[str]:
    """Generate n_variants alternative phrasings plus the original.

    Args:
        llm_fn: async callable(prompt: str) -> str. Supplied by caller so this
                module stays agnostic of internal/external routing.
    Returns:
        [original_query, variant_1, ..., variant_N]. If generation fails or
        returns fewer lines than requested, falls back to what was returned
        (never below just the original).
    """
    try:
        raw = await llm_fn(_EXPANSION_PROMPT.format(query=query, n=n_variants))
    except Exception as e:
        logger.warning("Query expansion failed: %s — falling back to original only", e)
        return [query]
    variants = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    variants = [v for v in variants if v.lower() != query.lower()][:n_variants]
    return [query] + variants

# --- retrieval + union ----------------------------------------------------

async def _search_one(
    vector: list[float], user_id: str | None, limit: int,
    qdrant_client: QdrantClient,
) -> list[dict]:
    """Single Qdrant query. Returns raw hits with vectors for MMR."""
    query_filter = Filter(
        must=[FieldCondition(
            key=FIELD_OWNER,
            match=MatchAny(any=[user_id, "la-familia"]),
        )]
    ) if user_id else None
    result = qdrant_client.query_points(
        collection_name=DOCUMENTS_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=True,   # needed for MMR
    )
    return [
        {
            "point_id":    hit.id,
            "vector":      hit.vector,                      # list[float]
            "score":       hit.score,
            "content":     (hit.payload or {}).get(FIELD_PARENT_TEXT, ""),
            "source_path": (hit.payload or {}).get(FIELD_SOURCE_PATH, ""),
            "file_name":   (hit.payload or {}).get(FIELD_FILE_NAME, ""),
            "file_type":   (hit.payload or {}).get(FIELD_FILE_TYPE, ""),
        }
        for hit in result.points
    ]

def _union_by_point_id(result_lists: list[list[dict]]) -> list[dict]:
    """Union multiple result lists, dedup by point_id, keep max score."""
    merged: dict[Any, dict] = {}
    for results in result_lists:
        for r in results:
            pid = r["point_id"]
            if pid not in merged or r["score"] > merged[pid]["score"]:
                merged[pid] = r
    return list(merged.values())

# --- MMR reranking --------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.dot(a, b) / denom)

def mmr_rerank(
    candidates: list[dict],
    query_vector: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict]:
    """MMR: iterative selection balancing relevance with diversity.

    Uses 'vector' and 'score' fields on each candidate. Returns a new list
    of the top_k picks in selection order (most-relevant-and-diverse first).
    """
    if not candidates:
        return []
    if top_k >= len(candidates):
        return sorted(candidates, key=lambda c: -c["score"])

    q = np.array(query_vector, dtype=np.float32)
    # Precompute query similarity (already in 'score' from Qdrant — use it)
    # but for variants the score is against the variant, not original q. For
    # MMR relevance we want sim(q, d), so recompute.
    vecs = {c["point_id"]: np.array(c["vector"], dtype=np.float32) for c in candidates}
    rel = {pid: _cosine(q, v) for pid, v in vecs.items()}

    selected: list[dict] = []
    remaining = {c["point_id"]: c for c in candidates}

    # First pick: highest relevance (no diversity term to evaluate yet)
    first_pid = max(remaining, key=lambda pid: rel[pid])
    selected.append(remaining.pop(first_pid))

    while len(selected) < top_k and remaining:
        best_pid = None
        best_mmr = -float("inf")
        for pid, cand in remaining.items():
            div = max(_cosine(vecs[pid], vecs[s["point_id"]]) for s in selected)
            mmr = lambda_ * rel[pid] - (1.0 - lambda_) * div
            if mmr > best_mmr:
                best_mmr = mmr
                best_pid = pid
        selected.append(remaining.pop(best_pid))

    # Attach the MMR-relevance score (not original Qdrant score) for visibility
    for c in selected:
        c["mmr_relevance"] = rel[c["point_id"]]
    return selected

# --- orchestrator ---------------------------------------------------------

async def multi_query_search(
    query: str,
    user_id: str | None,
    limit: int,
    qdrant_client: QdrantClient,
    rag_cfg: dict,
    llm_fn,
) -> list[dict]:
    """Drop-in enhanced replacement for search_rag() when multi_query is enabled.

    Shape of return value is identical to search_rag() except each dict also
    has 'point_id' and optionally 'mmr_relevance'. Existing callers (tool_use,
    _build_rag_prompt, rag_smoke_test) work unchanged — they only read
    'content', 'source_path', 'file_name', 'file_type', 'score'.
    """
    mq_cfg      = rag_cfg.get("multi_query", {})
    n_variants  = int(mq_cfg.get("n_variants", 3))
    lambda_     = float(mq_cfg.get("lambda", 0.5))
    pool_mult   = int(mq_cfg.get("pool_multiplier", 4))
    per_query_k = max(limit, pool_mult * limit // (n_variants + 1))

    # -- 1. Generate variants ------------------------------------------------
    queries = await expand_query(query, n_variants=n_variants, llm_fn=llm_fn)
    logger.info("multi_query: %d queries (1 original + %d variants)",
                len(queries), len(queries) - 1)

    # -- 2. Embed the user's ORIGINAL query once (for MMR relevance) --------
    model      = rag_cfg.get("embedding_model", "bge-m3")
    ollama_url = rag_cfg.get("ollama_url", "http://localhost:11434/api/embeddings")
    original_vector = await embed_text(query, model=model, ollama_url=ollama_url)

    # -- 3. Embed variants + parallel Qdrant search --------------------------
    async def _embed_and_search(q: str) -> list[dict]:
        vec = original_vector if q == query else await embed_text(q, model=model, ollama_url=ollama_url)
        return await _search_one(vec, user_id, per_query_k, qdrant_client)

    ensure_collection(qdrant_client)
    result_lists = await asyncio.gather(*(_embed_and_search(q) for q in queries))

    # -- 4. Union + MMR rerank -----------------------------------------------
    pool = _union_by_point_id(result_lists)
    logger.info("multi_query: pool size = %d (before MMR top_k=%d, λ=%.2f)",
                len(pool), limit, lambda_)
    return mmr_rerank(pool, original_vector, top_k=limit, lambda_=lambda_)
```

#### Integration with `main.py`

The `_retrieve(query, user_id, state)` helper that the §8.2 tool-use loop calls picks
between the two paths:

```python
async def _retrieve(query: str, user_id: str | None, state) -> list[dict]:
    mq_enabled = state.rag_cfg.get("multi_query", {}).get("enabled", False)
    if mq_enabled:
        return await multi_query_search(
            query=query, user_id=user_id, limit=state.rag_cfg.get("top_k", 5),
            qdrant_client=state.qdrant_client, rag_cfg=state.rag_cfg,
            llm_fn=state.variant_llm_fn,   # closure bound at app startup
        )
    return await search_rag(
        query=query, user_id=user_id, limit=state.rag_cfg.get("top_k", 5),
        qdrant_client=state.qdrant_client, rag_cfg=state.rag_cfg,
    )
```

`state.variant_llm_fn` is set in `main.py` lifespan startup. Both the internal and
external variants route through `SmartRouter.complete()` — no KV-slot management needed:

```python
_variant_model = rag_cfg.get("multi_query", {}).get("variant_model", "internal")

async def variant_llm_fn(prompt: str) -> str:
    resp = await app.state.router.complete(
        messages=[{"role": "user", "content": prompt}],
        model=_variant_model,
        force_cloud=(_variant_model == "external"),
        stream=False,
    )
    return resp.choices[0].message.content.strip()

app.state.variant_llm_fn = variant_llm_fn
```

**Why `router.complete()` instead of `/completion` + ephemeral slot (original plan):**
`SmartRouter.complete()` goes through LiteLLM to llama-server's `/v1/chat/completions`
endpoint, which is stateless with respect to KV slots — no slot management needed, no
risk of poisoning a persistent chat slot, no new `inference_ephemeral()` method to
implement. Variant generation is ~500ms either way; the KV-cache reuse benefit that
matters on the main chat path would be wasted here because the variant-gen prompt has
no shared prefix with the chat history. The `external` branch is identical to the
internal branch — both just pass `model=_variant_model` and let `SmartRouter` dispatch.

#### Config additions

```yaml
rag:
  multi_query:
    enabled:          false     # master toggle — start disabled; flip after bench
    n_variants:       3         # LLM-generated rephrasings (plus the original = 4 queries)
    lambda:           0.5       # MMR relevance/diversity balance [0, 1]
    pool_multiplier:  4         # candidate pool size = pool_multiplier * top_k
    variant_model:    internal  # "internal" or "external"
```

#### CLI changes

File: `utils/cli_chat.py`, `/rag status` output — ✅ implemented:

```
RAG: off  user=florian  top_k=5  multi_query=true (n=3, λ=0.50)  (98737 points)
```

(When `multi_query.enabled: false` the suffix is absent.)

File: `utils/rag_search_cli.py` — **TODO (not yet implemented)**: add `--multi-query`
flag that runs `multi_query_search()` instead of `search_rag()`. Print variants and
pool size in diagnostic output.

File: `utils/rag_smoke_test.py` — **TODO (not yet implemented)**: add `--multi-query`
flag symmetric to the above. Verdict section should print both pre-MMR pool size and
post-MMR selected size.

**Workaround**: use `utils/rag_mmr_bench.py` for complete multi-query evaluation (see
below). The two missing `--multi-query` flags are cosmetic gaps that do not block
production use; the feature gate (`rag.multi_query.enabled`) is fully functional.

#### Benchmark harness — `utils/rag_mmr_bench.py` ✅ implemented 2026-04-21

Go/no-go tool before enabling multi-query in production. Uses `utils/rag_bench_queries.txt`
(12 verified queries with expected path substrings).

```bash
# Full comparison: single vs multi, verdict + diversity stats
python utils/rag_mmr_bench.py --user florian

# Show variants generated by each model per query
python utils/rag_mmr_bench.py --user florian --show-variants

# Compare internal vs external variant LLM side-by-side
python utils/rag_mmr_bench.py --user florian --compare

# Tune λ (lower = more diversity, higher = more relevance)
python utils/rag_mmr_bench.py --user florian --lambda 0.3
```

Go/no-go criteria (hardcoded in the script):
- No recall regressions (multi ≥ single for every query)
- ≥ 2 queries gain recall from multi-query
- Max multi-query latency ≤ 2.5 s

#### Verification

1. `rag.multi_query.enabled: false` — existing behaviour preserved byte-identically.
2. `rag.multi_query.enabled: true`, `rag_mmr_bench.py --mode both`: output shows 4
   queries executed per query (original + 3 variants), pool size > limit, MMR prints
   `mmr_relevance` scores alongside standard scores.
3. Redundancy test (`rag_mmr_bench.py`): "dup" column shows `N/5 → M/5` where M < N
   for queries with high redundancy in single-query mode. At `λ=0.5` the Cisco DCNIDS
   query confirmed `3/5 → 2/5` diversity improvement (measured 2026-04-21).
4. Recall test (`rag_mmr_bench.py`): 12-query labelled set in `rag_bench_queries.txt`.
   Go = no regressions + ≥ 2 gains + max latency ≤ 2.5 s.
5. Latency: internal variant generation measured ~0.9 s per query; external ~2.6 s.
   Specialist stays well within 2.5 s threshold.
6. Failure mode: if `expand_query` fails (LLM timeout), `multi_query_search` falls
   back to a single original-query search — no user-facing error.
7. Provider comparison (`--compare`): showed Sonnet generates more domain-specific
   variants (e.g. "AWS MLA-C01 certified machine learning engineer" vs Gemma's
   "AWS ML certification exam"). Recall outcome identical at current corpus size;
   difference expected to surface on harder queries or larger corpora.

### 8.4 Contextual retrieval — design note, NOT to implement yet

**Origin:** Anthropic, "Contextual Retrieval" (2024-09,
<https://www.anthropic.com/news/contextual-retrieval>).

#### Concept

Ingestion-time technique. Before embedding each chunk, prepend a short LLM-generated
context string that explains where the chunk sits in its source document:

```
Original chunk:
    "Revenue grew 3% quarter-over-quarter."

Contextualised chunk (what gets embedded and stored):
    "This excerpt is from ACME Corp's Q2 2023 10-Q filing (SEC EDGAR);
    the prior quarter's revenue was $314M. Revenue grew 3% quarter-over-quarter."
```

Anthropic's published benchmarks show 35% reduction in retrieval failures when
combined with embedding search alone; 49% when combined with hybrid (embedding + BM25);
67% with reranking on top.

#### Why it is a strong fit for SoHoAI

The corpus is dominated by dense technical content (certifications, notebooks, work
docs) where individual chunks are meaningless without the surrounding document: a
code cell without its explanatory markdown, a "Chapter 4" table without the chapter
title, an exam answer without the question. The current narrow score margins
(0.51–0.55 range for relevant; 0.48–0.51 for noise; observed 2026-04-21) are a direct
consequence of embedding short decontextualised chunks. Prepending explicit context
before embedding should widen the relevant/irrelevant gap substantially.

#### Why NOT to implement now

Implementing contextual retrieval requires re-embedding all 98,737 points:

- One LLM call per child chunk (minimum). At 2891 files × ~34 chunks = 98,737 calls.
- On internal (~2s per ~200-token output × 2 slots parallel): ~27 hours.
- On Sonnet 4.6 with prompt caching of the full parent document per chunk cluster:
  ~3–5 hours, estimated cost $20–60 depending on parent size distribution.
- Full re-embed pass against bge-m3: ~9 hours at current ingest speed (already
  measured 2026-04-21).

Total serial wall time: 30–40 hours on internal, or ~10–15 hours on Sonnet (which
also spends money). Not appropriate until:

1. ✅ **Resolved 2026-04-22** — primary is now **external (Sonnet 4.6)** with prompt caching enabled. Contextualisation via Sonnet is cache-amortised across chunks of the same document, making it relatively cheap per-chunk. If contextual retrieval is revisited, use Sonnet and exploit prompt caching on the shared document prefix.
2. §8.1–§8.3 are implemented and the new baseline (mode=on + tool-use + multi-query +
   MMR) has been measured. Contextual retrieval's 35% retrieval-failure improvement
   is relative to a baseline without these improvements; the marginal gain on top of
   §8.1–§8.3 must be measured against the new baseline, not the 2026-04-21 numbers.
3. A benchmark harness exists (a fixed set of ≥ 30 known-answer queries with labelled
   expected sources) so the before/after delta can be measured, not guessed at.

#### What the implementation will look like (when approved)

- **Where**: `rag_engine/ingest.py`, between the chunking step (step 3) and the
  embedding step (step 4). New helper `_contextualise_chunks(parent_text, child_chunks,
  llm_fn) -> list[str]`.
- **Prompt** (per Anthropic, adapted):
  ```
  <document>
  {full parent chunk text}
  </document>
  Here is one sub-chunk of the document:
  <chunk>
  {child chunk text}
  </chunk>
  Give a short (1–2 sentence) context situating this chunk within the document,
  for the purpose of improving search retrieval. Answer ONLY with the context;
  do not repeat the chunk content.
  ```
- **Storage**: prepend context to the *text that is embedded*. Both text and
  `parent_text` in the Qdrant payload should remain the **original** (un-contextualised)
  chunk content — what gets returned to the LLM as RAG context is the original prose,
  not the contextualised version. This is Anthropic's published approach. Consequence:
  re-embedding is required; payload-only fields (parent_text, source_path, etc.) can
  stay as-is.
- **Prompt caching**: the parent chunk is identical across all its child chunks. When
  contextualising N children of the same parent, cache the parent block once and vary
  only the child. On Anthropic: use `cache_control: ephemeral` on the parent block.
  On internal: KV-cache prefix reuse via a pinned slot with `cache_prompt: true`
  (llama-server native).
- **Crash safety**: reuse the existing `ingestion_queue` with `rag_state.db`. Add a
  new column `contextualised: bool` so we can distinguish fully-processed rows from
  pre-contextual-retrieval completions; run the contextualisation pass as an
  incremental backfill rather than a flag-day re-ingest.
- **Hybrid search (BM25 + dense)**: Anthropic's published benchmark uses both. The
  Qdrant 1.10+ sparse-vector feature handles BM25 natively alongside dense vectors in
  the same collection. This is a natural add-on to contextual retrieval and should be
  designed as part of the same work package, not bolted on afterwards.

#### Decision gate

Before any implementation work begins on §8.4, require:
- Written outcome of the primary-model decision.
- §8.1–§8.3 shipped and measured.
- A benchmark harness with ≥ 30 labelled queries.
- Approval that a 10–40 hour ingest window is acceptable for the corpus (re-running
  against any new user's corpus once OAuth2 is live will take proportional time).

Nothing in this subsection should be implemented ahead of that gate.

---

## 9. Model tier policy (2026-04-22)

### Context

SoHoAI originally architected with Gemma 4 E4B on local RTX 5070 as primary interactive inference, with Claude Sonnet as fallback-only. This reflected two assumptions circa 2024:

1. Cloud API cost would be prohibitive for family-scale daily use (~50–100 interactive turns/day).
2. Quality gap between 4B local model and frontier cloud was narrow enough that local-first was defensible.

Both assumptions have since shifted:

- **Cost**: Sonnet 4.6 with prompt caching is ~$30–60/month at 50–100 turns/day — tolerable for a home lab. The "prohibitive cost" premise no longer holds.
- **Quality**: Gemma 4 E4B substantially lags Sonnet 4.6 on reasoning, code generation, and especially tool-use fidelity. Tool-use (RAG retrieval loop, §8.2) is core to this project, and a model that reliably emits well-formed tool calls is worth more than a fast-but-shaky local one.

### Decision: Flip primary to Sonnet 4.6, demote Gemma to internal roles

**Primary interactive chat default**: Sonnet 4.6 (external, cloud) — higher quality, prompt caching cost-efficient.

**Specialist (Gemma 4) roles**:
- **Fallback** — local inference when Anthropic API is unreachable; house still works offline
- **Summarization** — rolling summarization at ~100K token threshold; Gemma sufficient quality, cheaper than cloud
- **Multi-query variant expansion** — generate alternative query phrasings for RAG (§8.3); local, fast, low cost
- **Background/offline tasks** — RAG ingestion, RL data collection, maintenance jobs
- **Privacy-tagged content** (future) — conversations tagged to never leave local storage

### Mechanism

- **Config**: `routing.default_model: "external"` (was `"internal"`), `fallback_chain: ["external", "internal"]` (was reversed)
- **Router**: LiteLLM fallbacks dict: `{"external": ["internal"]}` (if Anthropic fails, try Gemma)
- **Summarization**: Explicitly config-driven at `routing.summarization_model: "internal"`; wired in `main.py:128`
- **Cold-resume**: Summary persistence to SQLite (new columns `summary_text`, `summary_covers_through_message_id`) allows reconstruction of summarized conversations on cold start from Redis expiry

### Cost impact

| Scenario | Cost/turn (Sonnet only) | Cost/turn (w/ caching) | Breaks even vs local infrastructure |
|---|---|---|---|
| 200 chars input + 200 output, cache miss | ~$0.004 | ~$0.002 | ~5000 turns = 50 days |
| 5K chars input + 200 output, cache hit | ~$0.024 | ~$0.003 | ~2000 turns = 20 days |
| 100+ turn chat session (prefix cache accrues) | accumulated $0.08/session | accumulated $0.01/session | n/a — cloud wins every time after turn 5 |

**Conclusion**: Prompt caching makes Sonnet 4.6 cost-competitive with Gemma 4 within 20–50 days of normal use. At 50–100 turns/day for a 4-user family, the payoff is achieved in 1–2 weeks. After that, cloud cost is **lower** than the sunk cost of the RTX 5070 and electricity to run it 24/7 for the fallback path.

### Fallback resilience

**Fallback chain is now reversed**: Anthropic outage → fall back to local Gemma, not the other way around.

If Anthropic API is down (502, timeout, rate-limited):
1. LiteLLM retries with exponential backoff (2 attempts)
2. Falls back to internal (Gemma 4 on Server 2)
3. User gets a response from local inference (may be lower quality, but functional)
4. No user-visible degradation once Anthropic recovers (next turn routes to Sonnet again)

If both fail (Anthropic + Server 2 down), orchestrator returns HTTP 502 — acceptable,  as this is the house is in severe infrastructure failure anyway (no local inference, no cloud access, no fallback).

### Design tension: Prompt caching vs summarization

Prompt caching and rolling summarization are mildly antagonistic:
- **Prompt caching** — LiteLLM maintains a rolling prefix cache of the system message + conversation context on Anthropic's servers. On each new turn, only the latest user message and assistant response are uncached input; everything before is read from cache.
- **Summarization** — when conversation exceeds ~100K tokens, `maybe_summarize()` re-builds Redis with `[system_msg, summary_msg, recent_turns]`, invalidating the entire prefix cache (summary is new, summary boundary is new).

**Impact**: Once per ~50-turn chat, a summarization event fires. That one turn has ~$0.05 extra cost (full non-cached input read). All subsequent turns on that same chat see a smaller cached prefix (from the summary point onward), saving on average ~$0.002 per turn. Payoff is achieved within 25 more turns. This is **not a performance problem** — it's a documented, accepted trade-off. The summarization threshold (400,000 chars ≈ 100K tokens) is deliberately chosen to stay well inside Gemma 4's 110K context window, so if Anthropic is down and we fall back to internal mid-chat, the conversation still fits.

### Tool-use on Sonnet vs Gemma fallback

**Sonnet 4.6 path**: Will eventually support native Anthropic tool-use (tool-use blocks in request → tool-use blocks in response). For now, tool-use still uses the `<tool_call>` XML sentinel (legacy, inherited from Gemma).

**Gemma 4 fallback path**: Stays on XML sentinel (`<tool_call>…</tool_call>` in both system prompt and response parsing). When fallback occurs mid-RAG-loop, the XML sentinel still works — `parse_tool_call()` in `rag_engine/tool_use.py` handles both paths.

### Future: premium/high-stakes mode

A future enhancement (not implemented): optional per-request flag `"model": "opus-4.7"` to route critical or review-heavy tasks to Opus 4.7 (most expensive). This would be explicitly opt-in by the user, logged, and clearly documented as a cost-incurring choice. Useful for Phase 4 RL review, sensitive document synthesis, or user-triggered "generate my best answer" mode.

---

## 10. RAG Ingestion Service (2026-05-05)

Automated replacement for manual `rag_sync_nfs.py && rag_ingest_daemon.py` invocations.
Implemented as a **systemd timer + oneshot service** with an NFS-backed distributed lock
preventing concurrent daemon instances across both servers.

### 10.1 Service architecture

| File | Role |
|------|------|
| `scripts/rag-ingest.timer` | Fires at 01:00, 07:00, 13:00, 19:00 local time; `Persistent=true` catches missed slots after reboot |
| `scripts/rag-ingest.service` | `Type=oneshot`; `Wants=network-online.target` (NFS safety at boot); `Environment=HOME=/home/florian` (reliable venv path) |
| `scripts/rag-ingest-run.sh` | Shell wrapper: multi-user sync loop → daemon |
| `scripts/rag-ingest-logrotate` | Daily rotation, 7 days, `copytruncate` |

The service is installed on one server only. Installing on both servers works (both timers
fire, one daemon wins the lock, the other skips) but doubles chatter in the log.

**systemd oneshot semantics**: the timer does not fire a second slot while the service is
still running from a previous slot. This prevents buildup of concurrent instances for free,
even before the lock is checked.

### 10.2 Shell wrapper — `scripts/rag-ingest-run.sh`

The wrapper is the sole entry point — invoked by the service and by operators for manual runs.

Key design choices:
- **No wrapper-level lock**: only the daemon holds the NFS lock. If two wrapper invocations
  race to the sync step, both syncs run (safe — idempotent). The second daemon then exits
  immediately on lock detection. Keeping one lock avoids the deadlock that would occur if the
  wrapper held a lock and the daemon (a subprocess) tried to acquire the same lock.
- **`RAG_SYNC_USERS` array**: one `rag_sync_nfs.py --user <u>` invocation per entry, run
  sequentially before the daemon. To add a second user, add one array entry.
- **Log path**: `/mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log` (NAS, alongside
  other SoHoAI databases). `mkdir -p` on startup creates the `logs/` subdirectory on first run.
- **Env overrides**: `RAG_WORKERS`, `RAG_BATCH`, `RAG_LOGFILE` — useful for one-off manual
  runs with different concurrency (e.g. `RAG_WORKERS=1 RAG_BATCH=5` for CPU-embed mode).

### 10.3 NFS distributed lock — `rag.ingest_lock`

**Config** (`config.yaml`): `rag.ingest_lock: "/mnt/nfs/__Backups/SoHoAI--databases/rag-ingest.lock"`

The daemon (`utils/rag_ingest_daemon.py`) acquires an exclusive lock on this NFS file at
startup, before any ingestion work begins. The lock prevents two daemon instances from
processing the same pending files simultaneously — which would otherwise produce **duplicate
Qdrant vectors** (see race condition analysis below).

**Why `fcntl.lockf()` not `fcntl.flock()`**:
Both use different kernel lock types on Linux. On this Synology NAS (NFSv4.1,
`local_lock=none`), `flock(LOCK_EX | LOCK_NB)` does not return `EAGAIN` immediately when
the lock is held — it blocks indefinitely (uninterruptible sleep, unkillable by SIGTERM).
`fcntl.lockf()` uses `F_SETLK` (POSIX record locks), which the NFS server honors correctly
as a non-blocking request, returning `BlockingIOError` in ~0–17s. The ~17s latency is the
NFS lockd response time and is acceptable for this use case (ingestion is a background job).

**Why NFS locking is safe here**: The NAS is mounted as NFSv4.1 with `local_lock=none`
(verified via `mount | grep nfs`). This routes flock/lockf calls to the NFS server's lock
manager rather than handling them locally — giving genuine cross-machine advisory locking.
This is distinct from the Qdrant/RocksDB NFS incompatibility (which involves RocksDB's
internal WAL lock patterns, not simple advisory file locks).

**Fallback path**: if `rag.ingest_lock` is absent from config, the daemon falls back to
`/tmp/rag-ingest.lock` (local, not cross-machine).

### 10.4 Race condition analysis

The critical race is two concurrent daemon instances processing the same files:

`fetch_pending_full()` (SELECT) is **not atomic** with `mark_processing()` (UPDATE).
Between these two calls, a second daemon can pick up the same rows. Both then:
1. Delete existing Qdrant points for the file (step 0 — idempotent)
2. Parse, chunk, embed, and upsert

Result: **duplicate Qdrant vectors** for every contested file, degrading cosine similarity
scores and search quality. The NFS lock prevents this entirely.

Other concurrency scenarios:

| Scenario | Risk | Outcome |
|----------|------|---------|
| Two `rag_sync_nfs.py` in parallel | Low | SQLite WAL + PRIMARY KEY serialize writes; idempotent |
| Sync while daemon processes a file | Low | If file mtime changed mid-processing: sync resets to `pending`; daemon calls `mark_completed()` overwriting; next sync re-queues | 
| Two daemons in parallel | **Critical** | Duplicate Qdrant vectors — prevented by NFS lockf |
| Manual daemon invocation while service active | Medium | Daemon sees lock held, prints "already running", exits |
| Cross-server conflict | Medium | NFS lockf blocks the second daemon on any machine |

### 10.5 Installation and operations

**Install (run once on target server):**
```bash
sudo cp scripts/rag-ingest.service /etc/systemd/system/
sudo cp scripts/rag-ingest.timer   /etc/systemd/system/
sudo cp scripts/rag-ingest-logrotate /etc/logrotate.d/rag-ingest
sudo systemctl daemon-reload
sudo systemctl enable --now rag-ingest.timer
```

**Check status:**
```bash
systemctl status rag-ingest.timer
systemctl list-timers rag-ingest.timer
journalctl -u rag-ingest.service -f          # live output from last/current run
```

**Manual trigger (same code path as timer):**
```bash
bash scripts/rag-ingest-run.sh
tail -f /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log
python utils/rag_status.py --watch /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log
```

**Debug (stop timer, run scripts individually):**
```bash
sudo systemctl stop rag-ingest.timer         # prevent auto-start during debug session
sudo systemctl stop rag-ingest.service       # stop any active run (SIGTERM; daemon finishes current file)

source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
python utils/rag_sync_nfs.py --user florian  # safe to run standalone, idempotent
python utils/rag_ingest_daemon.py --workers 1 --batch 5 \
    --log-file /tmp/rag-debug.log            # daemon acquires lock; second invocation exits cleanly

sudo systemctl start rag-ingest.timer        # re-enable when done
```

**Add a second user:**  
Edit `scripts/rag-ingest-run.sh`, line `RAG_SYNC_USERS=("florian")` → `RAG_SYNC_USERS=("florian" "eva")`.
No other changes needed — the wrapper loops over the array sequentially.

**CPU-embed mode** (if Ollama GPU on Server 2 is unavailable):
```bash
RAG_WORKERS=1 RAG_BATCH=5 bash scripts/rag-ingest-run.sh
```
Or set `Environment=RAG_WORKERS=1` / `Environment=RAG_BATCH=5` in the service unit and
`sudo systemctl daemon-reload`.

---

## 11. RAG Recovery and Rollback (2026-05-05)

### 11.1 Design goal

`rag_state.db` (SQLite, on NFS) records which files have been ingested and their status.
Qdrant's **active** vector store lives on local NVMe (`/var/lib/qdrant/storage`, Server 1
only — NFS file locking is incompatible with RocksDB). The NVMe is not independently
snapshotted; only Qdrant's **snapshot archives** (`.snapshot` files) are persisted to NFS.

For any rollback to be coherent, `rag_state.db` and the Qdrant snapshot archive must
represent the **same ingestion state** — the same set of files with `status=completed`.
This section documents how that consistency is maintained and how to exploit it for
recovery.

### 11.2 How consistency is guaranteed

After every ingestion run `scripts/rag-ingest-run.sh` executes two additional steps
(inserted after the daemon exits and before the final log line):

**Step A — SQLite WAL checkpoint**
```bash
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "PRAGMA wal_checkpoint(TRUNCATE);"
```
SQLite runs in WAL mode. Writes accumulate in `rag_state.db-wal`; readers merge the WAL
on the fly. The `TRUNCATE` checkpoint flushes the WAL into the main db file and clears it,
making `rag_state.db` **fully self-contained** — no `-wal` or `-shm` file is needed to
restore a consistent snapshot.

**Step B — Qdrant snapshot**
```bash
bash scripts/qdrant/qdrant-snapshot.sh --keep 12
```
Creates a new snapshot via `POST http://192.168.1.93:6333/collections/documents/snapshots`.
Qdrant's `snapshots_path` in `scripts/qdrant/qdrant-config.yaml` is
`/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots`, so the `.snapshot` file lands
directly on NFS under `…/qdrant-snapshots/documents/`. The script then deletes old
snapshots, keeping the 12 most recent (3 days × 4 runs/day).

**Result**: immediately after each ingest run both `rag_state.db` and the newest
`.snapshot` file on NFS reflect the same ingestion state. Any NFS point-in-time snapshot
taken *after* a run completes captures both files in sync.

### 11.3 Where everything lives under the backup directory

All relevant files are under a single directory that the NFS server snapshots automatically:

```
/mnt/nfs/__Backups/SoHoAI--databases/
├── sqlite/
│   ├── rag_state.db          ← ingestion queue (SQLite, WAL-checkpointed after each run)
│   ├── rag_state.db-wal      ← should be empty / absent after a run; safe to ignore on restore
│   └── chats.db              ← conversation history (unrelated to RAG rollback)
├── qdrant-snapshots/
│   └── documents/
│       ├── documents-<id>-<YYYY-MM-DD-HH-MM-SS>.snapshot          ← Qdrant archive
│       └── documents-<id>-<YYYY-MM-DD-HH-MM-SS>.snapshot.checksum ← SHA256
├── redis/                    ← Redis AOF persistence (unrelated to RAG rollback)
├── logs/
│   ├── rag-ingest.log        ← ingestion service log (rotation: daily, 7 days)
│   └── rag-ingest.lock       ← NFS advisory lock held by the running daemon
└── (qdrant/ is intentionally empty — not used)
```

The NFS server snapshots this entire directory tree on the following schedule:
- **Hourly**: 1 snapshot/hour, retained for 24 hours (24 recovery points)
- **Daily**: 1 snapshot/day, retained for 7 days (7 recovery points)
- **Monthly**: 31 daily snapshots retained (31 recovery points)

### 11.4 Identifying a matching pair at a given point in time

To roll back to time **T**, you need the `rag_state.db` and Qdrant snapshot that were
both written during the same ingest run, i.e., the run that completed closest to (but
before) T.

**Step 1 — Find the NFS snapshot at time T**

The Synology NAS exposes past snapshots as read-only directories (or via the Snapshot
Replication UI). Locate the snapshot volume for `/mnt/nfs/__Backups/SoHoAI--databases/`
at time T.

**Step 2 — Find the matching Qdrant snapshot**

Inside the NFS snapshot, list the Qdrant snapshot files:
```bash
ls -lt /path/to/nfs-snapshot-at-T/qdrant-snapshots/documents/*.snapshot
```
The filename encodes the creation timestamp:
`documents-<collection-id>-YYYY-MM-DD-HH-MM-SS.snapshot`

Choose the **most recent** `.snapshot` file whose timestamp is ≤ T. This is the Qdrant
archive that was written during the last ingest run before T.

**Step 3 — Confirm alignment with rag_state.db**

As a sanity check, compare the `completed_at` timestamp of the most recently ingested
file in `rag_state.db` with the Qdrant snapshot creation time — they should be within
seconds of each other (WAL checkpoint runs immediately before snapshot creation):
```bash
sqlite3 /path/to/nfs-snapshot-at-T/sqlite/rag_state.db \
  "SELECT MAX(completed_at) FROM ingestion_queue WHERE status='completed';"
```
If the Qdrant snapshot timestamp and this `completed_at` are within a few minutes,
the pair is consistent.

### 11.5 Full rollback procedure

Use this when Qdrant's local NVMe storage (Server 1) is corrupted or lost. `rag_state.db`
and the Qdrant `.snapshot` files are on NFS and are recoverable from any NFS snapshot.

#### Prerequisites
- SSH access to both Server 1 (192.168.1.93) and Server 2 (192.168.1.95)
- Qdrant binary installed on Server 1 (recoverable from package if lost)
- Access to the Synology snapshot browser (or NFS snapshot mount)

#### Step 1 — Stop the ingestion service (Server 2)

Prevents new data from being written to `rag_state.db` or Qdrant while restoring.
```bash
# On Server 2 (192.168.1.95)
sudo systemctl stop rag-ingest.timer rag-ingest.service
```
Confirm no daemon is running (the lock file should be absent or unlocked):
```bash
ls -la /mnt/nfs/__Backups/SoHoAI--databases/rag-ingest.lock 2>/dev/null || echo "no lock file"
```

#### Step 2 — Stop Qdrant (Server 1)

```bash
# On Server 1 (192.168.1.93)
sudo systemctl stop qdrant
```

#### Step 3 — Identify and mount the target NFS snapshot

Use the Synology Snapshot Replication UI or the NAS CLI to locate the snapshot at
(or just after) time T. Note the path to the snapshot's version of
`/mnt/nfs/__Backups/SoHoAI--databases/`.

Find the correct Qdrant snapshot and note its full filename:
```bash
ls -lt /path/to/nfs-snapshot-at-T/qdrant-snapshots/documents/*.snapshot
# Example: documents-8393594205839792-2026-05-05-01-32-10.snapshot
```

#### Step 4 — Clear Qdrant's local storage and restore (Server 1)

```bash
# On Server 1
sudo rm -rf /var/lib/qdrant/storage
sudo systemctl start qdrant

# Wait for Qdrant to be ready (~5–10 s)
until curl -sf http://192.168.1.93:6333/healthz; do sleep 2; done

# Restore the collection from the NFS snapshot file
curl -X PUT "http://192.168.1.93:6333/collections/documents/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///path/to/nfs-snapshot-at-T/qdrant-snapshots/documents/documents-...-YYYY-MM-DD-HH-MM-SS.snapshot"}'
```

Monitor recovery progress (for a 5+ GB snapshot this takes several minutes):
```bash
watch -n 5 'curl -sf http://192.168.1.93:6333/collections/documents | python3 -m json.tool | grep -E "status|points_count|optimizer"'
```
Wait until `status: "green"` and `optimizer_status: "ok"`.

#### Step 5 — Restore rag_state.db (any machine with NFS write access)

Copy only the main db file from the NFS snapshot. The `-wal` file is not needed — the
WAL was truncated to zero after the last run:
```bash
cp /path/to/nfs-snapshot-at-T/sqlite/rag_state.db \
   /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db
# Remove any stale WAL/SHM files that may be present
rm -f /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db-{wal,shm}
```

#### Step 6 — Verify consistency

```bash
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
cd /mnt/nfs/Florian/Gin-AI/projects/SoHoAI

# Queue counts — completed should match the Qdrant point count roughly
python utils/rag_status.py

# Qdrant collection health
curl -s http://192.168.1.93:6333/collections/documents | python3 -m json.tool | grep -E "status|points_count"

# Quick retrieval smoke test
python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS"
```

#### Step 7 — Re-ingest the gap (optional)

If files were ingested between the rollback point and the failure, re-sync to pick them
up. The daemon is idempotent — it re-ingests any file not already `completed` in the
restored `rag_state.db`:
```bash
python utils/rag_sync_nfs.py --user florian
python utils/rag_ingest_daemon.py --workers 3 --batch 20 \
  --log-file /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log
```

#### Step 8 — Re-enable the ingestion service (Server 2)

```bash
# On Server 2
sudo systemctl start rag-ingest.timer
systemctl list-timers rag-ingest.timer
```

### 11.6 SQLite-only rollback (Qdrant healthy)

Use this when `rag_state.db` is corrupted but Qdrant's NVMe storage is intact.

```bash
# 1. Stop ingestion on Server 2
sudo systemctl stop rag-ingest.timer rag-ingest.service

# 2. Restore rag_state.db from NFS snapshot
cp /path/to/nfs-snapshot-at-T/sqlite/rag_state.db \
   /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db
rm -f /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db-{wal,shm}

# 3. Reconcile: rag_sync_nfs.py will re-queue any file in SQLite marked 'completed'
#    that is absent from Qdrant, and remove SQLite rows for files no longer on NFS
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
cd /mnt/nfs/Florian/Gin-AI/projects/SoHoAI
python utils/rag_sync_nfs.py --user florian

# 4. Re-ingest any gaps
python utils/rag_ingest_daemon.py --workers 3 --batch 20 \
  --log-file /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log

# 5. Restart ingestion timer
sudo systemctl start rag-ingest.timer
```

### 11.7 Portability — moving snapshot ownership to another server

All snapshot automation (`rag-ingest-run.sh` tail steps + 03:00 cron) currently runs on
**Server 2** (192.168.1.95, NucBoxK11). The wrapper script is on NFS (shared), so only
the systemd unit activation and crontab entry are server-local.

To migrate to a different server:
```bash
# On Server 2 — disable the timer and remove the cron
sudo systemctl disable rag-ingest.timer
crontab -l | grep -v qdrant-snapshot | crontab -

# On the new server — enable the timer and add the cron
sudo systemctl enable --now rag-ingest.timer
(crontab -l 2>/dev/null; echo "0 3 * * * bash /mnt/nfs/Florian/Gin-AI/projects/SoHoAI/scripts/qdrant/qdrant-snapshot.sh --keep 12 >> /var/log/qdrant-snapshot.log 2>&1") | crontab -
```

Ensure the new server can reach Qdrant at `http://192.168.1.93:6333` and has the NFS
paths mounted. No code changes are required.

