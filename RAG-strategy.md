---
title: "HomeAI-Lab — RAG Strategy"
date: 2026-03-30
last_updated: 2026-04-17 (Qdrant server mode, NFS persistence, snapshot DR)
created_by: Florian Otel
last_updated_by: Claude Code (Claude Sonnet 4.6)
context: >
  HomeAI-Lab project (https://github.com/FlorianOtel/HomeAI-Lab);
  RAG pipeline design: embedding model, vector DB, chunking strategy,
  NFS corpus survey, Qdrant payload schema, multi-tenancy (Google OAuth2),
  rag_engine/ package layout, fail-safe ingestion (crash recovery, retry,
  delete-before-insert idempotency), Phase 2 implementation plan
---

# RAG Strategy — HomeAI-Lab

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

### 1.2 Exclusion filters (applied at ingestion time)

| Filter | Reason |
|--------|--------|
| `*/Gin-AI/.Gin-AI-python-3.12/**` | Python virtualenv (~46K `.py`, 37K `.pyc`, 15K `.h`/`.hpp`) |
| `**/*.pyc`, `**/*.so`, `**/*.mo` | Compiled artifacts |
| `**/*@synoeastream` | Synology NAS streaming metadata |
| `**/*.dist-info/**` | Package metadata |

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

### 3.1 Embedding model — bge-m3 via Ollama (updated 2026-04-17)

**Model**: `bge-m3` — 1024 dimensions, 570M params, ~1.2GB on disk.
Top MTEB retrieval scores; 8192-token context window.

**Server**: Ollama on **Server 1** (192.168.1.93) CPU.

**API**: `POST http://localhost:11434/api/embeddings`
No batch endpoint — `embed_batch()` runs 5 concurrent requests via asyncio semaphore.

**Config keys**:
```yaml
embedding_model: bge-m3
ollama_url: http://localhost:11434/api/embeddings
```

**Query latency**: ~650ms/chunk on Server 1 CPU.
**Batch ingestion**: async background job (~9 hours estimated for full NFS corpus).

**Why not Server 2?**
Server 2's RTX 5070 has only ~576 MiB free VRAM at idle (llama-server uses 11,642/12,227 MiB).
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
no shared process state. HomeAI-Lab uses `qdrant-client` directly — no LangChain default-name trap.

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
`/mnt/nfs/__Backups/HomeAI-lab--databases/qdrant/` was cleared of old SQLite data after
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
├── /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots/
│     ├── documents/
│     │     ├── documents-<id>-<timestamp>.snapshot      ← snapshot archive
│     │     ├── documents-<id>-<timestamp>.snapshot.checksum
│     │     └── ... (up to 3 kept)
│     └── (future: images/, videos/ as Phase 4 collections are added)
│
└── /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant/
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
  snapshots_path: /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots
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
2. GET  /collections/documents/snapshots  → list all snapshots, sorted by creation_time
3. DELETE oldest snapshots beyond KEEP=3  → via DELETE /collections/.../snapshots/{name}
```

All API calls use plain `curl` + `python3` (stdlib only, no extra dependencies).
Run manually: `bash scripts/qdrant/qdrant-snapshot.sh [--keep N]`

#### Recovery procedure

If Server 1 is rebuilt or `/var/lib/qdrant/storage` is lost:

```bash
# 1. Install Qdrant and start the service (empty storage)
sudo systemctl start qdrant

# 2. Identify the latest snapshot on NFS
ls -lt /mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots/documents/

# 3. Restore via the API (Qdrant reads the file path directly)
curl -X PUT "http://192.168.1.93:6333/collections/documents/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "file:///mnt/nfs/__Backups/HomeAI-lab--databases/qdrant-snapshots/documents/<snapshot-name>.snapshot"}'

# 4. Verify
curl -s http://192.168.1.93:6333/collections/documents | python3 -m json.tool
python utils/rag_status.py
```

After recovery, only documents ingested after the last snapshot are missing. Run
`rag_sync_nfs.py` (scanner detects mtime-changed files → resets to `pending`) followed by
`rag_ingest_daemon.py` to re-ingest the gap.

---

### 3.3 Document parsing — docling + dedicated ipynb parser (updated 2026-04-17)

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
negligible at this scale. Mistral Nemo's 53,248-token context window handles 800–1200 token
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

HomeAI-Lab serves a family of users, each with private NFS storage and a shared directory.
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
| Run initial NFS scan + data ingestion | ⏳ pending — see `RAG-ingestion-process.md` |
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

A dedicated SQLite database (`/mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/rag_state.db`) guarantees a fail-safe process that can pause and resume seamlessly.

#### `ingestion_queue` table schema

```sql
CREATE TABLE ingestion_queue (
    file_path       TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,           -- derived from NFS root at discovery
    last_modified   REAL NOT NULL,           -- os.path.getmtime() at discovery time
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | completed | failed
    error_msg       TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    started_at      TEXT,                    -- ISO timestamp; set when status → processing
    completed_at    TEXT,                    -- ISO timestamp; set when status → completed
    progress_detail TEXT                     -- e.g. "parsing", "embedding 34/120", "upserting"
);
```

**Status transitions:**
- `pending → processing` — worker picks up file, sets `started_at`
- `processing → completed` — all 7 steps succeeded, sets `completed_at`
- `processing → failed` — any step failed, writes `error_msg`, increments `retry_count`
- `failed → pending` — auto-retry if `retry_count < max_retries`; otherwise stays `failed` (permanent)
- `completed → pending` — re-discovery detects `last_modified` on disk > `last_modified` in SQLite

#### Crash recovery

On daemon startup, before entering the worker loop:
1. Query all rows where `status = 'processing'`
2. Reset them to `pending` (daemon was killed mid-file; work is incomplete)
3. Log which files were reset for operator visibility

This prevents files from being permanently stuck in `processing` after a crash, OOM, or NFS timeout.

#### Discovery function

Scans all configured NFS roots (per-user + shared), applies exclusion filters, derives `owner`
from path prefix via `schema.py:derive_owner()`, and:
- **New files:** inserts as `pending`
- **Modified files:** if `os.path.getmtime()` > stored `last_modified`, resets to `pending`
  (the worker loop handles Qdrant cleanup before re-ingestion — see §4.4 step 0)
- **Deleted files:** if a `completed` file no longer exists on disk, marks for Qdrant point
  deletion (filter by `source_path`) and removes the SQLite row

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
   `_parse_ipynb()` (ipynb), or direct UTF-8 read (MD/TXT/YAML/CSV).
   Update `progress_detail = "parsing"`.
3. **Generate Chunks:** Execute parent-child split logic (or flat 512-token chunks) entirely
   in memory. Update `progress_detail = "chunking ({n} chunks)"`.
4. **Vectorize:** Send child chunks to `bge-m3` via Ollama using 5-concurrent
   `asyncio` batching. Update `progress_detail = "embedding {i}/{n}"` periodically.
5. **Build Payloads:** Construct Qdrant point objects, binding the vector, UUID, and payload
   (`source_path`, `parent_text`, `owner` from SQLite row). All field names imported from
   `rag_engine/schema.py` constants — no string literals.
6. **Atomic Upsert:** Execute a *single* `client.upsert()` call to Qdrant containing all
   points for the document. Update `progress_detail = "upserting"`.
7. **Finalize State:** Upon successful upsert, update the SQLite row to `completed` with
   `completed_at` timestamp. If any step fails: write the exception to `error_msg`, increment
   `retry_count`, and set status to `pending` if `retry_count < max_retries`, else `failed`.

**Idempotency guarantee:** Step 0 (delete-before-insert) ensures that re-processing a file
— whether due to modification, crash recovery, or retry after partial failure — always
produces a clean result. Even if step 6 succeeds but step 7 fails (SQLite write error),
the next retry will delete the stale points before re-inserting.

**Point IDs:** Use random UUIDs (not deterministic hashes). Since step 0 deletes all
existing points for the file before inserting, there are no orphan-ID concerns even when
chunking changes produce a different number of chunks.

### 4.5 Standalone Utilities (`utils/`)

Standalone CLI scripts enable independent progress monitoring and RAG pipeline management without launching the FastAPI server:
* `utils/rag_sync_nfs.py`: Scans all configured NFS roots (per-user + shared), derives `owner` per file, applies filters, and populates SQLite with `pending` files. Accepts `--user florian` to scan a single user's root only.
* `utils/rag_ingest_daemon.py`: The worker loop executing the 7-step atomic embedding process (includes `owner` in every Qdrant payload).
* `utils/rag_status.py`: Dashboard querying SQLite/Qdrant to output ingestion metrics (`pending`, `completed`, `failed`). Accepts `--user` to filter by owner.
* `utils/rag_search_cli.py`: Query tester returning `parent_text` and cosine similarity scores. Requires `--user` flag to apply the ownership filter (simulates authenticated search).
* `utils/rag_reset.py`: Drops the Qdrant collection and resets SQLite to `pending` for clean re-ingestion. Accepts `--user` to reset a single user's documents only.

### 4.6 APIs for Control & Monitoring

The FastAPI orchestrator exposes the SQLite tracker state for client interfaces.
All endpoints require an authenticated user (Google OAuth2 JWT). Ingestion endpoints
are admin-only (Florian); search is scoped to the authenticated user's `owner` + `"la-familia"`.

* `POST /v1/rag/ingest/sync`: Triggers the NFS scanner across all configured roots.
* `POST /v1/rag/ingest/start`: Spawns the ingestion daemon as an asyncio background task.
* `POST /v1/rag/ingest/stop`: Gracefully halts the ingestion worker.
* `GET /v1/rag/ingest/status`: Returns metrics (`total_files`, `progress_percentage`, etc.) based on SQLite rows. Accepts optional `?user=florian` filter.



---

## 5. Phase 4 — Images and videos (future)

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

## 6. Use cases supported

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
