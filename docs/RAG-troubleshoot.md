---
title: "SoHoAI RAG Pipeline — Troubleshooting"
created_at: 20260422-000000
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: OpenCode (Claude Sonnet 5)
updated_at: 2026-09-13--21-55
context: >
  Consolidated RAG pipeline troubleshooting reference for SoHoAI.
  Originally two files: TROUBLESHOOTING.md (Qdrant timeout + project rename migration,
  sessions 2026-04-22 and 2026-04-30) and RAG-troubleshoot.md (Qdrant file-presence checks,
  session 2026-05-01). Merged 2026-05-01. Covers: Qdrant HTTP timeouts during bulk ingestion,
  re-queue after db_base_path rename, verifying specific files in the vector store,
  ignored file retry procedures, and clean restart procedures.
  Updated 2026-05-12: rag_sync_nfs.py sequential-delete timeout fix — wait=False on every
  delete call (fire-and-forget, no blocking on index re-optimization); get_client()
  timeout parameter added.
  Updated 2026-05-14: hybrid search activation — three qdrant-client API name mismatches
  (sparse_vectors attribute, Prefetch.filter, FusionQuery); corpus migration completed.
  Updated 2026-08-27: cross-encoder reranker had been down since the 2026-08-21 reboot —
  it was never a managed service. Now a llama-swap model reached at :8000/v1/rerank.
  Corrects two false claims in the 2026-05-14 reranker section (that it shared the chat
  llama-server process, and its Gemma-era VRAM budget) and records measured GPU tenancy:
  qwen3.5-9b-q4 (10,796 MiB) cannot coexist with the reranker.
  Updated 2026-09-13: recurring `history.jsonl` ingestion timeouts traced to two compounding
  bugs — a config exclude-pattern that never matched the real (symlinked) `.claude` dirname,
  and a missing Qdrant payload index on `source_path` that made every delete-before-insert an
  unindexed scan across the full (2.5M-point) collection. Refines the 2026-04-22 timeout
  diagnosis below: raising the client timeout treated the symptom at 473K points; the real
  fix at multi-million-point scale is the payload index.
---

# SoHoAI RAG Pipeline — Troubleshooting

---

## 2026-09-13 — Recurring `history.jsonl` Timeouts: Exclude-Pattern Miss + Missing Payload Index

### Symptom

`rag_ingest_daemon.py` repeatedly failed to ingest
`/mnt/nfs/Florian/Gin-AI/dot.claude-Linux/history.jsonl` with:

```
ERROR: Ingestion failed for .../dot.claude-Linux/history.jsonl: timed out
qdrant_client.http.exceptions.ResponseHandlingException: timed out
```

Most cycles logged `WARNING: Empty document, skipping` for the same file instead (harmless
but wasteful) — the timeout was intermittent, appearing whenever the file happened to be
re-queued during a slow Qdrant window. `grep -c "Ingestion failed.*timed out"` against
`rag-ingest.log` found **1,003** occurrences across dozens of unrelated files (`.md`, `.yaml`,
`.html`, `.docx`, `.jsonl`), confirming this was never specific to `history.jsonl` — that file
just had far more "at-bats" than any other (see root cause #1).

### Root cause #1 — `exclude_dir_names: ".claude/"` doesn't match the real on-disk dirname

`/home/florian/.claude` is a symlink → `/home/florian/Gin-AI/dot.claude-Linux` → (NFS)
`/mnt/nfs/Florian/Gin-AI/dot.claude-Linux`. `scan_nfs_roots()` walks the NFS root directly, so
`os.walk()` sees the real directory name `dot.claude-Linux`, never `.claude`.
`_is_excluded_dir()` (`rag_engine/scanner.py`) does exact trailing-slash name matching, so the
existing `.claude/` entry in `rag.scanner.exclude_dir_names` never fires for this tree.

Result: the generic NFS scanner freely ingested, as ordinary documents:
- `history.jsonl` — Claude Code's CLI *input-history* log (schema: `display`/`pastedContents`/
  `timestamp`/`project`/`sessionId` — not a session transcript). `_parse_claude_chat()` expects
  `type: user/assistant` entries, finds none, so text is always empty.
- `native-sessions/sessions.jsonl`, `native-sessions/telemetry.jsonl` — internal telemetry.
- `shell-snapshots/*.sh` — ephemeral files Claude Code deletes right after use (separate
  `[Errno 2] No such file or directory` errors for the same tree).
- `projects/**/*.jsonl` — the **same** session transcripts already correctly ingested via the
  dedicated `claude_chats.roots: /home/florian/.claude/projects` scanner, now duplicated a
  second time under the NFS path with a different `source_path`.

`history.jsonl` specifically is live-appended (every prompt Florian types), so its mtime
advances on almost every scan cycle — it's reset to `pending` and reprocessed far more often
than any static file, which is why it disproportionately hit the timeout window in root cause #2.

**Fix:** added `"dot.claude-Linux/"` to `rag.scanner.exclude_dir_names` in
`SoHoAI-config.yaml`. `scan_claude_chats()` is a separate function scanning
`/home/florian/.claude/projects` directly — unaffected by this change.

### Root cause #2 — no Qdrant payload index on `source_path`

`GET /collections/documents` returned `"payload_schema": {}` — no payload index existed on
any field, despite the collection having grown to **2,561,962 points** (vs. 98,737 at Phase 2
completion in April; the duplication in root cause #1 contributed to this growth). Every
ingestion's step 0 (`rag_engine/ingest.py::ingest_file()`) deletes stale points via:

```python
qdrant_client.delete(
    points_selector=FilterSelector(filter=Filter(must=[
        FieldCondition(key=FIELD_SOURCE_PATH, match=MatchValue(value=file_path))
    ]))
)
```

Without an index this is an unindexed linear scan across every point in the collection —
routinely exceeding the `QdrantClient(timeout=60)` set in `get_client()` (`rag_engine/
collection.py`) at this scale. This is a **refinement**, not a contradiction, of the
2026-04-22 entry below: at 473K points the 60s timeout bump was sufficient because Qdrant's
own index-optimization pauses were the bottleneck; at 2.5M+ points the *lack of a payload
index* became the dominant cost, and no client timeout increase fixes an O(n) scan that grows
with corpus size.

**Fix:** `rag_engine/collection.py::ensure_payload_indexes()` — called from `ensure_collection()`
on every process, cached per-process after the first check — creates a `KEYWORD` index on
`source_path` via `create_payload_index()` (idempotent; Qdrant no-ops if already present).
Verified on the live collection:

```bash
$ curl -sS http://192.168.1.93:6333/collections/documents | python3 -c "..."
payload_schema: {'source_path': {'data_type': 'keyword', 'points': 2561962}}
```

— built in the background across the full existing collection with no downtime.

### Verification

1. `python3 -c "from rag_engine.collection import get_client, ensure_collection; ..."` — index
   created and confirmed via `get_collection().payload_schema`.
2. `python utils/rag_sync_nfs.py --user florian` — with the exclude fix live, the scanner
   correctly flagged 2,629 now-excluded files under `dot.claude-Linux/` (including
   `history.jsonl`) as stale and queued fast, non-blocking Qdrant deletes for all of them
   (`wait=False`); point count dropped from 2,561,962 → 2,452,671.
3. `python utils/rag_ingest_daemon.py --workers 1 --batch 5` — drained the remaining queue
   (`pending: 0, failed: 0`) with no further `timed out` errors.
4. Ad-hoc `points/count` filtered query against `source_path` for `history.jsonl` returned in
   ~0.03s post-fix (vs. routinely >60s before).

### Known pre-existing, unrelated finding

`utils/rag_status.py` reports `"Qdrant: collection 'documents' does not exist yet"` even
though the collection is healthy — it checks `get_collections()` (real collection names only)
and doesn't resolve the `documents` → `documents_new` alias created by a prior
`rag_sparse_migrate.py --swap`. `ensure_collection()` already resolves aliases correctly; this
is a cosmetic bug in `rag_status.py`'s own existence check, pre-dating this session and out of
scope for this fix.

---

## 2026-08-27 — Reranker Down Since Reboot: llama-swap Now Owns Its Lifecycle

### Symptom

`curl http://192.168.1.95:8001/v1/rerank` → `Failed to connect ... Couldn't connect to server`.
Nothing was listening on port 8001 on Server 2. RAG kept answering, so nothing looked broken
from the outside — the gateway log just carried a steady drip of:

```
[rag_engine.rerank] WARNING: reranker failed: All connection attempts failed — falling back to Qdrant order
```

That is the designed fallback (RAG-strategy.md §13.5). No outage, only silently unranked results.

### Root cause

The reranker had **never** been a managed service. From its introduction (2026-05-14) it was a
hand-started `llama-server`:

```bash
llama-server -m .../bge-reranker-v2-m3-q8_0.gguf \
  -c 768 -b 768 -ub 768 --reranking --pooling rank --host 0.0.0.0 --port 8001
```

No systemd unit, no start script, no entry in `llama-swap--config.yaml` — nothing that would
ever restart it. Server 2 rebooted 2026-08-21 12:14; llama-swap was restarted by hand in tmux
at 23:28, the reranker was not.

### Fix — llama-swap manages it now

`bge-reranker-v2-m3` is a llama-swap model (backing port 8012), reached through llama-swap's
proxy at **`http://192.168.1.95:8000/v1/rerank`**. llama-swap routes `/v1/rerank` by the
request's `model` field, so the llama-swap model key must stay identical to `rag.rerank.model`
in `SoHoAI-config.yaml`. `rag.rerank.server_url` moved from `:8001` to `:8000` to match.

### GPU tenancy — measured, not estimated

Card is 12,227 MiB. Measured 2026-08-27:

| Resident | VRAM |
|---|---|
| `qwen3.5-4b-q6` @ 262144 ctx | 9,600 MiB |
| `bge-reranker-v2-m3` | 518–590 MiB |
| Ollama `bge-m3` embeddings (now on Server 2 — see `rag.ollama_url`) | 826 MiB |
| **total / free** | **11,040 MiB / 735 MiB** |
| `qwen3.5-9b-q4` @ 262144 ctx | **10,796 MiB** |

**The 9B and the reranker cannot coexist.** Verified in both directions: with the reranker
resident, the 9B dies with `ExitError >> signal: segmentation fault` during CUDA init; with the
reranker stopped, the same 9B loads in 5.4 s. The 9B plus Ollama's bge-m3 leaves 137 MiB free —
the reranker needs 518 MiB plus a CUDA context.

This differs from May, when the chat model was Gemma 4 E4B (~4.8 GB) and embeddings ran on
Server 1. Two later changes closed the gap: Qwen3.5-9B at the full 262144 ctx, and
`rag.ollama_url` moving to Server 2.

### Group policy

`llama-swap--config.yaml` encodes the constraint as two exclusive groups:

```yaml
groups:
  resident:                                    # default GPU tenants
    members: ["qwen3.5-4b-q6", "bge-reranker-v2-m3"]
    swap: false        # coexist — do not evict each other
    exclusive: true    # requesting either evicts the heavy group
  heavy:
    members: ["qwen3.5-9b-q4"]
    swap: true
    exclusive: true    # requesting the 9B evicts BOTH resident members
```

Both resident members are in `hooks.on_startup.preload`, so they come up with llama-swap.

**Known gap:** llama-swap v214 has no eviction or idle hook — `hooks.on_startup.preload` is the
only hook in its config schema. When the 9B's 300 s TTL expires the GPU is left empty and the
resident pair returns **lazily**, on the next request needing each one (~45 s cold load for the
4B over NFS, ~15 s for the reranker). Making that eager would need an external timer pinging
both endpoints.

### Checking and restarting the reranker

```bash
# Is it up? Go through llama-swap, NOT a direct port.
curl http://192.168.1.95:8000/v1/rerank -sS \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3","query":"test","documents":["foo"]}'

# What is llama-swap serving, and what is actually on the GPU?
curl -sS http://192.168.1.95:8000/v1/models
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Restart the lot — llama-swap runs in tmux window "llama-swap" on Server 2
tmux send-keys -t llama-swap:0.0 C-c
tmux send-keys -t llama-swap:0.0 'source ./llama-swap--start.sh' Enter
```

llama-swap is itself only started by hand in tmux, so a Server 2 reboot still leaves the whole
stack down until someone reattaches. That remains unfixed.

### Gotcha — the gateway does not hot-reload config edited over NFS

After `SoHoAI-config.yaml` was edited, the gateway kept calling the old `:8001`. `main.py` reads
the config once at import (~line 138) and exposes no reload endpoint; `uvicorn --reload` watches
via inotify, which **does not fire for NFS writes made from another host**. Editing the config
from Server 2 leaves Server 1's gateway on stale values indefinitely. After any remote config
edit, restart the gateway on Server 1 (or `touch` the file from Server 1 to fire its watcher).

---

## 2026-05-14 — Hybrid Search Activation: qdrant-client API Name Mismatches

Three bugs were discovered when hybrid queries first fired against the migrated corpus.
All are qdrant-client API mismatches — names that differ between documentation/examples
and the installed library version.

### Bug 1 — `collection_has_sparse()` always returns False

**Symptom:** `python utils/rag_sparse_migrate.py status` shows `sparse vectors: ✗ NO`
even after a successful migration. Hybrid search silently falls back to dense-only.

**Root cause:** `CollectionParams` attribute is named `sparse_vectors` in the Qdrant
Python client, but the code checked `sparse_vectors_config` (which doesn't exist, so
`getattr(..., None)` always returns `None`).

**Fix** (`rag_engine/collection.py`, `utils/rag_sparse_migrate.py`):
```python
# Wrong:
sparse = getattr(info.config.params, "sparse_vectors_config", None)

# Correct:
sparse = getattr(info.config.params, "sparse_vectors", None)
```

**Verification:**
```bash
python utils/rag_sparse_migrate.py status
# → sparse vectors: ✓ YES

# Or directly via REST:
curl -s http://192.168.1.93:6333/collections/documents_new | python3 -m json.tool | grep -A3 sparse_vectors
# → "sparse_vectors": { "sparse_text": { "modifier": "idf" } }
```

### Bug 2 — `Prefetch` rejects `query_filter` kwarg

**Symptom:** `pydantic_core.ValidationError: Extra inputs are not permitted` on any
hybrid search call. Traceback points to `Prefetch` model construction.

**Root cause:** `Prefetch` Pydantic model uses `filter` (not `query_filter`) for the
ownership/file-type filter parameter.

**Fix** (`rag_engine/search.py`):
```python
# Wrong:
Prefetch(query=vector, using="", limit=30, query_filter=query_filter)

# Correct:
Prefetch(query=vector, using="", limit=30, filter=query_filter)
```

### Bug 3 — `query=Fusion.RRF` causes Qdrant 400

**Symptom:** `qdrant_client.http.exceptions.UnexpectedResponse: 400 Bad Request —
"Expected some form of vector, id, or a type of query"` deep in the request JSON.

**Root cause:** `query_points()` `query` parameter accepts `FusionQuery` (a Pydantic
model), not the raw `Fusion` enum value.

**Fix** (`rag_engine/search.py`):
```python
from qdrant_client.models import FusionQuery

# Wrong:
query_points(..., query=Fusion.RRF)

# Correct:
query_points(..., query=FusionQuery(fusion=Fusion.RRF))
```

**Post-fix smoke test:**
```bash
python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS"
# → PASS
```

---

## 2026-05-14 — Cross-Encoder Reranker: What to Check When It Misbehaves

### Reachability

Check that the reranker server on Server 2 is responding to HTTP requests:

```bash
# NOTE: port 8001 is historical. Since 2026-08-27 the reranker is served through
# llama-swap on port 8000 — see the 2026-08-27 entry at the top of this file.
curl http://192.168.1.95:8000/v1/rerank -sS \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3","query":"test","documents":["foo"]}'
```

Expected response: `{"results":[{"index":0,"relevance_score":<float>}]}`. If you get
a connection timeout or "connection refused," the reranker is down.

### Reranker failure fallback

If the orchestrator logs show `[WARNING] reranker failed: ...`, the reranker is either
unavailable or timing out. **No action is required for chat continuity:** the orchestrator
automatically falls back to Qdrant cosine-ordered results. The WARN line is for debugging
only; users do not see an error.

To get reranking back: restart the reranker process on Server 2 (see below) and re-run
the failing query.

### Context limits and batch size

The bge-reranker-v2-m3 model on Server 2 is now launched with `-c 768 -b 768`, ensuring
both context and physical batch sizes are consistently 768 tokens. This eliminates the
previous 512-token physical batch limit issue.

**Previously** (before 2026-05-14): The physical batch limit was 512 tokens, causing HTTP 500
errors on dense CSV content or long child chunks. Client-side truncation (query: 64 tokens +
doc: 380 tokens) was used as a workaround.

**Now**: The server handles all typical queries and child chunks without truncation. No client-side
truncation is needed or performed.

To verify reranking is working: run `python utils/rag_search_cli.py --query '$CWD blindly' --user florian`
and check that `rerank_score` is present in the output (indicates successful reranking).
If HTTP 500 is still returned, check that Server 2's llama-server was restarted with the new `-b 768` flag.

### Reranker VRAM usage

> **Superseded 2026-08-27 — the numbers below are for a configuration that no longer exists.**
> They were measured against Gemma 4 E4B (~4.8 GB) with 2 KV slots and embeddings running on
> Server 1. The chat model is now Qwen3.5-4B at 262144 ctx (9,600 MiB) and embeddings run on
> Server 2. For current measured figures, and for why the 9B cannot share the card with the
> reranker, see the 2026-08-27 entry at the top of this file.

Historical (2026-05-14), both models resident alongside Gemma 4 E4B:
- Gemma 4 E4B Q8_0: ~4.8 GB
- bge-reranker-v2-m3: ~0.5 GB
- KV cache slots (2 × 110024 ctx): ~1.76 GB
- Overhead: ~1 GB
- **Total: ~9.3 GB / 12 GB available**

If you observe OOM or a segfault during CUDA init, check:
1. What is actually on the card: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
2. Whether `qwen3.5-9b-q4` is resident — it needs 10.8 GB and cannot share with the reranker.
3. Whether Ollama's `bge-m3` is holding VRAM on Server 2: `curl -sS http://192.168.1.95:11434/api/ps`

### Reranker process restart

> **Corrected 2026-08-27.** This section previously claimed the reranker was "served by the same
> process via model routing" as the chat endpoint, and that restarting the chat `llama-server`
> would bring port 8001 back. **That was never true** — `llama-server` serves exactly one model
> on one port, and the reranker was always a separate, hand-started process. Following the old
> procedure would never have restored reranking.

The reranker is now a llama-swap-managed model. To check and restart it, see
"Checking and restarting the reranker" in the 2026-08-27 entry at the top of this file.

---

## 2026-05-12--20-00 — RAG Search Quality: Corpus Cleanup, Search Improvements, Snapshot Fix

### Context and root cause

Symptom: searching for the literal phrase `$CWD blindly` (a specific code-pattern
mention from a claude-orchestra session) via `rag_search_cli.py` and `cli_chat.py`
returned only irrelevant results — Nokia/Scan PDF pages, COI-EMEA CSV rows. The
same corpus contained the relevant content but it never surfaced.

**Root cause 1 — zombie vectors from old `handle_deleted()` race condition:**

The old `handle_deleted()` method in `rag_engine/state.py` had the wrong crash-safe
ordering: it purged SQLite rows **before** the caller had a chance to delete the
corresponding Qdrant vectors. Any crash (OOM, kill, network drop) between those two
steps left vectors in Qdrant with no SQLite row — "zombie vectors." `find_deleted()`
can never detect these because they have no SQLite row to compare against. They
accumulate silently with every crash.

In this corpus (1.46M points), ~290K points (~20%) were zombies. The largest
offenders by point count: `Dave_Snowden--SAFe-podcast.pdf` (46,980 pts), Nokia/Scan
PDFs (~80K pts total), COI-EMEA customer CSVs (~110K pts).

**Root cause 2 — corrupt binary content produces noise embeddings:**

Several of these zombie vectors came from files that docling parsed as binary garbage
(65–88% printable chars). Their pseudo-random dense embeddings happened to score
0.55–0.58 cosine similarity for many query embeddings, consistently crowding out
legitimate content (which scored in the same 0.51–0.58 range). With `top_k=5` and
no score threshold, these zombies filled all 5 result slots.

**Root cause 3 — `SNAPSHOTS_NFS_DIR` never used in snapshot script:**

`scripts/sqlite-qdrant-snapshot.sh` defined `SNAPSHOTS_NFS_DIR` but never used it.
Snapshots were created and rotated only in Qdrant's local NVMe on Server 1 — zero
DR copies on NFS. Cron was running (daily 01:00 UTC from Server 2), but all 12
retained snapshots were on NVMe only, invisible on NFS.

---

### What was done (in sequence)

1. **Identified corrupt files** — Scrolled all 1.46M Qdrant points, checked
   printable-char ratio of `parent_text`, found 194 unique corrupt source_paths
   with ~290K total corrupt points. Confirmed the high-impact ones were zombie
   vectors (NOT in SQLite).

2. **Corpus cleanup** — Renamed or deleted the corrupt files from disk (`.noRAG`
   extension trick for files the sync pipeline could detect; direct Qdrant deletion
   via `rag_purge_corrupt.py --confirm` for zombie vectors not in SQLite).

3. **Ran `rag_sync_nfs.py`** — Detected the renamed/deleted files as gone-from-disk,
   cleaned their SQLite rows and residual Qdrant points (for the 8 files that _were_
   in SQLite).

4. **SQLite corruption** — `rag_purge_corrupt.py` originally used `StateDB()` for
   cross-reference lookups. The `StateDB` constructor writes to the WAL (PRAGMA +
   schema migrations) even for read-only use. Running this over NFS while the Qdrant
   optimizer was mid-vacuum corrupted the WAL (B-tree page 1324 malformed).
   **Recovery:** restored `rag_state.db` from NFS snapshot (20:38 mtime, before
   corruption at ~21:04), verified with `PRAGMA integrity_check`, then re-ran
   `rag_sync_nfs.py` to re-detect the 8 cleaned files.

5. **Fixed `rag_purge_corrupt.py`** — Replaced `StateDB` with a direct
   `sqlite3.connect(uri, mode=ro)` connection. Zero WAL writes from this utility.

6. **Implemented search improvements** — score_threshold, expanded internal pool,
   multi-query opt-in for the search endpoint and CLI.

7. **Fixed snapshot script** — Now downloads each snapshot to NFS, deletes the
   Qdrant-local copy, rotates NFS files by mtime.

---

### Tools and utilities — created or modified

#### `utils/rag_purge_corrupt.py` (NEW — commit `94f9d56`)

Scans all Qdrant points for corrupt content (low printable-char ratio) and optionally
deletes them directly from Qdrant by `source_path` filter.

**When to use:**
- After any bulk corpus operation (mass deletes, re-indexing) to verify no garbage
  vectors crept in.
- When RAG search returns irrelevant results that look like binary garbage (unusual
  characters, scanning artifacts, CSV row noise).
- To identify and purge zombie vectors — Qdrant points that have no SQLite row
  (i.e., `NOT_IN_DB` in the report). These cannot be cleaned by `rag_sync_nfs.py`.
- Periodically as a corpus health check (e.g., after a crash during ingestion).

**How to run:**

```bash
# Dry-run (default) — scan and report, no changes
python utils/rag_purge_corrupt.py --dry-run --user florian

# Confirm deletion — directly delete corrupt Qdrant vectors
# Skips paths where the original file still exists on disk (safety guard)
python utils/rag_purge_corrupt.py --confirm

# Force deletion even if file exists on disk (e.g. re-ingested with bad content)
python utils/rag_purge_corrupt.py --confirm --force

# Lower the threshold (default 0.85) to catch marginally corrupt content
python utils/rag_purge_corrupt.py --threshold 0.75 --dry-run

# Save corrupt path list to file for batch processing
python utils/rag_purge_corrupt.py --save /tmp/corrupt_paths.txt

# Scan all users (omit --user to search entire corpus)
python utils/rag_purge_corrupt.py --dry-run
```

**Output columns:**
```
[N pts]  IN_DB / NOT_IN_DB   X.X%   /full/source/path
```
- `IN_DB` — file is tracked in SQLite (can be cleaned by rag_sync_nfs.py after disk deletion)
- `NOT_IN_DB` — zombie vector; only `--confirm` can remove it

**Important constraints:**
- Uses `sqlite3.connect(mode=ro)` — **never writes to SQLite**. Safe to run while
  the ingest daemon is active.
- `--confirm` deletes from Qdrant with `wait=True`, so it's synchronous but slow for
  large batches. Use during off-peak hours.
- Retry logic: retries up to 5× with 10s backoff on Qdrant HTTP 500 (optimizer lock).
  Qdrant can return "timed out after 0ns" during active segment merges — the retry
  handles this without crashing.
- Do NOT run during a Qdrant optimizer `yellow` burst (large active merges). Wait for
  `green` status or use `--watch` in `qdrant_status.py`.

---

#### `utils/qdrant_status.py` (NEW — commit `5d8efd1`)

Displays Qdrant collection status, optimizer lag, and segment info. Replaces ad-hoc
`curl | python3 -c` one-liners for monitoring the optimizer after bulk operations.

**When to use:**
- After any bulk delete operation (zombie purge, `rag_sync_nfs.py` with many
  deletions) to monitor the HNSW optimizer rebuild.
- Before running `rag_purge_corrupt.py --confirm` to confirm Qdrant is idle (`green`).
- After a `yellow` status to track optimizer progress.
- As a quick health check: `python utils/qdrant_status.py` exits 0 if green, 1 if not.

**How to run:**

```bash
# One-shot status check
python utils/qdrant_status.py

# Show optimizer configuration details (deleted_threshold, indexing_threshold, etc.)
python utils/qdrant_status.py --verbose

# Watch continuously, poll every 10s (default), stop when green+lag=0
python utils/qdrant_status.py --watch

# Poll every 30s
python utils/qdrant_status.py --watch 30

# Use as a script pre-flight check (exits 0=green, 1=yellow/red)
python utils/qdrant_status.py || echo "Qdrant not fully optimized — wait before running purge"
```

**Output interpretation:**
```
[22:11:24]  ✓ GREEN
  Points:      946,129  (live)          ← what search actually queries
  Indexed:   1,112,326  (HNSW, ...)    ← HNSW graph size (incl. soft-deleted)
  Lag:         166,197  (14.9%)        ← soft-deleted vectors pending vacuum
  Segments:          8
```
- **green + lag=0**: fully optimized, ideal state for search and maintenance ops.
- **green + lag>0**: optimizer idle, lag below vacuum threshold (~20% per segment).
  Search quality slightly degraded but functional. Resolves naturally on next write.
- **yellow**: optimizer actively running a merge. Reads may hit `timed out after 0ns`
  errors on locked segments. Wait for green before bulk operations.

**Why lag doesn't always reach 0:**
Qdrant's vacuum triggers when a segment has ≥`deleted_threshold` (20%) of its
vectors soft-deleted. If deletions are spread across segments such that no segment
reaches 20%, the optimizer stays idle — lag is stable and non-zero. This is benign.
New ingestion writes trigger compaction organically.

---

#### `rag_engine/ingest.py` — ingest quality gate (MODIFIED — commit `94f9d56`)

Added a printable-char-ratio check in `ingest_file()` after document parsing (step 2),
before chunking and embedding.

```python
_printable = sum(1 for c in text if c.isprintable()) / max(len(text), 1)
if _printable < 0.85:
    logger.warning("Skipping %s — parsed text is %.0f%% printable (binary/corrupt)", ...)
    state_db.mark_ignored(file_path, f"binary content: {_printable:.0%} printable chars")
    return
```

**When this fires:**
- Scanned PDFs that docling extracts as binary garbage (image-only pages, encrypted docs)
- Binary files with document extensions (`.pptx` with OLE2 binary, etc.)
- Corrupted files where the content is mostly encoding artifacts

**Effect:**
- File is marked `ignored` in SQLite with reason `"binary content: XX% printable chars"`
- No Qdrant points are created — no garbage vectors enter the index
- Visible in `rag_status.py --ignored`

**How to retry** if a file was incorrectly flagged (e.g., encoding was fixed):
```bash
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "UPDATE ingestion_queue SET status='pending', retry_count=0, skip_reason=NULL
   WHERE skip_reason LIKE 'binary content:%' AND file_path = '/path/to/file';"
```

---

#### `rag_engine/search.py` — score threshold + expanded pool (MODIFIED — commit `94f9d56`)

Two improvements to `search_rag()`:

**1. score_threshold parameter:**
```python
async def search_rag(..., score_threshold: float = 0.0) -> list[dict]:
```
Passed directly to Qdrant's `query_points(score_threshold=...)`. Qdrant filters
out points below this threshold before returning results. Default `0.0` = no filter
(backward compatible). Useful range: `0.45`–`0.55` for this corpus.

**2. Expanded internal search pool:**
Internally fetches `min(limit * 3, 50)` candidates from Qdrant, then slices to
`limit` after threshold filtering. This means a `top_k=5` request actually asks
Qdrant for up to 15 candidates — improving the chances of surfacing relevant content
that would have been at rank 6–10 with a tight top_k.

**When to tune score_threshold:**
- If results contain clearly irrelevant docs at low scores: raise threshold to 0.50–0.55
- If relevant results are missing: lower threshold or remove it (0.0)
- After a corpus cleanup (corrupt vectors removed): scores for real content are now
  uncontested, so lower thresholds are safe without noise

---

#### `utils/rag_search_cli.py` — new flags (MODIFIED — commit `94f9d56`)

Added `--score-threshold` and `--multi-query` flags:

```bash
# Filter results below score 0.50
python utils/rag_search_cli.py --query "your query" --user florian --score-threshold 0.50

# Enable multi-query expansion + MMR reranking (uses internal Gemma on Server 2)
# Gemma generates 3 alternative phrasings of the query, runs 4 parallel Qdrant searches,
# unions results, then MMR-reranks for relevance + diversity
python utils/rag_search_cli.py --query "your query" --user florian --multi-query

# Combine both
python utils/rag_search_cli.py --query '$CWD blindly' --user florian \
    --score-threshold 0.45 --multi-query
```

**When to use `--multi-query`:**
- When a single-phrasing search misses the target (e.g., `$CWD blindly` vs
  `blindly uses $cwd` — same concept, different tokens)
- For jargon-heavy or code-specific queries that dense embedding handles poorly
- Latency cost: ~1–2 extra seconds (Gemma expansion on Server 2 GPU, ~0.9s typical)

**When NOT to use `--multi-query`:**
- For broad semantic queries that already return good results — adds latency with no benefit
- If Server 2 (192.168.1.95) / llama-server is offline

---

#### `main.py` — `/v1/rag/search` new query params (MODIFIED — commit `94f9d56`)

`GET /v1/rag/search` accepts two new optional query parameters:

```bash
# Score threshold
curl "http://192.168.1.93:8000/v1/rag/search?q=your+query&user=florian&score_threshold=0.50"

# Multi-query expansion
curl "http://192.168.1.93:8000/v1/rag/search?q=your+query&user=florian&multi_query=true"
```

These are used by the `/user:rag` slash command in Claude Code and by any external
client calling the search endpoint directly.

---

#### `scripts/sqlite-qdrant-snapshot.sh` — NFS download fix (MODIFIED — commit `94f9d56`)

**Old behavior (broken):** `SNAPSHOTS_NFS_DIR` was defined but never used.
Snapshots were created and rotated in Qdrant's local NVMe storage only. Zero DR copies
on NFS. If server1's NVMe failed, all 12 rotating snapshots were lost.

**New behavior:**
1. Creates snapshot via `POST /collections/{COLLECTION}/snapshots` (unchanged)
2. Downloads the snapshot to NFS via `curl --max-time 3600 ...`
3. Verifies the downloaded file is non-empty
4. Deletes the Qdrant-local copy via `DELETE /collections/{COLLECTION}/snapshots/{NAME}`
5. Rotates NFS files by mtime (keeps last `KEEP`, default 3; crontab passes `--keep 12`)

The `--max-time 3600` handles the 5GB+ snapshot size (takes several minutes over
LAN). The temp file (`${NAME}.tmp`) is atomically renamed to `${NAME}` after
successful download, preventing partial files from appearing as valid snapshots.

**Verification after the fix:**
```bash
bash scripts/sqlite-qdrant-snapshot.sh --keep 12
ls -lh /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/
# → should show a fresh .snapshot file with today's date and ~5GB size
```

---

### SQLite corruption incident (2026-05-12 ~21:04–21:45)

**What happened:** `rag_purge_corrupt.py --confirm` called `StateDB(db_path)` for
SQLite cross-reference lookups. The `StateDB` constructor writes to the SQLite WAL
(via `PRAGMA journal_mode=WAL`, `CREATE TABLE IF NOT EXISTS`, and migrations) even
when only reads are needed. Running this over NFS while the Qdrant optimizer was
mid-vacuum caused WAL corruption: B-tree page 1324 malformed, wrong entry count
on `sqlite_autoindex_ingestion_queue_1`. `PRAGMA integrity_check` failed, `REINDEX`
failed, `.dump` terminated with `ROLLBACK`.

**Root cause:** The `StateDB` constructor is not safe to call from utilities that
only need to read SQLite — it always writes, and NFS WAL writes during concurrent
activity are fragile on this Synology setup.

**Fix applied:** `rag_purge_corrupt.py` now uses a direct read-only connection:
```python
state_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
```
This opens the database as truly read-only at the OS level — no WAL writes,
no schema migrations, no PRAGMA changes. Safe to run concurrently with anything.

**Recovery procedure used:**

1. Confirmed corruption:
   ```bash
   sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
     "PRAGMA integrity_check;"
   # → Error: database disk image is malformed
   ```

2. Checked NFS snapshot copy:
   ```bash
   sqlite3 ~/Gin-AI/tmp/SQLite-db-snapshots/rag_state.db "PRAGMA integrity_check;"
   # → ok
   sqlite3 ~/Gin-AI/tmp/SQLite-db-snapshots/rag_state.db \
     "SELECT status, COUNT(*) FROM ingestion_queue GROUP BY status;"
   # → completed|7934   pending|9
   ```

3. Stopped ingest timer, restored main file only (do NOT restore -shm/-wal from
   a snapshot taken at a different time):
   ```bash
   sudo systemctl stop rag-ingest.timer
   cp ~/Gin-AI/tmp/SQLite-db-snapshots/rag_state.db \
      /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db
   rm -f /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db-wal
   rm -f /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db-shm
   ```

4. Verified and re-synced:
   ```bash
   sqlite3 .../rag_state.db "PRAGMA integrity_check; SELECT status, COUNT(*) ..."
   # → ok / completed|7934 pending|9
   python utils/rag_sync_nfs.py --user florian
   # → detected 8 deleted files (AWS--fotel files), cleaned SQLite + Qdrant
   # → pending: 14, completed: 7925
   ```

**General rule for SQLite recovery:** Always restore only the main `.db` file from
a snapshot. Remove existing `-wal` and `-shm` files — SQLite recreates them fresh
on first open. The snapshot for restore must be from a `PRAGMA wal_checkpoint(TRUNCATE)`
checkpoint point (written after each ingest run by `rag-ingest-run.sh`).

---

### Post-session corpus state (2026-05-12 ~22:30)

| Metric | Before session | After session |
|--------|----------------|---------------|
| Qdrant points | ~1,462,185 | ~946,129 |
| Corrupt/zombie vectors | ~290,000 | ~0 |
| SQLite completed | 7,934 | 7,925 |
| SQLite pending | 9 | 14 (5 new files) |
| Qdrant status | yellow (mid-merge) | green |
| Snapshot NFS copies | stale (Apr 23) | live (daily via fixed script) |

Smoke test: `python utils/rag_smoke_test.py --query "AWS certifications" --user florian --expect "AWS"` → **PASS**

Golden-path query: `python utils/rag_search_cli.py --query '$CWD blindly' --user florian`
→ Rank 1: `brain-fix-for-telemetry--2026-05-06--12-37.md` (score 0.546).
Nokia/Scan garbage completely absent from results.

---

## Consistent Rollback Procedure

### How consistency is guaranteed

After each ingestion run, `rag-ingest-run.sh` (Server 2):
1. Checkpoints the SQLite WAL with `PRAGMA wal_checkpoint(TRUNCATE)` — the main
   `rag_state.db` file becomes fully self-contained (no WAL file needed to restore).
2. Creates a Qdrant snapshot via API — written directly to NFS at
   `/mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/`.

Any NFS hourly snapshot taken AFTER an ingest run completes captures both files at
the same ingestion state and is a valid rollback point. 12 Qdrant snapshots are
retained (3 days × 4 runs/day); a 03:00 cron on Server 2 provides a safety-net.

### Rollback to a past NFS snapshot (Qdrant NVMe corrupted or lost)

Qdrant's active storage is local NVMe on Server 1 (`/var/lib/qdrant/storage`).
`rag_state.db` and Qdrant `.snapshot` files are both on NFS.

1. Stop services:
   ```bash
   # On Server 2:
   sudo systemctl stop rag-ingest.timer rag-ingest.service
   # On Server 1:
   sudo systemctl stop qdrant
   ```

2. From the Synology snapshot browser (or NFS snapshot mount), identify the target
   snapshot at time T. Find the Qdrant `.snapshot` file created closest before T:
   ```bash
   ls -lt /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/*.snapshot
   ```
   The filename encodes the creation timestamp (e.g. `documents-...-2026-05-05-01-32-10.snapshot`).

3. Restore Qdrant on Server 1:
   ```bash
   # Clear local NVMe storage
   sudo rm -rf /var/lib/qdrant/storage
   sudo systemctl start qdrant
   # Recover from NFS snapshot
   curl -X PUT "http://192.168.1.93:6333/collections/documents/snapshots/recover" \
     -H "Content-Type: application/json" \
     -d '{"location": "file:///mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/<snapshot-name>.snapshot"}'
   ```

4. Restore `rag_state.db` from the same NFS snapshot point.
   (Synology: access the snapshot volume, copy `rag_state.db` to
   `/mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db`.
   Only the main db file is needed — the WAL is empty after each ingest run.)

5. Verify:
   ```bash
   source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
   python utils/rag_status.py
   curl -s http://192.168.1.93:6333/collections/documents | python3 -m json.tool
   ```

6. Re-ingest files added after the rollback point (on Server 2):
   ```bash
   python utils/rag_sync_nfs.py --user florian
   python utils/rag_ingest_daemon.py --workers 3 --batch 20 \
     --log-file /mnt/nfs/__Backups/SoHoAI--databases/logs/rag-ingest.log
   ```

7. Restart ingestion timer (Server 2):
   ```bash
   sudo systemctl start rag-ingest.timer
   ```

### Rollback SQLite only (Qdrant healthy, rag_state.db corrupted)

1. Stop the ingestion timer on Server 2.
2. Restore `rag_state.db` from the NFS snapshot point (main file only; WAL is empty).
3. Run `rag_sync_nfs.py` to reconcile — files marked `completed` in SQLite but
   absent from Qdrant will be re-queued automatically.
4. Run `rag_ingest_daemon.py` to fill any gaps.
5. Restart ingestion timer.

---

## Session — 2026-05-12

### rag_sync_nfs.py timeout on bulk Qdrant cleanup

#### Symptom

`rag_sync_nfs.py` ran for ~19 minutes before crashing with
`qdrant_client.http.exceptions.ResponseHandlingException: timed out` while deleting Qdrant
points for 289 removed files. The script processed roughly 280 files successfully (~4 s each)
then stalled on one request that exceeded the 60-second client timeout.

The crash happened inside the Qdrant delete call, so SQLite rows survived intact (correct
crash-safe ordering). But the cleanup was incomplete and the script had to be re-run manually.

#### Root cause

`rag_sync_nfs.py` deleted Qdrant points **one file at a time** in a sequential loop:

```python
for path in deleted_paths:
    qdrant_client.delete(
        collection_name=DOCUMENTS_COLLECTION,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key=FIELD_SOURCE_PATH, match=MatchValue(value=path))
        ])),
    )
```

Each call uses `wait=true` (Qdrant default), blocking until the collection's index is
fully re-optimized. On a large collection (98 K+ points), that optimization takes several
seconds per request. 289 sequential requests meant any single slow optimization pass could
exceed the 60-second timeout, crashing the script.

#### Fix

Two changes in `rag_sync_nfs.py` and `rag_engine/collection.py`:

**`wait=False` on every delete call** — Qdrant queues the operation and returns immediately;
the client never blocks on index re-optimization. No retry loop needed: with fire-and-forget
semantics, the only failure mode is the request not reaching Qdrant at all (server down,
network error), in which case the script aborts and SQLite rows survive as retry markers for
the next sync run.

```python
def _delete_path_from_qdrant(client, path: str, file_num: int, total: int) -> None:
    client.delete(
        collection_name=DOCUMENTS_COLLECTION,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key=FIELD_SOURCE_PATH, match=MatchValue(value=path))]
        )),
        wait=False,
    )
    logger.info("Qdrant cleanup %d/%d: queued delete for %s", file_num, total, path)
```

`get_client()` in `rag_engine/collection.py` now accepts an optional `timeout: int = 60`
parameter (added in the same session) — no other callers are affected.

#### Expected outcome

289 sequential deletes, each returning in milliseconds. Total cleanup time drops from 19+
minutes to seconds. Index re-optimization happens asynchronously in Qdrant; the cleanup
script is completely decoupled from it.

#### Related entries

- [Session 2026-04-22](#session--2026-04-22) — original `get_client()` 60 s timeout fix for
  ingestion daemon timeouts.

---

## Session — 2026-05-08

### Lock file mtime appears frozen (Synology NFS empty-truncate no-op)

#### Symptom

`/mnt/nfs/__Backups/SoHoAI--databases/rag-ingest.lock` showed mtime `2026-05-06 10:53` despite
the ingestion service running successfully every six hours.  The lock appeared stale.

#### Root cause

The lock is **not stale** — no process holds it.  The frozen mtime is a Synology NFS quirk.

`rag_ingest_daemon.py` acquires the lock by opening the file in `"w"` mode (`O_WRONLY|O_CREAT|O_TRUNC`)
then calling `fcntl.lockf(fd, LOCK_EX|LOCK_NB)`.  The file is intentionally never written to —
it stays 0 bytes.  On this Synology NFS (NFSv4.1), the Linux NFS client omits the SETATTR RPC
when `O_TRUNC` would not change the file size (already 0), so the NAS receives no mtime-update.

The lock file was created on May 6 at 10:53 during a manual daemon run before the timer was set
up.  Every subsequent run re-opened the empty file, acquired the advisory lock, and exited —
leaving the mtime frozen at the initial creation date.

Confirming the lock is free:
```bash
flock --nonblock /mnt/nfs/__Backups/SoHoAI--databases/rag-ingest.lock echo "lock-free" \
  || echo "lock-held"
# → lock-free
```

A genuinely stale lock would print `lock-held` and `lsof` + `ps aux | grep rag` would show a
live process.

#### Fix

Added `os.utime(_lock_path)` in `utils/rag_ingest_daemon.py` immediately after `fcntl.lockf`
succeeds.  `utime` issues an explicit `utimes` syscall → NFS SETATTR, advancing mtime on every
daemon invocation regardless of pending-file count.

```python
# utils/rag_ingest_daemon.py — after lock acquisition (~line 169)
os.utime(_lock_path)   # force SETATTR so mtime advances on NFS (empty-truncate is a no-op)
```

#### Related note — May 8 run left no "Starting ingest:" log line

On May 8, `rag_sync_nfs.py` found 0 pending files.  The daemon connected to Qdrant (for
`ensure_collection`), checked the queue, hit the early-exit at line 185–188, and returned.
The early-exit uses `print()`, not `logger`, so the message goes to stdout → journald — it
does **not** appear in `rag-ingest.log`.  The file-handler for `--log-file` is only set up
after the lock is acquired; the early-exit path bypasses it.  This is expected behaviour.

---

## Session — 2026-05-04

### Ollama bge-m3 CUDA OOM + broken rag_sync_nfs import

#### Symptoms

Two separate bugs discovered during a RAG ingestion run on Server 2 (192.168.1.95):

1. `rag_sync_nfs.py` crashed immediately with `ImportError` on every run — deleted files were
   never cleaned from Qdrant or SQLite, and failed rows were never re-queued.
2. Ollama embedding calls returned HTTP 500 for all chunks, crashing the ingest daemon.

#### Bug 1 — broken import in `rag_sync_nfs.py`

**Root cause:** `rag_sync_nfs.py` imported `re_queue_failed` as a module-level name, but it only
exists as a method on `StateDB`. The module-level wrapper was added in commit `b8e1f43`
(2026-04-30) then silently removed during the bulk project rename in commit `a3f2c91`
(2026-05-04), leaving the import broken.

Effect: `rag_sync_nfs.py` was a no-op since 2026-05-04. The shell command
`python utils/rag_sync_nfs.py ; python utils/rag_ingest_daemon.py ...` ran the daemon
unconditionally via `;` even though sync had crashed, so ghost files (deleted from disk but
still in the queue) were retried on every daemon run.

**Fix:** Remove `re_queue_failed` from the import; call `state_db.re_queue_failed()` directly.

```python
# Before (broken)
from rag_engine.state import StateDB, re_queue_failed
...
n_requeued = re_queue_failed()

# After (fixed)
from rag_engine.state import StateDB
...
n_requeued = state_db.re_queue_failed()
```

Committed in `7d4a1b9`.

#### Bug 2 — Ollama bge-m3 CUDA OOM (HTTP 500)

**Root cause:** The Ollama runner crashed with `cudaMalloc failed: out of memory` when trying
to allocate the compute graph buffer. The failure sequence from the journal:

```
gpu memory available="569.2 MiB" free="1.3 GiB"
CUDA0 model buffer size = 456.96 MiB          ← weights load OK
allocating 1170.00 MiB on device 0: cudaMalloc failed: out of memory
graph_reserve: failed to allocate compute buffers
llama runner terminated: exit status 2
HTTP 500
```

Ollama automatically sets `batch_size = context_length` for embedding models. With the default
`OLLAMA_CONTEXT_LENGTH=0` (→ 4096), the compute graph required **1170 MiB**. When
llama-server's KV cache fills under load (Gemma 4 Q8_0, 2×110024 ctx ≈ 9.3 GiB VRAM), the
remaining free VRAM drops below 1.3 GiB — not enough for weights (457 MiB) + compute buffer
(1170 MiB) = 1627 MiB.

The model reloads successfully when llama-server slots are idle (2.4 GiB free), but fails
under concurrent load — making this an intermittent, load-dependent failure.

**Fix:** Set `OLLAMA_CONTEXT_LENGTH=768` in `/etc/systemd/system/ollama.service`.
RAG child chunks are ~250 tokens; 768 is more than sufficient with headroom above the
512-token flat-chunk ceiling.

```
# /etc/systemd/system/ollama.service
Environment="OLLAMA_CONTEXT_LENGTH=768"
```

Effect on VRAM (measured before/after):

| Component | ctx=4096 (before) | ctx=768 (after) |
|---|---|---|
| batch_size | 4096 | 768 |
| kv cache | 19 MiB | 4.5 MiB |
| compute buffer | **1168 MiB** | **53 MiB** |
| **Total VRAM** | **~1.75 GiB** | **~635 MiB** |

The compute buffer dropped 22× from 1168 MiB to 53 MiB. bge-m3 now fits reliably alongside
llama-server even with both KV slots active.

**Immediate workaround applied in parallel:** `SoHoAI-config.yaml` `ollama_url` temporarily reverted to
Server 1 CPU embed (`192.168.1.93:11434`) while Server 2 was being fixed.
**Resolved 2026-05-04:** `ollama_url` restored to Server 2 GPU (`192.168.1.95:11434`) after
`OLLAMA_CONTEXT_LENGTH=768` was deployed. GPU embed is now the active configuration.

**Verification:**
```bash
# Confirm context_length=768 in the loaded model
curl -s http://localhost:11434/api/ps | python3 -m json.tool
# → "context_length": 768, "size_vram" > 0

# Confirm embeddings work
curl -s http://localhost:11434/api/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","prompt":"test"}' | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f'OK — {len(d[\"embedding\"])} dims')"
# → OK — 1024 dims
```

---

## Session — 2026-05-01

### Checking Qdrant for Specific Files

#### Quick summary

Two layers to check in order:

1. **SQLite ingestion queue** (`rag_state.db`) — tells you whether the file was *scanned and
   processed*. `status=completed` means the ingest daemon finished; the file has Qdrant points.
   `pending` means it hasn't been processed yet. `ignored` means it permanently failed.

2. **Qdrant REST API** — confirms that vector points actually exist for a given file.
   Use `scroll` with a `filter` on `source_path`, `session_id`, or `file_type`.

Claude Code session files live under `/home/florian/.claude/projects/` as `<uuid>.jsonl`.
In Qdrant they appear as `file_type = "claude_chat"` points with `session_id` and `project`
payload fields in addition to the standard `source_path`.

#### Concrete example — session `00c53ce0-b267-4f0f-8835-5f4b5bf56e2e`

**Step 1 — check the ingestion queue**

```bash
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT file_path, status, retry_count
   FROM ingestion_queue
   WHERE file_path LIKE '%00c53ce0%';"
```

Output:
```
/home/florian/.claude/projects/-mnt-nfs-Florian-Gin-AI-projects-SoHoAI/00c53ce0-b267-4f0f-8835-5f4b5bf56e2e.jsonl|completed|0
```

`completed` with `retry_count=0` — ingestion succeeded, no retries needed.

**Step 2 — verify Qdrant points exist**

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"must": [{"key": "source_path", "match": {
      "value": "/home/florian/.claude/projects/-mnt-nfs-Florian-Gin-AI-projects-SoHoAI/00c53ce0-b267-4f0f-8835-5f4b5bf56e2e.jsonl"
    }}]},
    "limit": 5,
    "with_payload": ["source_path", "session_id", "project", "chunk_index"],
    "with_vector": false
  }' | python3 -m json.tool
```

Output (truncated):
```json
{
  "result": {
    "points": [
      {
        "id": "3c4720be-f9c8-4a16-bb52-fa9efa07c7dc",
        "payload": {
          "source_path": "/home/florian/.claude/projects/-mnt-nfs-Florian-Gin-AI-projects-SoHoAI/00c53ce0-b267-4f0f-8835-5f4b5bf56e2e.jsonl",
          "chunk_index": 0,
          "session_id": "00c53ce0-b267-4f0f-8835-5f4b5bf56e2e",
          "project": "SoHoAI"
        }
      }
    ],
    "next_page_offset": null
  },
  "status": "ok"
}
```

Points found — this session is indexed and searchable.

#### All methods

**1. SQLite queue — fastest first check**

```bash
# All claude session files and their status
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT status, COUNT(*) FROM ingestion_queue
   WHERE file_path LIKE '%/.claude/projects/%'
   GROUP BY status;"

# Specific file by UUID substring
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT file_path, status, retry_count FROM ingestion_queue
   WHERE file_path LIKE '%<uuid>%';"

# All pending or ignored sessions (not yet in Qdrant)
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT file_path, status, retry_count, skip_reason FROM ingestion_queue
   WHERE file_path LIKE '%/.claude/projects/%'
   AND status != 'completed';"
```

As of 2026-05-01: **178 sessions completed** in the queue.

**2. Qdrant scroll — filter by `source_path`**

Use when you have an exact file path and want to confirm Qdrant points exist.

```bash
SOURCE="/home/florian/.claude/projects/-mnt-nfs-Florian-Gin-AI-projects-SoHoAI/<uuid>.jsonl"

curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/scroll" \
  -H "Content-Type: application/json" \
  -d "{
    \"filter\": {\"must\": [{\"key\": \"source_path\", \"match\": {\"value\": \"$SOURCE\"}}]},
    \"limit\": 10,
    \"with_payload\": [\"source_path\", \"session_id\", \"chunk_index\"],
    \"with_vector\": false
  }" | python3 -m json.tool
```

`"next_page_offset": null` in the response means you got all chunks for that file.
Empty `points` array means no Qdrant points — either not ingested, or ingestion failed.

**3. Qdrant scroll — filter by `session_id`**

Same result as above when you have the UUID but not the full path.

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"must": [{"key": "session_id", "match": {"value": "<uuid>"}}]},
    "limit": 50,
    "with_payload": ["source_path", "session_id", "project", "chunk_index"],
    "with_vector": false
  }' | python3 -m json.tool
```

**4. Qdrant count — all `claude_chat` points**

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/count" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "file_type", "match": {"value": "claude_chat"}}]}}' \
  | python3 -m json.tool
```

As of 2026-05-01: **3,369 points** from claude chat sessions in the `documents` collection.

**5. Qdrant count — filter by `project`**

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/count" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "project", "match": {"value": "SoHoAI"}}]}}' \
  | python3 -m json.tool
```

**6. Python one-liner — list all ingested sessions with chunk counts and titles**

```bash
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate && python3 -c "
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

c = QdrantClient(url='http://192.168.1.93:6333', timeout=60)
results, _ = c.scroll(
    'documents',
    scroll_filter=Filter(must=[FieldCondition(key='file_type', match=MatchValue(value='claude_chat'))]),
    limit=500,
    with_payload=['source_path', 'session_id', 'project', 'session_title', 'chunk_index'],
    with_vectors=False,
)
seen = {}
for r in results:
    p = r.payload or {}
    src = p.get('source_path', '?')
    if src not in seen:
        seen[src] = {'session_id': p.get('session_id', ''), 'project': p.get('project', ''),
                     'title': p.get('session_title', ''), 'chunks': 0}
    seen[src]['chunks'] += 1
for path, info in sorted(seen.items()):
    title = info['title'] or '(no title — re-ingest needed)'
    print(f\"{info['chunks']:3d} chunks  [{info['project']}]  {title}\")
print(f'\nTotal: {len(seen)} sessions, {sum(v[\"chunks\"] for v in seen.values())} points')
"
```

**7. Searching by file type (CLI)**

```bash
# Claude Code sessions only
python utils/rag_search_cli.py --query "cost distribution" --user florian --file-types claude_chat

# PowerPoint presentations only
python utils/rag_search_cli.py --query "quarterly review" --user florian --file-types pptx ppt

# PDFs only
python utils/rag_search_cli.py --query "AWS certification" --user florian --file-types pdf
```

Valid `--file-types` values: `pdf`, `docx`, `pptx`, `ppt`, `txt`, `md`, `yaml`, `ipynb`, `claude_chat`

#### Payload fields for `claude_chat` documents

| Field | Type | Description |
|---|---|---|
| `file_type` | str | Always `"claude_chat"` for session files |
| `source_path` | str | Full path to the `.jsonl` file — the unique key |
| `session_id` | str | Claude Code session UUID — filterable |
| `session_title` | str | Human-readable title (added 2026-05-01; requires re-ingest of existing points) |
| `project` | str | Project name derived from `cwd` (e.g. `"SoHoAI"`) |
| `owner` | str | User who owns the session (e.g. `"florian"`) |
| `chunk_index` | int | Chunk number within the session |
| `text` | str | Child chunk content (what was embedded) |
| `parent_text` | str | Parent chunk content (what gets injected into LLM) |

NFS document points do **not** carry `session_id`, `session_title`, or `project` — sparse payload is fine in Qdrant.

#### Fixing missing files

**File shows `pending` in SQLite, zero Qdrant points:**
```bash
# Just run the daemon
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
python utils/rag_ingest_daemon.py --workers 1 --batch 5 --log-file /tmp/rag-ingestion.log
```

**File shows `ignored` (exhausted retries):**
```bash
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "UPDATE ingestion_queue SET status='pending', retry_count=0, skip_reason=NULL
   WHERE file_path='<exact-path>';"
python utils/rag_ingest_daemon.py --workers 1 --batch 2 --log-file /tmp/rag-ingestion.log
```

**New sessions not in queue yet:**
```bash
python utils/rag_sync_nfs.py   # scans both NFS roots and ~/.claude/projects, enqueues new files
python utils/rag_ingest_daemon.py --workers 1 --batch 5 --log-file /tmp/rag-ingestion.log
```

**File deleted from disk but still in Qdrant:**
```bash
python utils/rag_sync_nfs.py   # find_deleted() handles purge automatically
```

---

## Session — 2026-04-22

### Qdrant HTTP Timeouts During Ingestion

#### Symptom
The `rag_ingest_daemon.py` script fails with `httpcore.ReadTimeout: timed out` errors during
bulk document ingestion. Errors occur at regular intervals (~1 per 3-5 minutes) during heavy
ingestion runs.

**Example error trace:**
```
ERROR: Ingestion failed for /path/to/file.pdf: timed out
qdrant_client.http.exceptions.ResponseHandlingException: timed out
```

#### Root Cause
The `QdrantClient` in `rag_engine/collection.py` was initialized with the httpx default timeout
(~5 seconds). During heavy ingestion:

1. Large bulk operations (e.g., ingesting 70K+ chunks from a single file) cause Qdrant to
   perform extensive index optimization.
2. Index optimization blocks response handling while maintaining internal consistency.
3. The 5-second client timeout fires before Qdrant can respond, even though the server is healthy.

**Timeline from 2026-04-22 ingestion run:**
- 14:15:24 — Completed ingestion of 70,084-chunk CSV file
- 14:15:46 — First timeout (22 seconds later, during Qdrant optimization)
- 14:15:46 to 14:55:03 — 21 timeout errors over 39 minutes (1 per 3.3 minutes)
- Database remained healthy throughout (473K points, green status)

#### Solution
Increase the HTTP timeout to 60 seconds in `rag_engine/collection.py`. The function now also
accepts an optional `timeout` parameter so callers with heavier workloads (e.g. bulk-delete
in `rag_sync_nfs.py`) can request a longer timeout without affecting ingestion:

```python
def get_client(url: str, timeout: int = 60) -> QdrantClient:
    """Connect to a running Qdrant server.

    Default timeout is 60 seconds to handle index optimization on large batches.
    Pass a higher value for bulk-delete operations that may trigger longer
    re-indexing passes.
    """
    return QdrantClient(url=url, timeout=timeout)
```

**Fix applied:** 2026-04-22 (timeout=60 default); `timeout` parameter added 2026-05-12.

#### Why 60 Seconds?
- Qdrant's default flush interval is 5 seconds (`flush_interval_sec: 5`)
- Index optimization can require multiple flush cycles during heavy ingestion
- Large index restructuring (> 10K points per operation) can take 10–30 seconds on typical hardware
- 60 seconds provides comfortable margin without being excessive

#### Verification
If timeouts persist after applying this fix:

1. **Check Qdrant server health:**
   ```bash
   curl http://192.168.1.93:6333/collections/documents
   ```
   Expected status: `green`, optimizer: `ok`, no pending updates

2. **Monitor Qdrant performance during ingestion:**
   ```bash
   curl http://192.168.1.93:6333/collections/documents | jq .result.status
   ```

3. **Check network latency to Qdrant:**
   ```bash
   ping -c 5 192.168.1.93
   ```
   If RTT > 50ms, network issues may be contributing.

4. **If timeouts persist:**
   - Increase timeout further (try 120 seconds)
   - Check Qdrant server CPU/memory during ingestion
   - Consider reducing `--workers` or `--batch` flags to decrease concurrency

#### Related Configuration
- `rag_engine/collection.py::get_client()` — Client initialization
- `utils/rag_ingest_daemon.py --workers` — Number of concurrent files
- `utils/rag_ingest_daemon.py --batch` — Concurrent embedding requests per file
