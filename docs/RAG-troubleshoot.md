---
title: "SoHoAI RAG Pipeline — Troubleshooting"
created_at: 20260422-000000
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 2026-05-12--19-47
context: >
  Consolidated RAG pipeline troubleshooting reference for SoHoAI.
  Originally two files: TROUBLESHOOTING.md (Qdrant timeout + project rename migration,
  sessions 2026-04-22 and 2026-04-30) and RAG-troubleshoot.md (Qdrant file-presence checks,
  session 2026-05-01). Merged 2026-05-01. Covers: Qdrant HTTP timeouts during bulk ingestion,
  re-queue after db_base_path rename, verifying specific files in the vector store,
  ignored file retry procedures, and clean restart procedures.
  Updated 2026-05-12: rag_sync_nfs.py sequential-delete timeout fix — batched Qdrant deletes
  (50 paths per request, OR filter) with retry+backoff; get_client() timeout parameter added.
---

# SoHoAI RAG Pipeline — Troubleshooting

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

**1. Batch deletes using a `should` (OR) filter** — 50 paths per request instead of 1.
Qdrant's index re-optimization is amortized across 50 files at once; 289 files → 6 requests
instead of 289.

```python
# New helper in rag_sync_nfs.py
_DELETE_BATCH_SIZE = 50
_DELETE_MAX_RETRIES = 3

def _delete_paths_from_qdrant(client, paths: list[str]) -> None:
    batches = [paths[i:i+_DELETE_BATCH_SIZE] for i in range(0, len(paths), _DELETE_BATCH_SIZE)]
    for batch_num, batch in enumerate(batches, 1):
        for attempt in range(1, _DELETE_MAX_RETRIES + 1):
            try:
                client.delete(
                    collection_name=DOCUMENTS_COLLECTION,
                    points_selector=FilterSelector(filter=Filter(
                        should=[FieldCondition(key=FIELD_SOURCE_PATH, match=MatchValue(value=p))
                                for p in batch]
                    )),
                )
                break
            except Exception as exc:
                if attempt < _DELETE_MAX_RETRIES:
                    time.sleep(2 ** attempt)   # 2 s, 4 s, 8 s
                else:
                    raise
```

**2. Longer per-request timeout for the sync client** — `get_client(qdrant_url, timeout=120)`
instead of the default 60 s. Batch deletes trigger a single larger re-indexing pass; 120 s
provides headroom without affecting the ingestion daemon (which retains its 60 s default).

`get_client()` in `rag_engine/collection.py` now accepts an optional `timeout: int = 60`
parameter — no other callers are affected.

#### Expected outcome

289 files → 6 batched requests, completing in ~1–3 minutes instead of 19+ minutes (and
failure). If a batch still times out, the retry loop fires up to 3 times with exponential
back-off. On persistent failure the script aborts cleanly; SQLite rows survive for the next
sync run.

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

**Immediate workaround applied in parallel:** `config.yaml` `ollama_url` temporarily reverted to
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
