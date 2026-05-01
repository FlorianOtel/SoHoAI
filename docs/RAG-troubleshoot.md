---
title: "SoHoAI RAG Pipeline — Troubleshooting"
created_at: 20260422-000000
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_at: 20260501-150624
context: >
  Consolidated RAG pipeline troubleshooting reference for SoHoAI.
  Originally two files: TROUBLESHOOTING.md (Qdrant timeout + project rename migration,
  sessions 2026-04-22 and 2026-04-30) and RAG-troubleshoot.md (Qdrant file-presence checks,
  session 2026-05-01). Merged 2026-05-01. Covers: Qdrant HTTP timeouts during bulk ingestion,
  re-queue after db_base_path rename, verifying specific files in the vector store,
  ignored file retry procedures, and clean restart procedures.
---

# SoHoAI RAG Pipeline — Troubleshooting

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
          "project": "HomeAI"
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

## Session — 2026-04-30

### All Files Re-queued for Ingestion After Project Rename

#### Symptom
Running `python utils/rag_sync_nfs.py` shows all files as `pending` even though they were
previously ingested. Every subsequent daemon run re-embeds and re-upserts files that are
already present in Qdrant, wasting hours of compute.

#### Root Cause
Renaming the project changed `db_base_path` in `config.yaml`:

```diff
-db_base_path: "/mnt/nfs/__Backups/HomeAI--databases"
+db_base_path: "/mnt/nfs/__Backups/SoHoAI--databases"
```

`StateDB.__init__` calls `sqlite3.connect(new_path)`, which **silently creates a new empty
database** when the path does not exist. All previously completed rows vanish — the next
`rag_sync_nfs.py` run inserts every scanned file as `pending`. Qdrant is unaffected (active
storage is on local NVMe and does not depend on `db_base_path`).

The same issue affects `chats.db` (SQLite chat history) and the Redis AOF directory, though
Redis creates its directory automatically.

#### Fix
Use SQLite's online backup API to copy each database from the old path to the new one.
The backup API correctly merges any uncommitted WAL data.

```python
import sqlite3, pathlib

for name in ["rag_state.db", "chats.db"]:
    src_path = f"/mnt/nfs/__Backups/HomeAI--databases/sqlite/{name}"
    dst_path = f"/mnt/nfs/__Backups/SoHoAI--databases/sqlite/{name}"
    pathlib.Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    dst.close(); src.close()
    print(f"Migrated {name}")
```

Also create the missing subdirectories and copy Qdrant snapshots:

```bash
# Create directory structure
mkdir -p /mnt/nfs/__Backups/SoHoAI--databases/{qdrant,qdrant-snapshots/documents,qdrant-snapshots/tmp/upload,redis}

# Copy Qdrant DR snapshots (NFS-resident, not the active NVMe storage)
cp --preserve /mnt/nfs/__Backups/HomeAI--databases/qdrant-snapshots/documents/* \
              /mnt/nfs/__Backups/SoHoAI--databases/qdrant-snapshots/documents/

# Update cron job (snapshot script path)
crontab -e   # change HomeAI → SoHoAI in the qdrant-snapshot.sh path

# Update deployed systemd unit and reload (no restart needed if configs are identical)
sudo cp /mnt/nfs/Florian/Gin-AI/projects/SoHoAI/scripts/qdrant/qdrant.service \
        /etc/systemd/system/qdrant.service
sudo systemctl daemon-reload
```

#### Verification

```bash
# Confirm row counts match the old database
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT status, COUNT(*) FROM ingestion_queue GROUP BY status;"
# Expected: completed|<N>  (should match the old DB count)

# Confirm no re-queuing — pending should be 0 (or only truly new files)
python utils/rag_sync_nfs.py
python utils/rag_status.py
```

#### Prevention
When renaming the project or changing `db_base_path`:
1. Update the config first (so the new path is known).
2. Migrate the SQLite databases **before** running `rag_sync_nfs.py` for the first time.
3. Also update cron and the deployed systemd unit — they are not managed by git and reference
   the project directory path explicitly.

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
Increase the HTTP timeout to 60 seconds in `rag_engine/collection.py`:

```python
def get_client(url: str) -> QdrantClient:
    """Connect to a running Qdrant server.

    Timeout set to 60 seconds to handle index optimization on large batches.
    During heavy ingestion (e.g., 70K+ points), Qdrant may take >5 seconds to
    respond to delete/upsert requests while it optimizes indexes. Default httpx
    timeout (~5s) is too short; 60s allows adequate time.
    """
    return QdrantClient(url=url, timeout=60)
```

**Fix applied:** 2026-04-22

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
