---
title: "SoHoAI — Troubleshooting Guide"
date: 2026-04-22
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_on: 2026-04-30
context: >
  SoHoAI RAG pipeline troubleshooting guide;
  Qdrant HTTP timeouts during ingestion, embedding failure modes,
  large PDF parsing, search latency under concurrent ingestion,
  ignored file retry procedures, clean restart procedures.
  Project rename (HomeAI → SoHoAI) database migration.
---

# Troubleshooting Guide — SoHoAI RAG Engine

## Qdrant HTTP Timeouts During Ingestion

### Symptom
The `rag_ingest_daemon.py` script fails with `httpcore.ReadTimeout: timed out` errors during bulk document ingestion. Errors occur at regular intervals (~1 per 3-5 minutes) during heavy ingestion runs.

**Example error trace:**
```
ERROR: Ingestion failed for /path/to/file.pdf: timed out
qdrant_client.http.exceptions.ResponseHandlingException: timed out
```

### Root Cause
The `QdrantClient` in `rag_engine/collection.py` was initialized with the httpx default timeout (~5 seconds). During heavy ingestion:

1. Large bulk operations (e.g., ingesting 70K+ chunks from a single file) cause Qdrant to perform extensive index optimization
2. Index optimization blocks response handling while maintaining internal consistency
3. The 5-second client timeout fires before Qdrant can respond, even though the server is healthy

**Timeline from 2026-04-22 ingestion run:**
- 14:15:24 — Completed ingestion of 70,084-chunk CSV file
- 14:15:46 — First timeout (22 seconds later, during Qdrant optimization)
- 14:15:46 to 14:55:03 — 21 timeout errors over 39 minutes (1 per 3.3 minutes)
- Database remained healthy throughout (473K points, green status)

### Solution
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

**Fix applied:** 2026-04-22 (commit: TBD)

### Why 60 Seconds?
- Qdrant's default flush interval is 5 seconds (`flush_interval_sec: 5`)
- Index optimization can require multiple flush cycles during heavy ingestion
- Large index restructuring (> 10K points per operation) can take 10–30 seconds on typical hardware
- 60 seconds provides comfortable margin without being excessive

### Verification
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
   If RTT > 50ms, network issues may be contributing

4. **If timeouts persist:**
   - Increase timeout further (try 120 seconds)
   - Check Qdrant server CPU/memory during ingestion
   - Consider reducing `--workers` or `--batch` flags to decrease concurrency

### Related Configuration
- `rag_engine/collection.py::get_client()` — Client initialization
- `utils/rag_ingest_daemon.py --workers` — Number of concurrent files
- `utils/rag_ingest_daemon.py --batch` — Concurrent embedding requests per file

### Historical Issues Log
| Date | Issue | Fix | Status |
|------|-------|-----|--------|
| 2026-04-22 | HTTP read timeout (5s) during 70K-point ingestion | Increase timeout to 60s | ✅ Implemented |

---

## All Files Re-queued for Ingestion After Project Rename

### Symptom
Running `python utils/rag_sync_nfs.py` shows all files as `pending` even though they were
previously ingested. Every subsequent daemon run re-embeds and re-upserts files that are
already present in Qdrant, wasting hours of compute.

### Root Cause
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

### Fix
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

### Verification

```bash
# Confirm row counts match the old database
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "SELECT status, COUNT(*) FROM ingestion_queue GROUP BY status;"
# Expected: completed|<N>  (should match the old DB count)

# Confirm no re-queuing — pending should be 0 (or only truly new files)
python utils/rag_sync_nfs.py
python utils/rag_status.py
```

### Prevention
When renaming the project or changing `db_base_path`:
1. Update the config first (so the new path is known).
2. Migrate the SQLite databases **before** running `rag_sync_nfs.py` for the first time.
3. Also update cron and the deployed systemd unit — they are not managed by git and reference
   the project directory path explicitly.

### Historical Issues Log
| Date | Issue | Fix | Status |
|------|-------|-----|--------|
| 2026-04-22 | HTTP read timeout (5s) during 70K-point ingestion | Increase timeout to 60s | ✅ Implemented |
| 2026-04-30 | All files re-queued after HomeAI → SoHoAI rename (empty new `rag_state.db`) | SQLite backup copy + directory migration | ✅ Fixed |
