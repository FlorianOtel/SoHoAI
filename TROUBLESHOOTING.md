# Troubleshooting Guide — HomeAI-Lab RAG Engine

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
