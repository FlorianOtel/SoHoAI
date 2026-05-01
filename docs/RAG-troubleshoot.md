---
title: "RAG Troubleshooting — Checking Qdrant for Specific Files"
date: 2026-05-01
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Sonnet 4.6)
updated_on: 2026-05-01
context: >
  Practical reference for verifying whether specific files — especially Claude Code session
  .jsonl files under ~/.claude/projects — are present in the Qdrant vector store.
  Covers SQLite queue checks, Qdrant REST scroll/filter/count queries, and a Python one-liner.
  Derived from a live troubleshooting session on 2026-05-01 against the production SoHoAI
  corpus (documents collection, Server 1 at 192.168.1.93:6333).
  Updated 2026-05-01 to reflect new session_title payload field and file_types search filter.
---

# RAG Troubleshooting — Checking Qdrant for Specific Files

## Quick summary

Two layers to check in order:

1. **SQLite ingestion queue** (`rag_state.db`) — tells you whether the file was *scanned and
   processed*. `status=completed` means the ingest daemon finished; the file has Qdrant points.
   `pending` means it hasn't been processed yet. `ignored` means it permanently failed.

2. **Qdrant REST API** — confirms that vector points actually exist for a given file.
   Use `scroll` with a `filter` on `source_path`, `session_id`, or `file_type`.

Claude Code session files live under `/home/florian/.claude/projects/` as `<uuid>.jsonl`.
In Qdrant they appear as `file_type = "claude_chat"` points with `session_id` and `project`
payload fields in addition to the standard `source_path`.

---

## Concrete example — session `00c53ce0-b267-4f0f-8835-5f4b5bf56e2e`

### Step 1 — check the ingestion queue

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

### Step 2 — verify Qdrant points exist

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

---

## All methods

### 1. SQLite queue — fastest first check

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

### 2. Qdrant scroll — filter by `source_path`

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

### 3. Qdrant scroll — filter by `session_id`

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

### 4. Qdrant count — all `claude_chat` points

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/count" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "file_type", "match": {"value": "claude_chat"}}]}}' \
  | python3 -m json.tool
```

As of 2026-05-01: **3,369 points** from claude chat sessions in the `documents` collection.

### 5. Qdrant count — filter by `project`

```bash
curl -s -X POST "http://192.168.1.93:6333/collections/documents/points/count" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"must": [{"key": "project", "match": {"value": "SoHoAI"}}]}}' \
  | python3 -m json.tool
```

### 6. Python one-liner — list all ingested sessions with chunk counts and titles

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

### 7. Searching by file type (CLI)

```bash
# Claude Code sessions only
python utils/rag_search_cli.py --query "cost distribution" --user florian --file-types claude_chat

# PowerPoint presentations only
python utils/rag_search_cli.py --query "quarterly review" --user florian --file-types pptx ppt

# PDFs only
python utils/rag_search_cli.py --query "AWS certification" --user florian --file-types pdf
```

Valid `--file-types` values: `pdf`, `docx`, `pptx`, `ppt`, `txt`, `md`, `yaml`, `ipynb`, `claude_chat`

---

## Payload fields for `claude_chat` documents

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

---

## Fixing missing files

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
