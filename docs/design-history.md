---
title: "SoHoAI Design History"
date: 2026-05-01
created_by: Claude Code (Claude Sonnet 4.6)
context: >
  Running log of significant design decisions, feature additions, and architectural
  changes to the SoHoAI project. Each entry is timestamped and includes rationale.
  Complements CLAUDE.md (which documents current state) by preserving the reasoning
  behind changes as they were made.
---

# SoHoAI Design History

---

## 2026-05-01 13:39 — `session_title` metadata + `file_types` search filter

### What was implemented

Two improvements to the RAG pipeline, strictly isolated to the claude_chat sub-pipeline
and the search layer. NFS document ingestion, scanner, and StateDB are unchanged.

**6 files modified** (`rag_engine/schema.py`, `rag_engine/ingest.py`, `rag_engine/search.py`,
`rag_engine/tool_use.py`, `main.py`, `utils/rag_search_cli.py`).

#### 1. `session_title` Qdrant payload field (claude_chat only)

Every claude_chat Qdrant point now carries a human-readable `session_title`. The field is
absent on NFS document points (sparse payload — Qdrant allows this per-point).

Title derivation priority:
1. **Named sessions** — symlink stem from `~/.claude/chats/<project>/<name>.jsonl` (user-chosen).
   Looked up via `_build_title_map(dot_claude_dir)` — LRU-cached per daemon run, walks the
   chats index once regardless of how many sessions are ingested. UUID-named symlinks
   (auto-created by Claude Code) are excluded from the map.
2. **Unnamed sessions** (covers ~174 of 178) — `_synthesize_title(path)` reads the first
   non-empty, non-command user message from the JSONL, truncated to 60 chars + `...`.
   Falls back to `path.stem` (UUID) only if the file is unreadable or has zero user turns.
3. **Subagent files** — derive parent session's title via the above logic, append `[subagent]`.
   Path depth: `.../projects/<mangled>/<uuid>/subagents/agent-xxx.jsonl` → `path.parents[4]`
   resolves to `.claude/`.

New constant: `FIELD_SESSION_TITLE = "session_title"` in `rag_engine/schema.py`.

#### 2. `file_types` search filter (all file types, not claude_chat-specific)

`search_rag()` now accepts an optional `file_types: list[str] | None` parameter. When
provided, a `FieldCondition(key="file_type", match=MatchAny(any=file_types))` is added
to the Qdrant filter alongside the existing ownership filter. Both conditions are in `must`.

Valid values match the existing `file_type` payload field: `pdf`, `docx`, `pptx`, `ppt`,
`txt`, `md`, `yaml`, `ipynb`, `claude_chat`.

A list (not a single string) was chosen deliberately so multi-type queries work in one
call: `["pptx", "ppt"]` covers both PPTX and PPT without a second round-trip.

The tool spec in `build_tool_spec()` was updated with three usage examples — general search,
presentations only, and claude_chat sessions only — so both Sonnet 4.6 and Gemma 4 know
when and how to apply the filter.

`format_tool_result()` now includes `session_title` in the LLM-facing output when present,
allowing the model to cite sessions by name rather than UUID path.

`utils/rag_search_cli.py` gains a `--file-types TYPE [TYPE ...]` flag and displays
`session_title` in the File/Title column for claude_chat results.

### Rationale

**Why `session_title`?** RAG search results for Claude Code sessions were displayed as
raw UUID-based paths (e.g. `.../projects/-mnt-nfs-.../721bd06d-....jsonl`). These are
unreadable to users and to the LLM — the model cannot cite a session meaningfully when
only a UUID is available.

**Why no StateDB schema change?** Title derivation happens entirely at ingest time inside
`_parse_claude_chat()`. The title map and synthesis logic only touch the claude_chat
parsing path. Storing the title in StateDB (as an additional column) would require a
migration and would add coupling between the scanner and the ingest worker. Building it
at parse time is simpler, self-contained, and zero-overhead for NFS documents.

**Why `file_types` as a list?** `.ppt` and `.pptx` are distinct `file_type` values in
the Qdrant index (file extension is preserved at ingestion). A user asking "search my
PowerPoint presentations" naturally means both. A single-string parameter would require
two separate tool calls or a special-cased comma-split convention. `MatchAny(any=file_types)`
with a list is the correct Qdrant primitive — one filter, one network round-trip.

**Why add it to the tool spec rather than a separate endpoint?** The existing XML sentinel
tool-use loop in `main.py` is the single retrieval path for both Sonnet 4.6 and Gemma 4.
Extending the `search_documents` tool arguments is backward-compatible (LLM omits
`file_types` for general search) and keeps both models on the same code path. Adding a
second endpoint would duplicate the tool-use loop and the LLM's system prompt.

### Re-ingestion required

Existing Qdrant points for claude_chat sessions (178 sessions, ~3,369 points as of
2026-05-01) have no `session_title` field. To populate:

```bash
# Reset all .jsonl rows to pending
sqlite3 /mnt/nfs/__Backups/SoHoAI--databases/sqlite/rag_state.db \
  "UPDATE ingestion_queue
   SET status='pending', retry_count=0, error_msg=NULL, skip_reason=NULL,
       started_at=NULL, completed_at=NULL, progress_detail=NULL
   WHERE file_path LIKE '%.jsonl';"

# Re-ingest (daemon step 0 deletes stale Qdrant points before upserting)
source ~/Gin-AI/.Gin-AI-python-3.12/bin/activate
cd /mnt/nfs/Florian/Gin-AI/projects/SoHoAI
python utils/rag_ingest_daemon.py --workers 1 --batch 5 --log-file /tmp/rag-ingestion.log
```

### Verification results (pre-reindex, 2026-05-01)

| Check | Result |
|---|---|
| All 6 modified modules import cleanly | ✅ |
| `_build_title_map()` returns 3 named sessions | ✅ (`claude-orchestra-full`, `troubleshoot-ollama-after-kernel-upgrade`, `updated-logic`) |
| Named session title lookup | ✅ `troubleshoot-ollama-after-kernel-upgrade` |
| Unnamed session synthesis | ✅ `"Using the files in the utils/ directory update NotebookLM"` (57 chars) |
| Subagent title + suffix | ✅ `"You currently run in a tmux session... [subagent]"` |
| `--file-types pptx ppt` returns only presentations | ✅ 5 PPTX results, no other types |
| `session_title` in File/Title column (post-reindex) | pending re-ingest |
