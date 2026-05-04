---
title: "SoHoAI Design History"
created_at: 2026-05-01--13-40
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

---

## 2026-05-04--12-46 — `.claude/` exclusion + `claude_chats` two-path scanner design

### What was documented

Clarification of a design pattern that was implicit in the code but not explicitly explained in the documentation. The pattern concerns how Claude Code session files (`~/.claude/projects/`) are handled by the RAG scanner: excluded from the generic NFS walker, but deliberately re-entered through a dedicated scanner with specialized parsing.

**1 file updated** (`docs/RAG-strategy.md` §1.2).
**1 new design history entry** (this entry).

### The design pattern

#### Three categories of content under `~/.claude/`

| Path | Content | RAG relevance | Treatment |
|---|---|---|---|
| `~/.claude/projects/<mangled>/<uuid>.jsonl` | Session transcripts (user/assistant turns) | ✅ High — real conversational knowledge | **Ingested via dedicated scanner + parser** |
| `~/.claude/chats/<category>/<name>.jsonl` | Symlinks to `projects/` files, with human-chosen names | Read-only for title lookup | **Never ingested; consulted at ingest-time only** |
| Everything else (settings, caches, IDE config, tool metadata, MCP configs) | Claude Code's internal tooling state | ❌ None — noise / sensitive | **Excluded entirely** |

#### Why `.claude/` appears in both `exclude_dir_names` and `claude_chats.roots`

**In `config.yaml` `rag.scanner.exclude_dir_names`:**
```yaml
exclude_dir_names:
  - ".claude/"    # ← blocks the entire directory from the generic NFS scanner
```

This tells `scan_nfs_roots()` to completely skip the `.claude/` subtree when walking the NFS. This prevents:
1. Ingesting Claude Code's own internal state files (config, caches, IDE metadata) as documents
2. Picking up `.jsonl` session files incidentally through the wrong code path (they need special parsing to extract coherent dialogue)
3. **Most critically**: ingesting the `.claude/chats/` symlink tree, which contains aliases to the same `.jsonl` files that live in `projects/`. Without exclusion, the same session would be ingested twice under different logical paths, producing duplicate Qdrant points.

**In `config.yaml` `claude_chats.roots`:**
```yaml
claude_chats:
  roots:
    - path: /home/florian/.claude/projects
      owner: florian
```

A **separate, dedicated scanner function** `scan_claude_chats()` re-enters the specific subdirectory `~/.claude/projects/` and walks it independently. This function:
1. Bypasses all generic exclusion rules (it's not called through `scan_nfs_roots`)
2. Scans *only* for `.jsonl` files (no other file types)
3. Uses a dedicated parser `_parse_claude_chat()` that understands the JSONL format

#### Why the dedicated parser is mandatory

A raw UTF-8 read of a Claude Code `.jsonl` session file produces unreadable output:

```json
{"type":"user","sessionId":"abc-123","timestamp":"2026-05-04T12:00Z","message":{"content":"What is RAG?"}}
{"type":"assistant","sessionId":"abc-123","timestamp":"2026-05-04T12:01Z","message":{"content":"RAG stands for..."}}
```

Embedding this as plain text would create chunks containing JSON syntax (`{"type":`, `"sessionId":`, etc.) rather than the actual conversation. The resulting embeddings would be useless for retrieval.

`_parse_claude_chat()` instead:
- Extracts user/assistant text turns only (skips meta entries, tool_use blocks, tool_result blocks, thinking blocks)
- Synthesizes a human-readable session title via either:
  - Looking up the symlink name from `~/.claude/chats/` if one exists (via `_build_title_map()`)
  - Reading the first user message from the JSONL as a fallback title
- Returns clean dialogue text and metadata for Qdrant ingestion

#### The title lookup side channel

`_build_title_map()` walks `~/.claude/chats/` as a **read-only lookup table**. It builds a mapping of session file paths to human-chosen names:

```python
# ~/.claude/chats/projects/my-project.jsonl → symlink to .../.../projects/<uuid>.jsonl
# _build_title_map() discovers: {"/real/path/.../projects/<uuid>.jsonl": "my-project"}
```

This map is cached per daemon run (LRU) and consulted during ingest to populate the `session_title` field in Qdrant. The `chats/` directory itself is **never ingested**—it's only used for metadata enrichment.

### Rationale

This two-path architecture (generic `scan_nfs_roots` + exclusion vs. dedicated `scan_claude_chats`) solves three competing constraints:

1. **Avoid duplicate ingestion**: Without excluding `.claude/`, the `chats/` symlink tree would cause the same sessions to be ingested twice (once per path alias). The exclusion prevents this.
2. **Ingest sessions at all**: The sessions have high RAG value (real conversations about real work), so excluding `.claude/` entirely would be a loss. The dedicated scanner ensures they are still processed.
3. **Process them correctly**: Sessions require structured parsing to extract dialogue turns, not generic text read. A dedicated parser produces RAG-quality chunks; a generic read produces unreadable JSON blobs.

### Code locations

- **Generic scanner (NFS roots)**: `rag_engine/scanner.py::scan_nfs_roots()` — reads `config["rag"]["scanner"]["exclude_dir_names"]`
- **Dedicated scanner (claude chats)**: `rag_engine/scanner.py::scan_claude_chats()` — reads `config["claude_chats"]["roots"]`
- **Session parser**: `rag_engine/ingest.py::_parse_claude_chat()` — structured dialogue extraction
- **Title map builder**: `rag_engine/ingest.py::_build_title_map()` — reads `~/.claude/chats/` for title lookup (cached, read-only)
- **Ingest orchestrator**: `utils/rag_sync_nfs.py` and `POST /v1/rag/ingest/sync` — call both scanners, merge `existing_paths`, call `find_deleted()` once, then purge Qdrant before SQLite cleanup

### Documentation impact

Added clarification to `docs/RAG-strategy.md` §1.2 ("Exclusion filters") with a subsection explaining the `.claude/` exclusion and `claude_chats` two-path design pattern. This makes the implicit design visible in the documentation.

### Future enhancements (not in scope)

- Allow `claude_chats.roots` to point to directories other than `.claude/projects/` (e.g. other users' sessions on the NAS)
- Support additional session formats (not just Claude Code `.jsonl`)
