---
title: "SoHoAI Design History"
created_at: 2026-05-01--13-40
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Opus 4.7)
updated_at: 2026-05-27--14-30
context: >
  Running log of significant design decisions, feature additions, and architectural
  changes to the SoHoAI project. Each entry is timestamped and includes rationale.
  Complements CLAUDE.md (which documents current state) by preserving the reasoning
  behind changes as they were made.
---

# SoHoAI Design History

---

## 2026-05-27 — Opencode RAG ingestion, single-slot llama-server, chat title rendering

Three related changes from one session, committed as `e1af344`, `306a99e`, `12df6a2`.

### Opencode session ingestion (commit `e1af344`)

OpenCode (the agent used by octmux) joins Claude Code as a second external LLM agent whose sessions get ingested into the `documents` Qdrant collection — same `session_title` + content treatment, queryable via `GET /v1/rag/search?file_types=opencode`.

Architectural shape diverges from Claude Code: OpenCode has no on-disk session files, so the scanner queries its HTTP API (`/project`, `/session?directory=...`) instead of walking NFS. Sessions are referenced via synthetic `opencode://{session_id}` keys (kept in `ingestion_queue.file_path` and Qdrant `source_path`); the parser then fetches `/session/{id}/message` to extract text-only parts (skipping file/tool/reasoning/patch/snapshot parts).

Critical invariant: if the API is unreachable, `scan_opencode_sessions()` returns `existing_paths=None` (not an empty set). Callers treat `None` as "skip this source in `find_deleted()`" so existing opencode Qdrant points are preserved instead of mass-purged when the opencode server happens to be down at sync time.

Two config keys must both point at the live opencode server (currently `http://192.168.1.95:4096`, not `localhost`):
- `opencode.api_url` — scanner uses this
- `rag.opencode_api_url` — parser uses this (injected at startup in `lifespan()` and at daemon entry in `rag_ingest_daemon.py`)

Initial run after activation: 30 opencode sessions discovered, 5 ingested as smoke test, remainder queued for the systemd-driven Server 2 ingestion service.

### llama-server: single-slot 262144 ctx (commit `306a99e`)

Qwen3.5-4B llama-server (and the llama-swap config that fronts it) had been running `--parallel 2 --ctx-size 262144`, giving two slots × 131072 ctx. KV cache slot session pinning is not yet implemented — a chat coming back into the conversation cache can land on whichever slot the scheduler picks, and miss its saved KV state. Stopgap: drop `--parallel 2` entirely, run a single slot at the full 262144 ctx. Proxy-advertised `max_tokens` / `max_input_tokens` / `context_window` bumped to 262144 to match what external clients (Cline, Claude Code) now see via `/proxy/v1/model/info`. Documented in CLAUDE.md's runbook command and RAG-troubleshoot.md's expected `ps aux` flags.

### Chat sessions render by title, not by UUID (commit `12df6a2`)

Search results for opencode hits were displaying the raw session id (`ses_1b5c9eed...`) in the File/Title column instead of the friendly title — the `utils/rag_search_cli.py` and `~/.claude/commands/rag.md` display paths special-cased `file_type == "claude_chat"` to swap in `session_title`, and opencode was omitted. Same root-cause as the Claude Code session_title work (2026-05-01): `file_name = Path(file_path).name` produces a meaningless identifier for chat sources (`UUID.jsonl` for CC, `ses_<id>` for opencode), so renderers needed special-casing to swap in `session_title`.

Root-cause fix in `rag_engine/ingest.py`: when a parser returns `chat_meta[FIELD_SESSION_TITLE]`, override `file_name` with it before the Qdrant upsert. Both chat types now store the human-readable title in `file_name` natively. The display fallback in `rag_search_cli.py` was also extended to recognise `file_type=opencode` alongside `claude_chat` as a belt-and-braces measure — covers ~16.8k pre-existing claude_chat points and 5 pre-existing opencode points whose stored `file_name` still holds the raw identifier (they pick up the new label on re-ingest).

---

## 2026-05-16 — OAuth Bearer auth support for Claude Code

Added OAuth authentication support to the gateway's transparent Anthropic forward path. `_anthropic_messages_forward()` and `count_tokens_endpoint()` now detect `Authorization: Bearer <token>` and forward it unchanged, enabling `claude login` as an alternative to API-key mode. Both auth modes coexist and are detected at request time. No changes to LiteLLM path or proxy endpoints.

---

## 2026-04-16 — Phase 1 core loop complete

### What was implemented

The foundational orchestrator loop for SoHoAI: FastAPI server on Server 1, LiteLLM routing with fallback, Anthropic prompt caching, llama-server on Server 2 GPU, per-conversation KV cache persistence, rolling summarization, and full conversation memory (Redis short-term + SQLite long-term + KV cache).

**All Phase 1 deliverables**:
- FastAPI orchestrator on Server 1
- LiteLLM routing with fallback chain: external (Sonnet 4.6, cloud) → internal (Gemma 4, local)
- Anthropic prompt caching on the external path (`SmartRouter._apply_cache_control()`) — ephemeral breakpoints on system + `messages[-2]` rolling prefix
- llama-server on Server 2 GPU: Gemma 4 E4B 7.52B Q8_0, 2 slots × 110024 ctx
- Per-conversation KV cache persisted to NAS via llama-server slot API (internal path only)
- Rolling summarization at ~100K token threshold; summary + boundary `message_id` persisted to SQLite
- Redis conversation cache with NAS AOF persistence
- SQLite chat store with full CRUD + `summary_text`/`summary_covers_through_message_id` columns
- Markdown and JSONL export
- CLI chat client (`utils/cli_chat.py`)
- Feedback collection for RL
- MCP server for Gin-AI filesystem (`NFS-files--MCP-server/`)

### Per-request flow (tool-use loop)

Both external (primary, Sonnet 4.6) and internal (fallback, Gemma 4) share the same loop structure. The only branch is the inference step at `main.py:333-347`:

```
resume(chat_id)           → assign slot + restore KV from NAS (no-op on external)
append(user_msg)          → Redis + SQLite
maybe_summarize(store=…)  → if >400K chars (~100K tokens): summarize old turns via
                            internal (Gemma), rebuild Redis, erase stale KV,
                            persist summary + boundary id to SQLite
get_context()             → condensed history from Redis
build_system_prompt()     → off/on/only mode → tool spec injected into system message
loop (max 2 iterations):
  # Branch on selected model:
  if target == internal (fallback):
    apply_gemma_template()   → <|turn>role\n… format; stop=["<|turn>"]
    inference(slot_id)       → POST /completion to llama-server GPU slot
  else (primary = external):
    router.complete()        → LiteLLM → Anthropic with cache_control markers
                               (cache_read grows each turn as prefix rolls forward)
  parse_tool_call()       → detect <tool_call>…</tool_call> in output
  if tool_call:
    _retrieve()           → search_rag() (multi-query permanently disabled)
    append tool result    → re-enter loop
  else: final answer → break
append(assistant)         → Redis + SQLite
park(chat_id)             → save KV slot to NAS + refresh Redis TTL
```

---

## 2026-04-21 — Phase 2 RAG + Advanced features complete

### What was implemented

Initial ingestion run produced **2891 files completed, 0 pending, 0 ignored**, yielding **98,737 Qdrant points** in the `documents` collection (avg ~34 chunks/file). End-to-end retrieval + chat injection verified via `utils/rag_smoke_test.py`.

**Multi-tenancy** — per-user document isolation via Google OAuth2:
- Each family member has a private NFS root (`/mnt/nfs/{Florian,Eva,Annika,Laura}`)
- Shared content under `/mnt/nfs/La-Familia` visible to all authenticated users
- `owner` field in every Qdrant point; search filtered by `MatchAny(any=[user_owner, "la-familia"])`
- `user_id` field added to `ChatRequest`, `SearchRequest`, and SQLite `chats` table
- User→NFS root mapping in `SoHoAI-config.yaml` (`users:` + `shared:` sections)

**Document ingestion**: `docling` (PDF, PPTX, DOCX) + dedicated ipynb cell extractor + `python-pptx` PPTX fallback + direct UTF-8 read (TXT, MD, YAML, CSV):
- **ipynb NOT handled by docling** — Fix: `_parse_ipynb()` in `ingest.py` parses JSON directly, extracts markdown cells as prose and code cells as fenced blocks, skips outputs and empty cells.
- **PPTX docling format detection failure** — Fix: `_parse_pptx()` in `ingest.py` uses `python-pptx` as secondary fallback; iterates slides/shapes, extracts all text frame content with `Slide N` headers. Raw UTF-8 read is last resort only if `python-pptx` also fails. 29 affected files force-re-queued (2026-04-19).

**Chunking — parent-child strategy**:
- Child chunks (~250 tokens, 20 overlap) → embedded, stored in Qdrant as search index
- Parent chunks (~800–1200 tokens, 100 overlap) → raw text only, stored in Qdrant payload (`parent_text` field)
- Flat 512-token chunks used only for PPTX slides and short TXT/YAML/config files

**Embeddings via bge-m3 served by Ollama** — 1024 dimensions, 570M params, ~1.2GB, top MTEB scores:
- Two modes: CPU (Server 1) ~650ms/chunk; GPU (Server 2, RTX 5070) ~10–20ms/chunk
- `embed_batch()` runs up to N concurrent requests via asyncio semaphore; N = `--batch` in daemon
- `embed_batch()` accepts optional `progress_cb(done, total)` — called every 50 chunks; used by `ingest.py` to log lines read by `rag_status.py --watch`
- HTTP timeout in `embed_text()`: **120s** — must survive concurrent embedding load

**Package**: `rag_engine/` — fully implemented (schema, collection, embeddings, state, scanner, ingest, search, multi_query, tool_use)

**Advanced RAG features (§8.1–§8.3)**:
- **`rag_mode` enum** (`off` / `on` / `only`) — legacy boolean toggle removed
- **Tool-use via system prompt** — LLM decides when to retrieve; no unconditional top-k injection
- **Multi-query + MMR** — `rag_engine/multi_query.py`; **permanently disabled** (`rag.multi_query.enabled: false`). Evaluated 2026-04-22: no-go verdict. Standard single-query retrieval is sufficient for this corpus.

**Phase 2 implementation checklist** — all 14 items completed and strikethrough:
~~1. Replace `unstructured` with `docling`~~ ~~2. Replace `sentence-transformers` with Ollama in `rag.py`~~ ~~3. Fix `SoHoAI-config.yaml` RAG section (bge-m3, ollama_url)~~ ~~4. Add `owner` to Qdrant payload schema; `user_id` to `ChatRequest`/`SearchRequest`~~ ~~5. Add multi-user config (`users:` + `shared:` sections) to `SoHoAI-config.yaml`~~ ~~6. Implement `rag_engine/` package (schema, collection, embeddings, state, scanner, ingest, search)~~ ~~7. Wire `rag_engine` into `main.py`; delete `rag.py`~~ ~~8. Implement standalone CLI utils (`utils/rag_*.py`)~~ ~~9. Add `POST /v1/rag/ingest/*` FastAPI endpoints~~ ~~10. Add `db_base_path` global config variable~~ ~~11. Configure `users:` section in `SoHoAI-config.yaml` with real Google emails~~ ~~12. Run initial NFS scan and ingestion~~ ~~13. Implement `rag_mode` (off/on/only), system-prompt tool-use loop, `prompts/` module~~ ~~14. Implement multi-query + MMR reranking (`rag_engine/multi_query.py`)~~

**→ NOW**: Phase 3 — Google OAuth2 middleware + OpenAI-compatible response format for Open WebUI (not blocking; RAG works end-to-end today via `--user florian`).

Full design details and worker loop spec: `RAG-strategy.md`.

---

## 2026-04-20 — Quantization benchmark: Q6_K → Q8_0

### Benchmark methodology

Tool: `~/Gin-AI/tools/llama-performance-test/llama_perf_test.py`

3 runs averaged per scenario, `/completion` native endpoint, `cache_prompt: false` (cold prefill every run), `ignore_eos: true`, `n_predict: 200`, `temperature: 0` (greedy).

Results: `~/Gin-AI/tools/llama-performance-test/results_q6k.json` (Q6_K baseline), `results_q8_0.json` (Q8_0 current)

### Q6_K → Q8_0 comparison

Measured 2026-04-20, RTX 5070 12 GB, 2 slots × 131072 ctx at time of measurement

| Scenario | Q6_K tok/s | Q8_0 tok/s | Δ |
|---|---|---|---|
| **Decode serial** | 109.4 | 96.2 | **−12%** |
| Decode parallel/slot0 | 101.1 | 89.4 | −12% |
| Decode parallel/slot1 | 101.5 | 89.8 | −12% |
| Decode parallel combined | 202.6 | 179.2 | −12% |
| Prefill short (13 tok) | 440 | 399 | −9% (noisy) |
| **Prefill long (4126 tok)** | 5420 | 6270 | **+16%** |
| Wall clock short serial | 1.86s | 2.12s | +14% |
| Wall clock long serial | 2.68s | 2.82s | +5% |
| Parallel wall clock | 2.03s | 2.29s | +13% |
| Parallel speedup ratio | 0.92× | 0.92× | identical |

### Key findings

- **Decode is ~12% slower** — Q8_0 weights are 27% larger (5.9 → 7.5 GB); decode is memory-bandwidth-bound so larger weights cost proportionally more per step.
- **Long-context prefill is +16% faster** — Q8_0's uniform INT8 maps cleanly to tensor core INT8 ops; Q6_K's mixed-precision K-quant requires a dequantize step that costs more at high token counts.
- **Real-world impact at 200 output tokens**: ~260ms extra latency per turn. At 100 tokens: ~130ms. Imperceptible in streaming mode.
- **Continuous batching unchanged**: 0.92× parallel speedup ratio is identical across both quantizations.
- **Conclusion**: ~12% decode regression is the cost of near-f16 weight quality. Acceptable for interactive use at 96 tok/s serial decode.

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

**In `SoHoAI-config.yaml` `rag.scanner.exclude_dir_names`:**
```yaml
exclude_dir_names:
  - ".claude/"    # ← blocks the entire directory from the generic NFS scanner
```

This tells `scan_nfs_roots()` to completely skip the `.claude/` subtree when walking the NFS. This prevents:
1. Ingesting Claude Code's own internal state files (config, caches, IDE metadata) as documents
2. Picking up `.jsonl` session files incidentally through the wrong code path (they need special parsing to extract coherent dialogue)
3. **Most critically**: ingesting the `.claude/chats/` symlink tree, which contains aliases to the same `.jsonl` files that live in `projects/`. Without exclusion, the same session would be ingested twice under different logical paths, producing duplicate Qdrant points.

**In `SoHoAI-config.yaml` `claude_chats.roots`:**
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

---

## 2026-05-15 — Proxy hardening: subagent blocking, geometric backoff, tool_use ID sanitization

### Problem

Three production issues surfaced during Claude Code sessions using `claude-code-kimi-k2.6`:

1. **Haiku subagent leak**: CC internally auto-spawns `claude-haiku-4-5-20251001` for lightweight tasks even when the main model is `claude-code-kimi-k2.6` (an Ollama Cloud model). These Haiku requests hit `/v1/messages`, resolved to `None` in `_resolve_proxy_model()`, and were transparently forwarded to Anthropic — incurring unexpected Anthropic API cost while the user believed they were on a $0 Ollama session.

2. **No backoff on Ollama Cloud timeouts**: When Ollama Cloud timed out (30s), the error was immediately surfaced to CC as HTTP 529. No retries. A transient slowness caused the whole task to fail.

3. **Invalid `tool_use.id` from Ollama models**: Kimi K2.6 (and likely other Ollama models) returns tool call IDs in the format `functions.Bash:38` — containing `.` and `:` which violate Anthropic's required pattern `^[a-zA-Z0-9_-]+$`. Our LiteLLM path passed these IDs through unchanged. CC stored them in its conversation history. On a subsequent turn using an Anthropic-native model (transparent forward), Anthropic returned HTTP 400. Confirmed via session `/home/florian/.claude/projects/-mnt-nfs-Florian-Gin-AI-projects-claude-orchestra/31111cfa-ca5d-4b9a-a87c-b25ebb3fbeff.jsonl`.

### Solutions

**Fix 1 — Subagent blocklist** (`SoHoAI-config.yaml` + `main.py`):

Added `proxy.blocked_models` list to `SoHoAI-config.yaml`. In `anthropic_messages()`, before any routing logic, the model is checked against this list and rejected with HTTP 400 (`not_supported_error`) if it matches. Default list: `[claude-haiku-4-5-20251001]`.

```yaml
proxy:
  blocked_models:
    - claude-haiku-4-5-20251001
```

CC receives a 400 (not retried) and falls back to using its main model for the task, or skips the subagent entirely. Additional model names (future versioned Haiku releases etc.) can be added without code changes.

**Fix 2 — 3-step geometric backoff** (`router.py`):

Replaced the single `request_timeout=30` for `ollama-cloud/*` targets with a 3-attempt retry loop using increasing timeouts: 30s → 60s → 90s. Only `litellm.Timeout` triggers a retry; auth errors and 4xx responses propagate immediately. Worst-case total: 180s (within CC's 300s httpx limit).

**Fix 3 — Tool_use ID sanitization** (`main.py`):

Added `_sanitize_tool_use_id()` helper and `_TOOL_ID_VALID = re.compile(r'^[a-zA-Z0-9_-]+$')` above the Anthropic utilities section. The helper replaces invalid characters with `_` (deterministic substitution — `functions.Bash:38` → `functions_Bash_38`). Applied in both the streaming SSE path (line ~1862) and non-streaming path (line ~1935) of `_anthropic_messages_litellm()`. The old `f"toolu_{uuid.uuid4().hex[:16]}"` fallbacks are unified into this helper.

### Design notes

- **Subagent blocklist** is intentionally a denylist (explicit opt-in). A wildcard block of all non-`claude-code-*` models would also block legitimate main-model Haiku sessions.
- **Backoff timeouts grow, not waits** — the 30s/60s/90s are per-attempt timeout durations, not sleep delays between retries. This avoids the 5-minute total that sleep-based backoff would produce.
- **Tool ID substitution is deterministic** — same input, same output. CC stores the sanitized ID and sends it back as `tool_use_id`; the round-trip is consistent. Most Ollama models don't validate `tool_call_id` strictly, so the tool-use loop continues to function.

### Files changed

| File | Change |
|------|--------|
| `SoHoAI-config.yaml` | New `proxy.blocked_models` key |
| `main.py` | Blocklist check in `anthropic_messages()`; `_TOOL_ID_VALID` + `_sanitize_tool_use_id()` helper; both tool_use ID sites updated |
| `router.py` | `import litellm`; 3-step timeout backoff for `ollama-cloud/*` |

---

## 2026-05-15 — Backoff threshold correction: 30s→60s→90s → 60s→90s→120s

### Problem

Production logs (`/var/tmp/SoHoAI-gateway-timeout.log`) showed that the 30s initial
timeout introduced in the proxy hardening commit was too aggressive for kimi-k2.6.
kimi-k2.6 is a large reasoning model that regularly takes 30–60 s for complex coding
tasks. As a result, every non-trivial request was guaranteed to fail attempt 1, then
succeed on the 60s retry — adding +30s latency to every complex request and filling
logs with spurious timeout errors.

Two representative events from the log (confirmed the code was loaded — no race condition):
- `12:08:01` call → `12:08:23` timeout at exactly 30.0s → `12:08:23` retry at 60s → `12:08:54` 200 OK (31s)
- `12:09:25` call → `12:09:55` timeout at exactly 30.0s → `12:09:55` retry at 60s starts

The backoff loop was functionally correct; only the starting threshold was wrong.

### Solution

Changed `_backoff_timeouts = [30, 60, 90]` → `[60, 90, 120]` in `router.py`.

- **60s first attempt** covers both fast (<10s) and normal-slow (30–60s) responses without failing
- **90s** covers unusually slow responses (60–90s) — second attempt
- **120s** covers extreme cases — third attempt
- Worst case: 270s (within CC's 300s httpx limit)

### Files changed

| File | Change |
|------|--------|
| `router.py` | `_backoff_timeouts = [60, 90, 120]`; updated comment |
| `docs/Model-routing.md` | §2.3 timeout table updated; note added explaining the correction |

---

## 2026-05-15 — Local inference model swap: Gemma 4 E4B → Qwen3.5-4B

### Motivation

Qwen3.5-4B-UD-Q6_K_XL targeted as drop-in local replacement for Haiku-4.5 in Claude Orchestra actor sub-agent role. Gemma4 E4B tool-use reliability was unvalidated; Qwen3.5 offers stronger instruction-following and standardized ChatML format compatible with llama.cpp `--jinja`.

### Changes

**Model file**: `google_gemma-4-E4B-it-Q8_0.gguf` → `Qwen3.5-4B-UD-Q6_K_XL.gguf`

**llama-server**:
- Added `--jinja` (ChatML template parsing)
- Changed `--cache-type-k/v f16` → `q8_0` (8-bit KV quantization)
- Removed `--cache-ram 0` (allows offload to system RAM if needed)

**Chat template**: `apply_gemma_template()` (Gemma `<|turn>` markers) → `apply_qwen3_template()` (ChatML `<|im_start|>/<|im_end|>`)

**Stop token**: `<|turn>` → `<|im_end|>`

**Model aliases**:
- `internal/gemma-4-e4b` → `internal/qwen3-4b`
- `openai/gemma4` → `openai/qwen3-4b`

**KV cache**: `.bin` files purged (format incompatible across models)

**VRAM**: ~9.3 GB (Gemma4 f16) → ~6 GB estimated (Qwen3.5 q8_0, not yet measured)

### Tool-use validation

Pending smoke test against live server.

---

## 2026-05-09 — Dynamic file discovery in `utils/snapshot_codebase.py`

### Problem

`snapshot_codebase.py` maintained a hardcoded `SNAPSHOT_FILES` list that had drifted
from the actual repo contents. Twelve committed files were absent:
`usage_tracker.py`, `prompts/rag_system_prompts.py`, `rag_engine/multi_query.py`,
`rag_engine/tool_use.py`, `rag_engine/main.py`, `utils/rag_smoke_test.py`,
`utils/rag_mmr_bench.py`, `scripts/rag-ingest-run.sh`,
`scripts/qdrant/{qdrant-config.yaml,qdrant-snapshot.sh}`, and
`NFS-files--MCP-server/{nfs_files_mcp_server,setup_mcp}.sh`.

### Solution

Replaced the static list with `discover_files()`, which calls
`git ls-files --cached --others --exclude-standard` filtered by
`INCLUDE_EXTENSIONS = {".py", ".yaml", ".yml", ".toml", ".sh"}`.

- `.gitignore` already excludes `.venv/`, `__pycache__/`, `.claude/`, and the generated
  `utils/codebase_snapshot.md` — no separate exclude list is needed.
- `_SNAPSHOT_FILES_FALLBACK` (the old static list) is retained for environments where
  `git` is unavailable.
- Added `.sh → bash` syntax highlighting to `_lang()`.
- Added `--extensions` CLI arg for runtime override.

**Result:** 39 files included automatically (up from 27), zero maintenance required.

---

## 2026-05-10 — Code-quality cleanup: indentation, dead code, config-driven model_info

### Problem

Three code-quality issues accumulated during the `claude-code-*` alias scheme +
`count_tokens` feature additions (commits 02ea419 and 5ba6971):

1. **Mixed indentation** — six newly added functions (`_extract_text_from_blocks`,
   `count_tokens_endpoint`, `_claude_code_alias_for`, `_claude_code_alias_to_public`,
   `_display_name_for`, `_LITELLM_ROUTED`) used hard-tab indentation; the rest of
   `main.py` uses 4-space. Python 3 permits mixed indentation across blocks but
   flake8 flags W191 and any auto-formatter would mass-rewrite unrelated lines.

2. **Dead but buggy code in `_display_name_for`** — the `anthropic/` branch contained
   a no-op ternary (`suffix[-8:].isdigit()` check whose two arms were identical),
   producing wrong display names for versioned IDs. Since commit 5ba6971 excluded
   `anthropic/*` from `list_models()`, this branch was unreachable — unreachable
   buggy code is harder to remove safely later.

3. **Hardcoded model_info fallbacks in `list_models()`** — all four `ollama-cloud/*`
   models lacked `model_info:` blocks in `SoHoAI-config.yaml`, so their context-window and
   max-token values were hardcoded in a Python fallback dict. The `internal/gemma-4-e4b`
   and `anthropic/claude-sonnet-4-6` entries already had proper `model_info:` blocks;
   the Ollama Cloud entries just weren't updated at introduction time.

### Fix

1. **Tabs → 4-space**: `expand -t 4` applied to `main.py`; all 169 tab-indented
   lines normalized. Zero tabs remain.

2. **Dead branch deleted**: the entire `if public_id.startswith("anthropic/"):` block
   removed from `_display_name_for`, replaced with a one-line comment explaining the
   exclusion rationale.

3. **Config-driven model_info**: added `model_info:` blocks to all four `ollama-cloud/*`
   entries in `SoHoAI-config.yaml` (same structure as existing Anthropic/Gemma entries:
   `id`, `description`, `max_tokens`, `max_input_tokens`, `context_window`), then
   deleted the 9-line hardcoded fallback dict from `list_models()`.

### Values recorded in SoHoAI-config.yaml

| Model | context_window | max_tokens |
|---|---|---|
| `ollama-cloud/deepseek-v4-pro` | 1 000 000 | 32 000 |
| `ollama-cloud/kimi-k2.6` | 256 000 | 32 000 |
| `ollama-cloud/glm-5.1` | 200 000 | 32 000 |
| `ollama-cloud/qwen3-coder-next` | 262 000 | 32 000 |

### Rationale

Two sources of truth for context-window values means one will get stale — if Ollama
bumps deepseek-v4-pro to 2M context, there are now two places to update and the
fallback dict is not where a maintainer would look. Config-as-the-single-source also
makes the values visible to any config-reading tool (health checks, docs generation,
etc.) without importing Python.

---

## 2026-05-16 — KV cache native API port fix after llama-swap deployment

### What changed

Deployed llama-swap on Server 2 to manage dual-model serving. The orchestrator's internal `KVCacheManager` was still pointing at port 8000 (llama-swap's OpenAI proxy), which does NOT forward native llama-server endpoints (`/completion`, `/slots/*`). This broke `utils/cli_chat.py` with 400 Bad Request on `/completion` and 404 Not Found on `/slots/0`.

**Three fixes:**

1. **`SoHoAI-config.yaml`**: `llama_server.base_url` changed from `8000` to `8010`
   — bypass llama-swap and hit the native 4B llama-server directly for KV operations.

2. **`kv_cache.py`**: `num_slots` default changed from `2` to `1`
   — cli_chat exclusively uses slot 0, leaving slot 1 free for OpenAI auto-assignment so LRU eviction for proxy requests does not collide with conversational KV.

3. **`kv_cache.py`**: defensive slot verification in `inference()`
   — queries `GET /slots` before inference. If the slot is empty or contains a different file, re-restores from NFS. Handles llama-server LRU eviction, restarts, and cross-path collisions.

### What does NOT change

- LiteLLM routing continues to use `http://192.168.1.95:8000/v1` for OpenAI-compatible inference via llama-swap
- Claude Code proxy endpoints unaffected
- llama-server on port 8002 (always-on)

### Code locations

- `SoHoAI-config.yaml`: `llama_server.base_url` (port 8000 → 8010)
- `kv_cache.py`: `__init__` num_slots default, `inference()` defensive slot check with `GET /slots`
- `main.py`: passes `chat_id` to `kv_cache.inference()`

---

### Code locations

- `main.py`: `_display_name_for()` (dead branch removed), `list_models()` (fallback removed),
  all six tab-indented functions (indentation normalized)
- `SoHoAI-config.yaml`: `ollama-cloud/*` entries (lines 83–125 after this change)

---

## 2026-05-22 — Fix gateway 404 / 400 for CC model annotations and Anthropic versioned IDs

### Problem

Two classes of HTTP errors were appearing in the gateway log:

**404 on `POST /v1/messages`** — Claude Code sends `model: "claude-sonnet-4-6[1m]"` (its
internal annotation for the 1M-context Sonnet variant). `_resolve_proxy_model` couldn't match
it, so the gateway fell through to transparent forwarding — but forwarded the body unchanged,
with the `[1m]` suffix intact. Anthropic's API doesn't recognise this model ID → 404.
Side-effect: CC's auto-mode **Bash safety classifier** calls `/v1/messages` to decide if a
shell command is safe; a 404 there causes `"claude-sonnet-4-6[1m] is temporarily unavailable"`
and blocks all Bash execution in `auto` permission mode (seen in claude-orchestra sessions).

**400 on `POST /v1/messages/count_tokens`** — CC sends `model: "claude-haiku-4-5-20251001"`
(Anthropic's date-versioned ID format). The gateway's validation gate only knew the bare alias
`claude-haiku-4-5`, so it returned HTTP 400 "model not exposed by this proxy".

### Root cause

`_resolve_proxy_model` normalised neither CC's `[…]` context-window annotations nor
Anthropic's `-YYYYMMDD` date suffixes before performing alias lookups. Both forms are
real, valid identifiers that resolve to a known model once stripped.

### Fix (3 changes to `main.py`)

1. **`_resolve_proxy_model`** — added a normalization fallback at the end: strip `[…]`
   and `-YYYYMMDD` suffixes, then retry recursively. Handles both annotation types at the
   single resolution point that all endpoints share.

2. **`anthropic_messages`** — strip `[…]` from the model name in the request body before
   forwarding to `api.anthropic.com`. Anthropic only accepts bare names; without this the
   404 persists even if resolution succeeds.

3. **`count_tokens_endpoint`** — same `[…]` stripping on the model name forwarded to
   Anthropic's count_tokens API.

### What does NOT change

- Date-versioned IDs (`claude-haiku-4-5-20251001`) are forwarded to Anthropic as-is after
  passing the resolution gate — Anthropic knows these natively.
- No changes to routing logic, LiteLLM path, or streaming path.

### Code locations

- `main.py`: `_resolve_proxy_model()`, `anthropic_messages()`, `count_tokens_endpoint()`

---

## 2026-05-22 — Remove `claude-code-*` alias scheme for Claude Code

**Context**: CC's native model picker contains all Anthropic models natively. The `claude-code-*`
alias scheme (exposing non-Anthropic models via `GET /v1/models` + `gateway-models.json` cache)
added complexity with minimal benefit: CC's local inference use-case is served by sub-agents
via the `/proxy/v1` path, not the CC model picker.

**What was removed**:
- `GET /v1/models` endpoint (`list_models()` in `main.py`)
- `_claude_code_alias_for()`, `_claude_code_alias_to_public()`, `_display_name_for()` helpers
- `claude-code-*` resolution branch in `_resolve_proxy_model()`
- `gateway-models.json` writing block in `start-sohoai.sh`
- `utils/alias_bijection_test.py`

**What is unaffected**: Cline / OpenCode via `/proxy/v1/*`; the Anthropic passthrough `/v1/messages`;
all telemetry. CC continues to use native Anthropic models via `ANTHROPIC_BASE_URL`.

**Operator action**: delete `~/.claude/cache/gateway-models.json` once to clear stale picker entries.
