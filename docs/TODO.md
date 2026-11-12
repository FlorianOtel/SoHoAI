---
title: "SoHoAI — Future work and deferred tasks"
created_at: 2026-05-04--17-30
created_by: Claude Code (Claude Sonnet 4.6)
updated_by: Claude Code (Claude Haiku 5.5)
updated_at: 2026-11-12--16-50
context: >
  Tracks deferred implementation work that is understood, scoped, and intentionally
  left for a future session. Each entry includes the motivation, the known approach,
  estimated effort, and the blocker or reason for deferral.
---

# SoHoAI — Future work

---

## Phase 3 — Web UI + Auth (dropped: chat UI deprecated in favour of OpenCode)

**Dropped 2026-11-12.** The stateful chat stack (Redis, KV cache, SQLite chat store, `/v1/chats*`)
was removed, so the web frontend, the MCP gateway, the nfs-files MCP server, the OpenAI-format
`ChatResponse` for Open WebUI, and the offline CLI fallback (whose only consumer was the removed
`/v1/messages` path) are all dropped with it.

**Retained (2026-11-12):** Google OAuth2 (OIDC) authentication. Not a chat-UI concern: the
family-member `users:` map drives RAG owner filtering in Qdrant, so identity still needs a
gateway-level answer. No code exists yet.

**Google OAuth2 (OIDC) authentication middleware** — family members authenticate with separate Google accounts within the same Google Family Group. User identity from JWT is mapped to `owner` via `SoHoAI-config.yaml` `users:` section; Qdrant search filtered by ownership.

---

## Phase 4 — Image search + RL (future)

**Status 2026-11-12:** no image files are ingested — scanner whitelist has no image extensions; 0 image points in Qdrant.

**CLIP model** (openai/clip-vit-base-patch32) on Server 2 GPU. Family photo ingestion → CLIP embeddings → separate Qdrant `images` collection (same Qdrant instance as `documents`). Text-to-image similarity search for family photo library.

**RL training data pipeline**: export conversations with feedback signals (thumbs up/down on turns) as DPO-format JSONL for training with TRL framework. Goal: fine-tune a smaller local model (e.g. Qwen3.5-4B or similar) on real family conversations to reduce API cost over time.
