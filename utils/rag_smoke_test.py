#!/usr/bin/env python3
"""
Smoke-test the RAG path end-to-end.

Exercises both layers with the same query and compares them:

  Phase 1 — retrieval       direct search_rag() call  → what Qdrant returns
  Phase 2 — chat completion POST /v1/chat/completions → what the orchestrator
                                                        injects + the LLM sees

If rag_sources from the chat response aren't a subset of the retrieval hits,
something broke between search and prompt injection (main.py:_build_rag_prompt).

Usage (run from project root):

    # Basic: retrieval + chat, Florian's corpus
    python utils/rag_smoke_test.py --query "what AWS certifications do I have" --user florian

    # Assert a known source shows up (exit 1 if not)
    python utils/rag_smoke_test.py \\
        --query "cisco DCNIDS certificate" --user florian \\
        --expect "cisco-DCNIDS"

    # Retrieval-only (skip the LLM call — faster)
    python utils/rag_smoke_test.py --query "..." --user florian --skip-chat

    # No ownership filter (pre-auth dev mode — searches all docs)
    python utils/rag_smoke_test.py --query "..." --no-filter
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import yaml  # noqa: E402

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client  # noqa: E402
from rag_engine.search import search_rag  # noqa: E402


# -- helpers ------------------------------------------------------------------

def _load_config() -> dict:
    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        return yaml.safe_load(f)


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * (78 - len(title)))


def _qdrant_stats(qdrant_url: str) -> tuple[bool, int]:
    try:
        r = httpx.get(f"{qdrant_url}/collections/{DOCUMENTS_COLLECTION}", timeout=5)
        r.raise_for_status()
        return True, r.json()["result"]["points_count"]
    except Exception as e:
        print(f"  Qdrant check failed: {e}")
        return False, 0


# -- phases -------------------------------------------------------------------

async def retrieval_phase(query: str, user_id: str | None, top_k: int, rag_cfg: dict) -> list[dict]:
    _section("Phase 1 — direct retrieval (search_rag)")
    qdrant_client = get_client(rag_cfg.get("qdrant_url", "http://localhost:6333"))
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
    )
    if not results:
        print("  (no results)")
        return []

    print(f"  {len(results)} hit(s) for {query!r}  user={user_id!r}\n")
    print(f"  {'#':<3} {'score':>6}  {'file':<40}  source")
    for i, r in enumerate(results, 1):
        print(f"  {i:<3} {r['score']:>6.4f}  {r['file_name'][:40]:<40}  {r['source_path']}")
    return results


def chat_phase(server: str, query: str, user_id: str | None, timeout: float) -> dict:
    _section("Phase 2 — /v1/chat/completions with use_rag=true")
    payload = {
        # Fresh chat_id so we don't pick up unrelated Redis history.
        "chat_id": str(uuid.uuid4()),
        "messages": [{"role": "user", "content": query}],
        "use_rag": True,
        "stream": False,
    }
    if user_id:
        payload["user_id"] = user_id

    try:
        r = httpx.post(f"{server}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  HTTP error: {e}")
        return {}

    data = r.json()
    model_used = data.get("model_used", "?")
    content = data.get("message", {}).get("content", "")
    sources = data.get("rag_sources") or []

    print(f"  model_used : {model_used}")
    print(f"  rag_sources: {len(sources)} source(s)")
    for s in sources:
        print(f"    - {s}")
    print(f"\n  assistant reply ({len(content)} chars):")
    preview = content[:600].rstrip()
    print("    " + preview.replace("\n", "\n    "))
    if len(content) > 600:
        print(f"    ... [{len(content)} chars total]")
    return data


# -- verdict ------------------------------------------------------------------

def verdict(
    retrieval: list[dict],
    chat_data: dict,
    expect: str | None,
    skipped_chat: bool,
    skipped_retrieval: bool,
) -> int:
    _section("Verdict")
    ok = True

    if not skipped_retrieval:
        if not retrieval:
            print("  ✗ retrieval returned 0 hits")
            ok = False
        else:
            print(f"  ✓ retrieval: {len(retrieval)} hit(s)")

    if not skipped_chat:
        sources = chat_data.get("rag_sources") or []
        if not chat_data:
            print("  ✗ chat endpoint failed")
            ok = False
        elif not sources:
            print("  ✗ chat returned no rag_sources (retrieval or injection failed)")
            ok = False
        else:
            print(f"  ✓ chat returned {len(sources)} rag_source(s)")

        if not skipped_retrieval and sources:
            retrieval_paths = {r["source_path"] for r in retrieval}
            overlap = [s for s in sources if s in retrieval_paths]
            if not overlap:
                print("  ✗ rag_sources don't match retrieval — injection path is wrong")
                ok = False
            else:
                print(f"  ✓ {len(overlap)}/{len(sources)} rag_sources match retrieval")

    if expect:
        sources = chat_data.get("rag_sources") or []
        retrieval_paths = [r["source_path"] for r in retrieval]
        haystack = sources if not skipped_chat else retrieval_paths
        if any(expect in path for path in haystack):
            print(f"  ✓ --expect {expect!r} found")
        else:
            print(f"  ✗ --expect {expect!r} NOT found in {'rag_sources' if not skipped_chat else 'retrieval'}")
            ok = False

    print(f"\n  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# -- main ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="RAG end-to-end smoke test")
    p.add_argument("--query", required=True)
    p.add_argument("--user", metavar="OWNER", default=None,
                   help="Owner filter, e.g. florian (required unless --no-filter)")
    p.add_argument("--no-filter", action="store_true",
                   help="Skip ownership filter (dev mode)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--server", default=None,
                   help="Orchestrator URL (default: http://<server1_ip>:8000 from config.yaml)")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Chat endpoint timeout in seconds (default 180)")
    p.add_argument("--expect", metavar="SUBSTR", default=None,
                   help="Assert this substring appears in a returned source path")
    p.add_argument("--skip-retrieval", action="store_true")
    p.add_argument("--skip-chat", action="store_true")
    args = p.parse_args()

    if not args.no_filter and not args.user:
        p.error("Provide --user OWNER or --no-filter")
    if args.skip_retrieval and args.skip_chat:
        p.error("Cannot skip both phases")

    user_id = None if args.no_filter else args.user
    config = _load_config()
    rag_cfg = config.get("rag", {})
    server = args.server or f"http://{config.get('server1_ip', '192.168.1.93')}:8000"

    # -- pre-flight --------------------------------------------------------
    _section("Pre-flight")
    qdrant_url = rag_cfg.get("qdrant_url", "http://localhost:6333")
    ok, points = _qdrant_stats(qdrant_url)
    print(f"  qdrant {qdrant_url}: {'up' if ok else 'DOWN'}, {points} point(s) in '{DOCUMENTS_COLLECTION}'")
    if not ok or points == 0:
        print("  (smoke test cannot proceed)")
        return 1

    # -- phases ------------------------------------------------------------
    retrieval: list[dict] = []
    chat_data: dict = {}

    if not args.skip_retrieval:
        retrieval = asyncio.run(retrieval_phase(args.query, user_id, args.top_k, rag_cfg))

    if not args.skip_chat:
        chat_data = chat_phase(server, args.query, user_id, args.timeout)

    return verdict(retrieval, chat_data, args.expect, args.skip_chat, args.skip_retrieval)


if __name__ == "__main__":
    sys.exit(main())
