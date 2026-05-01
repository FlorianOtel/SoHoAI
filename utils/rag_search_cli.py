#!/usr/bin/env python3
"""
Test the RAG search pipeline from the command line.

Embeds the query, applies the ownership filter, and prints the top-k results
with cosine similarity scores and source paths — identical to what the
orchestrator injects into the LLM prompt.

Usage (run from project root):
    python utils/rag_search_cli.py --query "what certifications do I have" --user florian
    python utils/rag_search_cli.py --query "family album" --user la-familia --top-k 10
    python utils/rag_search_cli.py --query "test" --no-filter   # search without owner filter
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rag_engine.collection import get_client
from rag_engine.search import search_rag


async def run(query: str, user_id: str | None, top_k: int, qdrant_url: str, rag_cfg: dict, file_types: list[str] | None = None) -> None:
    qdrant_client = get_client(qdrant_url)

    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
        file_types=file_types,
    )

    if not results:
        print("No results found.")
        return

    print(f"\n{len(results)} result(s) for query: {query!r}  (user_id={user_id})\n")
    print(f"{'#':<3}  {'Score':>6}  {'File/Title':<40}  Source")
    print("─" * 100)
    for i, r in enumerate(results, 1):
        ftype  = r.get("file_type", "")
        stitle = r.get("session_title", "")
        display = (stitle if ftype == "claude_chat" and stitle else r.get("file_name", ""))[:38]
        print(f"{i:<3}  {r['score']:>6.4f}  {display:<40}  {r['source_path']}")

    print()
    if results:
        print("── Top result content (parent_text) ─────────────────────────────────────")
        print(results[0]["content"][:800])
        if len(results[0]["content"]) > 800:
            print(f"  ... [{len(results[0]['content'])} chars total]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RAG search from the command line")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--user", metavar="OWNER", default=None,
                        help="Owner filter, e.g. florian (required unless --no-filter)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Search all documents regardless of owner")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--file-types", nargs="+", metavar="TYPE",
                        help="Filter by file type(s): pdf, pptx, ppt, docx, ipynb, md, yaml, txt, claude_chat")
    args = parser.parse_args()

    if not args.no_filter and not args.user:
        parser.error("Provide --user OWNER or --no-filter")

    user_id = None if args.no_filter else args.user

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})

    asyncio.run(run(args.query, user_id, args.top_k, rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"), rag_cfg, file_types=args.file_types or None))


if __name__ == "__main__":
    main()
