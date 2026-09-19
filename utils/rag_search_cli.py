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

import httpx  # noqa: E402
import yaml  # noqa: E402

from rag_engine.collection import get_client
from rag_engine.search import search_rag


async def run(query: str, user_id: str | None, top_k: int, qdrant_url: str, rag_cfg: dict, model_list: list[dict] | None = None, file_types: list[str] | None = None, score_threshold: float = 0.0, multi_query: bool = False, rerank: bool | None = None) -> None:
    qdrant_client = get_client(qdrant_url)

    if multi_query:
        from rag_engine.multi_query import multi_query_search

        variant_model_cfg = rag_cfg.get("multi_query", {}).get("variant_model")
        if not variant_model_cfg:
            raise RuntimeError(
                "rag.multi_query.variant_model is not set in SoHoAI-config.yaml — "
                "required for --multi-query."
            )
        # This CLI calls llama-swap on Server 2 directly, bypassing the SoHoAI proxy, so it needs
        # the raw llama-swap model key. That key is NOT simply the SoHoAI model_name minus its
        # "local/" prefix: local/qwen3.5-9b-agentic is a SoHoAI-side alias whose backing model is
        # openai/qwen3.5-9b-q4. Resolve via model_list so a future alias cannot silently break.
        if not model_list:
            raise RuntimeError(
                "model_list is required to resolve the llama-swap model key for multi-query."
            )
        _entry = next(
            (m for m in model_list if m.get("model_name") == variant_model_cfg), None
        )
        if _entry is None:
            raise RuntimeError(
                f"rag.multi_query.variant_model = {variant_model_cfg!r} does not match any "
                f"model_list entry in SoHoAI-config.yaml."
            )
        _backing = _entry.get("litellm_params", {}).get("model", "")
        if not _backing:
            raise RuntimeError(
                f"model_list entry {variant_model_cfg!r} has no litellm_params.model — "
                f"cannot determine the llama-swap model key."
            )
        variant_model = _backing.split("/", 1)[-1]

        async def _llm_fn(prompt: str) -> str:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "http://192.168.1.95:8000/v1/chat/completions",
                    json={"model": variant_model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 150, "temperature": 0.4},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

        results = await multi_query_search(
            query=query,
            user_id=user_id,
            limit=top_k,
            qdrant_client=qdrant_client,
            rag_cfg=rag_cfg,
            llm_fn=_llm_fn,
        )
    else:
        results = await search_rag(
            query=query,
            user_id=user_id,
            limit=top_k,
            qdrant_client=qdrant_client,
            rag_cfg=rag_cfg,
            file_types=file_types,
            score_threshold=score_threshold,
            rerank=rerank,
        )

    if not results:
        print("No results found.")
        return

    print(f"\n{len(results)} result(s) for query: {query!r}  (user_id={user_id})\n")

    # Check if any result has a rerank_score field
    has_rerank = any("rerank_score" in r for r in results)

    if has_rerank:
        print(f"{'#':<3}  {'Score':>6}  {'Rerank':>7}  {'File/Title':<40}  Source")
        print("─" * 110)
        for i, r in enumerate(results, 1):
            ftype  = r.get("file_type", "")
            stitle = r.get("session_title", "")
            display = (stitle if ftype in ("claude_chat", "opencode") and stitle else r.get("file_name", ""))[:38]
            rr = r.get("rerank_score", 0.0)
            print(f"{i:<3}  {r['score']:>6.4f}  {rr:>+7.2f}  {display:<40}  {r['source_path']}")
    else:
        print(f"{'#':<3}  {'Score':>6}  {'File/Title':<40}  Source")
        print("─" * 100)
        for i, r in enumerate(results, 1):
            ftype  = r.get("file_type", "")
            stitle = r.get("session_title", "")
            display = (stitle if ftype in ("claude_chat", "opencode") and stitle else r.get("file_name", ""))[:38]
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
    parser.add_argument("--score-threshold", type=float, default=0.0, metavar="FLOAT",
                        help="Minimum cosine score to include (0=no filter, e.g. 0.50)")
    parser.add_argument("--multi-query", action="store_true",
                        help="Enable multi-query expansion + MMR reranking (uses internal Qwen3.5)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable cross-encoder reranking (use Qdrant cosine order only)")
    args = parser.parse_args()

    if not args.no_filter and not args.user:
        parser.error("Provide --user OWNER or --no-filter")

    user_id = None if args.no_filter else args.user

    with open(Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    model_list = config.get("model_list", [])

    rerank = False if args.no_rerank else None

    asyncio.run(run(args.query, user_id, args.top_k, rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"), rag_cfg, model_list=model_list, file_types=args.file_types or None, score_threshold=args.score_threshold, multi_query=args.multi_query, rerank=rerank))


if __name__ == "__main__":
    main()
