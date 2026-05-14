#!/usr/bin/env python3
"""
Cross-encoder reranking benchmark harness.

Compares retrieval results across different modes on a labelled set of known-good
queries. For each query it measures recall@top_k, wall-clock latency, and rank
changes, then prints a side-by-side comparison table and summary statistics.

Usage (run from project root):
    python utils/rag_rerank_bench.py --user florian
    python utils/rag_rerank_bench.py --user florian --mode dense            # dense only
    python utils/rag_rerank_bench.py --user florian --mode rerank           # rerank only
    python utils/rag_rerank_bench.py --user florian --mode hybrid           # hybrid only
    python utils/rag_rerank_bench.py --user florian --mode hybrid+rerank    # hybrid+rerank only
    python utils/rag_rerank_bench.py --user florian --top-k 10              # fetch more
    python utils/rag_rerank_bench.py --user florian --rerank-url http://192.168.1.95:8001/v1/rerank

Queries file format (utils/rag_bench_queries.txt by default):
    query text | expected_path_substring   # lines starting with # are comments

Comparison modes:
    both: dense vs reranked (cross-encoder rescoring)
    dense: dense-only results (Qdrant cosine order)
    rerank: reranked results (cross-encoder order)
    hybrid: dense vs hybrid (bge-m3 sparse + dense combined, no reranking)
    hybrid+rerank: dense vs hybrid with reranking applied

Printed summary: recall counts, average latency overhead, and rank delta stats.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.collection import get_client
from rag_engine.search import search_rag

# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_queries(path: Path) -> list[tuple[str, str]]:
    queries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            print(f"  [WARN] Skipping malformed line: {line!r}")
            continue
        q, expected = line.split("|", 1)
        queries.append((q.strip(), expected.strip()))
    return queries


def _recall(results: list[dict], expected: str) -> bool:
    """Check if expected substring appears in any result's source_path."""
    return any(expected.lower() in r["source_path"].lower() for r in results)


def _rank_delta(dense_results: list[dict], rerank_results: list[dict]) -> dict:
    """Analyze rank changes between dense and reranked results.

    Returns dict with:
      - moved: number of results that moved in rank
      - avg_delta: average absolute rank change
      - improved: number of results with rank improving (moving up)
      - degraded: number of results with rank moving down
    """
    if not dense_results or not rerank_results:
        return {"moved": 0, "avg_delta": 0.0, "improved": 0, "degraded": 0}

    # Build a map of source_path -> dense rank
    dense_rank = {r["source_path"]: i for i, r in enumerate(dense_results)}

    moved = 0
    deltas = []
    improved = 0
    degraded = 0

    for rerank_idx, rerank_result in enumerate(rerank_results):
        src = rerank_result["source_path"]
        if src in dense_rank:
            dense_idx = dense_rank[src]
            delta = abs(dense_idx - rerank_idx)
            if delta > 0:
                moved += 1
                deltas.append(delta)
                if rerank_idx < dense_idx:
                    improved += 1
                else:
                    degraded += 1

    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "moved": moved,
        "avg_delta": avg_delta,
        "improved": improved,
        "degraded": degraded,
    }


def _bar(n: int, total: int, width: int = 5) -> str:
    """Simple bar chart for visual feedback."""
    filled = round(n / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Per-query runners
# ---------------------------------------------------------------------------

async def run_dense(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
) -> tuple[list[dict], float]:
    """Run retrieval with reranking disabled (Qdrant cosine order)."""
    t0 = time.perf_counter()
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
        rerank=False,  # Force disable reranking
    )
    return results, time.perf_counter() - t0


async def run_rerank(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
) -> tuple[list[dict], float]:
    """Run retrieval with reranking enabled (cross-encoder rescoring)."""
    t0 = time.perf_counter()
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
        rerank=True,  # Force enable reranking
    )
    return results, time.perf_counter() - t0


async def run_hybrid(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
) -> tuple[list[dict], float]:
    """Run retrieval with hybrid mode enabled (bge-m3 sparse + dense combined, no reranking)."""
    t0 = time.perf_counter()
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
        hybrid=True,  # Enable hybrid search
        rerank=False,  # Disable reranking
    )
    return results, time.perf_counter() - t0


async def run_hybrid_rerank(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
) -> tuple[list[dict], float]:
    """Run retrieval with hybrid mode and reranking enabled."""
    t0 = time.perf_counter()
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
        hybrid=True,  # Enable hybrid search
        rerank=True,  # Enable reranking
    )
    return results, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Table + summary
# ---------------------------------------------------------------------------

_W_QUERY = 42
_W_SCORE = 6


def _print_header(right_label: str = "Reranked"):
    q = "Query".ljust(_W_QUERY)
    print(f"\n  {q}  {'Dense (Qdrant)':^16}  {right_label:^16}  Rank Delta")
    print(f"  {'─'*_W_QUERY}  {'─'*16}  {'─'*16}  {'─'*20}")


def _print_row(
    query: str, expected: str,
    d_hit: bool, d_time: float,
    r_hit: bool, r_time: float,
    rank_stats: dict,
):
    label = query[:_W_QUERY].ljust(_W_QUERY)
    d_hit_sym = "✓" if d_hit else "✗"
    r_hit_sym = "✓" if r_hit else "✗"
    gain_sym = "↑" if (r_hit and not d_hit) else ("↓" if (d_hit and not r_hit) else " ")

    moved = rank_stats.get("moved", 0)
    avg_delta = rank_stats.get("avg_delta", 0.0)
    improved = rank_stats.get("improved", 0)
    degraded = rank_stats.get("degraded", 0)

    d_col = f"{d_hit_sym} {d_time:6.3f}s"
    r_col = f"{r_hit_sym}{gain_sym} {r_time:6.3f}s"
    delta_col = f"{moved} moved (Δ{avg_delta:.1f}) ↑{improved} ↓{degraded}"

    print(f"  {label}  {d_col}  {r_col}  {delta_col}")


def _print_summary(rows: list[dict]):
    print(f"\n{'─'*100}")
    total = len(rows)
    d_recall = sum(1 for r in rows if r["d_hit"])
    r_recall = sum(1 for r in rows if r["r_hit"])
    gains = sum(1 for r in rows if r["r_hit"] and not r["d_hit"])
    regressions = sum(1 for r in rows if r["d_hit"] and not r["r_hit"])

    total_d_time = sum(r["d_time"] for r in rows)
    total_r_time = sum(r["r_time"] for r in rows)
    avg_d_time = total_d_time / total if total else 0.0
    avg_r_time = total_r_time / total if total else 0.0
    latency_overhead = avg_r_time - avg_d_time

    total_moved = sum(r["rank_stats"].get("moved", 0) for r in rows)
    total_improved = sum(r["rank_stats"].get("improved", 0) for r in rows)
    total_degraded = sum(r["rank_stats"].get("degraded", 0) for r in rows)
    avg_delta = sum(r["rank_stats"].get("avg_delta", 0.0) for r in rows) / total if total else 0.0

    print(f"\n  Recall:     dense {d_recall}/{total}   reranked {r_recall}/{total}"
          f"   gains={gains}  regressions={regressions}")
    print(f"  Latency:    dense avg {avg_d_time:.3f}s   reranked avg {avg_r_time:.3f}s"
          f"   overhead {latency_overhead:+.3f}s")
    print(f"  Rank changes: {total_moved} results moved across queries"
          f"   avg Δ rank {avg_delta:.1f}   ↑{total_improved} ↓{total_degraded}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Reranking benchmark")
    p.add_argument("--user", default="florian")
    p.add_argument("--queries", default=str(Path(__file__).parent / "rag_bench_queries.txt"),
                   help="Path to queries file (query | expected_substring)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Number of results to fetch (default: 5)")
    p.add_argument("--mode", choices=["dense", "rerank", "both", "hybrid", "hybrid+rerank"], default="both",
                   help="Run mode (default: both)")
    p.add_argument("--rerank-url", default=None,
                   help="Override reranker server URL")
    args = p.parse_args()

    config = _load_config()
    rag_cfg = config.get("rag", {})

    # Override rerank URL if provided
    if args.rerank_url:
        if "rerank" not in rag_cfg:
            rag_cfg["rerank"] = {}
        rag_cfg["rerank"]["server_url"] = args.rerank_url

    queries = _load_queries(Path(args.queries))

    if not queries:
        print("No queries loaded — check the file format.")
        return 1

    qdrant_url = rag_cfg.get("qdrant_url", "http://192.168.1.93:6333")
    qdrant_client = get_client(qdrant_url)

    rerank_cfg = rag_cfg.get("rerank", {})
    rerank_url = args.rerank_url or rerank_cfg.get("server_url", "http://192.168.1.95:8001/v1/rerank")

    print(f"\n  Reranking benchmark  user={args.user}  top_k={args.top_k}")
    print(f"  Qdrant: {qdrant_url}  |  queries: {len(queries)}  |  mode: {args.mode}")
    print(f"  Reranker: {rerank_url}")

    async def run() -> list[dict]:
        rows = []

        # Determine right column label and runner based on mode
        if args.mode == "both":
            right_label = "Reranked"
            right_runner = run_rerank
        elif args.mode == "dense":
            right_label = None  # Single mode, no comparison
            right_runner = None
        elif args.mode == "rerank":
            right_label = None  # Single mode, no comparison
            right_runner = None
        elif args.mode == "hybrid":
            right_label = "Hybrid"
            right_runner = run_hybrid
        elif args.mode == "hybrid+rerank":
            right_label = "Hybrid+RR"
            right_runner = run_hybrid_rerank

        if right_label:
            _print_header(right_label)

        for query, expected in queries:
            d_results, d_time = ([], 0.0)
            r_results, r_time = ([], 0.0)

            # Always run dense for comparison modes; single-mode runs only for their type
            if args.mode in ("dense", "both", "hybrid", "hybrid+rerank"):
                d_results, d_time = await run_dense(query, args.user, qdrant_client, rag_cfg, args.top_k)

            # Run the right-side variant if in a comparison mode
            if right_runner:
                r_results, r_time = await right_runner(query, args.user, qdrant_client, rag_cfg, args.top_k)

            d_hit = _recall(d_results, expected) if d_results else False
            r_hit = _recall(r_results, expected) if r_results else False
            rank_stats = _rank_delta(d_results, r_results) if right_runner else {}

            # Only print row if we have results to display
            if d_results or r_results:
                _print_row(query, expected,
                           d_hit, d_time,
                           r_hit, r_time,
                           rank_stats)

                rows.append(dict(
                    query=query, expected=expected,
                    d_hit=d_hit, d_time=d_time,
                    r_hit=r_hit, r_time=r_time,
                    rank_stats=rank_stats,
                ))

        return rows

    rows = asyncio.run(run())

    # Print summary for comparison modes only (where we have left/right comparison)
    if args.mode in ("both", "hybrid", "hybrid+rerank"):
        _print_summary(rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
