#!/usr/bin/env python3
"""
MMR benchmark harness.

Compares single-query retrieval vs multi-query + MMR on a labelled set of
known-good queries.  For each query it measures recall@top_k, wall-clock
latency, and source diversity, then prints a side-by-side table and a
go/no-go verdict for enabling rag.multi_query in production.

Usage (run from project root):
    python utils/rag_mmr_bench.py --user florian
    python utils/rag_mmr_bench.py --user florian --lambda 0.3   # tune MMR
    python utils/rag_mmr_bench.py --user florian --variants 4   # more queries
    python utils/rag_mmr_bench.py --user florian --mode single  # single only

Queries file format  (utils/rag_bench_queries.txt by default):
    query text | expected_path_substring   # lines starting with # are comments

Go/no-go criteria (documented in RAG-strategy.md §8.3):
    ✓  No query regresses  (multi recall ≥ single recall for every query)
    ✓  ≥ 2 queries gain recall  (multi hit where single missed)
    ✓  Max latency per query ≤ 2.5 s
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.collection import get_client
from rag_engine.multi_query import multi_query_search
from rag_engine.search import search_rag

# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml"

# Go/no-go thresholds
_LATENCY_LIMIT   = 2.5    # seconds per multi-query call
_MIN_RECALL_GAIN = 2      # minimum queries that must gain from multi-query


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


def _diversity(results: list[dict]) -> tuple[int, int]:
    """Return (max_same_file_count, total_results).

    Lower max_same_file_count = more diverse.
    """
    if not results:
        return 0, 0
    # Normalise: strip chunk index from path (same file = same base path)
    sources = [r["source_path"] for r in results]
    c = Counter(sources)
    return c.most_common(1)[0][1], len(results)


def _recall(results: list[dict], expected: str) -> bool:
    return any(expected.lower() in r["source_path"].lower() for r in results)


def _bar(n: int, total: int, width: int = 5) -> str:
    filled = round(n / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# LLM fn factories for variant generation
# ---------------------------------------------------------------------------

def _make_internal_llm_fn(server2_ip: str, model: str) -> callable:
    """Calls Qwen3.5 on Server 2 via llama-server /v1/chat/completions."""
    url = f"http://{server2_ip}:8000/v1/chat/completions"

    async def llm_fn(prompt: str) -> str:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150,
                "temperature": 0.4,
            })
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    return llm_fn


def _make_external_llm_fn(orchestrator_url: str) -> callable:
    """Calls Claude Sonnet via the orchestrator with force_cloud=true.

    Requires the orchestrator to be running at orchestrator_url.
    Uses rag_mode=off so no retrieval side-effects.
    """
    url = f"{orchestrator_url}/v1/chat/completions"

    async def llm_fn(prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json={
                "messages":   [{"role": "user", "content": prompt}],
                "rag_mode":   "off",
                "force_cloud": True,
                "stream":      False,
            })
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    return llm_fn


def _make_llm_fn(variant_model: str, config: dict) -> callable:
    """Return the right llm_fn based on --variant-model choice."""
    if variant_model == "external":
        server1_ip = config.get("server1_ip", "192.168.1.93")
        return _make_external_llm_fn(f"http://{server1_ip}:8000")
    # default: internal
    server2_ip = config.get("server2_ip", "192.168.1.95")
    model      = config.get("model_list", [{}])[0].get("litellm_params", {}).get("model", "qwen3-4b")
    return _make_internal_llm_fn(server2_ip, model)


def _capturing_llm_fn(llm_fn: callable) -> tuple[callable, list]:
    """Wrap llm_fn so the raw LLM response is captured for --show-variants."""
    captured: list[str] = []

    async def wrapper(prompt: str) -> str:
        result = await llm_fn(prompt)
        captured.append(result)
        return result

    return wrapper, captured


# ---------------------------------------------------------------------------
# Per-query runners
# ---------------------------------------------------------------------------

async def run_single(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    results = await search_rag(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=rag_cfg,
    )
    return results, time.perf_counter() - t0


async def run_multi(
    query: str, user_id: str | None,
    qdrant_client, rag_cfg: dict, top_k: int,
    llm_fn: callable,
    lambda_override: float | None,
    variants_override: int | None,
    capture_variants: bool = False,
) -> tuple[list[dict], float, list[str]]:
    # Build a rag_cfg copy that reflects any CLI overrides
    cfg = {**rag_cfg}
    mq = {**cfg.get("multi_query", {})}
    if lambda_override is not None:
        mq["lambda"] = lambda_override
    if variants_override is not None:
        mq["n_variants"] = variants_override
    mq["enabled"] = True   # always on for benchmark
    cfg["multi_query"] = mq

    active_fn, captured = _capturing_llm_fn(llm_fn) if capture_variants else (llm_fn, [])

    t0 = time.perf_counter()
    results = await multi_query_search(
        query=query,
        user_id=user_id,
        limit=top_k,
        qdrant_client=qdrant_client,
        rag_cfg=cfg,
        llm_fn=active_fn,
    )
    elapsed = time.perf_counter() - t0

    # Parse variants from the raw LLM response (same logic as expand_query)
    variants: list[str] = []
    if captured:
        raw = captured[0]
        variants = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        variants = [v for v in variants if v.lower() != query.lower()]

    return results, elapsed, variants


# ---------------------------------------------------------------------------
# Table + verdict
# ---------------------------------------------------------------------------

_W_QUERY    = 42
_W_SCORE    = 6
_W_TIME     = 7
_W_DIV      = 7

def _print_header():
    q = "Query".ljust(_W_QUERY)
    print(f"\n  {q}  {'Single':^14}  {'Multi-MMR':^14}  Notes")
    print(f"  {'─'*_W_QUERY}  {'─'*14}  {'─'*14}  {'─'*30}")


def _print_row(
    query: str, expected: str,
    s_hit: bool, s_time: float, s_max: int, s_total: int,
    m_hit: bool, m_time: float, m_max: int, m_total: int,
):
    label = query[:_W_QUERY].ljust(_W_QUERY)
    s_hit_sym = "✓" if s_hit else "✗"
    m_hit_sym = "✓" if m_hit else "✗"
    # gain_sym only meaningful when both modes ran
    if m_total == 0 or s_total == 0:
        gain_sym = " "
    else:
        gain_sym = "↑" if (m_hit and not s_hit) else ("↓" if (s_hit and not m_hit) else " ")
    div_note  = f"dup {s_max}/{s_total}→{m_max}/{m_total}" if s_max != m_max else f"div same ({s_max}/{s_total})"
    s_col = f"{s_hit_sym} {s_time:5.2f}s  {_bar(s_max, s_total)}"
    m_col = f"{m_hit_sym} {m_time:5.2f}s {gain_sym} {_bar(m_max, m_total)}"
    print(f"  {label}  {s_col}  {m_col}  {div_note}")


def _print_verdict(rows: list[dict]):
    print(f"\n{'─'*90}")
    total       = len(rows)
    s_recall    = sum(1 for r in rows if r["s_hit"])
    m_recall    = sum(1 for r in rows if r["m_hit"])
    gains       = sum(1 for r in rows if r["m_hit"] and not r["s_hit"])
    regressions = sum(1 for r in rows if r["s_hit"] and not r["m_hit"])
    max_m_time  = max(r["m_time"] for r in rows)
    avg_m_time  = sum(r["m_time"] for r in rows) / total
    avg_s_time  = sum(r["s_time"] for r in rows) / total
    div_improved = sum(1 for r in rows if r["m_max"] < r["s_max"])
    div_regressed = sum(1 for r in rows if r["m_max"] > r["s_max"])

    print(f"\n  Recall:   single {s_recall}/{total}   multi {m_recall}/{total}"
          f"   gains={gains}  regressions={regressions}")
    print(f"  Latency:  single avg {avg_s_time:.2f}s   multi avg {avg_m_time:.2f}s"
          f"   multi max {max_m_time:.2f}s")
    print(f"  Diversity improved: {div_improved}/{total} queries"
          f"   regressed: {div_regressed}/{total}")

    ok_no_regression = regressions == 0
    ok_gains         = gains >= _MIN_RECALL_GAIN
    ok_latency       = max_m_time <= _LATENCY_LIMIT

    print(f"\n  Criteria ({_LATENCY_LIMIT}s latency cap, ≥{_MIN_RECALL_GAIN} gains, 0 regressions):")
    print(f"    {'✓' if ok_no_regression else '✗'}  No regressions      ({regressions} found)")
    print(f"    {'✓' if ok_gains else '✗'}  ≥{_MIN_RECALL_GAIN} recall gains       ({gains} found)")
    print(f"    {'✓' if ok_latency else '✗'}  Max latency ≤{_LATENCY_LIMIT}s   ({max_m_time:.2f}s)")

    verdict = "GO  — flip rag.multi_query.enabled: true" if (ok_no_regression and ok_gains and ok_latency) \
              else "NO-GO — do not flip yet (see criteria above)"
    print(f"\n  Verdict:  {verdict}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="MMR recall benchmark")
    p.add_argument("--user",     default="florian")
    p.add_argument("--queries",  default=str(Path(__file__).parent / "rag_bench_queries.txt"),
                   help="Path to queries file  (query | expected_substring)")
    p.add_argument("--top-k",    type=int, default=5)
    p.add_argument("--lambda",   dest="lam", type=float, default=None,
                   help="Override rag.multi_query.lambda  (e.g. 0.3)")
    p.add_argument("--variants", type=int, default=None,
                   help="Override rag.multi_query.n_variants")
    p.add_argument("--mode",          choices=["single", "multi", "both"], default="both")
    p.add_argument("--variant-model", choices=["internal", "external"], default="internal",
                   help="LLM used for query expansion (internal=Qwen3.5, external=Sonnet)")
    p.add_argument("--show-variants", action="store_true",
                   help="Print the LLM-generated query variants for each query")
    p.add_argument("--compare",       action="store_true",
                   help="Run multi-query with BOTH internal AND external; "
                        "shows variants + recall side-by-side. Requires orchestrator.")
    args = p.parse_args()

    config   = _load_config()
    rag_cfg  = config.get("rag", {})
    queries  = _load_queries(Path(args.queries))

    if not queries:
        print("No queries loaded — check the file format.")
        return 1

    qdrant_url    = rag_cfg.get("qdrant_url", "http://192.168.1.93:6333")
    qdrant_client = get_client(qdrant_url)
    llm_fn        = _make_llm_fn(args.variant_model, config)

    lam_str = f"λ={args.lam:.2f}" if args.lam is not None else f"λ={rag_cfg.get('multi_query',{}).get('lambda',0.5):.2f}"
    var_str = f"n={args.variants}" if args.variants is not None else f"n={rag_cfg.get('multi_query',{}).get('n_variants',3)}"
    mode_desc = "compare(internal vs external)" if args.compare else args.mode

    print(f"\n  MMR benchmark  user={args.user}  top_k={args.top_k}  {lam_str}  {var_str}")
    print(f"  Qdrant: {qdrant_url}  |  queries: {len(queries)}  |  mode: {mode_desc}")
    if not args.compare:
        print(f"  Variant LLM: {args.variant_model}")

    # ── compare mode: internal vs external side-by-side ──────────────────
    if args.compare:
        spec_fn = _make_llm_fn("internal", config)
        ext_fn  = _make_llm_fn("external", config)

        async def run_compare():
            print()
            for query, expected in queries:
                print(f"  {'─'*80}")
                print(f"  Query: {query!r}  (expect: {expected!r})")

                s_results, s_time = await run_single(query, args.user, qdrant_client, rag_cfg, args.top_k)
                sr, se = _diversity(s_results)
                s_hit = _recall(s_results, expected)
                print(f"  Single        {('✓' if s_hit else '✗')} {s_time:.2f}s  dup={sr}/{se}")

                for label, fn in [("Specialist    ", spec_fn), ("External      ", ext_fn)]:
                    m_results, m_time, variants = await run_multi(
                        query, args.user, qdrant_client, rag_cfg, args.top_k,
                        fn, args.lam, args.variants, capture_variants=True,
                    )
                    mr, me = _diversity(m_results)
                    m_hit = _recall(m_results, expected)
                    gain = "↑" if (m_hit and not s_hit) else ("↓" if (s_hit and not m_hit) else " ")
                    print(f"  {label}  {('✓' if m_hit else '✗')}{gain} {m_time:.2f}s  dup={mr}/{me}")
                    for i, v in enumerate(variants[:args.variants or 3], 1):
                        print(f"      variant {i}: {v}")
            print(f"  {'─'*80}\n")

        asyncio.run(run_compare())
        return 0

    # ── normal / both mode ─────────────────────────────────────────────────
    async def run() -> list[dict]:
        rows = []
        _print_header()
        for query, expected in queries:
            if args.mode in ("single", "both"):
                s_results, s_time = await run_single(query, args.user, qdrant_client, rag_cfg, args.top_k)
            else:
                s_results, s_time = [], 0.0

            variants: list[str] = []
            if args.mode in ("multi", "both"):
                m_results, m_time, variants = await run_multi(
                    query, args.user, qdrant_client, rag_cfg, args.top_k,
                    llm_fn, args.lam, args.variants,
                    capture_variants=args.show_variants,
                )
            else:
                m_results, m_time = [], 0.0

            s_hit = _recall(s_results, expected) if s_results else False
            m_hit = _recall(m_results, expected) if m_results else False
            s_max, s_total = _diversity(s_results)
            m_max, m_total = _diversity(m_results)

            _print_row(query, expected,
                       s_hit, s_time, s_max, s_total,
                       m_hit, m_time, m_max, m_total)

            if args.show_variants and variants:
                for i, v in enumerate(variants, 1):
                    print(f"      variant {i}: {v}")

            rows.append(dict(
                query=query, expected=expected,
                s_hit=s_hit, s_time=s_time, s_max=s_max, s_total=s_total,
                m_hit=m_hit, m_time=m_time, m_max=m_max, m_total=m_total,
            ))
        return rows

    rows = asyncio.run(run())

    if args.mode == "both":
        _print_verdict(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
