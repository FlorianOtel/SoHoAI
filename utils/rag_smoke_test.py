"""
rag_smoke_test.py — end-to-end RAG smoke test.

Phase 1 — Retrieval: GET /v1/rag/search?q=...

Exit 0 on pass, non-zero on failure.
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

DEFAULT_SERVER = "http://192.168.1.93:8000"
DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 30


def _get(url: str, timeout: int) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def phase1_retrieval(args) -> bool:
    """GET /v1/rag/search and optionally assert --expect substring."""
    params: dict = {"q": args.query, "top_k": args.top_k}
    if args.user and not args.no_filter:
        params["user"] = args.user
    if args.file_type:
        # file_types is a repeated query param
        qs = urllib.parse.urlencode(params)
        for ft in args.file_type:
            qs += f"&file_types={urllib.parse.quote(ft)}"
        url = f"{args.server}/v1/rag/search?{qs}"
    else:
        url = f"{args.server}/v1/rag/search?{urllib.parse.urlencode(params)}"

    print(f"\n[Phase 1] GET {url}")
    try:
        resp = _get(url, args.timeout)
    except Exception as e:
        print(f"  FAIL: HTTP error: {e}")
        return False

    results = resp.get("results", [])
    print(f"  {len(results)} result(s) returned")
    for i, r in enumerate(results[:5], 1):
        print(f"  [{i}] score={r.get('score', 0):.3f}  {r.get('source_path', '')}")
        preview = r.get("content", "")[:120].replace("\n", " ")
        print(f"       {preview}")

    if args.expect:
        haystack = json.dumps(resp)
        if args.expect not in haystack:
            print(f"  FAIL: expected substring {args.expect!r} not found in results")
            return False
        print(f"  PASS: found {args.expect!r} in results")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="SoHoAI RAG smoke test")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--user", default=None, help="Owner filter (e.g. florian)")
    parser.add_argument("--no-filter", action="store_true", help="Skip user ownership filter")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Max retrieval results")
    parser.add_argument("--file-type", action="append", metavar="TYPE", help="Filter by file type (repeatable)")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="SoHoAI server base URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout seconds")
    parser.add_argument("--expect", default=None, help="Assert this substring appears in the retrieval results")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip Phase 1 (retrieval)")
    args = parser.parse_args()

    passed = True

    if not args.skip_retrieval:
        ok = phase1_retrieval(args)
        passed = passed and ok
    else:
        print("[Phase 1] skipped")

    if passed:
        print("\nSMOKE TEST PASSED")
        return 0
    else:
        print("\nSMOKE TEST FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
