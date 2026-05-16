#!/usr/bin/env python3
"""
Display Qdrant collection status, optimizer lag, and segment info.

Shows whether the background optimizer is idle (green, lag=0) or actively
merging/vacuuming segments after bulk deletes (yellow, lag>0).

Usage (run from project root):
    python utils/qdrant_status.py
    python utils/qdrant_status.py --watch        # poll every 10s until green
    python utils/qdrant_status.py --watch 30     # poll every 30s
    python utils/qdrant_status.py --verbose      # include optimizer config
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rag_engine.collection import get_client  # noqa: E402


def _fetch_status(client, collection: str) -> dict:
    info = client.get_collection(collection)
    pts     = info.points_count or 0
    indexed = info.indexed_vectors_count or 0
    return {
        "status":   info.status.value if hasattr(info.status, "value") else str(info.status),
        "points":   pts,
        "indexed":  indexed,
        "lag":      indexed - pts,
        "segments": info.segments_count or 0,
        "optimizer": {
            "deleted_threshold":      info.config.optimizer_config.deleted_threshold,
            "vacuum_min_vector_number": info.config.optimizer_config.vacuum_min_vector_number,
            "indexing_threshold":     info.config.optimizer_config.indexing_threshold,
            "flush_interval_sec":     info.config.optimizer_config.flush_interval_sec,
        } if info.config and info.config.optimizer_config else {},
    }


_STATUS_ICON = {"green": "✓", "yellow": "⟳", "red": "✗", "grey": "?"}


def _print_status(s: dict, verbose: bool = False) -> None:
    icon   = _STATUS_ICON.get(s["status"], "?")
    color  = "" if s["status"] == "green" else " (optimizer running)" if s["status"] == "yellow" else ""
    lag    = s["lag"]
    lag_pct = (lag / s["indexed"] * 100) if s["indexed"] else 0.0
    ts     = datetime.now().strftime("%H:%M:%S")

    print(f"[{ts}]  {icon} {s['status'].upper()}{color}")
    print(f"  Points:   {s['points']:>10,}  (live)")
    print(f"  Indexed:  {s['indexed']:>10,}  (HNSW, incl. soft-deleted)")
    if lag:
        print(f"  Lag:      {lag:>10,}  ({lag_pct:.1f}% — vectors pending vacuum)")
    else:
        print(f"  Lag:           0  (optimizer idle)")
    print(f"  Segments: {s['segments']:>10,}")

    if verbose and s["optimizer"]:
        o = s["optimizer"]
        print()
        print("  Optimizer config:")
        print(f"    deleted_threshold      = {o['deleted_threshold']}  "
              f"(vacuum segment when ≥{o['deleted_threshold']*100:.0f}% deleted)")
        print(f"    vacuum_min_vector_number = {o['vacuum_min_vector_number']}")
        print(f"    indexing_threshold     = {o['indexing_threshold']}")
        print(f"    flush_interval_sec     = {o['flush_interval_sec']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qdrant optimizer and lag status")
    parser.add_argument(
        "--watch", nargs="?", const=10, type=int, metavar="SECS",
        help="Poll every SECS seconds (default 10) until status is green",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show optimizer configuration details",
    )
    parser.add_argument(
        "--collection", default=None,
        help="Collection name (default: from SoHoAI-config.yaml rag.qdrant_collection)",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    qdrant_url = config.get("rag", {}).get("qdrant_url", "http://192.168.1.93:6333")
    collection = args.collection or config.get("rag", {}).get("qdrant_collection", "documents")

    client = get_client(qdrant_url, timeout=15)

    if args.watch is None:
        # One-shot
        try:
            s = _fetch_status(client, collection)
        except Exception as e:
            print(f"Error fetching Qdrant status: {e}", file=sys.stderr)
            sys.exit(1)
        _print_status(s, verbose=args.verbose)
        sys.exit(0 if s["status"] == "green" else 1)

    # Watch mode — poll until green
    interval = args.watch
    print(f"Watching '{collection}' every {interval}s — Ctrl-C to stop\n")
    prev_lag = None
    try:
        while True:
            try:
                s = _fetch_status(client, collection)
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}]  fetch error: {e}")
                time.sleep(interval)
                continue

            lag = s["lag"]
            delta = f"  (Δ {prev_lag - lag:+,})" if prev_lag is not None and lag != prev_lag else ""
            prev_lag = lag

            icon = _STATUS_ICON.get(s["status"], "?")
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}]  {icon} {s['status']:<6}"
                f"  pts={s['points']:,}  lag={lag:,}{delta}",
                flush=True,
            )

            if s["status"] == "green" and lag == 0:
                print("\nOptimizer is idle — collection fully optimized.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
