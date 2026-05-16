#!/usr/bin/env python3
"""
Scan Qdrant for corrupt content (low printable-char ratio) and optionally purge.

Files with parent_text containing mostly non-printable characters (formatting noise,
binary artifacts, encoding issues) are identified and can be deleted from Qdrant.

Corrupt points are cross-referenced with SQLite to show whether the file is still
queued for re-ingestion (IN_DB) or already completed (NOT_IN_DB).

Usage (run from project root):
    python utils/rag_purge_corrupt.py
    python utils/rag_purge_corrupt.py --dry-run
    python utils/rag_purge_corrupt.py --threshold 0.8
    python utils/rag_purge_corrupt.py --user florian
    python utils/rag_purge_corrupt.py --confirm
    python utils/rag_purge_corrupt.py --confirm --force
    python utils/rag_purge_corrupt.py --save /tmp/corrupt_paths.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client
from rag_engine.schema import (
    FIELD_SOURCE_PATH,
    FIELD_PARENT_TEXT,
    FIELD_OWNER,
)
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _printable_ratio(text: str) -> float:
    """Return fraction of printable characters in text (0.0 to 1.0)."""
    if not text:
        return 1.0
    printable_count = sum(1 for c in text if c.isprintable())
    return printable_count / len(text)


def scan_qdrant_for_corrupt(
    client,
    threshold: float,
    user_filter: str | None = None,
    batch_size: int = 500,
) -> dict:
    """
    Scroll all Qdrant points, identify corrupt ones, and return metadata.

    Returns:
        {
            "corrupt_paths": {path: printable_ratio, ...},
            "corrupt_points": {path: count, ...},
            "total_points": int,
            "total_corrupt_points": int,
        }
    """
    corrupt_paths = {}  # path -> min_printable_ratio
    corrupt_points = {}  # path -> point count
    total_points = 0
    total_corrupt_points = 0

    # Build filter if --user is given
    filters = None
    if user_filter:
        filters = Filter(
            must=[
                FieldCondition(
                    key=FIELD_OWNER,
                    match=MatchValue(value=user_filter),
                )
            ]
        )

    # Scroll all points
    offset = 0
    while True:
        # Retry on transient Qdrant 500s ("timed out after 0ns") that occur when
        # the optimizer holds a segment lock during merge/vacuum.
        last_exc = None
        for attempt in range(5):
            try:
                points, next_offset = client.scroll(
                    collection_name=DOCUMENTS_COLLECTION,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    scroll_filter=filters,
                )
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                wait = 10 * (attempt + 1)
                logger.warning(
                    "Scroll attempt %d/5 failed (optimizer busy?): %s — retrying in %ds",
                    attempt + 1, e, wait,
                )
                import time
                time.sleep(wait)
        if last_exc is not None:
            logger.error("Failed to scroll Qdrant after 5 attempts: %s", last_exc)
            raise last_exc

        if not points:
            break

        for point in points:
            total_points += 1
            if total_points % 50000 == 0:
                logger.info("Scanned %d points so far...", total_points)

            payload = point.payload or {}
            parent_text = payload.get(FIELD_PARENT_TEXT, "")
            source_path = payload.get(FIELD_SOURCE_PATH, "")

            if not source_path or not parent_text:
                continue

            ratio = _printable_ratio(parent_text)
            if ratio < threshold:
                total_corrupt_points += 1

                # Track the minimum ratio seen for this path
                if source_path not in corrupt_paths:
                    corrupt_paths[source_path] = ratio
                    corrupt_points[source_path] = 1
                else:
                    corrupt_paths[source_path] = min(corrupt_paths[source_path], ratio)
                    corrupt_points[source_path] += 1

        offset = next_offset
        if not next_offset:
            break

    return {
        "corrupt_paths": corrupt_paths,
        "corrupt_points": corrupt_points,
        "total_points": total_points,
        "total_corrupt_points": total_corrupt_points,
    }


def file_exists(path: str) -> bool:
    """Check if the file still exists on disk."""
    try:
        return os.path.exists(path)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find and optionally purge corrupt Qdrant points"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Scan and report (default behavior)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Scan and DELETE corrupt points from Qdrant",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete even if file still exists on disk (use with --confirm)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Printable-char ratio threshold (default 0.85); points below this are corrupt",
    )
    parser.add_argument(
        "--user",
        metavar="OWNER",
        help="Filter to a specific owner (e.g. florian)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Write corrupt source_path list (one per line) to FILE",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/SoHoAI--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    qdrant_url = config.get("rag", {}).get("qdrant_url", "http://192.168.1.93:6333")

    # Open Qdrant client and a read-only SQLite connection.
    # Use read-only URI to avoid any WAL writes on NFS — this utility never
    # modifies SQLite, only reads it for cross-reference reporting.
    qdrant_client = get_client(qdrant_url, timeout=120)
    state_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    state_conn.row_factory = sqlite3.Row

    print()
    print(f"Scanning Qdrant for corrupt content (threshold: {args.threshold:.2f})")
    if args.user:
        print(f"  Filter: owner={args.user}")
    print()

    # Scan
    result = scan_qdrant_for_corrupt(
        qdrant_client,
        threshold=args.threshold,
        user_filter=args.user,
        batch_size=500,
    )

    corrupt_paths = result["corrupt_paths"]
    corrupt_points = result["corrupt_points"]
    total_points = result["total_points"]
    total_corrupt_points = result["total_corrupt_points"]

    print(f"Total points scanned: {total_points:,}")
    print(f"Total corrupt points: {total_corrupt_points:,}")
    print()

    if not corrupt_paths:
        print("No corrupt content detected.")
        state_conn.close()
        return

    # Build report: check SQLite for each path (read-only)
    report = []
    for path in sorted(corrupt_paths.keys()):
        ratio = corrupt_paths[path]
        point_count = corrupt_points[path]

        try:
            cur = state_conn.execute(
                "SELECT status FROM ingestion_queue WHERE file_path = ?",
                (path,),
            )
            row = cur.fetchone()
            in_db = "IN_DB" if row else "NOT_IN_DB"
        except Exception as e:
            logger.warning("Failed to check SQLite for %s: %s", path, e)
            in_db = "UNKNOWN"

        pct = ratio * 100
        report.append((path, point_count, in_db, pct, ratio))

    # Print report (sorted by path)
    print(f"Corrupt source paths ({len(report)} unique):")
    print()
    for path, point_count, in_db, pct, ratio in report:
        print(f"  [{point_count:>4} pts]  {in_db:<12}  {pct:>5.1f}%  {path}")
    print()

    # Save to file if requested
    if args.save:
        with open(args.save, "w") as f:
            for path, _, _, _, _ in report:
                f.write(path + "\n")
        print(f"Saved corrupt path list to {args.save}")
        print()

    # --confirm: delete from Qdrant
    if args.confirm:
        print("Deleting corrupt points from Qdrant...")
        deleted_count = 0
        failed_count = 0

        for path, point_count, in_db, pct, ratio in report:
            # Skip if file still exists (safety check) unless --force
            if not args.force and file_exists(path):
                logger.info("SKIP: file still exists on disk: %s", path)
                continue

            try:
                qdrant_client.delete(
                    collection_name=DOCUMENTS_COLLECTION,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key=FIELD_SOURCE_PATH,
                                    match=MatchValue(value=path),
                                )
                            ]
                        )
                    ),
                    wait=True,
                )
                logger.info("DEL OK: %s (%d points)", path, point_count)
                deleted_count += point_count
            except Exception as e:
                logger.error("FAIL: %s: %s", path, e)
                failed_count += point_count

        print(f"Deleted {deleted_count} points ({failed_count} failed)")
        print()
    else:
        print("(dry-run mode: pass --confirm to delete)")
        print()

    state_conn.close()

if __name__ == "__main__":
    main()
