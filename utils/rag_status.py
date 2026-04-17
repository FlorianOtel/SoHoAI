#!/usr/bin/env python3
"""
Show RAG ingestion queue status and Qdrant collection stats.

Usage (run from project root):
    python utils/rag_status.py
    python utils/rag_status.py --user florian
    python utils/rag_status.py --failed        # list permanently failed files
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client
from rag_engine.schema import FIELD_OWNER
from rag_engine.state import StateDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Show RAG pipeline status")
    parser.add_argument("--user", metavar="OWNER", help="Filter counts by owner (e.g. florian)")
    parser.add_argument("--failed", action="store_true", help="List permanently failed files")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI-lab--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    qdrant_path = f"{db_base}/qdrant"

    state_db = StateDB(db_path)

    # --- SQLite queue counts ---
    if args.user:
        # Count only for this owner by querying directly
        conn = state_db._conn
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_queue "
            "WHERE owner = ? GROUP BY status",
            (args.user,),
        )
        raw = {row["status"]: row["n"] for row in cur.fetchall()}
        counts = {
            "pending":    raw.get("pending", 0),
            "processing": raw.get("processing", 0),
            "completed":  raw.get("completed", 0),
            "failed":     raw.get("failed", 0),
            "total":      sum(raw.values()),
        }
        scope = f"owner={args.user}"
    else:
        counts = state_db.get_counts()
        scope = "all users"

    print(f"\nIngestion queue ({scope}):")
    print(f"  pending    : {counts['pending']}")
    print(f"  processing : {counts['processing']}")
    print(f"  completed  : {counts['completed']}")
    print(f"  failed     : {counts['failed']}")
    print(f"  ─────────────────")
    print(f"  total      : {counts['total']}")

    if counts["total"] > 0:
        pct = counts["completed"] / counts["total"] * 100
        print(f"  progress   : {pct:.1f}%")

    # --- Qdrant stats ---
    try:
        qdrant = get_client(qdrant_path)
        existing = {c.name for c in qdrant.get_collections().collections}
        if DOCUMENTS_COLLECTION in existing:
            total_pts = qdrant.count(DOCUMENTS_COLLECTION, exact=True).count
            if args.user:
                user_pts = qdrant.count(
                    DOCUMENTS_COLLECTION,
                    count_filter=Filter(must=[
                        FieldCondition(key=FIELD_OWNER, match=MatchValue(value=args.user))
                    ]),
                    exact=True,
                ).count
                print(f"\nQdrant '{DOCUMENTS_COLLECTION}' collection:")
                print(f"  total points    : {total_pts}")
                print(f"  {args.user:15s} : {user_pts}")
            else:
                print(f"\nQdrant '{DOCUMENTS_COLLECTION}' collection:")
                print(f"  total points    : {total_pts}")
        else:
            print(f"\nQdrant: collection '{DOCUMENTS_COLLECTION}' does not exist yet")
    except Exception as exc:
        print(f"\nQdrant: unavailable ({exc})")

    # --- Failed files ---
    if args.failed:
        failed = state_db.get_failed()
        if failed:
            print(f"\nPermanently failed files ({len(failed)}):")
            for row in failed:
                print(f"  [{row['retry_count']} retries] {row['file_path']}")
                if row["error_msg"]:
                    print(f"    → {row['error_msg'][:120]}")
        else:
            print("\nNo permanently failed files.")

    state_db.close()


if __name__ == "__main__":
    main()
