#!/usr/bin/env python3
"""
Scan configured NFS roots and populate the RAG ingestion queue.

Handles all cases uniformly:

  - New file:    not yet in SQLite → inserted as 'pending'.
  - Modified:    disk mtime > stored mtime → reset to 'pending' (any status,
                 including 'ignored' — the file was replaced on disk).
  - Failed:      status = 'failed', mtime unchanged → reset to 'pending'
                 with retry_count = 0 so the daemon gives it a fresh run.
  - Ignored:     status = 'ignored', mtime unchanged → no-op (permanent skip).
  - Completed / pending / processing, mtime unchanged: no-op.

Files removed from NFS (or newly matched by an exclusion filter in config.yaml)
are deleted from SQLite and their Qdrant points purged.

Usage (run from project root):
    python utils/rag_sync_nfs.py
    python utils/rag_sync_nfs.py --user florian
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Project root is one directory up from utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402  (after sys.path fix)

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client
from rag_engine.scanner import scan_nfs_roots
from rag_engine.schema import FIELD_SOURCE_PATH
from rag_engine.state import StateDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan NFS roots and populate the ingestion queue")
    parser.add_argument("--user", metavar="OWNER", help="Scan only this owner's roots (e.g. florian)")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI-lab--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    state_db = StateDB(db_path)

    result = scan_nfs_roots(state_db, config, user_filter=args.user)

    deleted_paths = result.get("deleted_paths", [])
    if deleted_paths:
        qdrant_url = config.get("rag", {}).get("qdrant_url", "http://localhost:6333")
        qdrant_client = get_client(qdrant_url)
        for path in deleted_paths:
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
            )
        logger.info("Deleted Qdrant points for %d removed file(s)", len(deleted_paths))

    counts = state_db.get_counts()
    print()
    print(f"Scan complete:")
    print(f"  Files discovered  : {result['scanned']}")
    print(f"  Stale rows removed: {result['deleted']}")
    print()
    print(f"Queue status:")
    print(f"  pending    : {counts['pending']}")
    print(f"  completed  : {counts['completed']}")
    print(f"  failed     : {counts['failed']}")
    print(f"  ignored    : {counts['ignored']}")
    print(f"  total      : {counts['total']}")

    state_db.close()


if __name__ == "__main__":
    main()
