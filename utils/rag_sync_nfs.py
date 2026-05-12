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
import time
from pathlib import Path

# Project root is one directory up from utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402  (after sys.path fix)

from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, get_client
from rag_engine.scanner import scan_nfs_roots, scan_claude_chats
from rag_engine.schema import FIELD_SOURCE_PATH
from rag_engine.state import StateDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Qdrant bulk-delete tuning: 50 paths per request keeps payloads small while
# dramatically reducing round-trips (e.g. 289 files → 6 requests instead of 289).
_DELETE_BATCH_SIZE = 50
_DELETE_MAX_RETRIES = 3


def _delete_paths_from_qdrant(client, paths: list[str]) -> None:
    """Delete Qdrant points for all paths in batched requests, with retry on timeout.

    Uses a `should` (OR) filter so one HTTP call covers many source paths at once.
    Qdrant's index re-optimization is triggered once per batch rather than once per
    file, cutting total time from O(N) sequential waits to O(N/batch_size) waits.

    Retries each batch up to _DELETE_MAX_RETRIES times with exponential back-off
    before re-raising. If a batch ultimately fails, the script aborts: SQLite rows
    survive intact so the next sync will retry the Qdrant cleanup automatically.
    """
    batches = [
        paths[i : i + _DELETE_BATCH_SIZE]
        for i in range(0, len(paths), _DELETE_BATCH_SIZE)
    ]
    for batch_num, batch in enumerate(batches, 1):
        for attempt in range(1, _DELETE_MAX_RETRIES + 1):
            try:
                client.delete(
                    collection_name=DOCUMENTS_COLLECTION,
                    points_selector=FilterSelector(
                        filter=Filter(
                            should=[
                                FieldCondition(
                                    key=FIELD_SOURCE_PATH,
                                    match=MatchValue(value=p),
                                )
                                for p in batch
                            ]
                        )
                    ),
                )
                logger.info(
                    "Qdrant batch %d/%d: removed points for %d path(s)",
                    batch_num, len(batches), len(batch),
                )
                break
            except (ResponseHandlingException, Exception) as exc:
                if attempt < _DELETE_MAX_RETRIES:
                    wait = 2 ** attempt  # 2 s, 4 s, 8 s
                    logger.warning(
                        "Qdrant batch %d/%d attempt %d failed (%s) — retrying in %ds",
                        batch_num, len(batches), attempt, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Qdrant batch %d/%d failed after %d attempts — aborting",
                        batch_num, len(batches), _DELETE_MAX_RETRIES,
                    )
                    raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan NFS roots and populate the ingestion queue")
    parser.add_argument("--user", metavar="OWNER", help="Scan only this owner's roots (e.g. florian)")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/SoHoAI--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    state_db = StateDB(db_path)

    result_nfs = scan_nfs_roots(state_db, config, user_filter=args.user)
    result_chats = scan_claude_chats(state_db, config, user_filter=args.user)

    all_existing = result_nfs["existing_paths"] | result_chats["existing_paths"]
    deleted_paths, stale_paths = state_db.find_deleted(all_existing)
    if all_existing and deleted_paths:
        logger.info("Found %d completed file(s) pending Qdrant cleanup", len(deleted_paths))
    if stale_paths:
        logger.info("Found %d deleted file(s) pending queue removal", len(stale_paths))

    # Qdrant cleanup BEFORE SQLite deletion — if killed mid-loop, the SQLite rows
    # survive intact and the next sync retries automatically (no orphaned Qdrant points).
    if deleted_paths:
        qdrant_url = config.get("rag", {}).get("qdrant_url", "http://192.168.1.93:6333")
        # 120 s timeout: batch deletes on a large collection can trigger index
        # re-optimization that takes longer than the default 60 s per request.
        qdrant_client = get_client(qdrant_url, timeout=120)
        _delete_paths_from_qdrant(qdrant_client, deleted_paths)
        logger.info("Deleted Qdrant points for %d removed file(s)", len(deleted_paths))
    if stale_paths:
        state_db.purge_deleted(stale_paths)

    counts = state_db.get_counts()
    print()
    print(f"Scan complete:")
    print(f"  NFS files discovered  : {result_nfs['scanned']}")
    print(f"  Chat sessions found   : {result_chats['scanned']}")
    print(f"  Stale rows removed    : {len(stale_paths)}")
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
