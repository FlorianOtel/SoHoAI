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

_DELETE_MAX_RETRIES = 5


def _delete_path_from_qdrant(client, path: str, file_num: int, total: int) -> None:
    """Delete Qdrant points for one source path, with exponential back-off on failure.

    Uses wait=False so Qdrant queues the operation and returns immediately — the
    caller never blocks on index re-optimization, which can take minutes on a large
    collection. The crash-safe ordering is preserved: all deletes are submitted
    before purge_deleted() removes the SQLite rows; surviving rows act as retry
    markers if the process is killed mid-loop.
    """
    for attempt in range(1, _DELETE_MAX_RETRIES + 1):
        try:
            client.delete(
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
                wait=False,
            )
            logger.info("Qdrant cleanup %d/%d: queued delete for %s", file_num, total, path)
            return
        except (ResponseHandlingException, Exception) as exc:
            if attempt < _DELETE_MAX_RETRIES:
                backoff = 2 ** attempt  # 2, 4, 8, 16 s
                logger.warning(
                    "Qdrant delete attempt %d/%d failed (%s) — retrying in %ds",
                    attempt, _DELETE_MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
            else:
                logger.error(
                    "Qdrant delete failed after %d attempts — aborting", _DELETE_MAX_RETRIES
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
        qdrant_client = get_client(qdrant_url, timeout=120)
        for i, path in enumerate(deleted_paths, 1):
            _delete_path_from_qdrant(qdrant_client, path, i, len(deleted_paths))
        logger.info("Queued Qdrant point deletion for %d removed file(s)", len(deleted_paths))
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
