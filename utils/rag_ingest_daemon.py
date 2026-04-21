#!/usr/bin/env python3
"""
Process pending files from the RAG ingestion queue.

Runs the 7-step atomic ingestion loop (parse → chunk → embed → upsert) until
the queue is empty. Files are fetched from SQLite in batches of 10 (hardcoded).

--batch controls the Ollama embedding concurrency (queue depth), NOT the number
of files fetched. Lower values reduce httpx.ReadTimeout errors when Ollama is
under load; higher values improve throughput when Ollama has headroom.

Usage (run from project root):
    python utils/rag_ingest_daemon.py            # default concurrency=5
    python utils/rag_ingest_daemon.py --batch 2  # conservative; fewer timeouts
    python utils/rag_ingest_daemon.py --batch 8  # aggressive; more throughput
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from rag_engine.collection import ensure_collection, get_client
from rag_engine.ingest import ingest_file
from rag_engine.state import StateDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_FETCH_BATCH_SIZE = 10  # files fetched from SQLite per loop iteration (hardcoded)


async def run(state_db: StateDB, qdrant_client, rag_cfg: dict, embed_concurrency: int) -> None:
    """Main worker loop."""
    recovered = state_db.crash_recovery()
    if recovered:
        logger.info("Crash recovery: reset %d stuck file(s) to pending", len(recovered))

    processed = failed = skipped = 0

    while True:
        rows = state_db.fetch_pending_full(limit=_FETCH_BATCH_SIZE)
        if not rows:
            break

        for row in rows:
            file_path = row["file_path"]
            owner = row["owner"]
            logger.info("Processing [%d done / %d failed]: %s", processed, failed, file_path)
            try:
                await ingest_file(
                    file_path=file_path,
                    owner=owner,
                    rag_cfg=rag_cfg,
                    state_db=state_db,
                    qdrant_client=qdrant_client,
                    embed_concurrency=embed_concurrency,
                )
                processed += 1
            except Exception as exc:
                logger.error("Failed: %s — %s", file_path, exc)
                failed += 1

        # After processing a batch, re-fetch; loop exits when queue is empty
        counts = state_db.get_counts()
        logger.info(
            "Batch done — pending: %d  completed: %d  failed: %d",
            counts["pending"], counts["completed"], counts["failed"],
        )

    logger.info("Worker finished: %d processed, %d failed", processed, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process pending RAG ingestion files")
    parser.add_argument(
        "--batch", type=int, default=5,
        help="Ollama embedding concurrency — max parallel requests per file (default: 5). "
             "Lower to reduce httpx.ReadTimeout errors under load. "
             "Files fetched from SQLite per iteration is hardcoded to 10.",
    )
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI-lab--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    state_db = StateDB(db_path)
    qdrant_client = get_client(rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"))
    ensure_collection(qdrant_client)

    counts_before = state_db.get_counts()
    if counts_before["pending"] == 0:
        print("No pending files. Run rag_sync_nfs.py first.")
        state_db.close()
        return

    print(f"Starting ingest: {counts_before['pending']} files pending")

    try:
        asyncio.run(run(state_db, qdrant_client, rag_cfg, embed_concurrency=args.batch))
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    counts_after = state_db.get_counts()
    print()
    print("Final queue state:")
    print(f"  pending    : {counts_after['pending']}")
    print(f"  completed  : {counts_after['completed']}")
    print(f"  failed     : {counts_after['failed']}")

    state_db.close()


if __name__ == "__main__":
    main()
