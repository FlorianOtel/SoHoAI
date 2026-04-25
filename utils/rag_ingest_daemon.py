#!/usr/bin/env python3
"""
Process pending files from the RAG ingestion queue.

Runs the 7-step atomic ingestion loop (parse → chunk → embed → upsert) until
the queue is empty. Files are fetched from SQLite in batches of 10 (hardcoded).

Both --workers and --batch are required. They are separate knobs for different
bottlenecks — the right values depend on where the embedding server lives:

  --workers  Number of files processed concurrently. Each worker runs docling
             parsing in its own OS thread (thread-local DocumentConverter), so
             workers overlap CPU parse time with GPU/CPU embed time from other
             files. Set higher when the embedding server is a remote GPU (parse
             is the bottleneck); set to 1 when embedding is CPU-local (parse and
             embed compete for the same cores).

  --batch    Max concurrent Ollama embedding requests per file. Controls how
             many chunks are in-flight to Ollama at once. Set higher for a
             remote GPU (RTX 5070 can saturate many requests); set lower for
             CPU-local Ollama (it serialises inference anyway, high values just
             queue up).

Recommended operating points:
  GPU embed (Server 2, RTX 5070):  --workers 4 --batch 20
  CPU embed (Server 1, local):     --workers 1 --batch 5
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

_LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


def _add_file_handler(log_path: str) -> None:
    """Add a FileHandler to the root logger so all output also goes to log_path."""
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logging.getLogger().addHandler(handler)


_FETCH_BATCH_SIZE = 10  # files fetched from SQLite per loop iteration (hardcoded)


async def run(
    state_db: StateDB,
    qdrant_client,
    rag_cfg: dict,
    embed_concurrency: int,
    file_workers: int,
) -> None:
    """Main worker loop — processes files in parallel up to file_workers at a time."""
    recovered = state_db.crash_recovery()
    if recovered:
        logger.info("Crash recovery: reset %d stuck file(s) to pending", len(recovered))

    processed = failed = 0
    sem = asyncio.Semaphore(file_workers)

    async def ingest_one(row: dict) -> None:
        nonlocal processed, failed
        file_path = row["file_path"]
        async with sem:
            logger.info("Processing [%d done / %d failed]: %s", processed, failed, file_path)
            try:
                await ingest_file(
                    file_path=file_path,
                    owner=row["owner"],
                    rag_cfg=rag_cfg,
                    state_db=state_db,
                    qdrant_client=qdrant_client,
                    embed_concurrency=embed_concurrency,
                )
                processed += 1
            except Exception:
                failed += 1  # ingest_file already logged + marked failed

    while True:
        rows = state_db.fetch_pending_full(limit=_FETCH_BATCH_SIZE)
        if not rows:
            break

        await asyncio.gather(*[ingest_one(row) for row in rows])

        counts = state_db.get_counts()
        logger.info(
            "Batch done — pending: %d  completed: %d  failed: %d",
            counts["pending"], counts["completed"], counts["failed"],
        )

    logger.info("Worker finished: %d processed, %d failed", processed, failed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process pending RAG ingestion files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Both flags are required — their correct values depend on the embedding server:\n"
            "  GPU embed (Server 2):  --workers 4 --batch 20\n"
            "  CPU embed (Server 1):  --workers 1 --batch 5"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        metavar="N",
        help=(
            "Number of files to parse+embed concurrently. "
            "Higher values overlap docling CPU parse with remote GPU embed. "
            "Use 1 when embedding is CPU-local to avoid core contention."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        metavar="M",
        help=(
            "Max concurrent Ollama embedding requests per file (chunk-level parallelism). "
            "Higher values suit a remote GPU; keep low (3–5) for CPU-local Ollama."
        ),
    )
    parser.add_argument(
        "--log-file",
        metavar="PATH",
        default=None,
        help=(
            "Write all log output to this file in addition to stderr. "
            "Required for rag_status.py --watch to work."
        ),
    )
    args = parser.parse_args()

    if args.log_file:
        _add_file_handler(args.log_file)

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/HomeAI--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    state_db = StateDB(db_path)
    qdrant_client = get_client(rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"))
    ensure_collection(qdrant_client)

    counts_before = state_db.get_counts()
    if counts_before["pending"] == 0:
        print("No pending files. Run rag_sync_nfs.py first.")
        state_db.close()
        return

    logger.info(
        "Starting ingest: %d files pending  workers=%d  batch=%d",
        counts_before["pending"], args.workers, args.batch,
    )

    try:
        asyncio.run(
            run(
                state_db,
                qdrant_client,
                rag_cfg,
                embed_concurrency=args.batch,
                file_workers=args.workers,
            )
        )
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
