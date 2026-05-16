#!/usr/bin/env python3
"""
Reset the RAG pipeline — drop Qdrant collection and reset the ingestion queue.

Full reset (no --user): drops the entire Qdrant 'documents' collection and
resets all SQLite rows back to 'pending'.

Partial reset (--user florian): deletes only Florian's Qdrant points and
resets only Florian's SQLite rows to 'pending'. Other users are untouched.

Usage (run from project root):
    python utils/rag_reset.py                  # full reset (prompts for confirmation)
    python utils/rag_reset.py --user florian   # partial reset for one user
    python utils/rag_reset.py --yes            # skip confirmation prompt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

from rag_engine.collection import DOCUMENTS_COLLECTION, ensure_collection, get_client
from rag_engine.schema import FIELD_OWNER
from rag_engine.state import StateDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the RAG pipeline")
    parser.add_argument("--user", metavar="OWNER",
                        help="Partial reset: only this owner's data (e.g. florian)")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml") as f:
        config = yaml.safe_load(f)

    rag_cfg = config.get("rag", {})
    db_base = config.get("db_base_path", "/mnt/nfs/__Backups/SoHoAI--databases")
    db_path = f"{db_base}/sqlite/rag_state.db"
    scope = f"owner={args.user}" if args.user else "ALL USERS"
    action = (
        f"Delete Qdrant points for {scope} and reset SQLite rows to 'pending'."
        if args.user else
        f"DROP the entire Qdrant '{DOCUMENTS_COLLECTION}' collection and reset all SQLite rows."
    )

    print(f"\n{action}")
    if not args.yes:
        confirm = input("Confirm? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    state_db = StateDB(db_path)
    qdrant = get_client(rag_cfg.get("qdrant_url", "http://192.168.1.93:6333"))
    existing = {c.name for c in qdrant.get_collections().collections}

    if args.user:
        # --- Partial reset: one owner ---
        if DOCUMENTS_COLLECTION in existing:
            qdrant.delete(
                collection_name=DOCUMENTS_COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(key=FIELD_OWNER, match=MatchValue(value=args.user))
                    ])
                ),
            )
            print(f"Deleted Qdrant points for owner={args.user}")

        rows_reset = state_db._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = 'pending', retry_count = 0, error_msg = NULL, "
            "    started_at = NULL, completed_at = NULL, progress_detail = NULL "
            "WHERE owner = ?",
            (args.user,),
        ).rowcount
        state_db._conn.commit()
        print(f"Reset {rows_reset} SQLite row(s) to 'pending' for owner={args.user}")

    else:
        # --- Full reset ---
        if DOCUMENTS_COLLECTION in existing:
            qdrant.delete_collection(DOCUMENTS_COLLECTION)
            print(f"Dropped Qdrant collection '{DOCUMENTS_COLLECTION}'")
        ensure_collection(qdrant)
        print(f"Re-created empty collection '{DOCUMENTS_COLLECTION}'")

        rows_reset = state_db._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = 'pending', retry_count = 0, error_msg = NULL, "
            "    started_at = NULL, completed_at = NULL, progress_detail = NULL"
        ).rowcount
        state_db._conn.commit()
        print(f"Reset {rows_reset} SQLite row(s) to 'pending'")

    counts = state_db.get_counts()
    print(f"\nQueue: {counts['pending']} pending, {counts['completed']} completed, {counts['failed']} failed")
    state_db.close()


if __name__ == "__main__":
    main()
