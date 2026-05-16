#!/usr/bin/env python3
"""
Migrate Qdrant documents collection from dense-only to hybrid dense+sparse vectors.

Qdrant 1.17 does NOT support adding a sparse vector space to an existing collection
via update_collection. This script creates a new collection with the correct schema,
copies all points with computed sparse vectors, and swaps the alias.

Usage (run from project root):
    python utils/rag_sparse_migrate.py migrate [--batch-size 500] [--qdrant-url URL] [--force]
    python utils/rag_sparse_migrate.py swap [--confirm] [--qdrant-url URL]
    python utils/rag_sparse_migrate.py status [--qdrant-url URL]

Phases:
    migrate:  Safe to run while app is live. Creates documents_new collection,
              copies all points from documents with computed sparse vectors.
    swap:     Requires app to be stopped (brief downtime ~5s).
              Deletes documents, creates alias documents -> documents_new.
    status:   Read-only, can run anytime. Shows collection state.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.collection import DOCUMENTS_COLLECTION, VECTOR_SIZE
from rag_engine.embeddings import embed_sparse
from rag_engine.schema import FIELD_TEXT, SPARSE_VECTOR_NAME

logger = logging.getLogger(__name__)

# New collection name (hardcoded constant)
DOCUMENTS_NEW_COLLECTION = "documents_new"


def _load_config() -> dict:
    """Load SoHoAI-config.yaml from project root."""
    config_path = Path(__file__).resolve().parent.parent / "SoHoAI-config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"SoHoAI-config.yaml not found at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_qdrant_url(cli_url: str | None, config: dict) -> str:
    """Resolve Qdrant URL from CLI or config."""
    if cli_url:
        return cli_url
    url = config.get("rag", {}).get("qdrant_url")
    if not url:
        raise ValueError("qdrant_url not found in SoHoAI-config.yaml[rag] and not provided via --qdrant-url")
    return url


# ---------------------------------------------------------------------------
# Phase 1: Migrate (create new collection and copy all points with sparse vectors)
# ---------------------------------------------------------------------------


def migrate(
    qdrant_url: str,
    batch_size: int = 500,
    force: bool = False,
) -> None:
    """
    Create documents_new collection with dense+sparse schema and copy all points.

    Safe to run while app is live. Uses batch scrolling and upsert.

    Args:
        qdrant_url:   Qdrant server URL.
        batch_size:   Points per batch (default 500).
        force:        If True, skip prompt if documents_new already exists.
    """
    client = QdrantClient(url=qdrant_url)

    # Check if documents_new already exists
    real_collections = {c.name for c in client.get_collections().collections}
    if DOCUMENTS_NEW_COLLECTION in real_collections:
        if not force:
            response = input(
                f"{DOCUMENTS_NEW_COLLECTION} already exists. Overwrite? (y/n) "
            ).strip().lower()
            if response != "y":
                print("Aborted.")
                return
        print(f"Deleting existing {DOCUMENTS_NEW_COLLECTION}...")
        client.delete_collection(DOCUMENTS_NEW_COLLECTION)

    # Create documents_new with correct schema
    print(f"Creating {DOCUMENTS_NEW_COLLECTION} with dense + sparse vectors...")
    client.create_collection(
        collection_name=DOCUMENTS_NEW_COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
        },
    )
    print(f"✓ {DOCUMENTS_NEW_COLLECTION} collection created.")

    # Get the source collection info to estimate total
    source_info = client.get_collection(DOCUMENTS_COLLECTION)
    estimated_total = source_info.points_count
    print(f"Migrating {estimated_total} points from {DOCUMENTS_COLLECTION}...")

    # Scroll and migrate in batches
    total_migrated = 0
    batch_num = 0
    offset = None   # pagination cursor; None = start from beginning

    while True:
        batch_num += 1

        # Scroll from source collection; next_offset is the pagination cursor
        scroll_result, next_offset = client.scroll(
            DOCUMENTS_COLLECTION,
            offset=offset,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
        )
        offset = next_offset   # advance cursor for next iteration

        if not scroll_result:
            break

        # Build upsert batch with sparse vectors
        points_to_upsert: list[PointStruct] = []
        for point in scroll_result:
            # Extract text field for sparse embedding
            text = point.payload.get(FIELD_TEXT, "")

            # Compute sparse vector
            sparse_indices, sparse_values = embed_sparse(text)

            # Build PointStruct with both dense and sparse vectors
            point_struct = PointStruct(
                id=point.id,
                vector={
                    "": point.vector,  # Named dense vector (unnamed key "")
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                },
                payload=point.payload,
            )
            points_to_upsert.append(point_struct)

        # Upsert batch to new collection
        try:
            client.upsert(
                DOCUMENTS_NEW_COLLECTION,
                points=points_to_upsert,
            )
            total_migrated += len(points_to_upsert)
            print(
                f"batch {batch_num}: upserted {len(points_to_upsert)} points "
                f"(total migrated: {total_migrated}/{estimated_total})"
            )
        except Exception as e:
            logger.error(f"Batch {batch_num} upsert failed: {e}")
            print(f"⚠ Batch {batch_num} upsert failed (continuing): {e}")

        # Stop if we've reached the end
        if next_offset is None or not scroll_result:
            break

    # Final summary
    new_info = client.get_collection(DOCUMENTS_NEW_COLLECTION)
    print(
        f"✓ Migration complete. {DOCUMENTS_NEW_COLLECTION}: {new_info.points_count} points "
        f"({DOCUMENTS_COLLECTION}: {estimated_total} points)"
    )


# ---------------------------------------------------------------------------
# Phase 2: Swap (delete documents, create alias documents -> documents_new)
# ---------------------------------------------------------------------------


def swap(qdrant_url: str, confirm: bool = False) -> None:
    """
    Delete documents collection and create alias documents -> documents_new.

    Requires app to be stopped (brief downtime ~5 seconds).

    Args:
        qdrant_url:  Qdrant server URL.
        confirm:     If True, skip safety confirmation prompt.
    """
    client = QdrantClient(url=qdrant_url)

    # Verify documents_new exists and has sufficient points
    real_collections = {c.name for c in client.get_collections().collections}
    if DOCUMENTS_NEW_COLLECTION not in real_collections:
        print(f"✗ {DOCUMENTS_NEW_COLLECTION} does not exist. Run --migrate first.")
        sys.exit(1)

    new_info = client.get_collection(DOCUMENTS_NEW_COLLECTION)
    source_info = client.get_collection(DOCUMENTS_COLLECTION)
    source_count = source_info.points_count
    new_count = new_info.points_count

    # Sanity check: new collection has at least 95% of original points
    if new_count < source_count * 0.95:
        print(
            f"✗ Sanity check failed: {DOCUMENTS_NEW_COLLECTION} has {new_count} points "
            f"but {DOCUMENTS_COLLECTION} has {source_count} points. "
            f"Expected >= {int(source_count * 0.95)} in new collection."
        )
        sys.exit(1)

    # Safety warning
    print("=" * 70)
    print("⚠  WARNING: This will DELETE the 'documents' collection and create an alias.")
    print("    Stop the FastAPI app (uvicorn) before proceeding.")
    print("    Downtime: ~5 seconds.")
    print("=" * 70)

    if not confirm:
        response = input("Proceed with swap? (y/n) ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Delete documents collection
    print(f"Deleting {DOCUMENTS_COLLECTION}...")
    client.delete_collection(DOCUMENTS_COLLECTION)
    print(f"✓ {DOCUMENTS_COLLECTION} deleted.")

    # Create alias documents -> documents_new
    print(f"Creating alias {DOCUMENTS_COLLECTION} -> {DOCUMENTS_NEW_COLLECTION}...")
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=DOCUMENTS_NEW_COLLECTION,
                    alias_name=DOCUMENTS_COLLECTION,
                )
            )
        ]
    )

    # Verify alias resolves
    try:
        alias_info = client.get_collection(DOCUMENTS_COLLECTION)
        print(
            f"✓ Swap complete. '{DOCUMENTS_COLLECTION}' is now an alias to "
            f"'{DOCUMENTS_NEW_COLLECTION}' ({alias_info.points_count} points)."
        )
    except Exception as e:
        print(f"✗ Alias verification failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Phase 3: Status (show collection state)
# ---------------------------------------------------------------------------


def status(qdrant_url: str) -> None:
    """
    Show collection state: counts, alias status, sparse vector support.

    Read-only, can run anytime.

    Args:
        qdrant_url: Qdrant server URL.
    """
    client = QdrantClient(url=qdrant_url)

    real_collections = {c.name for c in client.get_collections().collections}
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}

    print("=" * 70)
    print("RAG SPARSE MIGRATION STATUS")
    print("=" * 70)

    # Status of documents (real or alias)
    if DOCUMENTS_COLLECTION in aliases:
        target_collection = aliases[DOCUMENTS_COLLECTION]
        print(f"documents: ALIAS -> {target_collection}")
    elif DOCUMENTS_COLLECTION in real_collections:
        print(f"documents: REAL COLLECTION")
    else:
        print(f"documents: NOT FOUND")

    # Show point counts
    for name in [DOCUMENTS_COLLECTION, DOCUMENTS_NEW_COLLECTION]:
        try:
            info = client.get_collection(name)
            print(f"  {name}: {info.points_count} points")
        except Exception as e:
            print(f"  {name}: not found ({e})")

    # Check sparse vector support
    try:
        info = client.get_collection(DOCUMENTS_COLLECTION)
        sparse_config = getattr(info.config.params, "sparse_vectors", None)  # qdrant-client uses "sparse_vectors", not "sparse_vectors_config"
        has_sparse = sparse_config is not None and SPARSE_VECTOR_NAME in sparse_config
        print(f"  sparse vectors: {'✓ YES' if has_sparse else '✗ NO'}")
    except Exception as e:
        print(f"  sparse vectors: ERROR ({e})")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args and execute the requested phase."""
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection to hybrid dense+sparse vectors.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --migrate
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Phase 1: Create documents_new and copy all points with sparse vectors",
    )
    migrate_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Points per batch (default 500)",
    )
    migrate_parser.add_argument(
        "--qdrant-url",
        help="Qdrant server URL (default from SoHoAI-config.yaml)",
    )
    migrate_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip prompt if documents_new already exists",
    )

    # --swap
    swap_parser = subparsers.add_parser(
        "swap",
        help="Phase 2: Delete documents and create alias (requires app stopped)",
    )
    swap_parser.add_argument(
        "--qdrant-url",
        help="Qdrant server URL (default from SoHoAI-config.yaml)",
    )
    swap_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip safety confirmation prompt",
    )

    # --status
    status_parser = subparsers.add_parser(
        "status",
        help="Phase 3: Show collection state (read-only)",
    )
    status_parser.add_argument(
        "--qdrant-url",
        help="Qdrant server URL (default from SoHoAI-config.yaml)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config for default qdrant_url
    try:
        config = _load_config()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        sys.exit(1)

    # Resolve Qdrant URL
    qdrant_url = _get_qdrant_url(args.qdrant_url, config)
    print(f"Using Qdrant: {qdrant_url}")

    # Execute command
    try:
        if args.command == "migrate":
            migrate(
                qdrant_url=qdrant_url,
                batch_size=args.batch_size,
                force=args.force,
            )
        elif args.command == "swap":
            swap(qdrant_url=qdrant_url, confirm=args.confirm)
        elif args.command == "status":
            status(qdrant_url=qdrant_url)
    except Exception as e:
        logger.exception(f"Command '{args.command}' failed")
        print(f"✗ {args.command} failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
