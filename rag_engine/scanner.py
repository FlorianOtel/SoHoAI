"""
NFS filesystem scanner — discovers files and populates the ingestion queue.

Shared by:
  - utils/rag_sync_nfs.py  (CLI invocation)
  - POST /v1/rag/ingest/sync  (FastAPI endpoint)

Exclusion rules (RAG-strategy.md §1.2):
  - .Gin-AI-python-3.12/ Python virtualenv subtree
  - *.dist-info/ package metadata directories
  - *@synoeastream Synology streaming metadata
  - __pycache__, .git, node_modules directories

Included extensions:
  .pdf  .pptx  .docx  .md  .ipynb  .txt  .csv  .yaml  .yml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .state import StateDB

logger = logging.getLogger(__name__)

INCLUDE_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".pptx", ".docx",     # structured documents
    ".md", ".txt", ".csv",        # text files
    ".yaml", ".yml",              # config / data
    ".ipynb",                     # Jupyter notebooks
})

# Directory names to skip entirely — os.walk will not descend into them
_SKIP_DIR_NAMES: frozenset[str] = frozenset({
    ".Gin-AI-python-3.12",
    "__pycache__",
    ".git",
    "node_modules",
})


def _is_excluded_dir(dirname: str) -> bool:
    if dirname in _SKIP_DIR_NAMES:
        return True
    if dirname.endswith(".dist-info"):
        return True
    return False


def _should_include(path: Path) -> bool:
    if path.suffix.lower() not in INCLUDE_EXTENSIONS:
        return False
    if "@synoeastream" in path.name:
        return False
    return True


def scan_nfs_roots(
    state_db: StateDB,
    config: dict,
    user_filter: str | None = None,
) -> dict[str, int]:
    """
    Walk all configured NFS roots and update the ingestion queue.

    Reads the top-level 'users' and 'shared' sections from config to discover
    which NFS paths to scan and which owner string to assign.

    For each discovered file:
      - New files      → inserted as 'pending'
      - Modified files → mtime on disk > stored mtime → reset to 'pending'
      - Unchanged      → no-op

    Completed rows whose files no longer exist are removed from SQLite;
    their Qdrant points should be cleaned up separately.

    Args:
        state_db:    StateDB instance to populate.
        config:      Full config dict (reads 'users' + 'shared' sections).
        user_filter: If set (e.g. "florian"), only scan that owner's NFS roots.

    Returns:
        {'scanned': N, 'deleted': N}
    """
    roots_to_scan: list[tuple[str, str]] = []  # (nfs_root_path, owner)

    for _email, cfg in config.get("users", {}).items():
        owner = cfg["owner"]
        if user_filter and owner != user_filter:
            continue
        for root in cfg.get("nfs_roots", []):
            roots_to_scan.append((root, owner))

    # Shared roots are included unless we're filtering to a specific user
    if not user_filter:
        shared = config.get("shared", {})
        for root in shared.get("nfs_roots", []):
            roots_to_scan.append((root, shared["owner"]))

    if not roots_to_scan:
        logger.warning(
            "No NFS roots to scan. "
            "Fill in 'users' / 'shared' sections in config.yaml."
        )

    scanned = 0
    existing_paths: set[str] = set()

    for root, owner in roots_to_scan:
        if not os.path.isdir(root):
            logger.warning("NFS root not accessible, skipping: %s", root)
            continue

        root_count = 0
        logger.info("Scanning %s  (owner=%s)", root, owner)

        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            # Prune excluded subtrees in-place — prevents os.walk from descending
            dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if _should_include(Path(filepath)):
                    try:
                        mtime = os.path.getmtime(filepath)
                    except OSError:
                        continue
                    state_db.discover_or_update(filepath, owner, mtime)
                    existing_paths.add(filepath)
                    scanned += 1
                    root_count += 1

        logger.info("  → %d file(s) found in %s", root_count, root)

    deleted = state_db.handle_deleted(existing_paths)
    if deleted:
        logger.info("Removed %d deleted file(s) from queue", len(deleted))

    return {"scanned": scanned, "deleted": len(deleted)}
