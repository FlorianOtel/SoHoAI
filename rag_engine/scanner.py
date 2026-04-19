"""
NFS filesystem scanner — discovers files and populates the ingestion queue.

Shared by:
  - utils/rag_sync_nfs.py  (CLI invocation)
  - POST /v1/rag/ingest/sync  (FastAPI endpoint)

Exclusion rules are read from config["rag"]["scanner"] in config.yaml.
All four keys (include_extensions, exclude_dir_names, exclude_dir_suffixes,
exclude_file_patterns) are required — missing config raises ValueError.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .state import StateDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter helpers — accept sets built from config
# ---------------------------------------------------------------------------

def _is_excluded_dir(
    dirname: str,
    skip_names: frozenset[str],
    skip_suffixes: tuple[str, ...],
) -> bool:
    if dirname in skip_names:
        return True
    for suffix in skip_suffixes:
        if dirname.endswith(suffix):
            return True
    return False


def _should_include(
    path: Path,
    include_exts: frozenset[str],
    exclude_patterns: tuple[str, ...],
) -> bool:
    if path.suffix.lower() not in include_exts:
        return False
    for pattern in exclude_patterns:
        if pattern in path.name:
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
    # -- Build filter sets from config (required — no silent fallback) ---------
    scanner_cfg = config.get("rag", {}).get("scanner")
    if not scanner_cfg:
        raise ValueError(
            "config.yaml is missing the 'rag.scanner' section. "
            "Add include_extensions, exclude_dir_names, exclude_dir_suffixes, "
            "and exclude_file_patterns."
        )

    _REQUIRED = ("include_extensions", "exclude_dir_names", "exclude_dir_suffixes", "exclude_file_patterns")
    missing = [k for k in _REQUIRED if k not in scanner_cfg]
    if missing:
        raise ValueError(
            f"config.yaml rag.scanner is missing required key(s): {', '.join(missing)}"
        )

    include_exts = frozenset(scanner_cfg["include_extensions"])
    skip_names = frozenset(scanner_cfg["exclude_dir_names"])
    skip_suffixes = tuple(scanner_cfg["exclude_dir_suffixes"])
    exclude_patterns = tuple(scanner_cfg["exclude_file_patterns"])

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
            dirnames[:] = [
                d for d in dirnames
                if not _is_excluded_dir(d, skip_names, skip_suffixes)
            ]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if _should_include(Path(filepath), include_exts, exclude_patterns):
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

    return {"scanned": scanned, "deleted": len(deleted), "deleted_paths": deleted}
