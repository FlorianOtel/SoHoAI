"""
NFS filesystem scanner — discovers files and populates the ingestion queue.

Shared by:
  - utils/rag_sync_nfs.py  (CLI invocation)
  - POST /v1/rag/ingest/sync  (FastAPI endpoint)

Exclusion rules are read from config["rag"]["scanner"] in SoHoAI-config.yaml.
All four keys (include_extensions, exclude_dir_names, exclude_dir_suffixes,
exclude_file_patterns) are required — missing config raises ValueError.

exclude_dir_names convention: every entry must have a trailing slash.
Single-component entries (e.g. "Library/") match any directory named exactly
"Library" at any depth. Multi-component entries (e.g. "Microsoft--flotel/Documents/")
match any directory whose full path ends with that exact path segment sequence.
Matching is done against the full child path, so "Library/" will NOT match
"PublicLibrary/" — the leading "/" in the suffix check enforces an exact boundary.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

from .state import StateDB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter helpers — accept sets built from config
# ---------------------------------------------------------------------------

def _is_excluded_dir(
    dirpath: str,
    dirname: str,
    skip_patterns: tuple[str, ...],
    skip_suffixes: tuple[str, ...],
) -> bool:
    full_child = os.path.join(dirpath, dirname)
    for pattern in skip_patterns:
        # Strip trailing slash, prepend "/" to enforce exact path-boundary match.
        # "Library/" matches ".../Library" but not ".../PublicLibrary".
        # "Microsoft--flotel/Documents/" matches ".../Microsoft--flotel/Documents".
        p = pattern.rstrip("/")
        if full_child.endswith("/" + p):
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
            "SoHoAI-config.yaml is missing the 'rag.scanner' section."
            "Add include_extensions, exclude_dir_names, exclude_dir_suffixes, "
            "and exclude_file_patterns."
        )

    _REQUIRED = ("include_extensions", "exclude_dir_names", "exclude_dir_suffixes", "exclude_file_patterns")
    missing = [k for k in _REQUIRED if k not in scanner_cfg]
    if missing:
        raise ValueError(
            f"SoHoAI-config.yaml rag.scanner is missing required key(s): {', '.join(missing)}"
        )

    include_exts = frozenset(scanner_cfg["include_extensions"])
    skip_patterns = tuple(scanner_cfg["exclude_dir_names"])
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
            "Fill in 'users' / 'shared' sections in SoHoAI-config.yaml."
        )

    scanned = 0
    existing_paths: set[str] = set()
    visited_real_dirs: set[str] = set()   # global — prevents re-walking the same real dir via different symlinks
    visited_real_files: set[str] = set()  # global — prevents ingesting the same real file via different symlink paths

    for root, owner in roots_to_scan:
        if not os.path.isdir(root):
            logger.warning("NFS root not accessible, skipping: %s", root)
            continue

        root_count = 0
        logger.info("Scanning %s  (owner=%s)", root, owner)

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=True):
            # Directory dedup — skip any real dir path already visited (handles circular symlinks
            # and the same dir reachable via multiple symlinks across roots)
            real_dirpath = os.path.realpath(dirpath)
            if real_dirpath in visited_real_dirs:
                dirnames.clear()
                continue
            visited_real_dirs.add(real_dirpath)

            # Prune excluded subtrees in-place — prevents os.walk from descending
            dirnames[:] = [
                d for d in dirnames
                if not _is_excluded_dir(dirpath, d, skip_patterns, skip_suffixes)
            ]

            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if _should_include(Path(filepath), include_exts, exclude_patterns):
                    real_filepath = os.path.realpath(filepath)
                    if real_filepath in visited_real_files:
                        continue
                    visited_real_files.add(real_filepath)
                    try:
                        mtime = os.path.getmtime(filepath)
                    except OSError:
                        continue
                    state_db.discover_or_update(filepath, owner, mtime)
                    existing_paths.add(filepath)
                    scanned += 1
                    root_count += 1

        logger.info("  → %d file(s) found in %s", root_count, root)

    return {
        "scanned": scanned,
        "existing_paths": existing_paths,
    }


def scan_claude_chats(
    state_db: StateDB,
    config: dict,
    user_filter: str | None = None,
) -> dict:
    """
    Walk configured claude_chats roots and update the ingestion queue with .jsonl session files.

    Reads config["claude_chats"]["roots"] (list of {path, owner} dicts).
    Skips gracefully if the key is absent.

    Returns:
        {'scanned': N, 'existing_paths': set[str]}
    """
    roots_cfg = config.get("claude_chats", {}).get("roots", [])
    if not roots_cfg:
        logger.info("No claude_chats roots configured — skipping chat scan")
        return {"scanned": 0, "existing_paths": set()}

    scanned = 0
    existing_paths: set[str] = set()
    visited_real_files: set[str] = set()

    for entry in roots_cfg:
        root = entry.get("path", "")
        owner = entry.get("owner", "")
        if user_filter and owner != user_filter:
            continue
        if not root or not os.path.isdir(root):
            logger.warning("claude_chats root not accessible, skipping: %s", root)
            continue

        root_count = 0
        logger.info("Scanning claude chats: %s  (owner=%s)", root, owner)

        for dirpath, _dirnames, filenames in os.walk(root, topdown=True, followlinks=True):
            for filename in filenames:
                if not filename.endswith(".jsonl"):
                    continue
                filepath = os.path.join(dirpath, filename)
                real_filepath = os.path.realpath(filepath)
                if real_filepath in visited_real_files:
                    continue
                visited_real_files.add(real_filepath)
                try:
                    mtime = os.path.getmtime(filepath)
                except OSError:
                    continue
                state_db.discover_or_update(filepath, owner, mtime)
                existing_paths.add(filepath)
                scanned += 1
                root_count += 1

        logger.info("  → %d session(s) found in %s", root_count, root)

    return {"scanned": scanned, "existing_paths": existing_paths}


def scan_opencode_sessions(
    state_db: StateDB,
    config: dict,
    user_filter: str | None = None,
) -> dict:
    """
    Discover opencode sessions via the opencode HTTP API and update the
    ingestion queue.

    Walks GET /api/session (v2) with cursor pagination until exhausted. This
    endpoint returns every session opencode knows about (project-bound or
    loose, archived or active) — no directory filter, no /project enumeration.

    Reads config["opencode"]:
      - api_url: base URL of the opencode HTTP API
      - owner:   constant owner for all opencode sessions (single-user per host)

    Returns gracefully if the config key is absent. On any HTTP failure mid-
    walk, returns existing_paths=None to preserve already-indexed Qdrant points
    (callers must skip this source in find_deleted()).

    Synthesises file_path as "opencode://{session_id}". Mtime is derived from
    session["time"]["updated"] / 1000.0 (UNIX milliseconds).

    user_filter, if set, narrows the scan to that owner; if the configured
    opencode.owner doesn't match user_filter, the scan is skipped.

    Returns:
        {'scanned': N, 'existing_paths': set[str] | None}
    """
    opencode_cfg = config.get("opencode", {})
    if not opencode_cfg:
        logger.info("No opencode config — skipping opencode scan")
        return {"scanned": 0, "existing_paths": set()}

    api_url = opencode_cfg.get("api_url", "http://localhost:4096")
    owner = opencode_cfg.get("owner", "")
    if not owner:
        logger.info("No opencode owner configured — skipping opencode scan")
        return {"scanned": 0, "existing_paths": set()}

    if user_filter and user_filter != owner:
        logger.info(
            "Opencode owner %r doesn't match --user %r — skipping opencode scan",
            owner, user_filter,
        )
        return {"scanned": 0, "existing_paths": set()}

    scanned = 0
    existing_paths: set[str] = set()
    cursor: str | None = None
    page_limit = 100

    # Walk /api/session with cursor pagination. Cursor is opaque; walk until next is empty.
    # Cap pages to avoid an unbounded loop if the API returns a self-referential cursor.
    for _page_index in range(1000):
        params: dict[str, str] = {"limit": str(page_limit)}
        if cursor:
            params["cursor"] = cursor
        url = f"{api_url}/api/session?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.load(response)
        except Exception as exc:
            logger.warning(
                "opencode /api/session unreachable at %s: %s — skipping opencode scan, "
                "will NOT delete stale opencode Qdrant points",
                api_url, exc,
            )
            return {"scanned": scanned, "existing_paths": None}

        items = payload.get("items", []) if isinstance(payload, dict) else []
        for session in items:
            session_id = session.get("id", "")
            if not session_id:
                continue

            path = f"opencode://{session_id}"
            time_info = session.get("time", {})
            updated_ms = time_info.get("updated", 0)
            mtime = updated_ms / 1000.0

            state_db.discover_or_update(path, owner, mtime)
            existing_paths.add(path)
            scanned += 1

        cursor_info = payload.get("cursor", {}) if isinstance(payload, dict) else {}
        next_cursor = cursor_info.get("next") if isinstance(cursor_info, dict) else None
        if not next_cursor or not items:
            break
        cursor = next_cursor
    else:
        logger.warning(
            "opencode /api/session pagination exceeded 1000 pages — stopping; "
            "this likely indicates a self-referential cursor from the API"
        )

    logger.info("  → %d opencode session(s) discovered", scanned)

    return {"scanned": scanned, "existing_paths": existing_paths}
