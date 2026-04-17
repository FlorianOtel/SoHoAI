"""
End-of-session sync: push the latest codebase snapshot to NotebookLM.

Pipeline:
  1. Generate codebase snapshot  (snapshot_codebase.py)
  2. Open NotebookLM (headless, using saved session)
  3. Delete any previous snapshot and doc source(s)
  4. Upload the fresh code snapshot
  5. Upload documentation files as separate Markdown sources

Usage:
    python utils/sync_to_notebook.py

    # Skip snapshot regeneration (upload existing codebase_snapshot.md):
    python utils/sync_to_notebook.py --no-snapshot

    # Keep old sources instead of replacing them:
    python utils/sync_to_notebook.py --no-delete
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from project root or from utils/
sys.path.insert(0, str(Path(__file__).parent))

from notebooklm_auth import NotebookLMSession
from snapshot_codebase import DEFAULT_OUTPUT, PROJECT_ROOT, _sanitize_for_markdown, generate_snapshot

log = logging.getLogger(__name__)

# Source name stems to find and delete before uploading fresh copies.
# Matches any source whose display name contains the stem (case-insensitive).
SNAPSHOT_STEMS = ["codebase_snapshot"]

# Documentation Markdown files uploaded as separate NotebookLM sources.
DOC_FILES: list[Path] = [
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "RAG-strategy.md",
]
DOC_STEMS = [p.stem for p in DOC_FILES]


async def sync(regenerate: bool = True, delete_old: bool = True) -> None:
    # Step 1 — generate snapshot
    if regenerate:
        log.info("=== Step 1: Generating codebase snapshot ===")
        snapshot_path = generate_snapshot(output=DEFAULT_OUTPUT)
    else:
        if not DEFAULT_OUTPUT.exists():
            log.error("Snapshot not found: %s", DEFAULT_OUTPUT)
            sys.exit(1)
        snapshot_path = DEFAULT_OUTPUT
        log.info("=== Step 1: Using existing snapshot: %s ===", snapshot_path)

    # Step 2 — open NotebookLM
    log.info("=== Step 2: Opening NotebookLM ===")
    async with NotebookLMSession(headless=True) as session:
        try:
            await session.goto_notebook()
        except RuntimeError as exc:
            log.error("Cannot open notebook: %s", exc)
            sys.exit(1)

        log.info("Current sources: %s", await session.list_sources() or "(none)")

        # Step 3 — delete old sources.
        # Re-query the DOM between deletions; loop per stem until no matches remain.
        # Max 5 iterations guards against stale-DOM infinite loops.
        if delete_old:
            log.info("=== Step 3: Removing previous snapshot and doc source(s) ===")
            for stem in SNAPSHOT_STEMS + DOC_STEMS:
                for _ in range(5):
                    current = await session.list_sources()
                    if not any(stem.lower() in s.lower() for s in current):
                        break
                    if not await session.delete_source_by_name(stem):
                        log.warning("Could not delete %r — remove manually in NotebookLM UI.", stem)
                        break
                else:
                    log.warning("Gave up deleting %r after 5 attempts.", stem)
        else:
            log.info("=== Step 3: Skipping delete (--no-delete) ===")

        # Step 4 — upload fresh code snapshot
        log.info("=== Step 4: Uploading codebase snapshot ===")
        await session.upload_source(snapshot_path)

        # Step 5 — upload documentation files (sanitized to avoid stuck indexing)
        log.info("=== Step 5: Uploading documentation files ===")
        import tempfile, os
        for doc in DOC_FILES:
            if not doc.exists():
                log.warning("Doc file not found, skipping: %s", doc)
                continue
            raw = doc.read_text(encoding="utf-8")
            sanitized = _sanitize_for_markdown(raw)
            if sanitized != raw:
                log.info("Sanitizing %s (box-drawing/emoji chars removed)", doc.name)
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=doc.suffix,
                    prefix=doc.stem + "_", delete=False,
                ) as tmp:
                    tmp.write(sanitized)
                    tmp_path = Path(tmp.name)
                # NotebookLM uses the filename as the source title — rename to match original
                upload_path = tmp_path.parent / doc.name
                tmp_path.rename(upload_path)
                try:
                    await session.upload_source(upload_path)
                finally:
                    upload_path.unlink(missing_ok=True)
            else:
                await session.upload_source(doc)

        log.info("Sources after sync: %s", await session.list_sources())

    log.info("=== Sync complete. NotebookLM is up to date. ===")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Sync codebase snapshot to NotebookLM")
    parser.add_argument(
        "--no-snapshot", action="store_true",
        help="Skip regenerating the snapshot — upload the existing codebase_snapshot.md",
    )
    parser.add_argument(
        "--no-delete", action="store_true",
        help="Do not delete previous snapshot/doc sources before uploading",
    )
    args = parser.parse_args()
    asyncio.run(sync(regenerate=not args.no_snapshot, delete_old=not args.no_delete))


if __name__ == "__main__":
    main()
