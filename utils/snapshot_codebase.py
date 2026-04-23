"""
Codebase snapshot generator for NotebookLM.

Aggregates the project's key source files into a single Markdown document
suitable for upload as a NotebookLM source.

Usage:
    python utils/snapshot_codebase.py
    python utils/snapshot_codebase.py --output /tmp/snapshot.md

Note on encoding: NotebookLM's Markdown parser chokes on Unicode box-drawing
characters (U+2500-U+257F) and non-BMP emoji (U+10000+) even inside code
fences — they cause a perpetual stuck-indexing spinner with no visible error.
_sanitize_for_markdown() strips these before embedding file content.
File size is NOT a constraint — a 122 KB combined snapshot indexes fine.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path("/mnt/nfs/Florian/Gin-AI/projects/HomeAI")
DEFAULT_OUTPUT = PROJECT_ROOT / "utils" / "codebase_snapshot.md"

SNAPSHOT_FILES: list[str] = [
    "pyproject.toml",
    "config.yaml",
    "schemas.py",
    "main.py",
    "router.py",
    "conversation.py",
    "kv_cache.py",
    "chat_store.py",
    "mcp_gateway.py",
    # RAG engine (Phase 2)
    "rag_engine/__init__.py",
    "rag_engine/schema.py",
    "rag_engine/collection.py",
    "rag_engine/embeddings.py",
    "rag_engine/state.py",
    "rag_engine/scanner.py",
    "rag_engine/ingest.py",
    "rag_engine/search.py",
    # CLI utilities
    "utils/cli_chat.py",
    "utils/rag_sync_nfs.py",
    "utils/rag_ingest_daemon.py",
    "utils/rag_status.py",
    "utils/rag_search_cli.py",
    "utils/rag_reset.py",
    "utils/notebooklm_auth.py",
    "utils/sync_to_notebook.py",
    "utils/snapshot_codebase.py",
    "NFS-files--MCP-server/nfs_files_mcp_server.py",
]


def _lang(path: Path) -> str:
    return {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".md": ""}.get(
        path.suffix.lower(), ""
    )


def _sanitize_for_markdown(content: str) -> str:
    """
    Replace or strip characters that cause NotebookLM's indexing to hang.

    Empirically confirmed problematic ranges (stuck-spinner, no visible error):
      U+2500–U+257F  Box Drawing — ─, │, ┌, └, etc.  (replaced with dash)
      U+2580–U+259F  Block Elements — █, ▓, ░, etc.   (replaced with dash)
      U+2190–U+21FF  Arrows — →, ←, ↑, ↓, etc.        (stripped)
      U+2600–U+26FF  Misc Symbols — ☁, ☺, ♠, ⚠, etc.  (stripped)
      U+2700–U+27BF  Dingbats — ✅, ✓, ✗, ➜, etc.      (stripped)
      U+10000+       Non-BMP emoji                      (stripped)

    Box-drawing and block-elements are replaced with dash to preserve table
    and border structure as ASCII.  All other symbol ranges are removed outright
    because they have no prose equivalent.

    NOTE: add new ranges here whenever NotebookLM is found to choke on a new
    character class — do not rely on "it looked fine last time" since the
    backend parser changes without notice.
    """
    content = re.sub(r"[\u2500-\u259F]", "-", content)  # box-drawing + block elements → dash
    content = re.sub(r"[\u2190-\u21FF]", "", content)    # arrows → removed
    content = re.sub(r"[\u2600-\u27BF]", "", content)    # misc symbols + dingbats → removed
    content = re.sub(r"[^\u0000-\uFFFF]", "", content)   # non-BMP emoji → removed
    return content


def generate_snapshot(
    output: Path = DEFAULT_OUTPUT,
    files: list[str] | None = None,
) -> Path:
    """Write the snapshot Markdown to `output` and return the path."""
    targets = [PROJECT_ROOT / f for f in (files or SNAPSHOT_FILES)]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = [
        f"# HomeAI — Codebase Snapshot\n\nGenerated: {now}\n",
        "---\n",
    ]

    included = 0
    for path in targets:
        if not path.exists():
            log.warning("Skipping missing file: %s", path)
            continue
        rel = path.relative_to(PROJECT_ROOT)
        lang = _lang(path)
        content = _sanitize_for_markdown(
            path.read_text(encoding="utf-8", errors="replace")
        )

        # Use a fence longer than the longest backtick run inside the file so
        # the outer fence always closes correctly (standard Markdown rule).
        max_inner = max(
            (len(m.group(0)) for m in re.finditer(r"`+", content)),
            default=0,
        )
        fence = "`" * max(3, max_inner + 1)

        sections.append(f"## `{rel}`\n\n{fence}{lang}\n{content}\n{fence}\n")
        included += 1
        log.info("  + %s", rel)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections), encoding="utf-8")

    size_kb = output.stat().st_size / 1024
    log.info("Snapshot written: %s  (%d files, %.1f KB)", output, included, size_kb)
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate codebase snapshot for NotebookLM")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    generate_snapshot(output=Path(args.output))


if __name__ == "__main__":
    main()
