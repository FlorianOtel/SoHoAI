"""
NFS-files — Project Filesystem MCP Server

Exposes the Gin-AI project directory (and optionally NAS paths) to any
MCP-compatible client: Claude Desktop, Claude Code, this project's own
orchestrator, or any other tool.

Capabilities:
  - Browse directory trees
  - Read files (source code, config, docs, logs)
  - Write / update files
  - Search across files (grep-style)
  - Read document metadata (PDF, PPTX, DOCX page/slide counts)

Safety:
  - All paths are resolved and validated against an allowed root
  - Directory traversal is blocked
  - Binary files are detected and handled gracefully
  - Destructive operations are clearly annotated

Transport:
  - Default: stdio (for Claude Desktop / Claude Code)
  - Optional: streamable HTTP (for remote access from Server 2 or web UI)
7
Usage:
  # stdio (Claude Desktop / Claude Code)
  python nfs_files_mcp_server.py

  # streamable HTTP (remote access)
  python nfs_files_mcp_server.py --http --port 3001 --host 0.0.0.0
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, field_validator, ConfigDict

# =============================================================================
#  Configuration
# =============================================================================

# Root directories this server can access.
ALLOWED_ROOTS = [
    # Path(os.environ.get("/home/florian", Path.home() / "Gin-AI")).resolve(),
    # Uncomment to also expose NAS:
    Path("/mnt/nfs/Florian").resolve(),
]

# Files/dirs to exclude from listings and search
IGNORE_PATTERNS = [
    "LLMs-cache/*","__pycache__", "*.pyc", ".git", ".env", "node_modules",
    "*.egg-info", ".mypy_cache", ".pytest_cache", "*.sqlite3",
]

# Max file size for read operations (5 MB)
MAX_READ_BYTES = 5 * 1024 * 1024

# Max lines to return from search
MAX_SEARCH_RESULTS = 200


# =============================================================================
#  Server Setup
# =============================================================================

mcp = FastMCP(
    "nfs_files",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


# =============================================================================
#  Path Safety
# =============================================================================

def resolve_safe_path(raw_path: str) -> Path:
    """
    Resolve a path and verify it falls within an allowed root.
    Raises ValueError on traversal attempts or disallowed paths.
    """
    # Expand ~ and resolve to absolute
    candidate = Path(raw_path).expanduser().resolve()

    for root in ALLOWED_ROOTS:
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue

    # Also allow relative paths (resolved against first root)
    relative_try = (ALLOWED_ROOTS[0] / raw_path).resolve()
    for root in ALLOWED_ROOTS:
        try:
            relative_try.relative_to(root)
            return relative_try
        except ValueError:
            continue

    allowed = ", ".join(str(r) for r in ALLOWED_ROOTS)
    raise ValueError(
        f"Path '{raw_path}' resolves to '{candidate}' which is outside "
        f"allowed directories: {allowed}. Check your path or update ALLOWED_ROOTS."
    )


def is_ignored(path: Path) -> bool:
    """Check if a path matches any ignore pattern."""
    name = path.name
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORE_PATTERNS)


def is_binary(path: Path) -> bool:
    """Quick heuristic: read first 8KB and look for null bytes."""
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except Exception:
        return True


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
#  Input Models
# =============================================================================

class PathInput(BaseModel):
    """Input requiring a single file or directory path."""
    model_config = ConfigDict(str_strip_whitespace=True)
    path: str = Field(
        ...,
        description=(
            "File or directory path. Can be absolute or relative to project root. "
            "Examples: 'main.py', 'orchestrator/router.py', '~/Gin-AI/config.yaml'"
        ),
        min_length=1,
        max_length=500,
    )


class ListDirInput(BaseModel):
    """Input for directory listing."""
    model_config = ConfigDict(str_strip_whitespace=True)
    path: str = Field(
        default=".",
        description="Directory path (default: project root)",
        max_length=500,
    )
    depth: int = Field(
        default=3,
        description="How many levels deep to list. 0 = unlimited (full recursive). Max 20.",
        ge=0, le=20,
    )
    show_hidden: bool = Field(
        default=False,
        description="Include hidden files (dotfiles)",
    )


class WriteFileInput(BaseModel):
    """Input for writing/creating a file."""
    model_config = ConfigDict(str_strip_whitespace=True)
    path: str = Field(
        ...,
        description="Target file path (created if it doesn't exist, parent dirs auto-created)",
        min_length=1,
        max_length=500,
    )
    content: str = Field(
        ...,
        description="Content to write to the file",
    )
    create_dirs: bool = Field(
        default=True,
        description="Auto-create parent directories if they don't exist",
    )


class SearchInput(BaseModel):
    """Input for searching across files."""
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(
        ...,
        description="Search term (substring match) or regex pattern",
        min_length=1,
        max_length=200,
    )
    path: str = Field(
        default=".",
        description="Directory to search within (default: project root)",
    )
    file_pattern: str = Field(
        default="*",
        description="Glob pattern to filter files, e.g. '*.py', '*.yaml'",
    )
    regex: bool = Field(
        default=False,
        description="Treat query as a regex pattern",
    )
    max_results: int = Field(
        default=MAX_SEARCH_RESULTS,
        description="Maximum number of matching lines to return",
        ge=1, le=1000,
    )


class EditFileInput(BaseModel):
    """Input for find-and-replace editing."""
    model_config = ConfigDict(str_strip_whitespace=True)
    path: str = Field(
        ...,
        description="Path to the file to edit",
        min_length=1,
        max_length=500,
    )
    old_text: str = Field(
        ...,
        description="Exact text to find (must appear exactly once in the file)",
        min_length=1,
    )
    new_text: str = Field(
        default="",
        description="Replacement text (empty string to delete the match)",
    )


# =============================================================================
#  Tools — Read Operations
# =============================================================================

@mcp.tool(
    name="list_directory",
    annotations={
        "title": "List Directory Contents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def list_directory(params: ListDirInput) -> str:
    """List files and directories in the Gin AI project.

    Returns a tree-style listing with file sizes and types.
    Use this to understand project structure before reading specific files.

    Args:
        params (ListDirInput): Directory path, depth, and display options.

    Returns:
        str: Tree-formatted directory listing with metadata.
    """
    try:
        root = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not root.is_dir():
        return f"Error: '{params.path}' is not a directory."

    lines = [f"📁 {root.name}/  ({root})\n"]

    def walk(dir_path: Path, prefix: str, current_depth: int):
        if params.depth > 0 and current_depth > params.depth:
            return

        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}⚠️  [permission denied]")
            return

        entries = [e for e in entries if not is_ignored(e)]
        if not params.show_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}📁 {entry.name}/")
                walk(entry, next_prefix, current_depth + 1)
            else:
                size = format_size(entry.stat().st_size)
                lines.append(f"{prefix}{connector}📄 {entry.name}  ({size})")

    walk(root, "", 1)
    return "\n".join(lines)


@mcp.tool(
    name="read_file",
    annotations={
        "title": "Read File Contents",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def read_file(params: PathInput) -> str:
    """Read the contents of a file in the Gin AI project.

    Returns the full text content with line numbers for source code,
    or metadata summary for binary files.

    Args:
        params (PathInput): Path to the file to read.

    Returns:
        str: File contents (with line numbers for code) or metadata for binary files.
    """
    try:
        filepath = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not filepath.exists():
        return f"Error: File not found: '{params.path}'"

    if filepath.is_dir():
        return f"Error: '{params.path}' is a directory. Use list_directory instead."

    stat = filepath.stat()

    if stat.st_size > MAX_READ_BYTES:
        return (
            f"Error: File is {format_size(stat.st_size)} which exceeds the "
            f"{format_size(MAX_READ_BYTES)} limit. Consider reading specific "
            f"line ranges or using search_files to find relevant sections."
        )

    if is_binary(filepath):
        return (
            f"Binary file: {filepath.name}\n"
            f"Size: {format_size(stat.st_size)}\n"
            f"Modified: {datetime.fromtimestamp(stat.st_mtime).isoformat()}\n"
            f"Cannot display binary content — use appropriate tools for this file type."
        )

    try:
        content = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = filepath.read_text(encoding="latin-1")
        except Exception:
            return "Error: Unable to read file — unsupported encoding."

    # Add line numbers for code files
    code_extensions = {
        "*.md", ".py", ".js", ".ts", ".jsx", ".tsx", ".yaml", ".yml",
        ".json", ".toml", ".html", ".css", ".sh", ".bash",
        ".sql", ".dockerfile", ".env", ".cfg", ".ini", ".conf",
    }
    if filepath.suffix.lower() in code_extensions:
        numbered_lines = []
        for i, line in enumerate(content.splitlines(), 1):
            numbered_lines.append(f"{i:4d} │ {line}")
        content = "\n".join(numbered_lines)

    header = (
        f"File: {filepath.name}\n"
        f"Path: {filepath}\n"
        f"Size: {format_size(stat.st_size)} | "
        f"Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')}\n"
        f"{'─' * 60}\n"
    )
    return header + content


@mcp.tool(
    name="search_files",
    annotations={
        "title": "Search Across Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def search_files(params: SearchInput) -> str:
    """Search for text across files in the Gin AI project.

    Performs line-by-line search (substring or regex) across matching files.
    Returns matching lines with file paths and line numbers.

    Args:
        params (SearchInput): Search query, directory scope, file filter, and options.

    Returns:
        str: Matching lines with context, grouped by file.
    """
    try:
        search_root = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not search_root.is_dir():
        return f"Error: '{params.path}' is not a directory."

    if params.regex:
        try:
            pattern = re.compile(params.query, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"
        match_fn = lambda line: pattern.search(line) is not None
    else:
        query_lower = params.query.lower()
        match_fn = lambda line: query_lower in line.lower()

    results = []
    files_searched = 0

    for filepath in search_root.rglob(params.file_pattern):
        if not filepath.is_file() or is_ignored(filepath) or is_binary(filepath):
            continue
        if filepath.stat().st_size > MAX_READ_BYTES:
            continue

        files_searched += 1
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        file_matches = []
        for i, line in enumerate(lines, 1):
            if match_fn(line):
                file_matches.append(f"  L{i}: {line.rstrip()}")
                if len(results) + len(file_matches) >= params.max_results:
                    break

        if file_matches:
            rel = filepath.relative_to(search_root) if search_root in filepath.parents or search_root == filepath.parent else filepath
            results.append(f"📄 {rel}")
            results.extend(file_matches)
            results.append("")

        if len(results) >= params.max_results:
            break

    if not results:
        return f"No matches found for '{params.query}' in {files_searched} files."

    header = f"Search: '{params.query}' in {params.file_pattern} ({files_searched} files searched)\n{'─' * 60}\n"
    return header + "\n".join(results)


@mcp.tool(
    name="get_file_info",
    annotations={
        "title": "Get File Metadata",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def get_file_info(params: PathInput) -> str:
    """Get detailed metadata about a file or directory.

    Returns size, modification time, permissions, line count (for text),
    and git status if available.

    Args:
        params (PathInput): Path to inspect.

    Returns:
        str: JSON-formatted metadata about the file or directory.
    """
    try:
        filepath = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not filepath.exists():
        return f"Error: Path not found: '{params.path}'"

    stat = filepath.stat()
    info = {
        "name": filepath.name,
        "path": str(filepath),
        "type": "directory" if filepath.is_dir() else "file",
        "size": format_size(stat.st_size),
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "permissions": oct(stat.st_mode)[-3:],
    }

    if filepath.is_file() and not is_binary(filepath) and stat.st_size < MAX_READ_BYTES:
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
            info["line_count"] = content.count("\n") + 1
            info["encoding"] = "utf-8"
        except Exception:
            pass

    if filepath.is_dir():
        try:
            children = list(filepath.iterdir())
            info["children_count"] = len(children)
            info["files"] = sum(1 for c in children if c.is_file())
            info["subdirs"] = sum(1 for c in children if c.is_dir())
        except PermissionError:
            info["children_count"] = "permission denied"

    # Git status if available
    try:
        git_result = subprocess.run(
            ["git", "status", "--porcelain", str(filepath)],
            cwd=str(ALLOWED_ROOTS[0]),
            capture_output=True, text=True, timeout=5,
        )
        if git_result.returncode == 0:
            status = git_result.stdout.strip()
            info["git_status"] = status if status else "clean"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return json.dumps(info, indent=2)


# =============================================================================
#  Tools — Write Operations
# =============================================================================

@mcp.tool(
    name="write_file",
    annotations={
        "title": "Write or Create File",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def write_file(params: WriteFileInput) -> str:
    """Write content to a file, creating it if it doesn't exist.

    WARNING: This will overwrite existing file contents entirely.
    For partial edits, use edit_file instead.

    Args:
        params (WriteFileInput): Target path and content to write.

    Returns:
        str: Confirmation with file path and size.
    """
    try:
        filepath = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    existed = filepath.exists()

    if params.create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        filepath.write_text(params.content, encoding="utf-8")
    except PermissionError:
        return f"Error: Permission denied writing to '{filepath}'."
    except OSError as e:
        return f"Error: Failed to write file: {e}"

    size = format_size(filepath.stat().st_size)
    action = "Updated" if existed else "Created"
    line_count = params.content.count("\n") + 1

    return f"{action}: {filepath}\nSize: {size} ({line_count} lines)"


@mcp.tool(
    name="edit_file",
    annotations={
        "title": "Edit File (Find and Replace)",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def edit_file(params: EditFileInput) -> str:
    """Edit a file by finding and replacing a specific text passage.

    The old_text must appear exactly once in the file to avoid ambiguous edits.
    For full file rewrites, use write_file instead.

    Args:
        params (EditFileInput): File path, text to find, and replacement text.

    Returns:
        str: Confirmation showing what was changed.
    """
    try:
        filepath = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not filepath.exists():
        return f"Error: File not found: '{params.path}'"

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error: Cannot read file: {e}"

    count = content.count(params.old_text)
    if count == 0:
        # Show a snippet of the file to help debugging
        preview = content[:500] + ("..." if len(content) > 500 else "")
        return (
            f"Error: old_text not found in '{filepath.name}'.\n"
            f"File preview (first 500 chars):\n{preview}"
        )
    if count > 1:
        return (
            f"Error: old_text appears {count} times in '{filepath.name}'. "
            f"It must appear exactly once. Make the old_text more specific."
        )

    new_content = content.replace(params.old_text, params.new_text, 1)
    filepath.write_text(new_content, encoding="utf-8")

    old_lines = params.old_text.count("\n") + 1
    new_lines = params.new_text.count("\n") + 1

    return (
        f"Edited: {filepath}\n"
        f"Replaced {old_lines} line(s) with {new_lines} line(s)."
    )


@mcp.tool(
    name="delete_file",
    annotations={
        "title": "Delete File",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def delete_file(params: PathInput) -> str:
    """Delete a file from the project directory.

    Only deletes single files, not directories.
    This action cannot be undone (unless the file is tracked in git).

    Args:
        params (PathInput): Path to the file to delete.

    Returns:
        str: Confirmation of deletion.
    """
    try:
        filepath = resolve_safe_path(params.path)
    except ValueError as e:
        return f"Error: {e}"

    if not filepath.exists():
        return f"Error: File not found: '{params.path}'"

    if filepath.is_dir():
        return (
            "Error: Cannot delete directories through this tool. "
            "Delete files individually or use the shell directly."
        )

    name = filepath.name
    size = format_size(filepath.stat().st_size)
    filepath.unlink()

    return f"Deleted: {name} ({size})"


# =============================================================================
#  Resources — Expose key files for quick access
# =============================================================================

@mcp.resource("file:///nfs_files/config")
async def get_config() -> str:
    """Expose the project configuration file."""
    config_path = ALLOWED_ROOTS[0] / "config.yaml"
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")
    return "Config file not found."


@mcp.resource("file:///nfs_files/structure")
async def get_structure() -> str:
    """Expose the project directory structure."""
    result = await list_directory(ListDirInput(path=".", depth=7))
    return result


# =============================================================================
#  Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NFS-files MCP Server")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport (mounts at /mcp)")
    parser.add_argument("--port", type=int, default=3001, help="HTTP port (default: 3001)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"Gin-AI files MCP Server — Allowed roots: {[str(r) for r in ALLOWED_ROOTS]}", file=sys.stderr)

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        print(f"Starting streamable HTTP on {args.host}:{args.port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")
    else:
        print("Starting stdio transport", file=sys.stderr)
        mcp.run()
