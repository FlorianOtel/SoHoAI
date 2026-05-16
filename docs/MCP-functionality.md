---
title: "SoHoAI — MCP server (nfs-files)"
created_at: 2026-05-05--16-38
created_by: Claude Code (Claude Haiku 4.5)
context: >
  NFS-files MCP server implementation exposing the Gin-AI filesystem to MCP clients.
  Tools for file CRUD, directory listing, and search. HTTP streamable transport on port 3001.
  Path safety enforced via hardcoded ALLOWED_ROOTS; all operations validated against root.
---

# SoHoAI — MCP server (nfs-files)

The nfs-files MCP server exposes the Gin-AI filesystem (`/mnt/nfs/Florian/Gin-AI` and its subdirectories) to any MCP client for file access, directory exploration, and searching. Implemented in the `NFS-files--MCP-server/` subdirectory.

---

## Deployment

**Subdirectory**: `./NFS-files--MCP-server/`

**Files**:
- `nfs_files_mcp_server.py` — MCP server implementation (22.9 KB)
- `nfs_files_mcp_server.sh` — Launch script (HTTP mode, port 3001)
- `setup_mcp.sh` — Install + validation script
- `claude_desktop_config.json` — Config snippet for Claude Desktop
- `server_config.json` — Server configuration

**Server name**: `nfs-files` (as seen by MCP clients)

---

## Tools exposed

| Tool | Purpose | Signature |
|------|---------|-----------|
| `list_directory` | List files/subdirs at a path | `path: str, depth: int (0-20)` |
| `read_file` | Read file contents | `path: str` |
| `write_file` | Write file contents (create or overwrite) | `path: str, content: str` |
| `edit_file` | In-place edit: search-replace | `path: str, old_string: str, new_string: str` |
| `delete_file` | Delete file or directory | `path: str` |
| `search_files` | Full-text search by regex or substring | `query: str, root: str (optional)` |
| `get_file_info` | File metadata: size, mtime, is_dir | `path: str` |

---

## Resources

| Resource | Content |
|----------|---------|
| `file:///nfs_files/config` | Server configuration (read-only) |
| `file://structure` | Directory structure listing (read-only) |

---

## Transport

**HTTP mode** (default, via `nfs_files_mcp_server.sh`):
- Port 3001
- Streamable JSON responses (chunked encoding for large files)
- Used for remote access (Claude Code, Claude Desktop)

**Stdio mode** (alternative):
- Launched as a subprocess with stdin/stdout piping
- Useful for local-only scenarios

---

## Path safety and sandboxing

**ALLOWED_ROOTS** (hardcoded in `nfs_files_mcp_server.py`):
```python
ALLOWED_ROOTS = ["/mnt/nfs/Florian"]
```

All operations are validated against this root:
- Every path argument is resolved to an absolute path
- Symlinks are followed (normalized via `os.path.realpath()`)
- Traversal checks ensure the final path is within or under an ALLOWED_ROOT
- Any attempt to escape (e.g. `../../../etc/passwd`) is rejected with an error

**No file can be accessed outside `/mnt/nfs/Florian`** — the root is hardcoded and cannot be changed at runtime.

---

## Listing depth

- **Depth 0**: unlimited recursive (entire subtree)
- **Depth 1–20**: limit recursion to N levels
- Useful for paged exploration: start with depth 1, drill into specific subdirs with depth 0 or higher N

Example:
```
GET /files/list?path=projects/SoHoAI&depth=1
→ Lists `projects/`, `SoHoAI/`, and immediate children of `SoHoAI/` only (not grandchildren)
```

---

## Searching

`search_files` walks the filesystem recursively (always):
- Query can be a substring or regex pattern
- Default root is `ALLOWED_ROOTS[0]` (`/mnt/nfs/Florian`)
- Up to 1000 result lines returned (capped for large corpora)

Example:
```
GET /search?query=rag_strategy&root=projects/SoHoAI
→ Returns all files in SoHoAI/ matching "rag_strategy" (e.g. docs/RAG-strategy.md)
```

---

## Path notation

Relative paths resolve from the Gin-AI root. Project files require the `projects/SoHoAI/` prefix.

Examples:
- `projects/SoHoAI/main.py` → `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/main.py`
- `projects/SoHoAI/SoHoAI-config.yaml` → `/mnt/nfs/Florian/Gin-AI/projects/SoHoAI/SoHoAI-config.yaml`
- `Gin-AI/docs/` → `/mnt/nfs/Florian/Gin-AI/docs/`

---

## Design decision: Path sandboxing

All path operations are validated against `ALLOWED_ROOTS`. The hardcoded root prevents any configuration mistake or runtime override from exposing system files. Symlinks are followed (users can access symlinked content within the allowed tree), but escape attempts fail fast with a clear error. This design ensures that MCP clients cannot accidentally or maliciously access `/etc`, `/root`, or other sensitive areas.

---

## Status

✅ **Completed and working** as of 2026-04-16. Exposes all files under `/mnt/nfs/Florian/Gin-AI` to MCP clients. SoHoAI project files are accessed via the `projects/SoHoAI/` path prefix.
