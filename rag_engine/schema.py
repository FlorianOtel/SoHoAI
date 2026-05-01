"""
Shared payload field name constants and owner derivation.

All ingestion and search code must import field names from here — no
string literals for Qdrant payload keys anywhere else in the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Qdrant payload field name constants
# ---------------------------------------------------------------------------

FIELD_TEXT = "text"                 # child chunk content (what was embedded)
FIELD_PARENT_TEXT = "parent_text"   # parent chunk (what gets returned to LLM)
FIELD_OWNER = "owner"               # "florian" | "eva" | "annika" | "laura" | "la-familia"
FIELD_SOURCE_PATH = "source_path"   # full NFS path — provenance returned to user
FIELD_FILE_NAME = "file_name"
FIELD_FILE_TYPE = "file_type"       # pdf | docx | pptx | txt | ipynb | md | yaml
FIELD_PAGE = "page"                 # page number / slide index / notebook cell index
FIELD_CHUNK_INDEX = "chunk_index"   # child chunk index within its parent
FIELD_TAG = "tag"                   # e.g. "certifications", "cisco-backup", "family"
FIELD_SESSION_ID = "session_id"     # UUID of the Claude Code session — indexed for Qdrant filter queries
FIELD_PROJECT = "project"           # derived project name (Path(cwd).name, e.g. "SoHoAI") — indexed for filtering


# ---------------------------------------------------------------------------
# Owner derivation
# ---------------------------------------------------------------------------

def derive_owner(file_path: str, user_config: dict) -> str:
    """
    Map an NFS file path to its owner string.

    Checks per-user NFS roots first, then the shared root.
    Raises ValueError if the path is not under any configured root.

    Args:
        file_path:   Absolute NFS path (e.g. /mnt/nfs/Florian/docs/cert.pdf)
        user_config: Top-level config dict — must contain 'users' and 'shared' keys.

    Returns:
        Owner string, e.g. "florian", "eva", "la-familia".
    """
    for _email, cfg in user_config.get("users", {}).items():
        for root in cfg.get("nfs_roots", []):
            if file_path.startswith(root):
                return cfg["owner"]

    shared = user_config.get("shared", {})
    for root in shared.get("nfs_roots", []):
        if file_path.startswith(root):
            return shared["owner"]

    raise ValueError(
        f"File {file_path!r} is not under any configured NFS root. "
        f"Users: {list(user_config.get('users', {}).keys())}, "
        f"Shared: {shared.get('nfs_roots', [])}"
    )
