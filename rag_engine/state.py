"""
SQLite-backed ingestion queue for the RAG pipeline.

Tracks every NFS file through its ingestion lifecycle:
  pending → processing → completed
                       → pending  (auto-retry; retry_count < max_retries)
                       → ignored  (exhausted max_retries; skip_reason = last error)

'failed' is not used by the automatic path. It only appears if set manually via SQL.
discover_or_update() resets any 'failed' row to 'pending' as a safety net.

Uses a dedicated rag_state.db so chat_store.db stays unaffected.
All methods are synchronous, matching the ChatStore pattern in chat_store.py.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_queue (
    file_path       TEXT PRIMARY KEY,
    owner           TEXT NOT NULL,
    last_modified   REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_msg       TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 5,
    started_at      TEXT,
    completed_at    TEXT,
    progress_detail TEXT,
    skip_reason     TEXT
);
"""

# Schema migrations applied in __init__ after CREATE TABLE IF NOT EXISTS.
_MIGRATIONS = [
    # (1) Add skip_reason column introduced with the 'ignored' status.
    "ALTER TABLE ingestion_queue ADD COLUMN skip_reason TEXT",
    # (2) Raise max_retries from 3 to 5 for all existing rows that still have the old default.
    "UPDATE ingestion_queue SET max_retries = 5 WHERE max_retries < 5",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateDB:
    """Ingestion queue CRUD for the RAG pipeline."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        for migration in _MIGRATIONS:
            try:
                self._conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # ALTER TABLE fails if column already exists — safe to ignore
        self._conn.commit()

    # ------------------------------------------------------------------
    # Startup

    def crash_recovery(self) -> list[str]:
        """
        Reset any rows stuck in 'processing' back to 'pending'.

        Call once on daemon startup. Files left in 'processing' after a crash
        or OOM kill are re-queued rather than staying stuck indefinitely.

        Returns the list of file paths that were reset (for operator logging).
        """
        cur = self._conn.execute(
            "SELECT file_path FROM ingestion_queue WHERE status = 'processing'"
        )
        stuck = [row["file_path"] for row in cur.fetchall()]
        if stuck:
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'pending', started_at = NULL, progress_detail = NULL "
                "WHERE status = 'processing'"
            )
            self._conn.commit()
            logger.warning(
                "Crash recovery: reset %d stuck file(s) to pending: %s",
                len(stuck),
                stuck,
            )
        return stuck

    # ------------------------------------------------------------------
    # Discovery

    def discover_or_update(
        self, file_path: str, owner: str, last_modified: float
    ) -> None:
        """
        Register a newly discovered file or re-queue it for ingestion.

        - New file:    inserted as 'pending'.
        - Modified:    disk mtime > stored mtime → reset to 'pending' (any status,
                       including 'ignored' — the file was replaced on disk).
        - Failed:      status = 'failed' (mtime unchanged) → reset to 'pending'
                       with retry_count = 0 so the daemon gives it a fresh set of
                       retries. Failed files have no Qdrant points (step 0 of the
                       worker loop deleted them before the failure), so no Qdrant
                       cleanup is needed.
        - Ignored:     status = 'ignored' (mtime unchanged) → no-op. The file is
                       permanently skipped until it changes on disk.
        - Everything else (pending / processing / completed, mtime unchanged): no-op.
        """
        cur = self._conn.execute(
            "SELECT last_modified, status FROM ingestion_queue WHERE file_path = ?",
            (file_path,),
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO ingestion_queue (file_path, owner, last_modified, status) "
                "VALUES (?, ?, ?, 'pending')",
                (file_path, owner, last_modified),
            )
        elif last_modified > row["last_modified"]:
            # File changed on disk — re-queue regardless of current status (including ignored).
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'pending', last_modified = ?, "
                "    error_msg = NULL, retry_count = 0, skip_reason = NULL, "
                "    started_at = NULL, completed_at = NULL, progress_detail = NULL "
                "WHERE file_path = ?",
                (last_modified, file_path),
            )
        elif row["status"] == "failed":
            # Permanently failed but file unchanged — give it a fresh set of retries.
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'pending', last_modified = ?, "
                "    error_msg = NULL, retry_count = 0, "
                "    started_at = NULL, completed_at = NULL, progress_detail = NULL "
                "WHERE file_path = ?",
                (last_modified, file_path),
            )
        # ignored (mtime unchanged), pending, processing, completed → no-op
        self._conn.commit()

    def handle_deleted(self, existing_paths: set[str]) -> list[str]:
        """
        Remove rows for files that are no longer present in the scan.

        Deletes ALL rows (any status) whose file_path is absent from
        existing_paths. Returns only the paths that were 'completed' so
        the caller can delete the corresponding Qdrant points (only
        completed files have been ingested into Qdrant).

        Args:
            existing_paths: Set of all file paths currently present on NFS.
        """
        cur = self._conn.execute(
            "SELECT file_path, status FROM ingestion_queue"
        )
        all_rows = [(row["file_path"], row["status"]) for row in cur.fetchall()]
        gone = [(p, s) for p, s in all_rows if p not in existing_paths]
        if not gone:
            return []
        gone_paths = [p for p, _ in gone]
        self._conn.executemany(
            "DELETE FROM ingestion_queue WHERE file_path = ?",
            [(p,) for p in gone_paths],
        )
        self._conn.commit()
        logger.info("Removed %d deleted file(s) from queue", len(gone_paths))
        return [p for p, s in gone if s == "completed"]

    # ------------------------------------------------------------------
    # Worker state transitions

    def fetch_pending(self, limit: int = 10, owner: str | None = None) -> list[str]:
        """Return up to `limit` pending file paths, oldest first. Optionally filter by owner."""
        if owner:
            cur = self._conn.execute(
                "SELECT file_path FROM ingestion_queue "
                "WHERE status = 'pending' AND owner = ? ORDER BY rowid LIMIT ?",
                (owner, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT file_path FROM ingestion_queue "
                "WHERE status = 'pending' ORDER BY rowid LIMIT ?",
                (limit,),
            )
        return [row["file_path"] for row in cur.fetchall()]

    def fetch_pending_full(self, limit: int = 10) -> list[dict]:
        """Return up to `limit` pending rows with file_path and owner, oldest first."""
        cur = self._conn.execute(
            "SELECT file_path, owner FROM ingestion_queue "
            "WHERE status = 'pending' ORDER BY rowid LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def mark_processing(self, file_path: str) -> None:
        """Transition pending → processing; record start timestamp."""
        self._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = 'processing', started_at = ?, progress_detail = 'starting' "
            "WHERE file_path = ?",
            (_now(), file_path),
        )
        self._conn.commit()

    def set_progress(self, file_path: str, detail: str) -> None:
        """Update progress_detail while a file is being processed (mid-flight)."""
        self._conn.execute(
            "UPDATE ingestion_queue SET progress_detail = ? WHERE file_path = ?",
            (detail, file_path),
        )
        self._conn.commit()

    def mark_completed(self, file_path: str) -> None:
        """Transition processing → completed; record completion timestamp."""
        self._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = 'completed', completed_at = ?, "
            "    progress_detail = NULL, error_msg = NULL "
            "WHERE file_path = ?",
            (_now(), file_path),
        )
        self._conn.commit()

    def mark_failed(self, file_path: str, error: str) -> None:
        """
        Record a failure and decide whether to auto-retry.

        - retry_count < max_retries  → status = 'pending'   (will be retried)
        - retry_count >= max_retries → status = 'ignored'   (skip_reason = last error)

        'ignored' files are never re-queued by discover_or_update() unless the file
        changes on disk. Use rag_status.py --ignored to review them.
        """
        cur = self._conn.execute(
            "SELECT retry_count, max_retries FROM ingestion_queue WHERE file_path = ?",
            (file_path,),
        )
        row = cur.fetchone()
        if row is None:
            logger.error("mark_failed called for unknown file: %s", file_path)
            return

        new_count = row["retry_count"] + 1
        if new_count < row["max_retries"]:
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'pending', retry_count = ?, error_msg = ?, progress_detail = NULL "
                "WHERE file_path = ?",
                (new_count, error[:2000], file_path),
            )
            logger.warning(
                "Retry %d/%d queued: %s — %s",
                new_count, row["max_retries"], file_path, error[:120],
            )
        else:
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'ignored', retry_count = ?, skip_reason = ?, "
                "    error_msg = NULL, progress_detail = NULL "
                "WHERE file_path = ?",
                (new_count, error[:2000], file_path),
            )
            logger.error(
                "Ignored after %d retries: %s — %s",
                new_count, file_path, error[:200],
            )
        self._conn.commit()

    def mark_ignored(self, file_path: str, reason: str) -> None:
        """
        Permanently skip a file — never re-queued by discover_or_update() unless
        the file is replaced on disk (mtime change).

        Use for files that fundamentally cannot be parsed regardless of retries:
        IRM/DRM-encrypted, corrupted, unsupported proprietary format.

        The file must have no Qdrant points at the time of this call (i.e. it
        should currently be 'failed' or 'pending', not 'completed'). mark_ignored()
        does not touch Qdrant — the caller is responsible for any cleanup.
        """
        self._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = 'ignored', skip_reason = ?, "
            "    error_msg = NULL, progress_detail = NULL "
            "WHERE file_path = ?",
            (reason[:2000], file_path),
        )
        self._conn.commit()
        logger.info("Marked as ignored: %s — %s", file_path, reason)

    # ------------------------------------------------------------------
    # Reporting

    def get_counts(self) -> dict[str, int]:
        """Return row counts grouped by status."""
        cur = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ingestion_queue GROUP BY status"
        )
        counts = {row["status"]: row["n"] for row in cur.fetchall()}
        return {
            "pending":    counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "completed":  counts.get("completed", 0),
            "failed":     counts.get("failed", 0),
            "ignored":    counts.get("ignored", 0),
            "total":      sum(counts.values()),
        }

    def get_failed(self, owner: str | None = None) -> list[dict]:
        """Return all permanently failed files with their error messages."""
        if owner:
            cur = self._conn.execute(
                "SELECT file_path, owner, retry_count, error_msg "
                "FROM ingestion_queue WHERE status = 'failed' AND owner = ? "
                "ORDER BY file_path",
                (owner,),
            )
        else:
            cur = self._conn.execute(
                "SELECT file_path, owner, retry_count, error_msg "
                "FROM ingestion_queue WHERE status = 'failed' ORDER BY file_path"
            )
        return [dict(row) for row in cur.fetchall()]

    def get_ignored(self, owner: str | None = None) -> list[dict]:
        """Return all ignored files with their retry count and skip reason (last error)."""
        if owner:
            cur = self._conn.execute(
                "SELECT file_path, owner, retry_count, skip_reason "
                "FROM ingestion_queue WHERE status = 'ignored' AND owner = ? "
                "ORDER BY file_path",
                (owner,),
            )
        else:
            cur = self._conn.execute(
                "SELECT file_path, owner, retry_count, skip_reason "
                "FROM ingestion_queue WHERE status = 'ignored' ORDER BY file_path"
            )
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
