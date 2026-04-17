"""
SQLite-backed ingestion queue for the RAG pipeline.

Tracks every NFS file through its ingestion lifecycle:
  pending → processing → completed
                       → failed  (auto-retried up to max_retries, then permanent)

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
    max_retries     INTEGER NOT NULL DEFAULT 3,
    started_at      TEXT,
    completed_at    TEXT,
    progress_detail TEXT
);
"""


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
        Register a newly discovered file or re-queue a modified one.

        - New file:      inserted as 'pending'.
        - Modified file: disk mtime > stored mtime → reset to 'pending' so the
                         worker re-ingests it with fresh content.
        - Unchanged file (completed / pending / failed): no-op.
        """
        cur = self._conn.execute(
            "SELECT last_modified FROM ingestion_queue WHERE file_path = ?",
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
            self._conn.execute(
                "UPDATE ingestion_queue "
                "SET status = 'pending', last_modified = ?, "
                "    error_msg = NULL, retry_count = 0, "
                "    started_at = NULL, completed_at = NULL, progress_detail = NULL "
                "WHERE file_path = ?",
                (last_modified, file_path),
            )
        self._conn.commit()

    def handle_deleted(self, existing_paths: set[str]) -> list[str]:
        """
        Identify completed rows whose files no longer exist on disk.

        Removes those rows from SQLite and returns their file paths so the
        caller can delete the corresponding Qdrant points.

        Args:
            existing_paths: Set of all file paths currently present on NFS.
        """
        cur = self._conn.execute(
            "SELECT file_path FROM ingestion_queue WHERE status = 'completed'"
        )
        completed = [row["file_path"] for row in cur.fetchall()]
        deleted = [p for p in completed if p not in existing_paths]
        if deleted:
            self._conn.executemany(
                "DELETE FROM ingestion_queue WHERE file_path = ?",
                [(p,) for p in deleted],
            )
            self._conn.commit()
            logger.info("Removed %d deleted file(s) from queue", len(deleted))
        return deleted

    # ------------------------------------------------------------------
    # Worker state transitions

    def fetch_pending(self, limit: int = 10) -> list[str]:
        """Return up to `limit` pending file paths, oldest first."""
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

        - retry_count < max_retries  → status = 'pending'  (will be retried)
        - retry_count >= max_retries → status = 'failed'   (permanent until manual reset)
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
        new_status = "pending" if new_count < row["max_retries"] else "failed"
        self._conn.execute(
            "UPDATE ingestion_queue "
            "SET status = ?, retry_count = ?, error_msg = ?, progress_detail = NULL "
            "WHERE file_path = ?",
            (new_status, new_count, error[:2000], file_path),
        )
        self._conn.commit()
        if new_status == "failed":
            logger.error(
                "Permanently failed after %d retries: %s — %s",
                new_count,
                file_path,
                error[:200],
            )

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
            "total":      sum(counts.values()),
        }

    def get_failed(self) -> list[dict]:
        """Return all permanently failed files with their error messages."""
        cur = self._conn.execute(
            "SELECT file_path, owner, retry_count, error_msg "
            "FROM ingestion_queue WHERE status = 'failed' ORDER BY file_path"
        )
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
