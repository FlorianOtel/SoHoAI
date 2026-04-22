"""
Persistent chat storage — SQLite on NAS.

This is the durable layer that survives Redis TTL expiry.
Every completed conversation turn is written here for:
  - Chat history browsing & search
  - Markdown/JSONL export
  - Future RL training data extraction

SQLite is the right choice here: single-writer, ~10K chats is trivial,
and it lives on the NAS as a single portable file.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from schemas import ChatExport, ChatSummary, Message, Role


class ChatStore:
    """SQLite-backed persistent chat history."""

    def __init__(self, db_path: str = "/mnt/nfs/__Backups/HomeAI-lab--databases/sqlite/chats.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id     TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT 'New Chat',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    model_used  TEXT DEFAULT '',
                    metadata    TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id     TEXT NOT NULL REFERENCES chats(chat_id),
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    model_used  TEXT DEFAULT '',
                    token_count INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat
                    ON messages(chat_id, id);

                -- For RL export: track user feedback signals
                CREATE TABLE IF NOT EXISTS feedback (
                    message_id  INTEGER REFERENCES messages(id),
                    signal      TEXT NOT NULL,  -- 'thumbs_up', 'thumbs_down', 'edited', 'regenerated'
                    detail      TEXT DEFAULT '',
                    created_at  TEXT NOT NULL
                );
            """)
            
            # Idempotent migration: add summary columns if they don't exist.
            # SQLite has no ADD COLUMN IF NOT EXISTS, so we tolerate the duplicate column error.
            for ddl in [
                "ALTER TABLE chats ADD COLUMN summary_text TEXT",
                "ALTER TABLE chats ADD COLUMN summary_covers_through_message_id INTEGER",
            ]:
                try:
                    conn.execute(ddl)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise
            conn.commit()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- Write -----------------------------------------------------------------

    def ensure_chat(self, chat_id: str, title: Optional[str] = None):
        """Create chat record if it doesn't exist."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO chats (chat_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (chat_id, title or "New Chat", now, now),
            )

    def save_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        model_used: str = "",
        token_count: int = 0,
    ):
        """Persist a single message turn."""
        now = datetime.utcnow().isoformat()
        self.ensure_chat(chat_id)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO messages (chat_id, role, content, timestamp, model_used, token_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (chat_id, role, content, now, model_used, token_count),
            )
            conn.execute(
                "UPDATE chats SET updated_at = ?, model_used = ? WHERE chat_id = ?",
                (now, model_used, chat_id),
            )

    def save_feedback(self, chat_id: str, message_index: int, signal: str, detail: str = ""):
        """Record user feedback on a response (for RL data later)."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            # Find the message by position in chat
            row = conn.execute(
                """SELECT id FROM messages WHERE chat_id = ?
                   ORDER BY id LIMIT 1 OFFSET ?""",
                (chat_id, message_index),
            ).fetchone()
            if row:
                conn.execute(
                    "INSERT INTO feedback (message_id, signal, detail, created_at) VALUES (?, ?, ?, ?)",
                    (row["id"], signal, detail, now),
                )

    def auto_title(self, chat_id: str):
        """Set title from first user message (truncated)."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id LIMIT 1",
                (chat_id,),
            ).fetchone()
            if row:
                title = row["content"][:80].replace("\n", " ").strip()
                if len(row["content"]) > 80:
                    title += "..."
                conn.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (title, chat_id))

    # -- Read ------------------------------------------------------------------

    def list_chats(self, limit: int = 50, offset: int = 0) -> list[ChatSummary]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, COUNT(m.id) as turn_count
                   FROM chats c LEFT JOIN messages m ON c.chat_id = m.chat_id
                   GROUP BY c.chat_id
                   ORDER BY c.updated_at DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            return [
                ChatSummary(
                    chat_id=r["chat_id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    turn_count=r["turn_count"],
                    model_used=r["model_used"] or "",
                )
                for r in rows
            ]

    def get_chat(self, chat_id: str) -> Optional[ChatExport]:
        with self._conn() as conn:
            chat = conn.execute(
                "SELECT * FROM chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if not chat:
                return None
            msgs = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
            return ChatExport(
                chat_id=chat_id,
                title=chat["title"],
                messages=[
                    Message(
                        role=Role(m["role"]),
                        content=m["content"],
                        timestamp=m["timestamp"],
                    )
                    for m in msgs
                ],
                metadata=json.loads(chat["metadata"]),
            )

    def delete_chat(self, chat_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))

    def update_summary(self, chat_id: str, summary_text: str, covers_through_message_id: int) -> None:
        """Persist a summary and the message id boundary it covers."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE chats SET summary_text = ?, summary_covers_through_message_id = ?
                   WHERE chat_id = ?""",
                (summary_text, covers_through_message_id, chat_id),
            )

    def get_summary(self, chat_id: str) -> tuple[Optional[str], Optional[int]]:
        """Return (summary_text, covers_through_message_id) or (None, None) if no summary."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT summary_text, summary_covers_through_message_id FROM chats WHERE chat_id = ?""",
                (chat_id,),
            ).fetchone()
            if row and row["summary_text"]:
                return row["summary_text"], row["summary_covers_through_message_id"]
            return None, None

    def get_messages_after(self, chat_id: str, message_id: Optional[int] = None) -> list[Message]:
        """Return all messages after message_id, or all if message_id is None."""
        with self._conn() as conn:
            if message_id is None:
                # Return all messages (existing get_chat behavior)
                msgs = conn.execute(
                    "SELECT * FROM messages WHERE chat_id = ? ORDER BY id",
                    (chat_id,),
                ).fetchall()
            else:
                # Return only messages with id > message_id
                msgs = conn.execute(
                    "SELECT * FROM messages WHERE chat_id = ? AND id > ? ORDER BY id",
                    (chat_id, message_id),
                ).fetchall()
            
            return [
                Message(
                    role=Role(m["role"]),
                    content=m["content"],
                    timestamp=m["timestamp"],
                )
                for m in msgs
            ]

    # -- Export ----------------------------------------------------------------

    def export_markdown(self, chat_id: str) -> Optional[str]:
        """Export a chat as clean Markdown."""
        chat = self.get_chat(chat_id)
        if not chat:
            return None

        lines = [f"# {chat.title}\n"]
        for msg in chat.messages:
            if msg.role == Role.system:
                continue
            prefix = "**User:**" if msg.role == Role.user else "**Assistant:**"
            lines.append(f"{prefix}\n\n{msg.content}\n\n---\n")

        return "\n".join(lines)

    def export_rl_jsonl(self, chat_id: str) -> Optional[str]:
        """
        Export chat in DPO/RLHF-ready JSONL format.
        Each assistant response with feedback becomes a training row.
        """
        with self._conn() as conn:
            msgs = conn.execute(
                "SELECT m.*, f.signal FROM messages m LEFT JOIN feedback f ON m.id = f.message_id WHERE m.chat_id = ? ORDER BY m.id",
                (chat_id,),
            ).fetchall()

        if not msgs:
            return None

        lines = []
        context = []
        for msg in msgs:
            if msg["role"] == "assistant" and msg["signal"]:
                row = {
                    "prompt": list(context),
                    "response": msg["content"],
                    "feedback": msg["signal"],
                    "model": msg["model_used"],
                    "chat_id": chat_id,
                }
                lines.append(json.dumps(row))
            context.append({"role": msg["role"], "content": msg["content"]})

        return "\n".join(lines) if lines else None
