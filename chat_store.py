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

    def __init__(self, db_path: str = "/mnt/nfs/__Backups/SoHoAI--databases/sqlite/telemetry.db"):
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

            # Idempotent migration: add usage_events table for cost tracking & analytics
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id            TEXT NOT NULL UNIQUE,
                    created_at            TEXT NOT NULL,
                    source                TEXT NOT NULL,
                    user_id               TEXT,
                    chat_id               TEXT,
                    orchestra_session_id  TEXT,
                    model                 TEXT NOT NULL,
                    input_tokens          INTEGER NOT NULL DEFAULT 0,
                    output_tokens         INTEGER NOT NULL DEFAULT 0,
                    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
                    cost_usd              REAL NOT NULL DEFAULT 0.0,
                    provider              TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_usage_events_created_at
                    ON usage_events (created_at);

                CREATE INDEX IF NOT EXISTS idx_usage_events_user_created_at
                    ON usage_events (user_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_usage_events_source_created_at
                    ON usage_events (source, created_at);

                CREATE INDEX IF NOT EXISTS idx_usage_events_orchestra_session_id
                    ON usage_events (orchestra_session_id);

                CREATE INDEX IF NOT EXISTS idx_usage_events_model_created_at
                    ON usage_events (model, created_at);
            """)
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

    def record_usage_event(
        self,
        *,
        request_id: str,
        created_at: str,
        source: str,
        user_id: Optional[str],
        chat_id: Optional[str],
        orchestra_session_id: Optional[str],
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        cost_usd: float,
        provider: Optional[str],
    ) -> None:
        """Record a usage event for cost tracking and analytics."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO usage_events
                   (request_id, created_at, source, user_id, chat_id, orchestra_session_id,
                    model, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens,
                    cost_usd, provider)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    created_at,
                    source,
                    user_id,
                    chat_id,
                    orchestra_session_id,
                    model,
                    input_tokens,
                    output_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                    cost_usd,
                    provider,
                ),
            )

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

    def get_summary_boundary_id(self, chat_id: str, summarize_keep_turns: int) -> Optional[int]:
        """
        Find the message id that marks the boundary for summarization.

        Returns the id of the message at position (total - keep_turns - 1),
        i.e., the last message that will be included in the summary.

        Returns None if there aren't enough messages to summarize.
        """
        with self._conn() as conn:
            # Get the total count of messages for this chat
            count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            total = count_row["cnt"] if count_row else 0

            if total <= summarize_keep_turns:
                return None  # Not enough old messages to summarize

            # Get the id at offset (total - keep_turns - 1) when ordered by id
            # This means: skip (total - keep_turns - 1) rows and get the next one
            row = conn.execute(
                """SELECT id FROM messages WHERE chat_id = ?
                   ORDER BY id ASC LIMIT 1 OFFSET ?""",
                (chat_id, total - summarize_keep_turns - 1),
            ).fetchone()

            return row["id"] if row else None

    def query_usage_stats(
        self,
        *,
        since: str,
        until: str,
        user: Optional[str] = None,
        model: Optional[str] = None,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> dict:
        """
        Query usage statistics for a time window.

        Returns a dict with:
        - window: {since, until}
        - totals: {requests, input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens, cost_usd, cache_hit_rate}
        - by_model: [{model, requests, input_tokens, output_tokens, cost_usd}, ...]
        - by_source: [{source, requests, input_tokens, output_tokens, cost_usd}, ...]
        - by_day: (optional, if group_by="day") [{date, requests, input_tokens, output_tokens, cost_usd}, ...]
        """
        with self._conn() as conn:
            # Build dynamic WHERE clause
            conditions = ["created_at BETWEEN ? AND ?"]
            params = [since, until]

            if user is not None:
                conditions.append("user_id = ?")
                params.append(user)
            if model is not None:
                conditions.append("model = ?")
                params.append(model)
            if source is not None:
                conditions.append("source = ?")
                params.append(source)
            if session_id is not None:
                conditions.append("orchestra_session_id = ?")
                params.append(session_id)

            where_clause = " AND ".join(conditions)

            # Query totals
            totals_query = f"""
                SELECT
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(cache_creation_tokens), 0) as cache_creation_tokens,
                    COALESCE(SUM(cache_read_tokens), 0) as cache_read_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM usage_events
                WHERE {where_clause}
            """
            totals_row = conn.execute(totals_query, params).fetchone()

            # Calculate cache_hit_rate (handle division by zero)
            total_cache_tokens = (
                totals_row["cache_creation_tokens"] + totals_row["cache_read_tokens"]
            )
            cache_hit_rate = (
                totals_row["cache_read_tokens"] / total_cache_tokens
                if total_cache_tokens > 0
                else 0.0
            )

            totals = {
                "requests": totals_row["requests"],
                "input_tokens": totals_row["input_tokens"],
                "output_tokens": totals_row["output_tokens"],
                "cache_creation_tokens": totals_row["cache_creation_tokens"],
                "cache_read_tokens": totals_row["cache_read_tokens"],
                "cost_usd": totals_row["cost_usd"],
                "cache_hit_rate": cache_hit_rate,
            }

            # Query by_model
            by_model_query = f"""
                SELECT
                    model,
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM usage_events
                WHERE {where_clause}
                GROUP BY model
            """
            by_model_rows = conn.execute(by_model_query, params).fetchall()
            by_model = [dict(row) for row in by_model_rows]

            # Sort by cost_usd DESC if group_by="model"
            if group_by == "model":
                by_model.sort(key=lambda x: x["cost_usd"], reverse=True)

            # Query by_source
            by_source_query = f"""
                SELECT
                    source,
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(cost_usd), 0.0) as cost_usd
                FROM usage_events
                WHERE {where_clause}
                GROUP BY source
            """
            by_source_rows = conn.execute(by_source_query, params).fetchall()
            by_source = [dict(row) for row in by_source_rows]

            # Sort by cost_usd DESC if group_by="source"
            if group_by == "source":
                by_source.sort(key=lambda x: x["cost_usd"], reverse=True)

            result = {
                "window": {"since": since, "until": until},
                "totals": totals,
                "by_model": by_model,
                "by_source": by_source,
            }

            # Query by_day if group_by="day"
            if group_by == "day":
                by_day_query = f"""
                    SELECT
                        strftime('%Y-%m-%d', created_at) as date,
                        COUNT(*) as requests,
                        COALESCE(SUM(input_tokens), 0) as input_tokens,
                        COALESCE(SUM(output_tokens), 0) as output_tokens,
                        COALESCE(SUM(cost_usd), 0.0) as cost_usd
                    FROM usage_events
                    WHERE {where_clause}
                    GROUP BY date
                    ORDER BY date ASC
                """
                by_day_rows = conn.execute(by_day_query, params).fetchall()
                result["by_day"] = [dict(row) for row in by_day_rows]

            return result

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
