"""SQLite persistence layer for multi-turn chat sessions and message history."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def resolve_sqlite_path(database_url: str) -> str:
    """Convert a SQLite DATABASE_URL (e.g. sqlite:////data/app.db) into a filesystem path."""
    url = database_url.strip()
    if url == ":memory:" or url == "sqlite:///:memory:":
        return ":memory:"
    if url.startswith("sqlite:////"):
        # Four slashes = absolute path on Unix (e.g. sqlite:////data/app.db -> /data/app.db)
        return "/" + url[len("sqlite:////"):]
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


class Database:
    """Thin, thread-safe SQLite wrapper using stdlib sqlite3."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.db_path = resolve_sqlite_path(database_url)
        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent) and str(parent) != ".":
                os.makedirs(parent, exist_ok=True)
        # Keep a persistent connection only when using an in-memory database (tests)
        self._memory_conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if self.db_path == ":memory:"
            else None
        )
        if self._memory_conn is not None:
            self._memory_conn.row_factory = sqlite3.Row
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _close(self, conn: sqlite3.Connection) -> None:
        if self._memory_conn is None:
            conn.close()

    def init_schema(self) -> None:
        """Create required tables and indexes automatically on startup."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                    ON messages (session_id, id);
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        system_prompt TEXT,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
        finally:
            self._close(conn)

    def ping(self) -> bool:
        """Verify SQLite connectivity."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT 1 AS ok;").fetchone()
            return bool(row and row["ok"] == 1)
        except sqlite3.Error:
            return False
        finally:
            self._close(conn)

    def get_session_system_prompt(self, session_id: str) -> Optional[str]:
        """Return stored custom system prompt for a session, if any."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            ).fetchone()
            return row["system_prompt"] if row else None
        finally:
            self._close(conn)

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Return stored messages for a session, oldest first, without the system prompt."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC;
                """,
                (session_id,),
            ).fetchall()
            messages: List[Dict[str, Any]] = []
            for row in rows:
                entry: Dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                }
                if row["role"] == "assistant" and row["metadata"]:
                    try:
                        entry["metadata"] = json.loads(row["metadata"])
                    except json.JSONDecodeError:
                        entry["metadata"] = None
                messages.append(entry)
            return messages
        finally:
            self._close(conn)

    def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        assistant_metadata: Dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> None:
        """Atomically persist both the user message and assistant reply after a successful turn."""
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_json = json.dumps(assistant_metadata)
        conn = self._connect()
        try:
            with conn:
                if system_prompt is not None:
                    conn.execute(
                        """
                        INSERT INTO sessions (session_id, system_prompt, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            system_prompt = excluded.system_prompt,
                            updated_at = excluded.updated_at;
                        """,
                        (session_id, system_prompt, now_iso),
                    )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'user', ?, NULL, ?);
                    """,
                    (session_id, user_message, now_iso),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'assistant', ?, ?, ?);
                    """,
                    (session_id, assistant_reply, metadata_json, now_iso),
                )
        finally:
            self._close(conn)

    def close(self) -> None:
        """Close persistent in-memory connection if present."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
