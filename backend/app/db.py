"""
Lightweight SQLite-backed store for user-trained AI -> humanized phrase pairs.

Single-user, local tool — no ORM needed. One file, stdlib sqlite3 only.
"""
import sqlite3
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trained_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_phrase TEXT NOT NULL,
                humanized_phrase TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def create_trained_phrase(ai_phrase: str, humanized_phrase: str) -> dict:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO trained_phrases (ai_phrase, humanized_phrase) VALUES (?, ?)",
            (ai_phrase, humanized_phrase),
        )
        row = conn.execute(
            "SELECT * FROM trained_phrases WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


def list_trained_phrases() -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trained_phrases ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def get_trained_phrase(phrase_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM trained_phrases WHERE id = ?", (phrase_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_trained_phrase(phrase_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM trained_phrases WHERE id = ?", (phrase_id,))
        return cursor.rowcount > 0


init_db()
