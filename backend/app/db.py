"""
Lightweight SQLite-backed store for user-trained AI -> humanized phrase pairs.

Single-user, local tool — no ORM needed. One file, stdlib sqlite3 only.
"""
import json
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                overall_pct REAL NOT NULL,
                confidence TEXT NOT NULL,
                detected_patterns TEXT NOT NULL,
                sentence_scores TEXT NOT NULL,
                highlighted_phrases TEXT NOT NULL,
                suggestions TEXT NOT NULL,
                humanized_content TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
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


def update_trained_phrase(phrase_id: int, ai_phrase: str, humanized_phrase: str) -> Optional[dict]:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE trained_phrases SET ai_phrase = ?, humanized_phrase = ? WHERE id = ?",
            (ai_phrase, humanized_phrase, phrase_id),
        )
        if cursor.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT * FROM trained_phrases WHERE id = ?", (phrase_id,)
        ).fetchone()
        return dict(row)


def delete_trained_phrase(phrase_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM trained_phrases WHERE id = ?", (phrase_id,))
        return cursor.rowcount > 0


def _deserialize_project(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["detected_patterns"] = json.loads(d["detected_patterns"])
    d["sentence_scores"] = json.loads(d["sentence_scores"])
    d["highlighted_phrases"] = json.loads(d["highlighted_phrases"])
    d["suggestions"] = json.loads(d["suggestions"])
    return d


def create_project(
    content: str,
    overall_pct: float,
    confidence: str,
    detected_patterns: list,
    sentence_scores: list,
    highlighted_phrases: list,
    suggestions: list,
) -> dict:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO projects
                (content, overall_pct, confidence, detected_patterns,
                 sentence_scores, highlighted_phrases, suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                overall_pct,
                confidence,
                json.dumps(detected_patterns),
                json.dumps(sentence_scores),
                json.dumps(highlighted_phrases),
                json.dumps(suggestions),
            ),
        )
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _deserialize_project(row)


def list_projects() -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content, overall_pct, confidence, humanized_content, created_at
            FROM projects ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_project(project_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return _deserialize_project(row) if row else None


def update_project_humanized(project_id: int, humanized_content: str) -> Optional[dict]:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE projects SET humanized_content = ?, updated_at = datetime('now') WHERE id = ?",
            (humanized_content, project_id),
        )
        if cursor.rowcount == 0:
            return None
        row = conn.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        return _deserialize_project(row)


def delete_project(project_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return cursor.rowcount > 0


init_db()
