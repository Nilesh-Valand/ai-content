"""
Lightweight SQLite-backed store for user-trained AI -> humanized phrase pairs and profiles.

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
        # Create profiles table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

        # Check if default profile exists, if not seed "Default Style"
        cursor = conn.execute("SELECT id FROM profiles LIMIT 1")
        if cursor.fetchone() is None:
            conn.execute(
                "INSERT INTO profiles (name, description) VALUES (?, ?)",
                ("Default Style", "Default writing style profile for general content humanization."),
            )

        # Get default profile ID
        default_profile_row = conn.execute(
            "SELECT id FROM profiles WHERE name = 'Default Style' LIMIT 1"
        ).fetchone()
        default_profile_id = default_profile_row["id"] if default_profile_row else 1

        # Create trained_phrases table if not exists
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS trained_phrases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ai_phrase TEXT NOT NULL,
                humanized_phrase TEXT NOT NULL,
                profile_id INTEGER DEFAULT {default_profile_id},
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
            """
        )

        # Check if profile_id column exists in trained_phrases (migration for existing DBs)
        table_info = conn.execute("PRAGMA table_info(trained_phrases)").fetchall()
        columns = [col["name"] for col in table_info]
        if "profile_id" not in columns:
            conn.execute(f"ALTER TABLE trained_phrases ADD COLUMN profile_id INTEGER DEFAULT {default_profile_id}")

        # Ensure any null/0 profile_id phrases get assigned to default_profile_id
        conn.execute(
            "UPDATE trained_phrases SET profile_id = ? WHERE profile_id IS NULL OR profile_id = 0",
            (default_profile_id,),
        )

        # Projects table
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


# --- Profiles CRUD ---

def list_profiles() -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.name, p.description, p.created_at,
                   COUNT(tp.id) AS phrase_count
            FROM profiles p
            LEFT JOIN trained_phrases tp ON p.id = tp.profile_id
            GROUP BY p.id
            ORDER BY p.id ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_profile(profile_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT p.id, p.name, p.description, p.created_at,
                   COUNT(tp.id) AS phrase_count
            FROM profiles p
            LEFT JOIN trained_phrases tp ON p.id = tp.profile_id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (profile_id,),
        ).fetchone()
        return dict(row) if row else None


def create_profile(name: str, description: str = "") -> dict:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO profiles (name, description) VALUES (?, ?)",
            (name.strip(), description.strip()),
        )
        profile_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, name, description, created_at, 0 as phrase_count FROM profiles WHERE id = ?",
            (profile_id,),
        ).fetchone()
        return dict(row)


def update_profile(profile_id: int, name: str, description: str = "") -> Optional[dict]:
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE profiles SET name = ?, description = ? WHERE id = ?",
            (name.strip(), description.strip(), profile_id),
        )
        if cursor.rowcount == 0:
            return None
        return get_profile(profile_id)


def delete_profile(profile_id: int) -> bool:
    with get_connection() as conn:
        # Prevent deleting the last remaining profile
        count_row = conn.execute("SELECT COUNT(*) as cnt FROM profiles").fetchone()
        if count_row and count_row["cnt"] <= 1:
            raise ValueError("Cannot delete the only remaining profile")

        # Delete profile and its associated phrases
        conn.execute("DELETE FROM trained_phrases WHERE profile_id = ?", (profile_id,))
        cursor = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        return cursor.rowcount > 0


# --- Trained Phrases CRUD ---

def get_default_profile_id() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM profiles ORDER BY id ASC LIMIT 1").fetchone()
        return row["id"] if row else 1


def create_trained_phrase(ai_phrase: str, humanized_phrase: str, profile_id: Optional[int] = None) -> dict:
    if profile_id is None:
        profile_id = get_default_profile_id()

    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO trained_phrases (ai_phrase, humanized_phrase, profile_id) VALUES (?, ?, ?)",
            (ai_phrase.strip(), humanized_phrase.strip(), profile_id),
        )
        row = conn.execute(
            "SELECT * FROM trained_phrases WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


def list_trained_phrases(profile_id: Optional[int] = None) -> List[dict]:
    with get_connection() as conn:
        if profile_id is not None:
            rows = conn.execute(
                "SELECT * FROM trained_phrases WHERE profile_id = ? ORDER BY created_at DESC, id DESC",
                (profile_id,),
            ).fetchall()
        else:
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


def update_trained_phrase(phrase_id: int, ai_phrase: str, humanized_phrase: str, profile_id: Optional[int] = None) -> Optional[dict]:
    with get_connection() as conn:
        if profile_id is not None:
            cursor = conn.execute(
                "UPDATE trained_phrases SET ai_phrase = ?, humanized_phrase = ?, profile_id = ? WHERE id = ?",
                (ai_phrase.strip(), humanized_phrase.strip(), profile_id, phrase_id),
            )
        else:
            cursor = conn.execute(
                "UPDATE trained_phrases SET ai_phrase = ?, humanized_phrase = ? WHERE id = ?",
                (ai_phrase.strip(), humanized_phrase.strip(), phrase_id),
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

