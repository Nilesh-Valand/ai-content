"""
Lightweight SQLite-backed store for users, phrase profiles, trained phrases,
and analyzed-content projects.

No login/auth — "users" here are just named buckets you switch between in
the UI (picked from a dropdown, remembered in the browser), used purely to
keep each person's profiles/phrases/history separate on a shared install.
There's no password and no real security boundary.

Single-user-per-request, no ORM — one file, stdlib sqlite3 only.
"""
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data.db"

DEFAULT_USER_NAME = "Nilesh"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        # --- Users ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        if conn.execute("SELECT id FROM users LIMIT 1").fetchone() is None:
            conn.execute("INSERT INTO users (name) VALUES (?)", (DEFAULT_USER_NAME,))
        default_user_id = conn.execute(
            "SELECT id FROM users ORDER BY id ASC LIMIT 1"
        ).fetchone()["id"]

        # --- Profiles ---
        # Legacy schema (pre-users) had a single global UNIQUE(name). Once
        # multiple users exist, two different people both wanting a
        # "LinkedIn" profile need that uniqueness scoped per-user instead —
        # SQLite can't alter a constraint in place, so this recreates the
        # table when migrating from the old shape.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (user_id, name),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        profiles_columns = [c["name"] for c in conn.execute("PRAGMA table_info(profiles)")]
        if "user_id" not in profiles_columns:
            conn.execute("ALTER TABLE profiles RENAME TO profiles_old")
            conn.execute(
                """
                CREATE TABLE profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE (user_id, name),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO profiles (id, user_id, name, description, created_at)
                SELECT id, ?, name, description, created_at FROM profiles_old
                """,
                (default_user_id,),
            )
            conn.execute("DROP TABLE profiles_old")

        if conn.execute("SELECT id FROM profiles LIMIT 1").fetchone() is None:
            conn.execute(
                "INSERT INTO profiles (user_id, name, description) VALUES (?, ?, ?)",
                (default_user_id, "Default Style", "Default writing style profile for general content humanization."),
            )

        default_profile_row = conn.execute(
            "SELECT id FROM profiles WHERE user_id = ? ORDER BY id ASC LIMIT 1",
            (default_user_id,),
        ).fetchone()
        default_profile_id = default_profile_row["id"] if default_profile_row else 1

        # --- Trained phrases (scoped to a profile, which is itself scoped
        # to a user — no separate user_id column needed here) ---
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
        phrase_columns = [c["name"] for c in conn.execute("PRAGMA table_info(trained_phrases)")]
        if "profile_id" not in phrase_columns:
            conn.execute(f"ALTER TABLE trained_phrases ADD COLUMN profile_id INTEGER DEFAULT {default_profile_id}")
        conn.execute(
            "UPDATE trained_phrases SET profile_id = ? WHERE profile_id IS NULL OR profile_id = 0",
            (default_profile_id,),
        )

        # --- Projects (analyzed-content history, scoped to a user) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                content TEXT NOT NULL,
                overall_pct REAL NOT NULL,
                confidence TEXT NOT NULL,
                detected_patterns TEXT NOT NULL,
                sentence_scores TEXT NOT NULL,
                highlighted_phrases TEXT NOT NULL,
                suggestions TEXT NOT NULL,
                humanized_content TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        # SQLite can't add a FOREIGN KEY constraint via ALTER TABLE — a plain
        # ADD COLUMN gets the user_id column but silently skips the ON DELETE
        # CASCADE behavior, which would leave orphaned projects behind after
        # a user is deleted. Recreate the table (same technique as profiles
        # above) whenever it doesn't already carry that constraint, whether
        # this is a fresh migration or one that only got the column added.
        existing_fks = conn.execute("PRAGMA foreign_key_list(projects)").fetchall()
        has_user_fk = any(fk["table"] == "users" for fk in existing_fks)
        if not has_user_fk:
            old_columns = [c["name"] for c in conn.execute("PRAGMA table_info(projects)")]
            conn.execute("ALTER TABLE projects RENAME TO projects_old")
            conn.execute(
                """
                CREATE TABLE projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    content TEXT NOT NULL,
                    overall_pct REAL NOT NULL,
                    confidence TEXT NOT NULL,
                    detected_patterns TEXT NOT NULL,
                    sentence_scores TEXT NOT NULL,
                    highlighted_phrases TEXT NOT NULL,
                    suggestions TEXT NOT NULL,
                    humanized_content TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            user_id_expr = "COALESCE(user_id, ?)" if "user_id" in old_columns else "?"
            conn.execute(
                f"""
                INSERT INTO projects
                    (id, user_id, content, overall_pct, confidence, detected_patterns,
                     sentence_scores, highlighted_phrases, suggestions, humanized_content,
                     created_at, updated_at)
                SELECT id, {user_id_expr}, content, overall_pct, confidence, detected_patterns,
                       sentence_scores, highlighted_phrases, suggestions, humanized_content,
                       created_at, updated_at
                FROM projects_old
                """,
                (default_user_id,),
            )
            conn.execute("DROP TABLE projects_old")

        conn.execute(
            "UPDATE projects SET user_id = ? WHERE user_id IS NULL",
            (default_user_id,),
        )


# --- Users CRUD ---

def get_default_user_id() -> int:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM users ORDER BY id ASC LIMIT 1").fetchone()
        return row["id"] if row else 1


def list_users() -> List[dict]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id ASC").fetchall()
        return [dict(row) for row in rows]


def get_user(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_user(name: str) -> dict:
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO users (name) VALUES (?)", (name.strip(),))
        user_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO profiles (user_id, name, description) VALUES (?, ?, ?)",
            (user_id, "Default Style", "Default writing style profile for general content humanization."),
        )
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row)


def delete_user(user_id: int) -> bool:
    with get_connection() as conn:
        count_row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
        if count_row and count_row["cnt"] <= 1:
            raise ValueError("Cannot delete the only remaining user")
        cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cursor.rowcount > 0


# --- Profiles CRUD ---

def list_profiles(user_id: Optional[int] = None) -> List[dict]:
    if user_id is None:
        user_id = get_default_user_id()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT p.id, p.user_id, p.name, p.description, p.created_at,
                   COUNT(tp.id) AS phrase_count
            FROM profiles p
            LEFT JOIN trained_phrases tp ON p.id = tp.profile_id
            WHERE p.user_id = ?
            GROUP BY p.id
            ORDER BY p.id ASC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_profile(profile_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT p.id, p.user_id, p.name, p.description, p.created_at,
                   COUNT(tp.id) AS phrase_count
            FROM profiles p
            LEFT JOIN trained_phrases tp ON p.id = tp.profile_id
            WHERE p.id = ?
            GROUP BY p.id
            """,
            (profile_id,),
        ).fetchone()
        return dict(row) if row else None


def create_profile(name: str, description: str = "", user_id: Optional[int] = None) -> dict:
    if user_id is None:
        user_id = get_default_user_id()
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO profiles (user_id, name, description) VALUES (?, ?, ?)",
            (user_id, name.strip(), description.strip()),
        )
        profile_id = cursor.lastrowid
        row = conn.execute(
            "SELECT id, user_id, name, description, created_at, 0 as phrase_count FROM profiles WHERE id = ?",
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
    # No "last remaining profile" guard: humanize_content() already handles
    # profile_id=None gracefully (falls back to unscoped trained phrases, or
    # none at all), so a user is free to end up with zero profiles and
    # create a new one later whenever they want.
    with get_connection() as conn:
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
    user_id: Optional[int] = None,
) -> dict:
    if user_id is None:
        user_id = get_default_user_id()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO projects
                (user_id, content, overall_pct, confidence, detected_patterns,
                 sentence_scores, highlighted_phrases, suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
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


def list_projects(user_id: Optional[int] = None) -> List[dict]:
    if user_id is None:
        user_id = get_default_user_id()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, content, overall_pct, confidence, humanized_content, created_at
            FROM projects WHERE user_id = ? ORDER BY created_at DESC, id DESC
            """,
            (user_id,),
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
