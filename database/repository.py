
import pandas as pd

from database.connection import get_database_connection


def initialize_database():
    """Create tables and safely migrate the database schema."""

    with get_database_connection() as connection:

        
        # User accounts
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'client'
            )
        """)

        # Migrate existing accounts without deleting user data.
        user_columns = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(users)"
        ).fetchall()
        ]

        if "role" not in user_columns:
            connection.execute("""
                ALTER TABLE users
                ADD COLUMN role TEXT NOT NULL DEFAULT 'employee'
            """)

        # Analysis metadata
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                predicted_defaults INTEGER NOT NULL,
                average_probability REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                target_available INTEGER NOT NULL,
                actual_default_rate REAL,
                result_filename TEXT NOT NULL
            )
        """)

        # Add ownership to existing databases without deleting records.
        columns = [
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(analysis_runs)"
            ).fetchall()
        ]

        if "owner_id" not in columns:
            connection.execute("""
                ALTER TABLE analysis_runs
                ADD COLUMN owner_id TEXT
                REFERENCES users(user_id)
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_owner
            ON analysis_runs(owner_id)
        """)

        # Borrower predictions
        connection.execute("""
            CREATE TABLE IF NOT EXISTS borrower_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                borrower_reference TEXT NOT NULL,
                predicted_probability REAL NOT NULL,
                predicted_default INTEGER NOT NULL,
                risk_category TEXT NOT NULL,
                actual_outcome INTEGER,
                FOREIGN KEY(run_id)
                    REFERENCES analysis_runs(run_id)
                    ON DELETE CASCADE
            )
        """)


def create_user(user_id, email, password_hash, created_at, role="client"):
    """Create a user account with a selected role."""

    with get_database_connection() as connection:
        connection.execute("""
            INSERT INTO users (
                user_id,
                email,
                password_hash,
                created_at,
                role
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            email,
            password_hash,
            created_at,
            role
        ))


def get_user_by_email(email):
    """Find a user using their email address."""

    with get_database_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM users
            WHERE email = ?
        """, (email,)).fetchone()

    return dict(row) if row else None


def get_user_by_id(user_id):
    """Find a user using their unique ID."""

    with get_database_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM users
            WHERE user_id = ?
        """, (user_id,)).fetchone()

    return dict(row) if row else None


def get_analysis_history():
    """Return all saved analysis runs, newest first."""

    with get_database_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM analysis_runs
            ORDER BY uploaded_at DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_analysis_run(run_id):
    """Retrieve metadata for one analysis run."""

    with get_database_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM analysis_runs
            WHERE run_id = ?
        """, (run_id,)).fetchone()

    return dict(row) if row else None


def get_borrower_predictions(run_id):
    """Retrieve borrower-level predictions for one analysis."""

    with get_database_connection() as connection:
        rows = connection.execute("""
            SELECT
                borrower_reference,
                predicted_probability,
                predicted_default,
                risk_category,
                actual_outcome
            FROM borrower_predictions
            WHERE run_id = ?
            ORDER BY id
        """, (run_id,)).fetchall()

    return pd.DataFrame([dict(row) for row in rows])

def get_user_analysis_history(user_id):
    """Return only analyses belonging to the specified user."""

    with get_database_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM analysis_runs
            WHERE owner_id = ?
            ORDER BY uploaded_at DESC
        """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def get_user_analysis_run(run_id, user_id):
    """Retrieve an analysis only if it belongs to the specified user."""

    with get_database_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM analysis_runs
            WHERE run_id = ?
              AND owner_id = ?
        """, (run_id, user_id)).fetchone()

    return dict(row) if row else None


def get_user_borrower_predictions(run_id, user_id):
    """Retrieve predictions only if the analysis belongs to the specified user."""

    with get_database_connection() as connection:
        rows = connection.execute("""
            SELECT
                bp.borrower_reference,
                bp.predicted_probability,
                bp.predicted_default,
                bp.risk_category,
                bp.actual_outcome
            FROM borrower_predictions AS bp
            INNER JOIN analysis_runs AS ar
                ON bp.run_id = ar.run_id
            WHERE ar.run_id = ?
              AND ar.owner_id = ?
            ORDER BY bp.id
        """, (run_id, user_id)).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def save_analysis_run(
    run_id,
    original_filename,
    stored_filename,
    uploaded_at,
    record_count,
    predicted_defaults,
    average_probability,
    model_name,
    model_version,
    target_available,
    actual_default_rate,
    result_filename,
    prediction_rows,
    owner_id=None
):
    """Save analysis metadata and predictions in one transaction."""

    initialize_database()

    with get_database_connection() as connection:

        connection.execute("""
            INSERT INTO analysis_runs (
                run_id,
                original_filename,
                stored_filename,
                uploaded_at,
                record_count,
                predicted_defaults,
                average_probability,
                model_name,
                model_version,
                target_available,
                actual_default_rate,
                result_filename,
                owner_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            original_filename,
            stored_filename,
            uploaded_at,
            record_count,
            predicted_defaults,
            average_probability,
            model_name,
            model_version,
            int(target_available),
            actual_default_rate,
            result_filename,
            owner_id
        ))

        connection.executemany("""
            INSERT INTO borrower_predictions (
                run_id,
                borrower_reference,
                predicted_probability,
                predicted_default,
                risk_category,
                actual_outcome
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, prediction_rows)
