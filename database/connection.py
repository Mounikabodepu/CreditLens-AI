
from pathlib import Path
import sqlite3
from contextlib import contextmanager


# Project and database locations
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "data" / "creditlens.db"


@contextmanager
def get_database_connection():
    """
    Open a SQLite connection.

    Automatically:
    - Enables foreign-key enforcement.
    - Commits successful transactions.
    - Rolls back failed transactions.
    - Closes the connection.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
