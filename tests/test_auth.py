
import pytest

from database import connection as db_connection
from database.repository import initialize_database
from auth.service import register_user, login_user


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    monkeypatch.setattr(
        db_connection,
        "DB_PATH",
        tmp_path / "test_auth.db"
    )
    initialize_database()


def test_register_user(temporary_database):
    user = register_user(
        "test@example.com",
        "CreditLensTest123!"
    )

    assert user["email"] == "test@example.com"
    assert "user_id" in user
    assert "password_hash" not in user


def test_duplicate_registration(temporary_database):
    register_user("test@example.com", "CreditLensTest123!")

    with pytest.raises(ValueError):
        register_user("test@example.com", "AnotherPassword123!")


def test_successful_login(temporary_database):
    register_user("test@example.com", "CreditLensTest123!")

    result = login_user(
        "test@example.com",
        "CreditLensTest123!"
    )

    assert result is not None
    assert result["token_type"] == "bearer"
    assert "access_token" in result


def test_invalid_login(temporary_database):
    register_user("test@example.com", "CreditLensTest123!")

    result = login_user(
        "test@example.com",
        "WrongPassword123!"
    )

    assert result is None
