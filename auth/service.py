
import sqlite3
import uuid

from datetime import datetime, timezone

from database.repository import (
    create_user,
    get_user_by_email
)

from auth.security import (
    hash_password,
    verify_password,
    create_access_token
)


VALID_ROLES = {"employee", "client"}


def register_user(
    email: str,
    password: str,
    role: str = "client"
) -> dict:
    """
    Register a new employee or client account.
    """

    normalized_email = email.strip().lower()
    normalized_role = role.strip().lower()

    if normalized_role not in VALID_ROLES:
        raise ValueError("Please select a valid account type.")

    existing_user = get_user_by_email(normalized_email)

    if existing_user:
        raise ValueError(
            "An account with this email already exists."
        )

    user_id = str(uuid.uuid4())
    password_hash = hash_password(password)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        create_user(
            user_id=user_id,
            email=normalized_email,
            password_hash=password_hash,
            created_at=created_at,
            role=normalized_role
        )
    except sqlite3.IntegrityError:
        raise ValueError(
            "An account with this email already exists."
        )

    return {
        "user_id": user_id,
        "email": normalized_email,
        "created_at": created_at,
        "role": normalized_role
    }


def login_user(email: str, password: str) -> dict | None:
    """
    Verify credentials and return the authenticated user.
    """

    normalized_email = email.strip().lower()

    user = get_user_by_email(normalized_email)

    if not user:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    access_token = create_access_token(user["user_id"])

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user["user_id"],
        "email": user["email"],
        "role": user["role"]
    }
