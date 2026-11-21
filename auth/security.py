
import os
import jwt

from datetime import datetime, timedelta, timezone
from pwdlib import PasswordHash
from dotenv import load_dotenv

load_dotenv()

password_hash = PasswordHash.recommended()

import streamlit as st

try:
    JWT_SECRET_KEY = (
        os.getenv("CREDITLENS_JWT_SECRET")
        or st.secrets.get("CREDITLENS_JWT_SECRET")
    )
except Exception:
    JWT_SECRET_KEY = os.getenv("CREDITLENS_JWT_SECRET")

if not JWT_SECRET_KEY:
    raise RuntimeError(
        "CREDITLENS_JWT_SECRET environment variable is not configured."
    )

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def hash_password(password: str) -> str:
    """Convert a password into a secure Argon2 hash."""
    return password_hash.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Check a password against its stored hash."""
    return password_hash.verify(password, stored_hash)


def create_access_token(user_id: str) -> str:
    """Create a signed JWT access token."""

    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": user_id,
        "exp": expires_at
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )


def decode_access_token(token: str):
    """Verify a JWT and return its payload."""

    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM]
    )
