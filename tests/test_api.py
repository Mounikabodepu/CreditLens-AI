
import pytest

from fastapi.testclient import TestClient

from api.main import app
import jwt

from datetime import datetime, timedelta, timezone

from auth.security import JWT_SECRET_KEY, JWT_ALGORITHM
from database import connection as db_connection
from database.repository import (
    initialize_database,
    save_analysis_run
)

from auth.service import register_user, login_user


client = TestClient(app)


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    monkeypatch.setattr(
        db_connection,
        "DB_PATH",
        tmp_path / "test_api.db"
    )
    initialize_database()


@pytest.fixture
def authenticated_user(temporary_database):
    user = register_user(
        "usera@example.com",
        "CreditLensTest123!"
    )

    login = login_user(
        "usera@example.com",
        "CreditLensTest123!"
    )

    return {
        "user": user,
        "headers": {
            "Authorization": f"Bearer {login['access_token']}"
        }
    }


def save_demo_analysis(owner_id, run_id="test-run-001"):

    save_analysis_run(
        run_id=run_id,
        original_filename="demo.csv",
        stored_filename="demo-stored.csv",
        uploaded_at="2026-11-21T10:00:00+00:00",
        record_count=1,
        predicted_defaults=1,
        average_probability=0.80,
        model_name="Random Forest",
        model_version="test_v1",
        target_available=True,
        actual_default_rate=1.0,
        result_filename="demo-predictions.csv",
        prediction_rows=[
            (run_id, "BORROWER-001", 0.80, 1, "High", 1)
        ],
        owner_id=owner_id
    )


def test_health_check():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analysis_endpoint_requires_authentication():

    response = client.get("/analyses")

    assert response.status_code == 401


def test_analysis_history(authenticated_user):

    response = client.get(
        "/analyses",
        headers=authenticated_user["headers"]
    )

    assert response.status_code == 200

    data = response.json()

    assert "total_analyses" in data
    assert "analyses" in data
    assert data["total_analyses"] == 0


def test_invalid_analysis_id(authenticated_user):

    response = client.get(
        "/analyses/invalid-run-id",
        headers=authenticated_user["headers"]
    )

    assert response.status_code == 404


def test_invalid_prediction_run_id(authenticated_user):

    response = client.get(
        "/analyses/invalid-run-id/predictions",
        headers=authenticated_user["headers"]
    )

    assert response.status_code == 404


def test_upload_rejects_non_csv_file(authenticated_user):

    response = client.post(
        "/analyse",
        headers=authenticated_user["headers"],
        files={
            "file": (
                "sample.txt",
                b"This is not a CSV file",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are supported"


def test_upload_rejects_empty_csv(authenticated_user):

    response = client.post(
        "/analyse",
        headers=authenticated_user["headers"],
        files={
            "file": (
                "empty.csv",
                b"",
                "text/csv"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty"


def test_download_rejects_invalid_run_id(authenticated_user):

    response = client.get(
        "/analyses/invalid-run-id/download",
        headers=authenticated_user["headers"]
    )

    assert response.status_code == 404


def test_user_can_access_own_analysis(authenticated_user):

    user_id = authenticated_user["user"]["user_id"]

    save_demo_analysis(user_id)

    response = client.get(
        "/analyses/test-run-001",
        headers=authenticated_user["headers"]
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "test-run-001"


def test_user_cannot_access_another_users_analysis(authenticated_user):

    owner_id = authenticated_user["user"]["user_id"]

    save_demo_analysis(owner_id)

    other_user = register_user(
        "userb@example.com",
        "CreditLensTest456!"
    )

    other_login = login_user(
        "userb@example.com",
        "CreditLensTest456!"
    )

    other_headers = {
        "Authorization": f"Bearer {other_login['access_token']}"
    }

    response = client.get(
        "/analyses/test-run-001",
        headers=other_headers
    )

    assert response.status_code == 404


def test_user_cannot_see_another_users_analysis_in_history(authenticated_user):

    owner_id = authenticated_user["user"]["user_id"]

    save_demo_analysis(owner_id)

    other_user = register_user(
        "userb@example.com",
        "CreditLensTest456!"
    )

    other_login = login_user(
        "userb@example.com",
        "CreditLensTest456!"
    )

    other_headers = {
        "Authorization": f"Bearer {other_login['access_token']}"
    }

    response = client.get(
        "/analyses",
        headers=other_headers
    )

    assert response.status_code == 200
    assert response.json()["total_analyses"] == 0

def test_api_user_registration(temporary_database):

    response = client.post(
        "/auth/register",
        json={
            "email": "apiuser@example.com",
            "password": "CreditLensTest123!"
        }
    )

    assert response.status_code == 201
    assert response.json()["email"] == "apiuser@example.com"
    assert "user_id" in response.json()
    assert "password_hash" not in response.json()


def test_api_user_login(temporary_database):

    client.post(
        "/auth/register",
        json={
            "email": "loginuser@example.com",
            "password": "CreditLensTest123!"
        }
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "loginuser@example.com",
            "password": "CreditLensTest123!"
        }
    )

    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_api_login_rejects_wrong_password(temporary_database):

    client.post(
        "/auth/register",
        json={
            "email": "wrongpass@example.com",
            "password": "CreditLensTest123!"
        }
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "wrongpass@example.com",
            "password": "WrongPassword123!"
        }
    )

    assert response.status_code == 401

def test_upload_requires_authentication():

    response = client.post(
        "/analyse",
        files={
            "file": (
                "sample.csv",
                b"LIMIT_BAL,AGE\n100000,30",
                "text/csv"
            )
        }
    )

    assert response.status_code == 401


def test_user_cannot_access_another_users_predictions(authenticated_user):

    owner_id = authenticated_user["user"]["user_id"]

    save_demo_analysis(owner_id)

    register_user(
        "userc@example.com",
        "CreditLensTest789!"
    )

    other_login = login_user(
        "userc@example.com",
        "CreditLensTest789!"
    )

    other_headers = {
        "Authorization": f"Bearer {other_login['access_token']}"
    }

    response = client.get(
        "/analyses/test-run-001/predictions",
        headers=other_headers
    )

    assert response.status_code == 404


def test_user_cannot_download_another_users_analysis(authenticated_user):

    owner_id = authenticated_user["user"]["user_id"]

    save_demo_analysis(owner_id)

    register_user(
        "userd@example.com",
        "CreditLensTest999!"
    )

    other_login = login_user(
        "userd@example.com",
        "CreditLensTest999!"
    )

    other_headers = {
        "Authorization": f"Bearer {other_login['access_token']}"
    }

    response = client.get(
        "/analyses/test-run-001/download",
        headers=other_headers
    )

    assert response.status_code == 404
def test_analysis_details_requires_authentication():

    response = client.get("/analyses/test-run-001")

    assert response.status_code == 401


def test_predictions_require_authentication():

    response = client.get("/analyses/test-run-001/predictions")

    assert response.status_code == 401


def test_download_requires_authentication():

    response = client.get("/analyses/test-run-001/download")

    assert response.status_code == 401

def test_upload_rejects_file_larger_than_10_mb(authenticated_user):

    large_file = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/analyse",
        headers=authenticated_user["headers"],
        files={
            "file": (
                "large.csv",
                large_file,
                "text/csv"
            )
        }
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "File size exceeds the 10 MB limit"


def test_upload_rejects_csv_with_missing_columns(authenticated_user):

    response = client.post(
        "/analyse",
        headers=authenticated_user["headers"],
        files={
            "file": (
                "incomplete.csv",
                b"AGE,LIMIT_BAL\n30,100000\n",
                "text/csv"
            )
        }
    )

    assert response.status_code == 400


def test_upload_rejects_malformed_csv(authenticated_user):

    response = client.post(
        "/analyse",
        headers=authenticated_user["headers"],
        files={
            "file": (
                "malformed.csv",
                b'AGE,LIMIT_BAL\n"unterminated,100000\n',
                "text/csv"
            )
        }
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unable to read the CSV. Check its format and encoding."
    )

def test_api_rejects_duplicate_registration(temporary_database):

    registration_data = {
        "email": "duplicate@example.com",
        "password": "CreditLensTest123!"
    }

    first_response = client.post(
        "/auth/register",
        json=registration_data
    )

    second_response = client.post(
        "/auth/register",
        json=registration_data
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409


def test_api_rejects_invalid_login(temporary_database):

    client.post(
        "/auth/register",
        json={
            "email": "invalidlogin@example.com",
            "password": "CreditLensTest123!"
        }
    )

    response = client.post(
        "/auth/login",
        json={
            "email": "invalidlogin@example.com",
            "password": "WrongPassword123!"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password."


def test_api_rejects_invalid_jwt(authenticated_user):

    response = client.get(
        "/analyses",
        headers={
            "Authorization": "Bearer invalid.token.value"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Invalid or expired authentication token."
    )


def test_api_rejects_expired_jwt(authenticated_user):

    expired_payload = {
        "sub": authenticated_user["user"]["user_id"],
        "exp": datetime.now(timezone.utc) - timedelta(minutes=5)
    }

    expired_token = jwt.encode(
        expired_payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM
    )

    response = client.get(
        "/analyses",
        headers={
            "Authorization": f"Bearer {expired_token}"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == (
        "Invalid or expired authentication token."
    )

def test_api_rejects_tampered_jwt(authenticated_user):

    original_login = login_user(
        "usera@example.com",
        "CreditLensTest123!"
    )

    original_token = original_login["access_token"]

    tampered_payload = {
        "sub": "another-user-id",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=30)
    }

    tampered_token = jwt.encode(
        tampered_payload,
        "incorrect-secret-key-for-testing-only-32",
        algorithm=JWT_ALGORITHM
    )

    assert tampered_token != original_token

    response = client.get(
        "/analyses",
        headers={
            "Authorization": f"Bearer {tampered_token}"
        }
    )

    assert response.status_code == 401

