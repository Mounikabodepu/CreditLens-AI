
import pytest

from database import connection as db_connection
from database.repository import (
    initialize_database,
    get_analysis_history,
    get_analysis_run,
    get_borrower_predictions,
    save_analysis_run
)


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    """Give each test its own temporary SQLite database."""

    monkeypatch.setattr(
        db_connection,
        "DB_PATH",
        tmp_path / "test_creditlens.db"
    )

    initialize_database()


def test_database_initialization(temporary_database):
    history = get_analysis_history()
    assert history == []


def test_save_and_retrieve_analysis(temporary_database):

    save_analysis_run(
        run_id="test-run-001",
        original_filename="demo.csv",
        stored_filename="demo-stored.csv",
        uploaded_at="2026-11-21T10:00:00+00:00",
        record_count=2,
        predicted_defaults=1,
        average_probability=0.65,
        model_name="Random Forest",
        model_version="test_v1",
        target_available=True,
        actual_default_rate=0.5,
        result_filename="demo-predictions.csv",
        prediction_rows=[
            ("test-run-001", "BORROWER-001", 0.80, 1, "High", 1),
            ("test-run-001", "BORROWER-002", 0.20, 0, "Low", 0)
        ]
    )

    analysis = get_analysis_run("test-run-001")

    assert analysis is not None
    assert analysis["record_count"] == 2
    assert analysis["predicted_defaults"] == 1

    predictions = get_borrower_predictions("test-run-001")

    assert len(predictions) == 2
    assert list(predictions.columns) == [
        "borrower_reference",
        "predicted_probability",
        "predicted_default",
        "risk_category",
        "actual_outcome"
    ]


def test_failed_prediction_insert_rolls_back_analysis(
    temporary_database
):

    with pytest.raises(Exception):

        save_analysis_run(
            run_id="test-run-002",
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
                (
                    "wrong-run-id",
                    "BORROWER-001",
                    0.80,
                    1,
                    "High",
                    1
                )
            ]
        )

    assert get_analysis_run("test-run-002") is None
