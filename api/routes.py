
from io import BytesIO

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Response,
    Depends
)

from src.upload_service import (
    load_analysis_results,
    analyse_uploaded_dataset
)

from database.repository import (
    get_user_analysis_history,
    get_user_analysis_run,
    get_user_borrower_predictions
)

from auth.dependencies import get_current_user

from api.schemas import (
    HealthResponse,
    AnalysisHistoryItem,
    AnalysisHistoryResponse,
    PredictionResponse,
    AnalysisUploadResponse
)


router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check():

    return {
        "status": "healthy",
        "application": "CreditLens AI API",
        "version": "1.0.0"
    }


@router.get("/analyses", response_model=AnalysisHistoryResponse)
def list_analyses(
    current_user: dict = Depends(get_current_user)
):

    history = get_user_analysis_history(
        current_user["user_id"]
    )

    return {
        "total_analyses": len(history),
        "analyses": history
    }


@router.get(
    "/analyses/{run_id}",
    response_model=AnalysisHistoryItem
)
def retrieve_analysis(
    run_id: str,
    current_user: dict = Depends(get_current_user)
):

    analysis = get_user_analysis_run(
        run_id,
        current_user["user_id"]
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found"
        )

    return analysis


@router.get(
    "/analyses/{run_id}/predictions",
    response_model=PredictionResponse
)
def retrieve_predictions(
    run_id: str,
    current_user: dict = Depends(get_current_user)
):

    analysis = get_user_analysis_run(
        run_id,
        current_user["user_id"]
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found"
        )

    predictions = get_user_borrower_predictions(
        run_id,
        current_user["user_id"]
    )

    return {
        "run_id": run_id,
        "total_records": len(predictions),
        "predictions": predictions.to_dict(orient="records")
    }


@router.get("/analyses/{run_id}/download")
def download_analysis_results(
    run_id: str,
    current_user: dict = Depends(get_current_user)
):

    analysis = get_user_analysis_run(
        run_id,
        current_user["user_id"]
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis run not found"
        )

    try:
        results = load_analysis_results(run_id)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Saved analysis result file not found"
        )

    csv_content = results.to_csv(index=False)

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{analysis["result_filename"]}"'
            )
        }
    )


@router.post(
    "/analyse",
    response_model=AnalysisUploadResponse
)
async def analyse_dataset(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )

    file_contents = await file.read(10 * 1024 * 1024 + 1)

    if not file_contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty"
        )

    if len(file_contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the 10 MB limit"
        )

    try:
        data = pd.read_csv(BytesIO(file_contents))

    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError
    ):
        raise HTTPException(
            status_code=400,
            detail="Unable to read the CSV. Check its format and encoding."
        )

    try:
        result = analyse_uploaded_dataset(
            data=data,
            original_filename=file.filename,
            owner_id=current_user["user_id"]
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    return {
        "message": "Credit risk analysis completed successfully",
        "run_id": result["run_id"],
        "original_filename": file.filename,
        "record_count": result["record_count"],
        "predicted_defaults": result["predicted_defaults"],
        "average_probability": result["average_probability"],
        "target_available": result["target_available"],
        "actual_default_rate": result["actual_default_rate"],
        "result_filename": result["result_path"].name
    }
