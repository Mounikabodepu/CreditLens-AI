
from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    application: str
    version: str


class AnalysisHistoryItem(BaseModel):
    run_id: str
    original_filename: str
    stored_filename: str
    uploaded_at: str
    record_count: int
    predicted_defaults: int
    average_probability: float
    model_name: str
    model_version: str
    target_available: bool
    actual_default_rate: Optional[float] = None
    result_filename: str


class AnalysisHistoryResponse(BaseModel):
    total_analyses: int
    analyses: list[AnalysisHistoryItem]


class PredictionItem(BaseModel):
    borrower_reference: str
    predicted_probability: float
    predicted_default: int
    risk_category: str
    actual_outcome: Optional[int] = None


class PredictionResponse(BaseModel):
    run_id: str
    total_records: int
    predictions: list[PredictionItem]


class AnalysisUploadResponse(BaseModel):
    message: str
    run_id: str
    original_filename: str
    record_count: int
    predicted_defaults: int
    average_probability: float
    target_available: bool
    actual_default_rate: Optional[float] = None
    result_filename: str

from pydantic import EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    user_id: str
    email: EmailStr
    created_at: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str
    email: EmailStr
