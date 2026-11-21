
from fastapi import APIRouter, HTTPException, status

from auth.service import register_user, login_user

from api.schemas import (
    RegisterRequest,
    LoginRequest,
    UserResponse,
    LoginResponse
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
def register(request: RegisterRequest):
    try:
        user = register_user(
            email=str(request.email),
            password=request.password
        )

        return user

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error)
        )


@router.post(
    "/login",
    response_model=LoginResponse
)
def login(request: LoginRequest):
    result = login_user(
        email=str(request.email),
        password=request.password
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    return result
