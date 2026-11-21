
from fastapi import FastAPI

from api.routes import router
from auth.routes import router as auth_router


app = FastAPI(
    title="CreditLens AI API",
    description="Credit risk intelligence and analysis API",
    version="1.0.0"
)

app.include_router(router)
app.include_router(auth_router)
