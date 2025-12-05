from fastapi import FastAPI
from .inference.quantile_forecast import router as quantile_router

app = FastAPI(title="COVID Forecasting API")

app.include_router(quantile_router, prefix="/v1")
