from fastapi import APIRouter, Header, HTTPException, Depends, BackgroundTasks
from supabase import Client
from typing import Optional
import os

from app.main import get_supabase
from app.models.xgboost_predictor import train_and_predict_models
import logging

log = logging.getLogger("ml-backend.predict")

router = APIRouter()

CRON_SECRET = os.getenv("CRON_SECRET")

# En una app de producción esto idealmente se extraería de Supabase leyendo `all_tickers`.
DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "NVDA",
    "JPM",
    "V",
    "JNJ",
    "WMT",
    "PG",
    "MA",
    "UNH",
    "HD",
    "BAC",
    "DIS",
    "ADBE",
    "CRM",
    "NFLX",
]


@router.post("/")
async def run_ml_predictions(
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    supabase: Client = Depends(get_supabase),
):
    """
    Endpoint para ser llamado diariamente. Dispara el entrenamiento continuo
    (Rolling Window retraining) para la lista de tickers y guarda las predicciones en BD.
    Protegido por CRON_SECRET.
    """
    if not CRON_SECRET:
        log.error("CRON_SECRET no está configurado.")
        raise HTTPException(status_code=500, detail="Server misconfigured")

    expected = f"Bearer {CRON_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    background_tasks.add_task(train_and_predict_models, supabase, DEFAULT_TICKERS)

    return {
        "status": "accepted",
        "message": f"ML Model Training & Prediction dispatched for {len(DEFAULT_TICKERS)} assets.",
    }
