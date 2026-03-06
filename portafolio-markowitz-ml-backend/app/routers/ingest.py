from fastapi import APIRouter, Header, HTTPException, Depends, BackgroundTasks
from supabase import Client
from typing import Optional
import os

from app.main import get_supabase
from app.services.data_ingestion import fetch_and_ingest_prices
import logging

log = logging.getLogger("ml-backend.ingest")

router = APIRouter()

# Cargamos el cron secret para proteger el endpoint
CRON_SECRET = os.getenv("CRON_SECRET")

# Una lista estática de prueba de los tickers más populares para ingestar
# En un sistema real más grande, esto se podría leer desde otra tabla de Supabase o un JSON.
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
async def run_data_ingestion(
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(None),
    supabase: Client = Depends(get_supabase),
):
    """
    Endpoint diseñado para ser llamado diariamente por un Cron Job (ej. cron-job.org o Railway Cron).
    Verifica el Authorization header contra el CRON_SECRET y ejecuta la ingesta en background.
    """
    if not CRON_SECRET:
        log.error("CRON_SECRET no está configurado en el servidor.")
        raise HTTPException(status_code=500, detail="Server configuration error")

    expected_header = f"Bearer {CRON_SECRET}"
    if authorization != expected_header:
        log.warning("Intento de acceso denegado as cron job.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Ejecutamos la tarea pesada en background para que el cron job no sufra timeout
    background_tasks.add_task(fetch_and_ingest_prices, supabase, DEFAULT_TICKERS, "5y")

    return {
        "status": "accepted",
        "message": f"Data ingestion started in background for {len(DEFAULT_TICKERS)} tickers.",
    }
