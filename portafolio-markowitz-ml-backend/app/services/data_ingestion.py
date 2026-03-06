import httpx
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime  # noqa: F401 — used in type annotations
from supabase import Client
import logging
import os

log = logging.getLogger("ml-backend")


async def _fetch_fmp_direct(
    client: httpx.AsyncClient, ticker: str, api_key: str
) -> dict | None:
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/light?symbol={ticker}&apikey={api_key}"
    try:
        resp = await client.get(url, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return {"historical": data}
        return data
    except Exception as e:
        log.error(f"Error FMP API para {ticker}: {e}")
        return None


def fetch_and_ingest_prices(
    supabase: Client, tickers: list[str], period: str = "5y"
) -> dict:
    """
    Descarga datos históricos vía Financial Modeling Prep (FMP) y los almacena en la tabla activos_precios.
    Calcula los retornos diarios automáticamente.
    """
    ingestion_stats = {"success": 0, "failed": 0, "errors": []}

    if not tickers:
        log.warning("No se proporcionaron tickers para ingestar.")
        return ingestion_stats

    # Extraer API KEY
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    if not FMP_API_KEY:
        log.error("FMP_API_KEY no configurado en entorno.")
        ingestion_stats["errors"].append("FMP_API_KEY no configurado")
        return ingestion_stats

    log.info(f"Descargando datos desde FMP para {len(tickers)} tickers...")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    async def fetch_all(tkrs):
        async with httpx.AsyncClient() as client:
            tasks = [_fetch_fmp_direct(client, t, FMP_API_KEY) for t in tkrs]
            return await asyncio.gather(*tasks)

    results = loop.run_until_complete(fetch_all(tickers))

    for ticker, data in zip(tickers, results):
        if not data or "historical" not in data:
            log.warning(f"FMP no regresó datos completos para {ticker}.")
            ingestion_stats["failed"] += 1
            continue

        try:
            df_ticker = pd.DataFrame(data["historical"])
            if (
                df_ticker.empty
                or "date" not in df_ticker.columns
                or (
                    "adjClose" not in df_ticker.columns
                    and "price" not in df_ticker.columns
                )
            ):
                ingestion_stats["failed"] += 1
                continue

            # Preparar fechas
            df_ticker["date"] = pd.to_datetime(df_ticker["date"])
            df_ticker.sort_values(by="date", ascending=True, inplace=True)
            df_ticker.set_index("date", inplace=True)

            # Calcular retorno diario
            price_col = "price" if "price" in df_ticker.columns else "adjClose"
            df_ticker["retorno_diario"] = df_ticker[price_col].pct_change()
            df_ticker.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Preparar registros a BD
            records = []
            for date, row in df_ticker.iterrows():
                retorno = row.get("retorno_diario")
                if pd.isna(retorno):
                    retorno = None

                fecha_str = date.strftime("%Y-%m-%d")
                vol = row.get("volume", 0)
                records.append(
                    {
                        "ticker": ticker,
                        "fecha": fecha_str,
                        "precio_cierre": float(row[price_col]),
                        "volumen": int(vol) if pd.notna(vol) else 0,
                        "retorno_diario": (
                            float(retorno) if retorno is not None else None
                        ),
                    }
                )

            # Upsert en chunks para Supabase limit
            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                try:
                    supabase.table("activos_precios").upsert(batch).execute()
                except Exception as db_err:
                    log.error(f"Error guardando lote de {ticker} en BD: {db_err}")
                    raise db_err

            ingestion_stats["success"] += 1
            log.info(
                f"[{ticker}] ✅ Procesados y guardados {len(records)} días de historia (FMP)."
            )

        except Exception as e:
            log.error(f"Error procesando el ticker {ticker}: {e}")
            ingestion_stats["failed"] += 1
            ingestion_stats["errors"].append(f"[{ticker}] {str(e)}")

    return ingestion_stats
