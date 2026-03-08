import os
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd

from config import (
    FMP_API_KEY,
    FMP_BASE_URL,
    get_supabase_client,
    log,
)

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    def __init__(self, message: str, failed_tickers: list[str] | None = None):
        super().__init__(message)
        self.failed_tickers = failed_tickers or []


# ── Supabase cache helpers ──────────────────────────────────────


def _read_cache_supabase(ticker: str, fecha_inicio: date, fecha_fin: date) -> pd.DataFrame | None:
    """Lee datos cacheados de market_data_cache en Supabase.
    Retorna un DataFrame con columnas [date, close] o None si no hay datos suficientes.
    """
    try:
        sb = get_supabase_client()
        resp = (
            sb.table("market_data_cache")
            .select("fecha, close_price")
            .eq("ticker", ticker)
            .gte("fecha", fecha_inicio.isoformat())
            .lte("fecha", fecha_fin.isoformat())
            .order("fecha")
            .execute()
        )
        rows = resp.data
        if not rows:
            return None

        df = pd.DataFrame(rows)
        df.rename(columns={"fecha": "date", "close_price": "close"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df["close"] = df["close"].astype(float)
        return df
    except Exception as e:
        logger.warning(f"Error leyendo cache Supabase para {ticker}: {e}")
        return None


def _is_cache_fresh(cached_df: pd.DataFrame | None, fecha_fin: date) -> bool:
    """Verifica si el cache cubre hasta al menos ayer (o fecha_fin si es pasado)."""
    if cached_df is None or cached_df.empty:
        return False
    last_cached = cached_df.index.max().date()
    # Consideramos fresco si cubre hasta ayer o hasta fecha_fin (lo que sea menor)
    target = min(fecha_fin, date.today() - timedelta(days=1))
    # Tolerancia de 3 dias por fines de semana / festivos
    return (target - last_cached).days <= 3


def _upsert_cache_supabase(ticker: str, df: pd.DataFrame) -> None:
    """Guarda/actualiza datos en market_data_cache."""
    try:
        sb = get_supabase_client()
        records = []
        for idx, row in df.iterrows():
            records.append({
                "ticker": ticker,
                "fecha": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "close_price": float(row["close"]),
            })

        if not records:
            return

        # Upsert en lotes de 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            sb.table("market_data_cache").upsert(batch, on_conflict="ticker,fecha").execute()

        logger.info(f"Cache actualizado para {ticker}: {len(records)} registros.")
    except Exception as e:
        logger.warning(f"Error escribiendo cache Supabase para {ticker}: {e}")


# ── FMP API helpers ─────────────────────────────────────────────


def _fetch_fmp_historical(
    ticker: str, fecha_inicio: date, fecha_fin: date, retries: int = 2
) -> pd.DataFrame:
    """Descarga precios historicos desde Financial Modeling Prep (stable API).
    Retorna DataFrame con index=date, columna 'close'.
    Lanza DataFetchError si el ticker no existe o FMP falla tras reintentos.
    """
    import time

    if not FMP_API_KEY:
        raise DataFetchError("FMP_API_KEY no esta configurada. Agrega FMP_API_KEY al .env")

    url = f"{FMP_BASE_URL}/historical-price-eod/full"
    params = {
        "symbol": ticker,
        "from": fecha_inicio.isoformat(),
        "to": fecha_fin.isoformat(),
        "apikey": FMP_API_KEY,
    }

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, params=params)

            if resp.status_code == 404:
                raise DataFetchError(
                    f"Ticker '{ticker}' no encontrado en FMP (HTTP 404).",
                    failed_tickers=[ticker],
                )

            if resp.status_code != 200:
                msg = f"Error de FMP para '{ticker}': HTTP {resp.status_code}"
                if attempt < retries:
                    logger.warning(f"{msg} (intento {attempt}/{retries}, reintentando...)")
                    time.sleep(1.5)
                    continue
                raise DataFetchError(msg, failed_tickers=[ticker])

            data = resp.json()

            if not data:
                raise DataFetchError(
                    f"FMP retorno un array vacio para '{ticker}'. Verifica que el simbolo sea valido.",
                    failed_tickers=[ticker],
                )

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            df = df[["close"]].copy()
            df["close"] = df["close"].astype(float)
            return df

        except DataFetchError:
            raise
        except Exception as e:
            last_error = e
            if attempt < retries:
                logger.warning(f"Error de red FMP para '{ticker}' (intento {attempt}/{retries}): {e}")
                time.sleep(1.5)
            else:
                logger.error(f"FMP fallo tras {retries} intentos para '{ticker}': {e}")

    raise DataFetchError(
        f"No se pudieron obtener datos historicos de '{ticker}' tras {retries} intentos: {last_error}",
        failed_tickers=[ticker],
    )


# ── Public API ──────────────────────────────────────────────────


def get_historical_prices_fmp(
    tickers: list[str],
    fecha_inicio: date,
    fecha_fin: date,
) -> dict:
    """Descarga precios historicos usando el patron Cache-Aside con FMP + Supabase.

    Retorna:
        {
            "prices": pd.DataFrame (index=date, columnas=tickers),
            "returns": pd.DataFrame (log-returns),
            "tickers": list[str],
            "failed_tickers": list[str],
        }
    """
    if not tickers:
        raise DataFetchError("No se proporcionaron tickers.")

    prices_dict: dict[str, pd.Series] = {}
    failed_tickers: list[str] = []

    for ticker in tickers:
        try:
            # 1. Intentar cache
            cached = _read_cache_supabase(ticker, fecha_inicio, fecha_fin)

            if _is_cache_fresh(cached, fecha_fin):
                logger.info(f"Cache HIT para {ticker}")
                prices_dict[ticker] = cached["close"]
                continue

            # 2. Cache miss o desactualizado — pedir a FMP
            logger.info(f"Cache MISS para {ticker}, descargando de FMP...")
            fmp_df = _fetch_fmp_historical(ticker, fecha_inicio, fecha_fin)

            # 3. Guardar en cache
            _upsert_cache_supabase(ticker, fmp_df)

            prices_dict[ticker] = fmp_df["close"]

        except DataFetchError:
            failed_tickers.append(ticker)
            logger.warning(f"Ticker fallido: {ticker}")
        except Exception as e:
            failed_tickers.append(ticker)
            logger.error(f"Error inesperado para {ticker}: {e}")

    if not prices_dict:
        raise DataFetchError(
            "No se encontraron datos para ninguno de los tickers solicitados.",
            failed_tickers=failed_tickers,
        )

    # Construir DataFrame de precios alineados
    prices_df = pd.DataFrame(prices_dict)
    prices_df.sort_index(inplace=True)

    # Limpieza y validacion
    cleaned_prices, log_returns = clean_and_validate_data(prices_df)

    return {
        "prices": cleaned_prices,
        "returns": log_returns,
        "tickers": list(cleaned_prices.columns),
        "failed_tickers": failed_tickers,
    }


# ── Legacy wrappers (mantienen compatibilidad con api.py existente) ──


def get_historical_prices(tickers: list[str], period: str) -> dict:
    """Wrapper que traduce el formato period ('1y','3y','5y') a fechas y llama a FMP."""
    period_map = {"1y": 1, "3y": 3, "5y": 5}
    years = period_map.get(period)
    if years is None:
        raise ValueError(f"Period '{period}' no soportado. Usa uno de {set(period_map.keys())}")

    fecha_fin = date.today()
    fecha_inicio = date(fecha_fin.year - years, fecha_fin.month, fecha_fin.day)

    return get_historical_prices_fmp(tickers, fecha_inicio, fecha_fin)


def get_benchmarks(period: str) -> dict:
    """Descarga benchmarks (IPC y S&P500) usando FMP y calcula el crecimiento de $10,000."""
    INITIAL_AMOUNT = 10000.0

    cetes_rate_str = os.environ.get("CETES_RATE", "0.105")
    try:
        cetes_rate = float(cetes_rate_str)
    except ValueError:
        cetes_rate = 0.105

    period_map = {"1y": 1, "3y": 3, "5y": 5}
    years = period_map.get(period, 1)
    cetes_final = INITIAL_AMOUNT * ((1 + cetes_rate) ** years)

    fecha_fin = date.today()
    fecha_inicio = date(fecha_fin.year - years, fecha_fin.month, fecha_fin.day)

    benchmarks_map = {"^MXX": "IPC", "^GSPC": "S&P500"}
    # FMP usa formatos distintos para indices
    fmp_ticker_map = {"^MXX": "%5EMXX", "^GSPC": "%5EGSPC"}
    bm_results: dict[str, float | None] = {}

    for original_ticker, name in benchmarks_map.items():
        try:
            fmp_ticker = fmp_ticker_map.get(original_ticker, original_ticker)
            fmp_df = _fetch_fmp_historical(fmp_ticker, fecha_inicio, fecha_fin)
            if not fmp_df.empty:
                first_price = fmp_df["close"].iloc[0]
                last_price = fmp_df["close"].iloc[-1]
                growth = last_price / first_price
                bm_results[name] = INITIAL_AMOUNT * growth
            else:
                bm_results[name] = None
        except Exception as e:
            logger.warning(f"Error descargando benchmark {name}: {e}")
            bm_results[name] = None

    return {
        "IPC_final": bm_results.get("IPC"),
        "S&P500_final": bm_results.get("S&P500"),
        "CETES_final": cetes_final,
        "initial_amount": INITIAL_AMOUNT,
        "period": period,
        "cetes_rate_used": cetes_rate,
    }


def filter_illiquid_tickers(tickers: list[str], min_volume: float = 1_000_000) -> dict:
    """Placeholder — la API de FMP no provee volumen en el endpoint historico gratuito.
    Retorna todos los tickers como validos por ahora."""
    return {"valid_tickers": tickers, "rejected_tickers": {}}


def clean_and_validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Limpia y valida datos de precios:
    1. Forward fill NaN (max 5 dias)
    2. Alinear fechas (inner join)
    3. Validar minimo 126 dias (~6 meses)
    4. Detectar outliers (cambio > 50%)
    5. Calcular retornos logaritmicos
    """
    # 1. Forward fill
    df = df.ffill(limit=5)

    # 2. Alinear fechas
    df = df.dropna(how="any")

    # 3. Validar minimo 6 meses
    if len(df) < 126:
        raise DataFetchError(
            f"Datos insuficientes: solo {len(df)} dias disponibles, minimo 126 requeridos (~6 meses)."
        )

    # 4. Detectar outliers
    pct_change = df.pct_change()
    outlier_mask = (pct_change > 0.5) | (pct_change < -0.5)
    if outlier_mask.any().any():
        outliers = outlier_mask.sum()
        for col in outliers[outliers > 0].index:
            logger.warning(f"Outlier detectado en {col}: {outliers[col]} dias con > 50% de variacion.")

    # 5. Retornos logaritmicos
    log_returns = np.log(df / df.shift(1)).dropna(how="all")
    df = df.loc[log_returns.index]

    return df, log_returns
