import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio
from pydantic import BaseModel
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

# slowapi — Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import our backend logic
from data import (
    descargar_lotes, extraer_panel, limpiar_retornos, filtro_liquidez,
    calcular_perfil_por_meta, calcular_impacto_volatilidad, simular_crisis,
    comparar_instrumentos, obtener_dividend_yields_batch,
)
from optimizer import optimize_markowitz_lw, optimize_hrp, simulate_monte_carlo, smart_beta_filter
from config import log, CORS_ORIGINS, ENVIRONMENT, RATE_LIMIT_AUTH, RATE_LIMIT_ANON, CRON_SECRET
from auth import get_current_user, get_optional_user
from data_fetcher import get_historical_prices, get_benchmarks, DataFetchError
from ml_pipeline import ingest_daily_prices, train_and_predict

# ── Rate Limiter Setup ──────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Kaudal Portfolio API")
app.state.limiter = limiter


# Custom rate-limit error handler (Spanish)
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "⏱️ Has excedido el límite de solicitudes. Por favor, espera un momento antes de intentar de nuevo.",
            "retry_after": str(exc.detail),
        },
    )


# ── CORS — Solo orígenes permitidos ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# In-Memory Cache for Global Tickers
GLOBAL_TICKERS_CACHE = []
GLOBAL_TICKER_DETAILS = {}

# --- Stats Counter ---
STATS_FILE = os.path.join(os.path.dirname(__file__), "stats.json")


def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {"optimizations_count": 0}


def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)


def increment_optimization_count():
    stats = load_stats()
    stats["optimizations_count"] = stats.get("optimizations_count", 0) + 1
    save_stats(stats)
    return stats["optimizations_count"]


@app.on_event("startup")
def load_tickers_on_startup():
    global GLOBAL_TICKERS_CACHE, GLOBAL_TICKER_DETAILS

    # 1. Load base US Tickers from YAML
    try:
        with open("tickers.yaml", "r") as f:
            data = yaml.safe_load(f)
            GLOBAL_TICKERS_CACHE = data.get("tickers", [])
        log.info(f"Loaded {len(GLOBAL_TICKERS_CACHE)} base tickers from YAML.")
    except Exception as e:
        log.error(f"Failed to load tickers.yaml at startup: {e}")
        GLOBAL_TICKERS_CACHE = []

    # 2. Load enriched US and BMV tickers if available
    try:
        for json_file in ["bmv_tickers.json", "us_tickers.json"]:
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    ticker_list = file_data.get("tickers", [])

                    for t_obj in ticker_list:
                        symbol = t_obj["symbol"]
                        if symbol not in GLOBAL_TICKERS_CACHE:
                            GLOBAL_TICKERS_CACHE.append(symbol)

                        GLOBAL_TICKER_DETAILS[symbol] = {
                            "name": t_obj.get("name", symbol),
                            "sector": t_obj.get("sector", "N/A"),
                            "market": t_obj.get("market", "Unknown"),
                        }
                log.info(f"Loaded {len(ticker_list)} enriched tickers from {json_file}.")
    except Exception as e:
        log.error(f"Failed to load enriched ticker JSONs at startup: {e}")


# ════════════════════════════════════════════════════════════
#  PUBLIC ENDPOINTS (sin autenticación)
# ════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Endpoint público de salud — sin auth, sin rate limit."""
    return {"status": "ok", "environment": ENVIRONMENT}


@app.get("/api/example-fetch")
@limiter.limit("30/minute")
def example_fetch(request: Request, tickers: str = "AAPL,MSFT", period: str = "1y"):
    """Ejemplo de uso del nuevo módulo data_fetcher sin auth."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Debe proveer al menos 1 ticker.")
        
    try:
        data = get_historical_prices(ticker_list, period)
        benchmarks = get_benchmarks(period)
        
        # Convert DataFrames to JSON serializable dictionaries
        prices_json = {
            t: {date.strftime("%Y-%m-%d"): val for date, val in series.items()}
            for t, series in data["prices"].items()
        }
        returns_json = {
            t: {date.strftime("%Y-%m-%d"): val for date, val in series.items()}
            for t, series in data["returns"].items()
        }
        
        return {
            "status": "success",
            "historical_data": {
                "prices": prices_json,
                "returns": returns_json,
                "tickers": data["tickers"],
                "period": data["period"],
                "failed_tickers": data["failed_tickers"]
            },
            "benchmarks": benchmarks
        }
    except ValueError as e:
         raise HTTPException(status_code=400, detail=str(e))
    except DataFetchError as e:
        raise HTTPException(
            status_code=400, 
            detail={
                "message": str(e),
                "failed_tickers": e.failed_tickers
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@app.get("/api/validate_ticker/{ticker}")
@limiter.limit("30/minute")
def validate_ticker(request: Request, ticker: str):
    """Look up a ticker on Yahoo Finance and return its info for validation."""
    import yfinance as yf
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
        name = info.get("longName") or info.get("shortName") or ""
        sector = info.get("sector") or "N/A"
        exchange = info.get("exchange") or ""
        market_cap = info.get("marketCap")

        # If no name and no market cap, it's likely invalid
        if not name and not market_cap:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró información para '{ticker.upper()}' en Yahoo Finance.",
            )

        # Determine market label
        market = "MX" if ticker.upper().endswith(".MX") else "US"

        return {
            "valid": True,
            "ticker": ticker.upper(),
            "name": name,
            "sector": sector,
            "exchange": exchange,
            "market": market,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail=f"No se encontró el ticker '{ticker.upper()}'.")


@app.get("/api/tickers")
@limiter.limit("30/minute")
def get_tickers(request: Request):
    """Retorna la lista de tickers desde el caché en memoria junto con metadatos."""
    if not GLOBAL_TICKERS_CACHE:
        raise HTTPException(status_code=500, detail="Error leyendo el caché de tickers.")
    return {"tickers": GLOBAL_TICKERS_CACHE, "details": GLOBAL_TICKER_DETAILS}


@app.get("/api/stats")
@limiter.limit("30/minute")
def get_stats(request: Request):
    """Returns optimization counter and server time for freshness indicator."""
    stats = load_stats()
    return {
        "optimizations_count": stats.get("optimizations_count", 0),
        "server_time": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
#  PROTECTED ENDPOINTS (requieren autenticación)
# ════════════════════════════════════════════════════════════

# ── CRON JOBS (Background Tasks invoked by Railway) ──

def verify_cron_secret(authorization: str | None = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Header Authorization inválido o ausente.")
    token = authorization.split("Bearer ")[1]
    if token != CRON_SECRET:
        raise HTTPException(status_code=403, detail="CRON_SECRET inválido.")
    return True


@app.post("/api/cron/ingest", dependencies=[Depends(verify_cron_secret)])
def cron_ingest_prices():
    """Descarga los últimos precios diarios y los upsert en Supabase 'activos_precios'."""
    try:
        resultado = ingest_daily_prices()
        return resultado
    except Exception as e:
        log.error(f"Error en /api/cron/ingest: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingestando precios: {str(e)}")


@app.post("/api/cron/predict", dependencies=[Depends(verify_cron_secret)])
def cron_train_predict():
    """Entrena un modelo base de ML y predice los retornos mensuales en base al historico."""
    try:
        resultado = train_and_predict()
        return resultado
    except Exception as e:
        log.error(f"Error en /api/cron/predict: {e}")
        raise HTTPException(status_code=500, detail=f"Error entrenando modelo ML: {str(e)}")


class DreamsTestRequest(BaseModel):
    meta_costo: float
    años: int
    capital_inicial: float
    aporte_mensual: float


@app.post("/api/dreams_test")
@limiter.limit(RATE_LIMIT_AUTH)
def dreams_test(
    request: Request,
    req: DreamsTestRequest,
    user: dict = Depends(get_current_user),
):
    """Calcula la tasa anual requerida para lograr una meta, guiando a principiantes."""
    log.info(f"Dreams Test request from user={user['user_id']}")

    if req.meta_costo <= 0 or req.años <= 0:
        raise HTTPException(status_code=400, detail="El costo de la meta y los años deben ser mayores a 0.")

    resultado = calcular_perfil_por_meta(
        meta_costo=req.meta_costo,
        años=req.años,
        capital_inicial=req.capital_inicial,
        aporte_mensual=req.aporte_mensual,
    )

    if resultado["status"] == "impossible":
        return {
            "status": "impossible",
            "message": "Con estos parámetros, parece que no necesitas rendimiento (ya tienes el dinero) o los números no cuadran matemáticamente.",
            "tasa_objetivo_anual": 0.0,
        }

    tasa = resultado["tasa_objetivo_anual"]

    # Asignar un perfil sugerido basado en la tasa requerida
    if tasa < 4.0:
        perfil = "Conservador"
        desc = "¡Estás muy cerca de tu meta! Solo necesitas proteger tu dinero de la inflación."
    elif tasa < 8.0:
        perfil = "Moderado"
        desc = "Tienes una meta balanceada. Un portafolio mixto clásico te ayudará a llegar."
    elif tasa < 14.0:
        perfil = "Crecimiento"
        desc = "Para llegar a esta meta necesitas hacer crecer tu dinero activamente."
    elif tasa < 25.0:
        perfil = "Agresivo"
        desc = "Una meta ambiciosa. Necesitaremos buscar los activos con mayor crecimiento histórico asumiendo más volatilidad."
    else:
        perfil = "Muy Agresivo (O Imposible)"
        desc = "Atención: La tasa que necesitas es extremadamente alta. Te sugerimos aumentar tu aporte mensual o el tiempo de inversión."

    resultado["perfil_sugerido"] = perfil
    resultado["mensaje_perfil"] = desc

    return resultado


def normalize_currency(prices: pd.DataFrame, target_tickers: list[str]) -> pd.DataFrame:
    """Convierte los precios de activos mexicanos a USD usando el tipo de cambio."""
    mx_tickers = [t for t in target_tickers if t.endswith(".MX")]
    if mx_tickers and "USDMXN=X" in prices.columns:
        # FFill y BFill evitan NaNs al inicio o final de las series del FX
        usdmxn = prices["USDMXN=X"].ffill().bfill()
        for t in mx_tickers:
            if t in prices.columns:
                prices[t] = prices[t] / usdmxn
        # Evitar inplace drop en caso de views, y limpiar el FX del panel
        prices = prices.drop(columns=["USDMXN=X"])

    # Mantener sólo las columnas objetivo originales (en el orden original)
    valid_cols = [t for t in target_tickers if t in prices.columns]
    return prices[valid_cols]


class BacktestCrisisRequest(BaseModel):
    tickers: list[str]
    pesos: dict
    monto_inicial: float
    crisis: str


@app.post("/api/backtest_crisis")
@limiter.limit(RATE_LIMIT_AUTH)
def backtest_crisis(
    request: Request,
    req: BacktestCrisisRequest,
    user: dict = Depends(get_current_user),
):
    """Descarga los precios del periodo de crisis indicado y simula el portafolio."""
    log.info(f"Backtest crisis request from user={user['user_id']}, crisis={req.crisis}")

    if not req.tickers or not req.pesos:
        raise HTTPException(status_code=400, detail="Debe proveer tickers y pesos.")

    fechas = {
        "pandemia_2020": ("2020-01-01", "2020-12-31"),
        "crisis_2008": ("2007-08-01", "2009-12-31"),
    }

    if req.crisis not in fechas:
        raise HTTPException(status_code=400, detail="Crisis no soportada.")

    start_date, end_date = fechas[req.crisis]

    # Añadimos el benchmark para la comparativa
    download_tickers = req.tickers.copy()
    if "^GSPC" not in download_tickers:
        download_tickers.append("^GSPC")

    mx_tickers = [t for t in req.tickers if t.endswith(".MX")]
    if mx_tickers and "USDMXN=X" not in download_tickers:
        download_tickers.append("USDMXN=X")

    raw, fallidos = descargar_lotes(download_tickers, start=start_date, end=end_date)
    if raw.empty:
        raise HTTPException(
            status_code=500,
            detail="Fallo la descarga de datos (Dataframe vacío) para el periodo histórico de crisis.",
        )

    prices = extraer_panel(raw, download_tickers, "Adj Close")
    prices = normalize_currency(prices, req.tickers + ["^GSPC"])

    if prices.empty:
        raise HTTPException(status_code=400, detail="Panel de precios insuficiente.")

    # Llenado de vacíos
    idx_master = pd.date_range(prices.index.min(), prices.index.max(), freq="B")
    prices = prices.reindex(idx_master).ffill(limit=5)

    resultado = simular_crisis(prices, req.pesos, req.monto_inicial, req.crisis)

    if not resultado:
        raise HTTPException(
            status_code=400,
            detail="No se pudo simular la crisis (Los activos probablemente no existían en esa fecha).",
        )

    return {"status": "success", "crisis": req.crisis, "track_data": resultado}


class OptimizerRequest(BaseModel):
    tickers: list[str]
    budget: float
    period: str = "3y"
    max_weight: float = 0.25
    min_return: float | None = None
    max_volatility: float | None = None
    broker_commission: float = 0.0


@app.post("/api/optimizar")
@limiter.limit(RATE_LIMIT_AUTH)
async def optimize_portfolio_sse(
    request: Request,
    req: OptimizerRequest,
    user: dict = Depends(get_current_user),
):
    """Ejecuta la optimización de portafolios devolviendo estados vía SSE."""
    log.info(f"Optimize request from user={user['user_id']}, budget=${req.budget}")

    if len(req.tickers) < 2:
        raise HTTPException(status_code=400, detail="Debe proveer al menos 2 tickers.")
    if req.budget <= 0:
        raise HTTPException(status_code=400, detail="El presupuesto debe ser mayor a 0.")

    increment_optimization_count()

    async def event_generator():
        try:
            yield f'data: {{"stage": "downloading_data"}}\\n\\n'
            await asyncio.sleep(0.1)

            # 1. Fetch data
            data = await asyncio.to_thread(get_historical_prices, req.tickers, req.period)
            prices_df = data["prices"]
            returns_df = data["returns"]

            yield f'data: {{"stage": "cleaning_data"}}\\n\\n'
            await asyncio.sleep(0.1)

            # 2. Sanity Filters
            from optimizer import sanity_filters, optimize_markowitz, optimize_hrp, run_monte_carlo, calculate_positions
            
            filter_res = sanity_filters(returns_df)
            filtered_returns = filter_res["filtered_returns"]

            yield f'data: {{"stage": "calculating_covariance"}}\\n\\n'
            await asyncio.sleep(0.1)

            yield f'data: {{"stage": "optimizing"}}\\n\\n'
            await asyncio.sleep(0.1)

            constraints = {
                "max_weight": req.max_weight,
                "max_volatility": req.max_volatility,
                "min_return": req.min_return,
                "broker_commission": req.broker_commission
            }

            # 3. Math Optimization
            markowitz_res = await asyncio.to_thread(optimize_markowitz, filtered_returns, constraints)
            hrp_res = await asyncio.to_thread(optimize_hrp, filtered_returns, req.max_weight)
            mc_res = await asyncio.to_thread(run_monte_carlo, filtered_returns, 5000)

            yield f'data: {{"stage": "calculating_positions"}}\\n\\n'
            await asyncio.sleep(0.1)

            # 4. Positions
            last_prices = prices_df.ffill().iloc[-1]
            markowitz_alloc = calculate_positions(markowitz_res["weights"], req.budget, last_prices)
            hrp_alloc = calculate_positions(hrp_res["weights"], req.budget, last_prices)

            final_data = {
                "stage": "done",
                "result": {
                    "warnings": filter_res["messages"],
                    "excluded_tickers": filter_res["excluded_tickers"],
                    "markowitz": markowitz_res,
                    "markowitz_allocation": markowitz_alloc,
                    "hrp": hrp_res,
                    "hrp_allocation": hrp_alloc,
                    "monte_carlo": mc_res
                }
            }
            yield f'data: {json.dumps(final_data)}\\n\\n'
            
        except DataFetchError as e:
            err = {"stage": "error", "detail": str(e), "failed_tickers": e.failed_tickers}
            yield f'data: {json.dumps(err)}\\n\\n'
        except Exception as e:
            log.error(f"Error in optimizar: {e}", exc_info=True)
            err = {"stage": "error", "detail": str(e)}
            yield f'data: {json.dumps(err)}\\n\\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
