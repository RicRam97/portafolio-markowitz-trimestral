import os
import json
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
import asyncio
from pydantic import BaseModel, Field
from typing import Optional
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

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
from optimizer import optimize_markowitz, optimize_hrp, run_monte_carlo, smart_beta_filter, MarkowitzOptimizer, MonteCarloOptimizer, OptimizerError, calcular_acciones_y_efectivo, optimizar_efectivo_restante
from config import log, CORS_ORIGINS, ENVIRONMENT, RATE_LIMIT_AUTH, RATE_LIMIT_ANON, CRON_SECRET, get_supabase_client
from auth import get_current_user, get_optional_user
from data_fetcher import get_historical_prices, get_benchmarks, DataFetchError
from data_fetcher.fetcher import get_historical_prices_fmp
from ml_pipeline import ingest_daily_prices, train_and_predict
from error_codes import api_error

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
            "detail": {
                "code": "RATE_LIMITED",
                "message": "Has excedido el limite de solicitudes. Espera un momento antes de intentar de nuevo.",
            },
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


@app.get("/precio-actual/{ticker}")
@limiter.limit("60/minute")
def get_precio_actual(request: Request, ticker: str):
    """Retorna el precio actual de un ticker via FMP /quote."""
    import httpx
    from config import FMP_API_KEY, FMP_BASE_URL

    symbol = ticker.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Ticker vacio.")

    url = f"{FMP_BASE_URL}/quote/{symbol}"
    try:
        resp = httpx.get(url, params={"apikey": FMP_API_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data or not isinstance(data, list) or len(data) == 0:
            raise HTTPException(status_code=404, detail=f"No se encontro cotizacion para '{symbol}'.")

        quote = data[0]
        return {
            "ticker": symbol,
            "price": quote.get("price", 0),
            "name": quote.get("name", symbol),
            "change_pct": quote.get("changesPercentage", 0),
        }
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail=f"Error consultando precio de '{symbol}' en FMP.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error de red consultando FMP: {str(e)}")


@app.get("/precio-actual")
@limiter.limit("30/minute")
def get_precios_batch(request: Request, tickers: str = ""):
    """Retorna precios actuales de multiples tickers via FMP /quote (comma-separated)."""
    import httpx
    from config import FMP_API_KEY, FMP_BASE_URL

    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Debe proveer al menos 1 ticker.")
    if len(ticker_list) > 20:
        raise HTTPException(status_code=400, detail="Maximo 20 tickers por consulta.")

    symbols = ",".join(ticker_list)
    url = f"{FMP_BASE_URL}/quote/{symbols}"
    try:
        resp = httpx.get(url, params={"apikey": FMP_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            data = []

        result = {}
        for quote in data:
            sym = quote.get("symbol", "")
            result[sym] = {
                "ticker": sym,
                "price": quote.get("price", 0),
                "name": quote.get("name", sym),
                "change_pct": quote.get("changesPercentage", 0),
            }

        # Flag tickers not found
        for t in ticker_list:
            if t not in result:
                result[t] = {"ticker": t, "price": 0, "name": t, "change_pct": 0, "error": True}

        return {"prices": result}
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=502, detail="Error consultando precios en FMP.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Error de red consultando FMP: {str(e)}")


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
    """Look up a ticker on FMP and return its info for validation."""
    import httpx
    from config import FMP_API_KEY, FMP_BASE_URL

    symbol = ticker.strip().upper()
    url = f"{FMP_BASE_URL}/profile/{symbol}"
    try:
        resp = httpx.get(url, params={"apikey": FMP_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data or not isinstance(data, list) or len(data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontro informacion para '{symbol}' en FMP.",
            )

        info = data[0]
        name = info.get("companyName", "")
        sector = info.get("sector") or "N/A"
        exchange = info.get("exchangeShortName") or info.get("exchange", "")

        if not name:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontro informacion para '{symbol}' en FMP.",
            )

        market = "MX" if symbol.endswith(".MX") else "US"

        return {
            "valid": True,
            "ticker": symbol,
            "name": name,
            "sector": sector,
            "exchange": exchange,
            "market": market,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontro el ticker '{symbol}'.",
        )


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
        api_error(400, "OPTIMIZATION_MIN_TICKERS", "Debe proveer al menos 2 tickers.")
    if req.budget <= 0:
        api_error(400, "OPTIMIZATION_INVALID_BUDGET", "El presupuesto debe ser mayor a 0.")

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


# ════════════════════════════════════════════════════════════
#  POST /optimizar/markowitz — Endpoint dedicado Markowitz + FMP
# ════════════════════════════════════════════════════════════


def _fetch_current_prices_fmp(tickers: list[str]) -> dict[str, float]:
    """Obtiene precios actuales via FMP /quote batch. Retorna {ticker: price}."""
    import httpx
    from config import FMP_API_KEY, FMP_BASE_URL

    symbols = ",".join(tickers)
    url = f"{FMP_BASE_URL}/quote/{symbols}"
    try:
        resp = httpx.get(url, params={"apikey": FMP_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return {}
        return {q["symbol"]: q.get("price", 0) for q in data if "symbol" in q}
    except Exception as e:
        log.warning(f"No se pudieron obtener precios actuales de FMP: {e}")
        return {}


def _post_process_allocation(
    pesos_optimos: dict[str, float],
    tickers_incluidos: list[str],
    presupuesto: float | None,
    comision_broker: float,
    peso_maximo: float,
) -> dict | None:
    """Post-procesa los pesos optimos: calcula acciones, optimiza efectivo, genera warnings."""
    if presupuesto is None or presupuesto <= 0:
        return None

    precios = _fetch_current_prices_fmp(tickers_incluidos)
    if not precios:
        return None

    resultado = calcular_acciones_y_efectivo(
        pesos_optimos, precios, presupuesto, comision_broker
    )

    # Optimizar efectivo restante si > 5%
    if resultado["efectivo_restante"] > 0.05 * presupuesto:
        resultado["asignacion"], resultado["efectivo_restante"] = optimizar_efectivo_restante(
            resultado["asignacion"],
            resultado["efectivo_restante"],
            precios,
            presupuesto,
            max_weight=peso_maximo,
            comision_broker=comision_broker,
        )
        # Recalcular metricas
        resultado["inversion_total"] = round(
            sum(a["inversion"] for a in resultado["asignacion"].values()), 2
        )
        resultado["comisiones_totales"] = round(
            sum(a["comision"] for a in resultado["asignacion"].values()), 2
        )
        resultado["porcentaje_invertido"] = round(
            (resultado["inversion_total"] / presupuesto) * 100, 2
        )
        desviaciones = [
            abs(a["peso_real"] - a["peso_teorico"])
            for a in resultado["asignacion"].values()
        ]
        resultado["desviacion_maxima_peso"] = round(max(desviaciones) if desviaciones else 0.0, 6)

    # Warning si desviacion > 5%
    if resultado["desviacion_maxima_peso"] > 0.05:
        resultado["warning"] = (
            "Los pesos reales se desvian significativamente de los optimos "
            "debido al redondeo de acciones. Considera aumentar el presupuesto."
        )

    return resultado


class MarkowitzInput(BaseModel):
    tickers: list[str] = Field(..., min_length=2, description="Lista de simbolos, ej. ['AAPL', 'GOOGL', 'MSFT']")
    fecha_inicio: Optional[date] = Field(None, description="Fecha inicio (default: 3 anios atras)")
    fecha_fin: Optional[date] = Field(None, description="Fecha fin (default: hoy)")
    tasa_libre_riesgo: Optional[float] = Field(0.04, description="Tasa libre de riesgo anual (default 0.04)")
    peso_maximo: Optional[float] = Field(0.25, ge=0.05, le=1.0, description="Peso maximo por activo (default 0.25)")
    volatilidad_maxima: Optional[float] = Field(None, ge=0.01, le=1.0, description="Volatilidad maxima permitida del portafolio")
    presupuesto: Optional[float] = Field(None, ge=100, description="Presupuesto de inversion en USD (si se provee, calcula acciones)")
    comision_broker: Optional[float] = Field(0.0025, ge=0, le=0.1, description="Comision del broker (default 0.25%)")


class PortafolioOptimo(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class AsignacionReal(BaseModel):
    asignacion: dict[str, dict]
    efectivo_restante: float
    inversion_total: float
    comisiones_totales: float
    porcentaje_invertido: float
    desviacion_maxima_peso: float
    warning: Optional[str] = None


class MarkowitzOutput(BaseModel):
    portafolio_optimo: PortafolioOptimo
    frontera_eficiente: list[dict]
    tickers_incluidos: list[str]
    fecha_calculo: datetime
    asignacion_real: Optional[AsignacionReal] = None
    advertencias: list[str] = []
    compatible_con_perfil: bool = True


async def _validar_perfil_usuario(
    user_id: str,
    volatilidad_portafolio: float,
    retorno_esperado: float,
) -> tuple[list[str], bool]:
    """
    Consulta el perfil del inversor y compara con las metricas del portafolio.
    Retorna (advertencias, compatible_con_perfil).
    """
    advertencias: list[str] = []
    compatible = True

    try:
        sb = get_supabase_client()
        perfil_res = sb.table("perfil_combinado").select(
            "volatilidad_maxima, retorno_minimo"
        ).eq("user_id", user_id).order(
            "created_at", desc=True
        ).limit(1).execute()

        if not perfil_res.data:
            return advertencias, compatible

        perfil = perfil_res.data[0]
        vol_max = perfil.get("volatilidad_maxima")
        ret_min = perfil.get("retorno_minimo")

        if vol_max is not None and volatilidad_portafolio > vol_max * 1.1:
            exceso = ((volatilidad_portafolio / vol_max) - 1) * 100
            advertencias.append(
                f"La volatilidad del portafolio ({volatilidad_portafolio:.2%}) "
                f"excede tu tolerancia maxima ({vol_max:.2%}) por {exceso:.1f}%."
            )
            compatible = False

        if ret_min is not None and retorno_esperado < ret_min:
            advertencias.append(
                f"El retorno esperado ({retorno_esperado:.2%}) es menor que "
                f"tu retorno minimo requerido ({ret_min:.2%})."
            )
            compatible = False

    except Exception as e:
        log.warning(f"No se pudo validar perfil del usuario: {e}")

    return advertencias, compatible


@app.post("/optimizar/markowitz", response_model=MarkowitzOutput)
@limiter.limit(RATE_LIMIT_AUTH)
async def optimizar_markowitz_endpoint(
    request: Request,
    req: MarkowitzInput,
    user: dict = Depends(get_current_user),
):
    """Optimiza un portafolio Markowitz usando datos de FMP con cache en Supabase."""
    log.info(f"Markowitz request from user={user['user_id']}, tickers={req.tickers}")

    # Defaults de fechas
    fecha_fin = req.fecha_fin or date.today()
    fecha_inicio = req.fecha_inicio or date(fecha_fin.year - 3, fecha_fin.month, fecha_fin.day)
    tasa_libre_riesgo = req.tasa_libre_riesgo if req.tasa_libre_riesgo is not None else 0.04
    peso_maximo = req.peso_maximo if req.peso_maximo is not None else 0.25

    # Normalizar tickers
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    if len(tickers) < 2:
        api_error(422, "OPTIMIZATION_MIN_TICKERS", "Debe proveer al menos 2 tickers validos.")

    # ── 1. Obtener precios historicos (FMP + Cache-Aside) ─────────
    try:
        data = await asyncio.to_thread(
            get_historical_prices_fmp, tickers, fecha_inicio, fecha_fin
        )
    except DataFetchError as e:
        if e.failed_tickers:
            api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Tickers invalidos o sin datos: {', '.join(e.failed_tickers)}. {str(e)}")
        api_error(422, "DATA_FETCH_ERROR", str(e))

    log_returns = data["returns"]
    tickers_incluidos = data["tickers"]
    failed = data["failed_tickers"]

    # Validar datos suficientes (6 meses)
    if len(log_returns) < 126:
        api_error(422, "OPTIMIZATION_INSUFFICIENT_DATA", f"Datos insuficientes: solo {len(log_returns)} dias de historial. Se requieren al menos 126 (~6 meses).")

    if len(tickers_incluidos) < 2:
        api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Solo se obtuvieron datos para {len(tickers_incluidos)} ticker(s). Se necesitan al menos 2. Fallidos: {', '.join(failed)}")

    # ── 2. Optimizar ─────────────────────────────────────────────
    try:
        optimizer = MarkowitzOptimizer(
            log_returns=log_returns,
            tasa_libre_riesgo=tasa_libre_riesgo,
            peso_maximo=peso_maximo,
            volatilidad_maxima=req.volatilidad_maxima,
        )
        portafolio_optimo = await asyncio.to_thread(optimizer.optimize)
        frontera_eficiente = await asyncio.to_thread(optimizer.efficient_frontier, 50)
    except OptimizerError as e:
        api_error(500, "OPTIMIZATION_INFEASIBLE", f"La optimizacion no convergio. Intenta con otros tickers o parametros. Detalle: {str(e)}")

    fecha_calculo = datetime.now()

    # ── 3. Guardar en Supabase (portafolios_calculados, guardado=false) ──
    try:
        sb = get_supabase_client()
        sb.table("portafolios_calculados").insert({
            "user_id": user["user_id"],
            "tickers": tickers_incluidos,
            "parametros": {
                "modelo": "markowitz",
                "fecha_inicio": fecha_inicio.isoformat(),
                "fecha_fin": fecha_fin.isoformat(),
                "tasa_libre_riesgo": tasa_libre_riesgo,
                "peso_maximo": peso_maximo,
                "volatilidad_maxima": req.volatilidad_maxima,
            },
            "resultado": {
                "portafolio_optimo": portafolio_optimo,
                "frontera_eficiente": frontera_eficiente,
            },
            "guardado": False,
            "fecha_calculo": fecha_calculo.isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"No se pudo guardar en portafolios_calculados: {e}")

    increment_optimization_count()

    # ── 4. Post-procesamiento: calculo de acciones ────────────
    comision_broker = req.comision_broker if req.comision_broker is not None else 0.0025
    asignacion_real_data = None
    if req.presupuesto:
        asignacion_raw = await asyncio.to_thread(
            _post_process_allocation,
            portafolio_optimo["weights"],
            tickers_incluidos,
            req.presupuesto,
            comision_broker,
            peso_maximo,
        )
        if asignacion_raw:
            asignacion_real_data = AsignacionReal(**asignacion_raw)

    # ── 5. Validacion contra perfil del inversor ────────────
    advertencias, compatible = await _validar_perfil_usuario(
        user["user_id"],
        portafolio_optimo["volatility"],
        portafolio_optimo["expected_return"],
    )

    return MarkowitzOutput(
        portafolio_optimo=PortafolioOptimo(**portafolio_optimo),
        frontera_eficiente=frontera_eficiente,
        tickers_incluidos=tickers_incluidos,
        fecha_calculo=fecha_calculo,
        asignacion_real=asignacion_real_data,
        advertencias=advertencias,
        compatible_con_perfil=compatible,
    )


# ════════════════════════════════════════════════════════════
#  POST /optimizar/hrp — Hierarchical Risk Parity (Pro+)
# ════════════════════════════════════════════════════════════


class HRPInput(BaseModel):
    tickers: list[str] = Field(..., min_length=2, description="Lista de simbolos, ej. ['AAPL', 'GOOGL', 'MSFT']")
    fecha_inicio: Optional[date] = Field(None, description="Fecha inicio (default: 3 anios atras)")
    fecha_fin: Optional[date] = Field(None, description="Fecha fin (default: hoy)")
    peso_maximo: Optional[float] = Field(0.25, ge=0.05, le=1.0, description="Peso maximo por activo (default 0.25)")
    presupuesto: Optional[float] = Field(None, ge=100, description="Presupuesto de inversion en USD")
    comision_broker: Optional[float] = Field(0.0025, ge=0, le=0.1, description="Comision del broker (default 0.25%)")


class HRPOutput(BaseModel):
    portafolio_optimo: PortafolioOptimo
    clustering_data: Optional[list] = None
    tickers_incluidos: list[str]
    fecha_calculo: datetime
    asignacion_real: Optional[AsignacionReal] = None
    advertencias: list[str] = []
    compatible_con_perfil: bool = True


@app.post("/optimizar/hrp", response_model=HRPOutput)
@limiter.limit(RATE_LIMIT_AUTH)
async def optimizar_hrp_endpoint(
    request: Request,
    req: HRPInput,
    user: dict = Depends(get_current_user),
):
    """Optimiza un portafolio con Hierarchical Risk Parity (exclusivo plan Pro y Ultra)."""

    # ── Verificar tier Pro o Ultra ────────────────────────────
    user_plan = user.get("plan", "basico")
    if user_plan not in ("pro", "ultra"):
        api_error(403, "PLAN_UPGRADE_REQUIRED", "HRP es exclusivo de los planes Pro y Ultra. Actualiza tu plan para desbloquear esta funcionalidad.")

    log.info(f"HRP request from user={user['user_id']}, tickers={req.tickers}")

    # Defaults
    fecha_fin = req.fecha_fin or date.today()
    fecha_inicio = req.fecha_inicio or date(fecha_fin.year - 3, fecha_fin.month, fecha_fin.day)
    peso_maximo = req.peso_maximo if req.peso_maximo is not None else 0.25

    # Normalizar tickers
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    if len(tickers) < 2:
        api_error(422, "OPTIMIZATION_MIN_TICKERS", "Debe proveer al menos 2 tickers validos.")

    # ── 1. Obtener precios historicos (FMP + Cache-Aside) ─────
    try:
        data = await asyncio.to_thread(
            get_historical_prices_fmp, tickers, fecha_inicio, fecha_fin
        )
    except DataFetchError as e:
        if e.failed_tickers:
            api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Tickers invalidos o sin datos: {', '.join(e.failed_tickers)}. {str(e)}")
        api_error(422, "DATA_FETCH_ERROR", str(e))

    log_returns = data["returns"]
    tickers_incluidos = data["tickers"]
    failed = data["failed_tickers"]

    if len(log_returns) < 126:
        api_error(422, "OPTIMIZATION_INSUFFICIENT_DATA", f"Datos insuficientes: solo {len(log_returns)} dias de historial. Se requieren al menos 126 (~6 meses).")

    if len(tickers_incluidos) < 2:
        api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Solo se obtuvieron datos para {len(tickers_incluidos)} ticker(s). Se necesitan al menos 2. Fallidos: {', '.join(failed)}")

    # ── 2. Optimizar HRP ──────────────────────────────────────
    try:
        result = await asyncio.to_thread(optimize_hrp, log_returns, peso_maximo)
    except OptimizerError as e:
        api_error(500, "OPTIMIZATION_INFEASIBLE", f"La optimizacion HRP no convergio. Intenta con otros tickers o parametros. Detalle: {str(e)}")

    fecha_calculo = datetime.now()

    portafolio_optimo = {
        "weights": result["weights"],
        "expected_return": round(result["expected_return"], 6),
        "volatility": round(result["expected_volatility"], 6),
        "sharpe_ratio": round(result["sharpe_ratio"], 4),
    }

    # ── 3. Guardar en Supabase ────────────────────────────────
    try:
        sb = get_supabase_client()
        sb.table("portafolios_calculados").insert({
            "user_id": user["user_id"],
            "tickers": tickers_incluidos,
            "parametros": {
                "modelo": "hrp",
                "fecha_inicio": fecha_inicio.isoformat(),
                "fecha_fin": fecha_fin.isoformat(),
                "peso_maximo": peso_maximo,
            },
            "resultado": {
                "portafolio_optimo": portafolio_optimo,
                "clustering_data": result["clustering_data"],
            },
            "guardado": False,
            "fecha_calculo": fecha_calculo.isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"No se pudo guardar en portafolios_calculados: {e}")

    increment_optimization_count()

    # ── 4. Post-procesamiento: calculo de acciones ────────────
    comision_broker = req.comision_broker if req.comision_broker is not None else 0.0025
    asignacion_real_data = None
    if req.presupuesto:
        asignacion_raw = await asyncio.to_thread(
            _post_process_allocation,
            portafolio_optimo["weights"],
            tickers_incluidos,
            req.presupuesto,
            comision_broker,
            peso_maximo,
        )
        if asignacion_raw:
            asignacion_real_data = AsignacionReal(**asignacion_raw)

    # ── 5. Validacion contra perfil del inversor ────────────
    advertencias, compatible = await _validar_perfil_usuario(
        user["user_id"],
        portafolio_optimo["volatility"],
        portafolio_optimo["expected_return"],
    )

    return HRPOutput(
        portafolio_optimo=PortafolioOptimo(**portafolio_optimo),
        clustering_data=result["clustering_data"],
        tickers_incluidos=tickers_incluidos,
        fecha_calculo=fecha_calculo,
        asignacion_real=asignacion_real_data,
        advertencias=advertencias,
        compatible_con_perfil=compatible,
    )


# ════════════════════════════════════════════════════════════
#  POST /optimizar/montecarlo — Simulacion Monte Carlo (Ultra)
# ════════════════════════════════════════════════════════════


class MonteCarloInput(BaseModel):
    tickers: list[str] = Field(..., min_length=2, description="Lista de simbolos, ej. ['AAPL', 'GOOGL', 'MSFT']")
    num_simulaciones: Optional[int] = Field(10000, ge=1000, le=50000, description="Numero de portafolios a simular")
    fecha_inicio: Optional[date] = Field(None, description="Fecha inicio (default: 3 anios atras)")
    fecha_fin: Optional[date] = Field(None, description="Fecha fin (default: hoy)")
    tasa_libre_riesgo: Optional[float] = Field(0.04, description="Tasa libre de riesgo anual")
    peso_maximo: Optional[float] = Field(0.30, ge=0.05, le=1.0, description="Peso maximo por activo")
    presupuesto: Optional[float] = Field(None, ge=100, description="Presupuesto de inversion en USD")
    comision_broker: Optional[float] = Field(0.0025, ge=0, le=0.1, description="Comision del broker (default 0.25%)")


class MonteCarloCloud(BaseModel):
    num_portfolios: int
    cloud: list[dict]


class MonteCarloOutput(BaseModel):
    portafolio_optimo: PortafolioOptimo
    portafolio_min_vol: PortafolioOptimo
    simulacion: MonteCarloCloud
    tickers_incluidos: list[str]
    fecha_calculo: datetime
    asignacion_real: Optional[AsignacionReal] = None
    advertencias: list[str] = []
    compatible_con_perfil: bool = True


@app.post("/optimizar/montecarlo", response_model=MonteCarloOutput)
@limiter.limit(RATE_LIMIT_AUTH)
async def optimizar_montecarlo_endpoint(
    request: Request,
    req: MonteCarloInput,
    user: dict = Depends(get_current_user),
):
    """Optimiza un portafolio via simulacion Monte Carlo (exclusivo plan Ultra)."""

    # ── Verificar tier Ultra ──────────────────────────────────
    user_plan = user.get("plan", "basico")
    if user_plan != "ultra":
        api_error(403, "PLAN_UPGRADE_REQUIRED", "Monte Carlo es exclusivo del plan Ultra. Actualiza tu plan para desbloquear esta funcionalidad.")

    log.info(f"Monte Carlo request from user={user['user_id']}, tickers={req.tickers}, n={req.num_simulaciones}")

    # Defaults
    fecha_fin = req.fecha_fin or date.today()
    fecha_inicio = req.fecha_inicio or date(fecha_fin.year - 3, fecha_fin.month, fecha_fin.day)
    tasa_libre_riesgo = req.tasa_libre_riesgo if req.tasa_libre_riesgo is not None else 0.04
    peso_maximo = req.peso_maximo if req.peso_maximo is not None else 0.30
    num_simulaciones = req.num_simulaciones if req.num_simulaciones is not None else 10000

    # Normalizar tickers
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    if len(tickers) < 2:
        api_error(422, "OPTIMIZATION_MIN_TICKERS", "Debe proveer al menos 2 tickers validos.")

    # ── 1. Obtener precios historicos (FMP + Cache-Aside) ─────
    try:
        data = await asyncio.to_thread(
            get_historical_prices_fmp, tickers, fecha_inicio, fecha_fin
        )
    except DataFetchError as e:
        if e.failed_tickers:
            api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Tickers invalidos o sin datos: {', '.join(e.failed_tickers)}. {str(e)}")
        api_error(422, "DATA_FETCH_ERROR", str(e))

    log_returns = data["returns"]
    tickers_incluidos = data["tickers"]
    failed = data["failed_tickers"]

    if len(log_returns) < 126:
        api_error(422, "OPTIMIZATION_INSUFFICIENT_DATA", f"Datos insuficientes: solo {len(log_returns)} dias de historial. Se requieren al menos 126 (~6 meses).")

    if len(tickers_incluidos) < 2:
        api_error(422, "OPTIMIZATION_INVALID_TICKERS", f"Solo se obtuvieron datos para {len(tickers_incluidos)} ticker(s). Se necesitan al menos 2. Fallidos: {', '.join(failed)}")

    # ── 2. Simulacion Monte Carlo ─────────────────────────────
    try:
        optimizer = MonteCarloOptimizer(
            log_returns=log_returns,
            num_portfolios=num_simulaciones,
            tasa_libre_riesgo=tasa_libre_riesgo,
            peso_maximo=peso_maximo,
        )
        result = await asyncio.to_thread(optimizer.optimize)
    except Exception as e:
        log.error(f"Error en Monte Carlo: {e}", exc_info=True)
        api_error(500, "OPTIMIZATION_INFEASIBLE", f"Error en la simulacion Monte Carlo: {str(e)}")

    fecha_calculo = datetime.now()

    # ── 3. Guardar en Supabase ────────────────────────────────
    try:
        sb = get_supabase_client()
        sb.table("portafolios_calculados").insert({
            "user_id": user["user_id"],
            "tickers": tickers_incluidos,
            "parametros": {
                "modelo": "montecarlo",
                "fecha_inicio": fecha_inicio.isoformat(),
                "fecha_fin": fecha_fin.isoformat(),
                "tasa_libre_riesgo": tasa_libre_riesgo,
                "peso_maximo": peso_maximo,
                "num_simulaciones": num_simulaciones,
            },
            "resultado": {
                "portafolio_optimo": result["portafolio_optimo"],
                "portafolio_min_vol": result["portafolio_min_vol"],
            },
            "guardado": False,
            "fecha_calculo": fecha_calculo.isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"No se pudo guardar en portafolios_calculados: {e}")

    increment_optimization_count()

    # ── 4. Post-procesamiento: calculo de acciones ────────────
    comision_broker = req.comision_broker if req.comision_broker is not None else 0.0025
    asignacion_real_data = None
    if req.presupuesto:
        asignacion_raw = await asyncio.to_thread(
            _post_process_allocation,
            result["portafolio_optimo"]["weights"],
            tickers_incluidos,
            req.presupuesto,
            comision_broker,
            peso_maximo,
        )
        if asignacion_raw:
            asignacion_real_data = AsignacionReal(**asignacion_raw)

    # ── 5. Validacion contra perfil del inversor ────────────
    advertencias, compatible = await _validar_perfil_usuario(
        user["user_id"],
        result["portafolio_optimo"]["volatility"],
        result["portafolio_optimo"]["expected_return"],
    )

    return MonteCarloOutput(
        portafolio_optimo=PortafolioOptimo(**result["portafolio_optimo"]),
        portafolio_min_vol=PortafolioOptimo(**result["portafolio_min_vol"]),
        simulacion=MonteCarloCloud(**result["simulacion"]),
        tickers_incluidos=tickers_incluidos,
        fecha_calculo=fecha_calculo,
        asignacion_real=asignacion_real_data,
        advertencias=advertencias,
        compatible_con_perfil=compatible,
    )


# ════════════════════════════════════════════════════════════
#  GET /estrategias/{id}/check-drift — Drift / rebalanceo
# ════════════════════════════════════════════════════════════


class DriftItem(BaseModel):
    ticker: str
    peso_objetivo: float
    peso_actual: float
    desviacion: float
    necesita_rebalanceo: bool


class DriftOutput(BaseModel):
    estrategia_id: str
    necesita_rebalanceo: bool
    drift: list[DriftItem]


@app.get(
    "/estrategias/{estrategia_id}/check-drift",
    response_model=DriftOutput,
)
@limiter.limit(RATE_LIMIT_AUTH)
async def check_drift(
    request: Request,
    estrategia_id: str,
    user: dict = Depends(get_current_user),
):
    """Calcula la desviacion (drift) entre los pesos objetivo y los
    pesos actuales de una estrategia, usando precios de FMP."""
    sb = get_supabase_client()
    user_id = user["user_id"]

    # 1. Obtener la estrategia y verificar que pertenece al usuario
    est_res = (
        sb.table("estrategias")
        .select("id, parametros")
        .eq("id", estrategia_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not est_res.data:
        raise HTTPException(
            status_code=404,
            detail="Estrategia no encontrada o no pertenece al usuario.",
        )

    # 2. Obtener el portafolio mas reciente vinculado a la estrategia
    port_res = (
        sb.table("portafolios")
        .select("allocation")
        .eq("estrategia_id", estrategia_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not port_res.data or not port_res.data[0].get("allocation"):
        raise HTTPException(
            status_code=404,
            detail="No hay portafolios guardados para esta estrategia.",
        )

    allocation = port_res.data[0]["allocation"]
    # allocation: [{"ticker":"AAPL","weight_pct":35,"shares":10}, ...]

    tickers = [a["ticker"] for a in allocation if a.get("shares", 0) > 0]
    shares_map = {
        a["ticker"]: a["shares"] for a in allocation if a.get("shares", 0) > 0
    }
    weight_target = {
        a["ticker"]: a["weight_pct"] / 100.0
        for a in allocation
        if a.get("shares", 0) > 0
    }

    if not tickers:
        raise HTTPException(
            status_code=422,
            detail="La asignacion no contiene acciones compradas.",
        )

    # 3. Obtener precios actuales via FMP
    precios = await asyncio.to_thread(_fetch_current_prices_fmp, tickers)

    if not precios:
        raise HTTPException(
            status_code=502,
            detail="No se obtuvieron precios de cierre para los tickers.",
        )

    # 4. Calcular peso actual de cada activo
    valor_total = sum(
        shares_map[t] * precios[t]
        for t in tickers
        if t in precios
    )

    if valor_total <= 0:
        raise HTTPException(
            status_code=422,
            detail="El valor total del portafolio es cero.",
        )

    drift_items: list[DriftItem] = []
    rebalanceo_global = False

    for t in tickers:
        if t not in precios:
            continue
        peso_actual = (shares_map[t] * precios[t]) / valor_total
        peso_obj = weight_target.get(t, 0.0)
        desviacion = abs(peso_actual - peso_obj)
        necesita = desviacion > 0.05

        if necesita:
            rebalanceo_global = True

        drift_items.append(DriftItem(
            ticker=t,
            peso_objetivo=round(peso_obj, 6),
            peso_actual=round(peso_actual, 6),
            desviacion=round(desviacion, 6),
            necesita_rebalanceo=necesita,
        ))

    # Ordenar por mayor desviacion primero
    drift_items.sort(key=lambda d: d.desviacion, reverse=True)

    return DriftOutput(
        estrategia_id=estrategia_id,
        necesita_rebalanceo=rebalanceo_global,
        drift=drift_items,
    )


# ── PDF Report Generation (reportlab) ──────────────────────────
@app.get("/estrategias/{estrategia_id}/export-pdf")
@limiter.limit(RATE_LIMIT_AUTH)
async def export_estrategia_pdf(
    request: Request,
    estrategia_id: str,
    user: dict = Depends(get_current_user),
):
    """Genera un reporte PDF de la estrategia y su portafolio vigente."""
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    sb = get_supabase_client()
    user_id = user["user_id"]

    # ── 1. Consultar estrategia ──
    est_res = (
        sb.table("estrategias")
        .select("id, nombre, tipo, parametros, created_at")
        .eq("id", estrategia_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not est_res.data:
        raise HTTPException(
            status_code=404,
            detail="Estrategia no encontrada o no pertenece al usuario.",
        )
    est = est_res.data

    # ── 2. Consultar portafolio vigente (mas reciente) ──
    port_res = (
        sb.table("portafolios")
        .select(
            "presupuesto, rendimiento_pct, volatilidad_pct, "
            "sharpe_ratio, allocation, created_at"
        )
        .eq("estrategia_id", estrategia_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not port_res.data:
        raise HTTPException(
            status_code=404,
            detail="No hay portafolios guardados para esta estrategia.",
        )
    port = port_res.data[0]
    allocation = port.get("allocation", [])

    # ── 3. Generar PDF con reportlab ──
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.5 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    style_title = ParagraphStyle(
        "PDFTitle", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "PDFSubtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER, spaceAfter=16,
    )
    style_section = ParagraphStyle(
        "PDFSection", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#2563eb"),
        spaceBefore=14, spaceAfter=6,
        borderWidth=0,
    )
    style_body = ParagraphStyle(
        "PDFBody", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#374151"),
        spaceAfter=2,
    )
    style_footer = ParagraphStyle(
        "PDFFooter", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#9ca3af"),
        alignment=TA_CENTER, spaceBefore=20,
    )

    story = []

    # ── Header ──
    nombre = est.get("nombre", "Sin nombre")
    story.append(Paragraph(
        f"Reporte de Estrategia: {nombre}", style_title
    ))
    tipo_label = "Markowitz" if est.get("tipo") == "markowitz" else "HRP"
    story.append(Paragraph(
        f"Modelo: {tipo_label}", style_subtitle
    ))
    story.append(Spacer(1, 8))

    # ── Seccion 1: Resumen General ──
    story.append(Paragraph("Resumen General", style_section))

    presupuesto = port.get("presupuesto", 0) or 0
    moneda = "USD"
    fecha_creacion = est.get("created_at", "")[:10]
    estado = "Vigente"

    resumen_data = [
        ["Campo", "Valor"],
        ["Monto Inicial", f"${presupuesto:,.2f} {moneda}"],
        ["Moneda", moneda],
        ["Fecha de Creacion", fecha_creacion],
        ["Estado", estado],
    ]
    t_resumen = Table(resumen_data, colWidths=[2.5 * inch, 4 * inch])
    t_resumen.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f5ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_resumen)
    story.append(Spacer(1, 10))

    # ── Seccion 2: Metricas del Portafolio ──
    story.append(Paragraph("Metricas del Portafolio", style_section))

    sharpe = port.get("sharpe_ratio") or 0
    ret_pct = port.get("rendimiento_pct") or 0
    vol_pct = port.get("volatilidad_pct") or 0

    metricas_data = [
        ["Metrica", "Valor"],
        ["Sharpe Ratio", f"{sharpe:.4f}"],
        ["Retorno Esperado (anual)", f"{ret_pct:.2f}%"],
        ["Volatilidad (anual)", f"{vol_pct:.2f}%"],
    ]
    t_metricas = Table(metricas_data, colWidths=[2.5 * inch, 4 * inch])
    t_metricas.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f5ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_metricas)
    story.append(Spacer(1, 10))

    # ── Seccion 3: Tabla de Composicion ──
    story.append(Paragraph("Composicion del Portafolio", style_section))

    comp_data = [["Ticker", "Peso (%)", "Acciones Sugeridas", "Precio Referencia"]]
    for item in allocation:
        ticker = item.get("ticker", "")
        peso = item.get("weight_pct", 0)
        acciones = item.get("shares", 0)
        precio = item.get("precio_compra", "-")
        precio_str = (
            f"${precio:,.2f}" if isinstance(precio, (int, float)) else "-"
        )
        comp_data.append([ticker, f"{peso:.2f}%", str(acciones), precio_str])

    col_w = [1.4 * inch, 1.3 * inch, 1.8 * inch, 1.8 * inch]
    t_comp = Table(comp_data, colWidths=col_w)
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#374151")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f5ff")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_comp)

    # ── Footer ──
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(
        f"Generado el {now_str}", style_footer
    ))
    story.append(Paragraph(
        "Este reporte es solo informativo. No constituye asesoria financiera.",
        style_footer,
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="reporte_{nombre.replace(" ", "_")}.pdf"'
            ),
        },
    )


# ── Strategy Sharing ───────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


@app.post("/estrategias/{estrategia_id}/generar-link-compartido")
@limiter.limit(RATE_LIMIT_AUTH)
async def generar_link_compartido(
    request: Request,
    estrategia_id: str,
    user: dict = Depends(get_current_user),
):
    """Genera un token unico y habilita la estrategia como publica."""
    sb = get_supabase_client()
    user_id = user["user_id"]

    # Verificar propiedad
    est_res = (
        sb.table("estrategias")
        .select("id, compartir_token")
        .eq("id", estrategia_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if not est_res.data:
        raise HTTPException(
            status_code=404,
            detail="Estrategia no encontrada o no pertenece al usuario.",
        )

    # Reusar token existente si ya tiene uno
    existing_token = est_res.data.get("compartir_token")
    token = existing_token or str(uuid.uuid4())

    # Actualizar en Supabase
    sb.table("estrategias").update({
        "compartir_token": token,
        "es_publica": True,
    }).eq("id", estrategia_id).execute()

    link = f"{FRONTEND_URL}/estrategias/compartidas/{token}"
    return {"token": token, "link": link}


class SharedStrategyOutput(BaseModel):
    nombre: str
    tipo: str
    parametros: dict
    created_at: str
    portafolio: Optional[dict] = None


@app.get(
    "/estrategias/compartidas/{token}",
    response_model=SharedStrategyOutput,
)
async def ver_estrategia_compartida(
    request: Request,
    token: str,
):
    """Endpoint publico (sin auth) para ver una estrategia compartida."""
    sb = get_supabase_client()

    est_res = (
        sb.table("estrategias")
        .select("id, nombre, tipo, parametros, created_at")
        .eq("compartir_token", token)
        .eq("es_publica", True)
        .single()
        .execute()
    )
    if not est_res.data:
        raise HTTPException(
            status_code=404,
            detail="Estrategia no encontrada o no es publica.",
        )

    est = est_res.data

    # Obtener portafolio vigente (sin datos del propietario)
    port_res = (
        sb.table("portafolios")
        .select(
            "presupuesto, rendimiento_pct, volatilidad_pct, "
            "sharpe_ratio, allocation"
        )
        .eq("estrategia_id", est["id"])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    portafolio = port_res.data[0] if port_res.data else None

    return SharedStrategyOutput(
        nombre=est["nombre"],
        tipo=est["tipo"],
        parametros=est.get("parametros", {}),
        created_at=est["created_at"],
        portafolio=portafolio,
    )


class CloneInput(BaseModel):
    token: str


@app.post("/estrategias/clonar")
@limiter.limit(RATE_LIMIT_AUTH)
async def clonar_estrategia(
    request: Request,
    req: CloneInput,
    user: dict = Depends(get_current_user),
):
    """Clona una estrategia compartida a la cuenta del usuario autenticado."""
    sb = get_supabase_client()
    user_id = user["user_id"]

    # Buscar la estrategia publica
    est_res = (
        sb.table("estrategias")
        .select("nombre, tipo, parametros")
        .eq("compartir_token", req.token)
        .eq("es_publica", True)
        .single()
        .execute()
    )
    if not est_res.data:
        raise HTTPException(
            status_code=404,
            detail="Estrategia compartida no encontrada.",
        )

    src = est_res.data

    # Crear nueva estrategia independiente
    new_est = (
        sb.table("estrategias")
        .insert({
            "user_id": user_id,
            "nombre": f"{src['nombre']} (clon)",
            "tipo": src["tipo"],
            "parametros": src.get("parametros", {}),
        })
        .execute()
    )
    if not new_est.data:
        raise HTTPException(
            status_code=500,
            detail="Error al clonar la estrategia.",
        )

    return {
        "id": new_est.data[0]["id"],
        "nombre": new_est.data[0]["nombre"],
        "message": "Estrategia clonada exitosamente.",
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
