# data.py — Ingesta, limpieza y calidad de datos
import pandas as pd

import time
import hashlib
import pickle
from pathlib import Path

from config import log, BATCH_SIZE, COVERAGE_THRESHOLD, OUTLIER_SIGMA

# Directorio para cache en disco
CACHE_DIR = Path(".cache_fmp")
CACHE_DIR.mkdir(exist_ok=True)


def descargar_lotes(
    tickers: list[str],
    start,
    end,
    interval: str = "1d",
    lote: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    """Descarga precios historicos via FMP, con cache en disco.

    Retorna un DataFrame con MultiIndex (ticker, field) de columnas
    y una lista de tickers fallidos.
    """
    from datetime import date as _date
    from data_fetcher.fetcher import _fetch_fmp_historical

    # 1. Attempt Cache Retrieval
    cache_key = hashlib.md5(
        f"{','.join(sorted(tickers))}_{start}_{end}_{interval}".encode()
    ).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.pkl"

    if cache_file.exists():
        if time.time() - cache_file.stat().st_mtime < 43200:  # 12 hrs
            log.info("Hit de Cache! Cargando datos desde disco...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    # Normalizar fechas a date objects
    if isinstance(start, str):
        start = _date.fromisoformat(start)
    if isinstance(end, str):
        end = _date.fromisoformat(end)

    frames: list[pd.DataFrame] = []
    fallidos: list[str] = []

    for idx, ticker in enumerate(tickers, 1):
        log.info("Descargando %d/%d: %s ...", idx, len(tickers), ticker)
        try:
            df = _fetch_fmp_historical(ticker, start, end)
            # Convertir a MultiIndex (ticker, field) para compatibilidad
            # _fetch_fmp_historical retorna index=date, col='close'
            # Necesitamos crear columnas: (TICKER, 'Close'), (TICKER, 'Adj Close')
            close_series = df["close"]
            ticker_df = pd.DataFrame({
                (ticker, "Close"): close_series,
                (ticker, "Adj Close"): close_series,
            })
            ticker_df.columns = pd.MultiIndex.from_tuples(ticker_df.columns)
            frames.append(ticker_df)
        except Exception as e:
            log.warning("Fallo descargando %s: %s", ticker, e)
            fallidos.append(ticker)

    if not frames:
        log.error("Todos los tickers fallaron — no hay datos.")
        return pd.DataFrame(), tickers

    raw = pd.concat(frames, axis=1).sort_index()
    log.info(
        "Descarga completa: %d filas, %d columnas, %d fallidos",
        len(raw), raw.shape[1], len(fallidos),
    )

    # 2. Save Cache
    with open(cache_file, "wb") as f:
        pickle.dump((raw, fallidos), f)

    return raw, fallidos


def extraer_panel(
    raw: pd.DataFrame,
    tickers: list[str],
    field: str = "Adj Close",
) -> pd.DataFrame:
    """Extrae un panel de precios (una columna por ticker)."""
    if raw.empty:
        return pd.DataFrame()

    # Caso: un solo ticker → columnas planas
    if not isinstance(raw.columns, pd.MultiIndex):
        t = tickers[0]
        if field in raw.columns:
            return raw[[field]].rename(columns={field: t})
        col0 = raw.columns[0]
        return raw[[col0]].rename(columns={col0: t})

    # Caso: MultiIndex (ticker, field)
    cols = []
    for t in tickers:
        if (t, field) in raw.columns:
            cols.append(raw[(t, field)].rename(t))
    if not cols:
        log.warning("No se encontró campo '%s' en ningún ticker", field)
        return pd.DataFrame()
    panel = pd.concat(cols, axis=1).sort_index()
    if panel.index.tz is not None:
        panel = panel.tz_localize(None)
    log.info("Panel extraído: %d filas × %d tickers", len(panel), panel.shape[1])
    return panel


def filtro_liquidez(prices_df: pd.DataFrame, volumes_df: pd.DataFrame, min_adv_usd: float = 1_000_000, days: int = 30) -> list[str]:
    """
    Filtra activos ilíquidos usando el Volumen Comercializado (Average Daily Volume en USD).
    Retorna la lista de tickers válidos.
    """
    if prices_df.empty or volumes_df.empty:
        return list(prices_df.columns)
        
    # Extraer ultimos N dias
    p_tail = prices_df.tail(days)
    v_tail = volumes_df.tail(days)
    
    # Calcular el valor transado en dolares por dia
    usd_volumes = p_tail * v_tail
    adv = usd_volumes.mean(skipna=True)
    
    valid_tickers = []
    dropped = []
    for t in p_tail.columns:
        if t == "^GSPC":
            valid_tickers.append(t) # El benchmark siempre pasa
            continue
            
        if t in adv and pd.notna(adv[t]) and adv[t] >= min_adv_usd:
            valid_tickers.append(t)
        else:
            dropped.append(t)
            
    if dropped:
        log.info(f"Filtro Liquidez (ADV < ${min_adv_usd/1e6:.1f}M): se eliminaron {len(dropped)} activos ({', '.join(dropped[:5])}...)")
        
    return valid_tickers

def limpiar_retornos(
    prices: pd.DataFrame,
    sigma: int = OUTLIER_SIGMA,
    cov_min: float = COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    """Calcula retornos, clipa outliers y filtra por cobertura. 
    Aplica relleno de NaNs robusto."""
    # Forward fill maximo 3 dias para mitigar gaps
    prices = prices.ffill(limit=3)
    
    returns_raw = prices.pct_change()
    sigma_lim = sigma * returns_raw.std(skipna=True)
    returns = returns_raw.clip(lower=-sigma_lim, upper=sigma_lim, axis=1)

    coverage = returns.notna().mean()
    keep = coverage[coverage >= cov_min].index
    dropped = set(returns.columns) - set(keep)
    if dropped:
        log.info(
            "Eliminados %d tickers por baja cobertura (<%.0f%%): %s",
            len(dropped), cov_min * 100, ", ".join(sorted(dropped)),
        )
    rets_clean = returns[keep].dropna(how="any")
    log.info(
        "Retornos limpios: %d filas × %d tickers", len(rets_clean), rets_clean.shape[1]
    )
    return rets_clean


def reporte_calidad(prices: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Genera un DataFrame con métricas de calidad por ticker."""
    coverage = prices.notna().sum() / len(prices)
    start_dates = prices.apply(lambda s: s.first_valid_index())
    end_dates = prices.apply(lambda s: s.last_valid_index())
    gaps = prices.isna().sum()
    # Solo incluir tickers presentes en rets
    common = rets.columns.intersection(prices.columns)
    return pd.DataFrame(
        {
            "coverage_%": (coverage * 100).round(2),
            "gaps": gaps,
            "start": start_dates,
            "end": end_dates,
            "mean_ret_daily_%": (rets[common].mean() * 100).round(3),
            "vol_daily_%": (rets[common].std() * 100).round(3),
        }
    ).sort_values("coverage_%")

import numpy_financial as npf

def calcular_perfil_por_meta(meta_costo: float, años: int, capital_inicial: float, aporte_mensual: float):
    """
    Calcula la tasa anual requerida (TIR) para lograr una meta financiera
    usando la fórmula de Valor Futuro.
    """
    meses = años * 12
    # npf.rate(nper, pmt, pv, fv)
    # pmt y pv son negativos porque es dinero que "sale" de nuestro bolsillo (inversión)
    # y el costo futuro es positivo (lo que queremos tener al final)
    tasa_mensual = npf.rate(meses, -aporte_mensual, -capital_inicial, meta_costo)
    
    # Manejar imposibles
    if pd.isna(tasa_mensual) or tasa_mensual < -1:
        return {"tasa_objetivo_anual": 0.0, "status": "impossible"}
        
    # Anualizar la tasa
    tasa_anual_requerida = ((1 + tasa_mensual)**12 - 1)
    
    return {
        "tasa_objetivo_anual": round(tasa_anual_requerida * 100, 2),
        "status": "success"
    }

import scipy.stats as stats

def calcular_impacto_volatilidad(monto_inversion: float, retorno_esperado: float, volatilidad: float):
    """
    Calcula la ganancia esperada y el Value at Risk (VaR) paramétrico al 99%
    de confianza en moneda real.
    """
    ganancia_esperada = monto_inversion * retorno_esperado
    
    # 99% Nivel de confianza para el peor escenario (cola izquierda de la normal)
    # Z-score aprox -2.326
    z_score_99 = stats.norm.ppf(0.01) 
    
    retorno_estres = retorno_esperado + (z_score_99 * volatilidad)
    perdida_potencial = monto_inversion * retorno_estres
    
    return {
        "ganancia_esperada": round(ganancia_esperada, 2),
        "perdida_peor_escenario": round(perdida_potencial, 2)
    }

def simular_crisis(df_precios: pd.DataFrame, pesos: dict, monto_inicial: float, crisis: str = 'pandemia_2020'):
    """
    Simula el comportamiento de un portafolio durante un periodo histórico de crisis.
    """
    fechas = {
        'pandemia_2020': ('2020-02-01', '2020-12-31'),
        'crisis_2008': ('2007-10-01', '2009-12-31')
    }
    inicio, fin = fechas.get(crisis, fechas['pandemia_2020'])
    
    # Aislar data y calcular retornos
    try:
        df_crisis = df_precios.loc[inicio:fin]
        if df_crisis.empty:
            return None
            
        rets_diarios = df_crisis.pct_change().dropna()
        
        # Filtrar solo los pesos de los que tenemos datos en ese momento
        # (ej. algunos tickers recientes no existían en 2008)
        tickers_validos = [t for t in pesos.keys() if t in rets_diarios.columns]
        if not tickers_validos:
            return None
            
        pesos_ajustados = {t: pesos[t] for t in tickers_validos}
        # Re-normalizar a 1 si faltan
        suma = sum(pesos_ajustados.values())
        if suma == 0:
            return None
        pesos_ajustados = {t: v/suma for t, v in pesos_ajustados.items()}
        
        ps = pd.Series(pesos_ajustados)
        rets_portafolio = rets_diarios[tickers_validos].dot(ps)
        
        # Evolución de valor
        evolucion = monto_inicial * (1 + rets_portafolio).cumprod()
        
        # Benchmark S&P 500 si existe
        if "^GSPC" in rets_diarios.columns:
            evolucion_bench = monto_inicial * (1 + rets_diarios["^GSPC"]).cumprod()
            benchmark_data = [round(val, 2) for val in evolucion_bench]
        else:
            benchmark_data = [monto_inicial] * len(evolucion) # Fallback plana
            
        return {
            "dates": [str(d.date()) for d in evolucion.index],
            "portfolio": [round(val, 2) for val in evolucion],
            "benchmark": benchmark_data
        }
    except Exception as e:
        log.error(f"Error simulando crisis {crisis}: {e}")
        return None

def comparar_instrumentos(monto_inversion: float, años: int, rendimiento_portafolio: float, tasa_cetes=0.10, inflacion=0.045):
    """
    Calcula el interés compuesto para proyecciones a largo plazo.
    """
    proyecciones = []
    
    # Agregar el año 0
    proyecciones.append({
        "anio": "Hoy",
        "portafolio": monto_inversion,
        "cetes": monto_inversion,
        "bajo_colchon": monto_inversion
    })
    
    for anio in range(1, años + 1):
        proyecciones.append({
            "anio": f"Año {anio}",
            "portafolio": round(monto_inversion * ((1 + rendimiento_portafolio) ** anio), 2),
            "cetes": round(monto_inversion * ((1 + tasa_cetes) ** anio), 2),
            "bajo_colchon": round(monto_inversion * ((1 - inflacion) ** anio), 2)
        })
        
    return {
        "anios": [p["anio"] for p in proyecciones],
        "portafolio": [p["portafolio"] for p in proyecciones],
        "cetes": [p["cetes"] for p in proyecciones],
        "inflacion": [p["bajo_colchon"] for p in proyecciones]
    }

def obtener_dividend_yields_batch(tickers: list[str]) -> dict:
    """
    Obtiene el dividend yield actual via FMP /profile batch.
    """
    import httpx
    from config import FMP_API_KEY, FMP_BASE_URL

    yields: dict[str, float] = {}
    if not tickers:
        return yields

    symbols = ",".join(tickers)
    url = f"{FMP_BASE_URL}/profile/{symbols}"
    try:
        resp = httpx.get(url, params={"apikey": FMP_API_KEY}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            for item in data:
                symbol = item.get("symbol", "")
                # lastDiv es el dividendo anual en USD; dividimos entre precio
                last_div = item.get("lastDiv", 0) or 0
                price = item.get("price", 0) or 0
                dy = (last_div / price) if price > 0 else 0
                yields[symbol] = float(dy)
    except Exception as e:
        log.error(f"Fallo obteniendo dividend yields de FMP: {e}")

    # Asegurar que todos los tickers tengan entrada
    for t in tickers:
        if t not in yields:
            yields[t] = 0.0

    return yields
