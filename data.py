# data.py — Ingesta, limpieza y calidad de datos
import pandas as pd
import numpy as np
import yfinance as yf

import time
from traceback import format_exc

import time
from traceback import format_exc

from config import log, BATCH_SIZE, COVERAGE_THRESHOLD, OUTLIER_SIGMA


def descargar_lotes(
    tickers: list[str],
    start,
    end,
    interval: str = "1d",
    lote: int = BATCH_SIZE,
) -> tuple[pd.DataFrame, list[str]]:
    """Descarga precios en lotes para evitar rate-limits de yfinance."""
    grupos = [tickers[i : i + lote] for i in range(0, len(tickers), lote)]
    frames: list[pd.DataFrame] = []
    fallidos: list[str] = []
    for idx, g in enumerate(grupos, 1):
        log.info("Descargando lote %d/%d (%d tickers)…", idx, len(grupos), len(g))
        df = yf.download(
            g,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if df.empty:
            log.warning("Lote %d devolvió vacío. Intentando procesar tickers individualmente...", idx)
            time.sleep(1)
            for single_ticker in g:
                try:
                    single_df = yf.download(
                        single_ticker,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=True,
                        progress=False,
                    )
                    if single_df.empty:
                        fallidos.append(single_ticker)
                    else:
                        # Para mantener el formato de MultiIndex (ticker, field) que genera el group_by="ticker"
                        single_df.columns = pd.MultiIndex.from_product([[single_ticker], single_df.columns])
                        frames.append(single_df)
                except Exception as e:
                    log.error("Fallo descargando %s: %s", single_ticker, e)
                    fallidos.append(single_ticker)
            continue
        frames.append(df)
        time.sleep(2)  # Pausa entre lotes exitosos
    if not frames:
        log.error("Todos los lotes fallaron — no hay datos. Posible bloqueo de IP por Yahoo Finance.")
        return pd.DataFrame(), tickers
    raw = pd.concat(frames, axis=1).sort_index()
    log.info(
        "Descarga completa: %d filas, %d columnas, %d fallidos",
        len(raw), raw.shape[1], len(fallidos),
    )
    return raw, fallidos


def extraer_panel(
    raw: pd.DataFrame,
    tickers: list[str],
    field: str = "Close",
) -> pd.DataFrame:
    """Extrae un panel de precios (una columna por ticker).

    Con auto_adjust=True, yfinance ya ajusta los precios en 'Close',
    por lo que 'Adj Close' no existe y no debe usarse.
    """
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


def limpiar_retornos(
    prices: pd.DataFrame,
    sigma: int = OUTLIER_SIGMA,
    cov_min: float = COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    """Calcula retornos, clipa outliers y filtra por cobertura."""
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
