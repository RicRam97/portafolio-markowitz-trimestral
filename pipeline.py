# pipeline.py — Orquestador del pipeline trimestral
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import log, cargar_tickers, REPORTS_DIR, LOOKBACK_YEARS
from data import descargar_lotes, extraer_panel, limpiar_retornos, reporte_calidad
from optimizer import optimize_markowitz_lw, optimize_hrp, simulate_monte_carlo, smart_beta_filter
from pypfopt import expected_returns, risk_models
from report import generar_reporte_html


def pipeline_trimestral() -> str:
    """Ejecuta el pipeline completo: ingesta → optimización → reporte.

    Returns:
        Ruta del archivo HTML generado.

    Raises:
        RuntimeError: si el pipeline falla en algún paso crítico.
    """
    log.info("=" * 60)
    log.info("INICIO del pipeline trimestral")
    log.info("=" * 60)

    try:
        tickers = cargar_tickers()
        log.info("Tickers cargados: %d", len(tickers))

        # ── Ingesta ──────────────────────────────────────────
        start = (pd.Timestamp.today(tz="UTC") - pd.DateOffset(years=LOOKBACK_YEARS)).date()
        end = pd.Timestamp.today(tz="UTC").date()
        log.info("Ventana: %s → %s", start, end)

        raw, fallidos = descargar_lotes(tickers, start, end)
        if raw.empty:
            raise RuntimeError("La descarga de datos devolvió un DataFrame vacío")
        if fallidos:
            log.warning("Tickers fallidos (%d): %s", len(fallidos), ", ".join(fallidos))

        prices = extraer_panel(raw, tickers, "Close")
        if prices.empty or prices.shape[1] < 2:
            raise RuntimeError(
                f"Panel de precios insuficiente: {prices.shape[1]} tickers"
            )

        # Rellenar huecos pequeños (≤3 días hábiles)
        idx_master = pd.date_range(prices.index.min(), prices.index.max(), freq="B")
        prices = prices.reindex(idx_master).ffill(limit=3)

        # ── Limpieza ─────────────────────────────────────────
        # ── Limpieza y Filtrado ──────────────────────────────
        rets_clean = limpiar_retornos(prices)
        if rets_clean.shape[1] < 2:
            raise RuntimeError(
                f"Muy pocos tickers tras limpieza: {rets_clean.shape[1]}"
            )
            
        rets_clean = smart_beta_filter(rets_clean, strategy="momentum", top_n=20)

        # ── Estimación ───────────────────────────────────────
        mu_shrunk = expected_returns.mean_historical_return(rets_clean, returns_data=True, frequency=252)
        cov_annual = risk_models.CovarianceShrinkage(rets_clean, returns_data=True).ledoit_wolf()
        cov_annual = pd.DataFrame(cov_annual, index=rets_clean.columns, columns=rets_clean.columns)

        # ── Optimización ─────────────────────────────────────
        w_best, _, _, _ = optimize_markowitz_lw(rets_clean)
        w_rp, _, _, _ = optimize_hrp(rets_clean)
        ef = simulate_monte_carlo(rets_clean, num_portfolios=5000)

        # ── Calidad ──────────────────────────────────────────
        quality = reporte_calidad(prices, rets_clean)

        # ── Guardado ─────────────────────────────────────────
        timestamp = datetime.utcnow().strftime("%Y_%m_%d")
        run_dir = REPORTS_DIR / timestamp
        os.makedirs(run_dir, exist_ok=True)

        ef.to_csv(run_dir / "efficient_frontier.csv", index=False)
        w_best.to_csv(run_dir / "weights_best.csv", header=["weight"])
        mu_shrunk.to_csv(run_dir / "mu_shrunk.csv", header=["mu_shrunk"])
        cov_annual.to_csv(run_dir / "cov_shrunk.csv")
        quality.to_csv(run_dir / "quality_report.csv")
        w_rp.to_csv(run_dir / "weights_risk_parity.csv", header=["weight"])
        log.info("CSVs guardados en %s", run_dir)

        # ── Reporte HTML ─────────────────────────────────────
        report_path = str(run_dir / f"reporte_trimestral_{timestamp}.html")
        generar_reporte_html(ef, w_best, mu_shrunk, cov_annual, quality, w_rp, prices, report_path)

        log.info("=" * 60)
        log.info("Pipeline COMPLETADO — reporte: %s", report_path)
        log.info("=" * 60)
        return report_path

    except Exception:
        log.exception("Pipeline FALLIDO")
        raise


if __name__ == "__main__":
    try:
        path = pipeline_trimestral()
        print("Reporte listo en:", path)
    except Exception as exc:
        log.critical("Ejecución fallida: %s", exc)
        sys.exit(1)
