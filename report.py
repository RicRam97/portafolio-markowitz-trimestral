# report.py — Generación del reporte HTML trimestral
import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import log, RF


def _fig_to_b64(fig) -> str:
    """Convierte una figura matplotlib a base64 PNG."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _chart_frontera(ef: pd.DataFrame, vol_opt: float, ret_opt: float) -> str:
    """Gráfica de la frontera eficiente con punto óptimo."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(ef["vol"], ef["ret"], c=ef["sharpe"], cmap="viridis", s=30)
    fig.colorbar(sc, ax=ax, label="Sharpe")
    ax.scatter([vol_opt], [ret_opt], color="red", marker="x", s=140, zorder=5, label="Mejor Sharpe")
    ax.set_xlabel("Volatilidad anual")
    ax.set_ylabel("Retorno anual")
    ax.set_title("Frontera Eficiente")
    ax.legend()
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def _chart_correlacion(cov: pd.DataFrame, top_tickers: list[str]) -> str:
    """Heatmap de correlación de los top holdings."""
    sub = cov.loc[top_tickers, top_tickers]
    # Convertir covarianza → correlación
    vols = np.sqrt(np.diag(sub.values))
    corr = sub.values / np.outer(vols, vols + 1e-12)
    corr = pd.DataFrame(corr, index=top_tickers, columns=top_tickers)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="Correlación")
    ax.set_xticks(range(len(top_tickers)))
    ax.set_xticklabels(top_tickers, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_tickers)))
    ax.set_yticklabels(top_tickers, fontsize=7)
    ax.set_title("Correlación — Top 20 Holdings")
    return _fig_to_b64(fig)


def _chart_drawdown(prices: pd.DataFrame, w_best: pd.Series) -> str:
    """Drawdown del portafolio óptimo en el período histórico."""
    common = w_best.index.intersection(prices.columns)
    w = w_best.reindex(common).fillna(0)
    w = w / w.sum()  # renormalizar
    portfolio = (prices[common] * w).sum(axis=1).dropna()
    cummax = portfolio.cummax()
    dd = (portfolio - cummax) / (cummax + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.plot(dd.index, dd.values, color="crimson", linewidth=0.8)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown del Portafolio Óptimo")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    return _fig_to_b64(fig)


def _metricas_portafolio(w: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, rf: float = RF):
    """Retorno, volatilidad y Sharpe de un vector de pesos."""
    ret = float(np.dot(w, mu))
    vol = float(np.sqrt(np.dot(w, Sigma @ w)))
    sh = (ret - rf) / (vol + 1e-12)
    return ret, vol, sh


def generar_reporte_html(
    ef: pd.DataFrame,
    w_best: pd.Series,
    mu_shrunk: pd.Series,
    cov: pd.DataFrame,
    quality: pd.DataFrame,
    w_rp: pd.Series,
    prices: pd.DataFrame,
    filename: str,
) -> None:
    """Genera el reporte HTML completo y lo guarda en `filename`."""
    assets = list(mu_shrunk.index)
    Sigma = cov.reindex(index=assets, columns=assets).values
    mu = mu_shrunk.reindex(assets).values
    w = w_best.reindex(assets).fillna(0).values
    n = len(assets)

    # Métricas: óptimo, equiponderado, SPY, risk-parity
    ret_opt, vol_opt, sh_opt = _metricas_portafolio(w, mu, Sigma)

    w_eq = np.repeat(1 / n, n)
    ret_eq, vol_eq, sh_eq = _metricas_portafolio(w_eq, mu, Sigma)

    w_rp_arr = w_rp.reindex(assets).fillna(0).values
    ret_rp, vol_rp, sh_rp = _metricas_portafolio(w_rp_arr, mu, Sigma)

    if "SPY" in assets:
        spy = np.zeros(n)
        spy[assets.index("SPY")] = 1.0
        ret_spy, vol_spy, sh_spy = _metricas_portafolio(spy, mu, Sigma)
    else:
        ret_spy = vol_spy = sh_spy = float("nan")

    # Contribución a riesgo
    marginal = Sigma @ w
    rc = (w * marginal) / (vol_opt + 1e-12)
    rc_series = pd.Series(rc, index=assets).sort_values(ascending=False)
    top_w = w_best.head(15).round(4)
    top_rc = rc_series.head(15).round(4)

    # Gráficas
    img_ef = _chart_frontera(ef, vol_opt, ret_opt)
    top_20 = w_best.head(20).index.tolist()
    img_corr = _chart_correlacion(cov, top_20)
    img_dd = _chart_drawdown(prices, w_best)

    today = datetime.utcnow().strftime("%Y-%m-%d")

    # ── HTML ────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 24px; color: #222; }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }}
  h2 {{ color: #2c3e50; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #f2f2f2; font-weight: 600; }}
  .left {{ text-align: left; }}
  .summary-table td {{ font-size: 14px; }}
  .highlight {{ background: #e8f5e9; font-weight: bold; }}
  img {{ max-width: 700px; margin: 12px 0; }}
  .footer {{ font-size: 12px; color: #666; margin-top: 24px; border-top: 1px solid #eee; padding-top: 8px; }}
</style>
<title>Reporte Trimestral — {today}</title>
</head>
<body>
<h1>📊 Reporte Trimestral de Portafolio</h1>
<div>Generado: {today}</div>

<h2>Resumen Comparativo</h2>
<table class="summary-table">
  <tr><th class="left">Portafolio</th><th>Retorno</th><th>Volatilidad</th><th>Sharpe</th></tr>
  <tr class="highlight"><td class="left">Óptimo (Max Sharpe)</td><td>{ret_opt:.2%}</td><td>{vol_opt:.2%}</td><td>{sh_opt:.3f}</td></tr>
  <tr><td class="left">Risk-Parity</td><td>{ret_rp:.2%}</td><td>{vol_rp:.2%}</td><td>{sh_rp:.3f}</td></tr>
  <tr><td class="left">Equiponderado</td><td>{ret_eq:.2%}</td><td>{vol_eq:.2%}</td><td>{sh_eq:.3f}</td></tr>
  <tr><td class="left">SPY (100%)</td><td>{ret_spy:.2%}</td><td>{vol_spy:.2%}</td><td>{sh_spy:.3f}</td></tr>
</table>

<h2>Frontera Eficiente</h2>
<img src="data:image/png;base64,{img_ef}" alt="Frontera Eficiente"/>

<h2>Drawdown del Portafolio Óptimo</h2>
<img src="data:image/png;base64,{img_dd}" alt="Drawdown"/>

<h2>Correlación — Top 20 Holdings</h2>
<img src="data:image/png;base64,{img_corr}" alt="Correlación"/>

<h2>Top 15 Pesos</h2>
<table>
  <tr><th class="left">Ticker</th><th>Peso</th></tr>
  {''.join(f'<tr><td class="left">{k}</td><td>{v:.4f}</td></tr>' for k, v in top_w.items())}
</table>

<h2>Top 15 Contribuciones a Riesgo</h2>
<table>
  <tr><th class="left">Ticker</th><th>RC</th></tr>
  {''.join(f'<tr><td class="left">{k}</td><td>{v:.4f}</td></tr>' for k, v in top_rc.items())}
</table>

<h2>Calidad de Datos — Peores 10 Coberturas</h2>
<table>
  <tr><th class="left">Ticker</th><th>Coverage %</th><th>Gaps</th><th class="left">Start</th><th class="left">End</th></tr>
  {''.join(f'<tr><td class="left">{idx}</td><td>{row["coverage_%"]:.2f}</td><td>{int(row["gaps"])}</td><td class="left">{row["start"]}</td><td class="left">{row["end"]}</td></tr>' for idx, row in quality.sort_values("coverage_%").head(10).iterrows())}
</table>

<div class="footer">
  Metodología: Ledoit–Wolf shrinkage + medias shrunk (α={0.25}), SLSQP con límites 0–15% por activo.<br/>
  Risk-parity: método iterativo de inversión de riesgo marginal.
</div>
</body>
</html>"""

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Reporte HTML generado: %s", filename)
