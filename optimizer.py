# optimizer.py — Estimación y optimización de portafolios
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

from config import log, SHRINKAGE_ALPHA, MAX_WEIGHT, LAMBDA_L2, EF_POINTS, RF


def estimar_parametros(
    rets_clean: pd.DataFrame,
    shrinkage_alpha: float = SHRINKAGE_ALPHA,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimación robusta: Ledoit-Wolf cov + medias shrunk.

    Nota: esto NO es Black-Litterman (no hay prior de equilibrio ni views).
    Es simplemente un shrinkage conservador sobre las medias muestrales.
    """
    assert len(rets_clean) > 30, (
        f"Insuficientes observaciones ({len(rets_clean)}) para estimar covarianza"
    )
    assert rets_clean.shape[1] >= 5, (
        f"Insuficientes activos ({rets_clean.shape[1]}) para optimizar"
    )

    lw = LedoitWolf().fit(rets_clean.values)
    cov_annual = pd.DataFrame(
        lw.covariance_ * 252,
        index=rets_clean.columns,
        columns=rets_clean.columns,
    )
    mu_annual = rets_clean.mean() * 252
    mu_shrunk = (1 - shrinkage_alpha) * mu_annual

    log.info(
        "Parámetros estimados: %d activos, shrinkage LW = %.3f",
        len(mu_shrunk), lw.shrinkage_,
    )
    return mu_shrunk, cov_annual


def optimizar_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    max_weight: float = MAX_WEIGHT,
    rf: float = RF,
    lambda_l2: float = LAMBDA_L2,
) -> pd.Series:
    """Maximiza el Sharpe ratio vía SLSQP con límites por activo."""
    assets = list(mu.index)
    n = len(assets)
    Sigma = cov.reindex(index=assets, columns=assets).values
    mu_np = mu.reindex(assets).values
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def obj(w):
        r = float(np.dot(w, mu_np))
        v = float(np.sqrt(np.dot(w, Sigma @ w)))
        return -(r - rf) / (v + 1e-12) + lambda_l2 * np.dot(w, w)

    w0 = np.repeat(1 / n, n)
    res = minimize(
        obj, w0, method="SLSQP", bounds=bounds,
        constraints=cons, options={"maxiter": 1000},
    )
    if not res.success:
        log.warning("Optimizador no convergió: %s", res.message)
    else:
        log.info("Optimización convergida en %d iteraciones", res.nit)

    w_best = pd.Series(res.x, index=assets).sort_values(ascending=False)

    # Métricas del portafolio óptimo
    ret_opt = float(np.dot(res.x, mu_np))
    vol_opt = float(np.sqrt(np.dot(res.x, Sigma @ res.x)))
    sharpe_opt = (ret_opt - rf) / (vol_opt + 1e-12)
    log.info(
        "Portafolio óptimo: ret=%.2f%%, vol=%.2f%%, sharpe=%.3f",
        ret_opt * 100, vol_opt * 100, sharpe_opt,
    )
    return w_best


def frontera_eficiente(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = EF_POINTS,
    max_weight: float = MAX_WEIGHT,
    rf: float = RF,
    lambda_l2: float = LAMBDA_L2,
) -> pd.DataFrame:
    """Calcula n_points puntos de la frontera eficiente (min-var sujeto a retorno mín)."""
    assets = list(mu.index)
    n = len(assets)
    Sigma = cov.reindex(index=assets, columns=assets).values
    mu_np = mu.reindex(assets).values
    w0 = np.repeat(1 / n, n)
    bounds = [(0.0, max_weight)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    mu_lo, mu_hi = np.percentile(mu_np, 20), np.percentile(mu_np, 80)
    targets = np.linspace(mu_lo, mu_hi, n_points)

    def obj_minvar(w):
        return float(np.dot(w, Sigma @ w) + lambda_l2 * np.dot(w, w))

    rows = []
    for t in targets:
        cons_t = cons + [
            {"type": "ineq", "fun": lambda w, tt=t: np.dot(w, mu_np) - tt}
        ]
        resf = minimize(
            obj_minvar, w0, method="SLSQP", bounds=bounds,
            constraints=cons_t, options={"maxiter": 1000},
        )
        if resf.success:
            r = float(np.dot(resf.x, mu_np))
            v = float(np.sqrt(np.dot(resf.x, Sigma @ resf.x)))
            rows.append([t, r, v, (r - rf) / (v + 1e-12)])

    ef = pd.DataFrame(rows, columns=["target", "ret", "vol", "sharpe"])
    log.info("Frontera eficiente: %d/%d puntos exitosos", len(ef), n_points)
    return ef


def pesos_risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Calcula pesos de risk-parity (contribución de riesgo igualitaria).

    Usa el método iterativo de Spinu (2013):
      w_i ∝ 1 / (Σw)_i  →  normalizar a suma 1.
    """
    assets = list(cov.columns)
    Sigma = cov.values
    n = len(assets)
    w = np.repeat(1 / n, n)

    for _ in range(500):
        marginal = Sigma @ w
        w_new = 1.0 / (marginal + 1e-12)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < 1e-10:
            break
        w = w_new

    rp = pd.Series(w_new, index=assets).sort_values(ascending=False)
    vol_rp = float(np.sqrt(np.dot(w_new, Sigma @ w_new)))
    log.info("Risk-parity: vol=%.2f%%, max peso=%.2f%%", vol_rp * 100, rp.max() * 100)
    return rp
