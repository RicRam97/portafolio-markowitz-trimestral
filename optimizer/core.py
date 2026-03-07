import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
import logging

logger = logging.getLogger(__name__)

class OptimizerError(Exception):
    pass

def smart_beta_filter(
    rets_clean: pd.DataFrame,
    strategy: str = "momentum",
    top_n: int = 15,
    momentum_window: int = 252
) -> pd.DataFrame:
    """
    Filtra el universo de activos por un factor Smart Beta.
    """
    if rets_clean.empty or rets_clean.shape[1] <= top_n:
        return rets_clean

    if strategy == "momentum":
        recent_rets = rets_clean.tail(momentum_window)
        cum_ret = (1 + recent_rets).prod() - 1
        selected = cum_ret.nlargest(top_n).index
    elif strategy == "low_volatility":
        vol = rets_clean.std() * np.sqrt(252)
        selected = vol.nsmallest(top_n).index
    else:
        selected = rets_clean.columns

    return rets_clean[selected]

def sanity_filters(returns_df: pd.DataFrame) -> dict:
    """
    Applies sanity filters on the returns DataFrame.
    Returns a dict with flags and potentially modified dataframe.
    """
    flags = {
        "outlier_warning": False,
        "insufficient_data": False,
        "valid": True,
        "messages": [],
        "excluded_tickers": []
    }
    
    if returns_df.empty or returns_df.shape[1] < 2:
        raise OptimizerError("Mínimo 2 activos requeridos para diversificación.")
        
    if len(returns_df) < 252:
        flags["insufficient_data"] = True
        flags["messages"].append(f"Solo {len(returns_df)} días de historia (ideal > 252).")
        
    # Calculate geometric mean return annualized correctly (simple returns from pct_change)
    cum_returns = (1 + returns_df).prod() - 1
    years = len(returns_df) / 252
    ann_returns = (1 + cum_returns) ** (1 / years) - 1 if years > 0 else cum_returns * 0.0
    ann_vols = returns_df.std() * np.sqrt(252)
    
    tickers_to_keep = []
    
    for ticker in returns_df.columns:
        ret = ann_returns[ticker]
        vol = ann_vols[ticker]
        
        if ret > 2.0: # > 200%
            flags["outlier_warning"] = True
            flags["messages"].append(f"Retorno extremo detectado en {ticker} (> 200% anual).")
            
        if vol > 1.0: # > 100%
            flags["excluded_tickers"].append({
                "ticker": ticker,
                "reason": f"Volatilidad anualizada de {vol*100:.1f}% supera el 100%."
            })
            logger.warning(f"Excluyendo {ticker} por volatilidad extrema ({vol*100:.1f}%).")
        else:
            tickers_to_keep.append(ticker)
            
    filtered_df = returns_df[tickers_to_keep]
    if filtered_df.shape[1] < 2:
         raise OptimizerError("Tras aplicar filtros de cordura (volatilidad < 100%), quedaron menos de 2 activos.")

    flags["filtered_returns"] = filtered_df
    return flags

def optimize_markowitz(returns_df: pd.DataFrame, constraints: dict) -> dict:
    """
    Optimizes portfolio using Markowitz mean-variance and Ledoit-Wolf shrinkage.
    Constraints: 
      max_weight (default 0.25)
      max_volatility (from user profile)
      min_return (from dreams test)
      broker_commission (cost modeling)
    """
    max_w = constraints.get("max_weight", 0.25)
    target_vol = constraints.get("max_volatility")
    target_ret = constraints.get("min_return")
    rf = constraints.get("risk_free_rate", 0.04)

    mu = expected_returns.mean_historical_return(returns_df, returns_data=True, frequency=252)
    cov = risk_models.CovarianceShrinkage(returns_df, returns_data=True).ledoit_wolf()

    safe_max_weight = max(max_w, 1.0 / len(mu) + 0.05) if len(mu) > 0 else max_w

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))

    try:
        if target_ret is not None and target_vol is not None:
            # Minimize risk for target return, then verify volatility constraint
            ef.efficient_return(target_ret)
            _, vol_check, _ = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
            if vol_check > target_vol:
                logger.warning(
                    f"Volatilidad resultante ({vol_check:.4f}) excede límite ({target_vol:.4f}), "
                    "reoptimizando con efficient_risk."
                )
                ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
                ef.efficient_risk(target_vol)
        elif target_ret is not None:
            ef.efficient_return(target_ret)
        elif target_vol is not None:
            ef.efficient_risk(target_vol)
        else:
            try:
                ef.max_sharpe(risk_free_rate=rf)
            except ValueError:
                logger.warning("max_sharpe falló (retornos < rf), usando min_volatility.")
                ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
                ef.min_volatility()
    except Exception as e:
        logger.warning(f"Falla optimización con restricciones específicas, cayendo a max_sharpe: {e}")
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
        try:
            ef.max_sharpe(risk_free_rate=rf)
        except ValueError:
            logger.warning("max_sharpe falló (retornos < rf), usando min_volatility.")
            ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
            ef.min_volatility()

    raw_weights = ef.clean_weights()
    w_series = pd.Series(raw_weights)
    
    ret_opt, vol_opt, sharpe_opt = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
    
    # Calculate Max Drawdown Estimation (historical based on daily returns dot weights)
    port_rets = returns_df.dot(w_series)
    cum_returns = (1 + port_rets).cumprod()
    running_max = cum_returns.cummax()
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Subtract commissions from expected returns structurally
    commission = constraints.get("broker_commission", 0.0)
    # Approx: 1 buy + 1 sell in N years = net return drop. For simplicity, drop annual expected return by the commission
    net_ret = ret_opt - commission

    return {
        "weights": w_series.to_dict(),
        "expected_return_net": net_ret,
        "expected_volatility": vol_opt,
        "sharpe_ratio": sharpe_opt,
        "max_drawdown_est": max_drawdown
    }

def optimize_hrp(returns_df: pd.DataFrame, max_weight: float = 0.25) -> dict:
    """
    Hierarchical Risk Parity.
    Returns: weights, clustering_data (for dendrogram), expected_return, expected_volatility
    """
    try:
        hrp = HRPOpt(returns_df)
        hrp.optimize()
        
        # HRP doesn't natively support max_weight well without internal iteration,
        # but PyPortfolioOpt doesn't take weight_bounds for HRP easily.
        # We will extract weights and clip/re-normalize manually if strictly needed,
        # but usually HRP avoids extreme concentration naturally.
        cleaned_weights = hrp.clean_weights()
        w_series = pd.Series(cleaned_weights)
        
        if w_series.max() > max_weight:
            w_series = w_series.clip(upper=max_weight)
            w_series = w_series / w_series.sum()

        # Recalculate performance from actual (possibly clipped) weights
        mu = expected_returns.mean_historical_return(returns_df, returns_data=True, frequency=252)
        cov = risk_models.CovarianceShrinkage(returns_df, returns_data=True).ledoit_wolf()
        ret_hrp = float(w_series.dot(mu))
        vol_hrp = float(np.sqrt(w_series.dot(cov).dot(w_series)))
        sharpe_hrp = ret_hrp / vol_hrp if vol_hrp > 0 else 0.0
        
        # Extract clusters for dendrogram
        linkage_matrix = None
        if hasattr(hrp, 'clusters'):
            # Convert scipy linkage array to JSON serializable list of lists
            # format: [[idx1, idx2, distance, sample_count], ...]
            linkage_matrix = hrp.clusters.tolist()

        return {
            "weights": w_series.to_dict(),
            "clustering_data": linkage_matrix,
            "expected_return": ret_hrp,
            "expected_volatility": vol_hrp,
            "sharpe_ratio": sharpe_hrp
        }
    except Exception as e:
         raise OptimizerError(f"HRP falló: {e}")

def run_monte_carlo(returns_df: pd.DataFrame, n_portfolios: int = 5000) -> dict:
    """
    Generates random portfolios.
    Returns efficient frontier cloud points.
    """
    mu = expected_returns.mean_historical_return(returns_df, returns_data=True, frequency=252)
    cov = risk_models.CovarianceShrinkage(returns_df, returns_data=True).ledoit_wolf()
    
    num_assets = len(mu)
    mu_np = mu.values
    cov_np = cov.values
    
    weights = np.random.random((n_portfolios, num_assets))
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    
    rets = weights @ mu_np
    vols = np.sqrt((weights * (weights @ cov_np)).sum(axis=1))
    sharpes = np.where(vols > 0, rets / vols, 0.0)
    
    # Identify key portfolios
    max_sharpe_idx = np.argmax(sharpes)
    min_vol_idx = np.argmin(vols)
    
    # Compact scatter points for UI (subset to avoid huge payloads)
    # take max 500 points representing the cloud roughly
    subset_indices = np.random.choice(n_portfolios, min(500, n_portfolios), replace=False)
    # Ensure key points are in
    subset_indices = list(set(subset_indices).union({max_sharpe_idx, min_vol_idx}))
    
    cloud_points = [
        {"ret": float(rets[i]), "vol": float(vols[i]), "sharpe": float(sharpes[i])} 
        for i in subset_indices
    ]
    
    return {
        "cloud": cloud_points,
        "max_sharpe_portfolio": {
            "ret": float(rets[max_sharpe_idx]),
            "vol": float(vols[max_sharpe_idx]),
            "sharpe": float(sharpes[max_sharpe_idx]),
            "weights": {returns_df.columns[k]: float(weights[max_sharpe_idx][k]) for k in range(num_assets)}
        },
        "min_vol_portfolio": {
            "ret": float(rets[min_vol_idx]),
            "vol": float(vols[min_vol_idx]),
            "sharpe": float(sharpes[min_vol_idx]),
            "weights": {returns_df.columns[k]: float(weights[min_vol_idx][k]) for k in range(num_assets)}
        }
    }

def calculate_positions(weights: dict, budget: float, last_prices: pd.Series) -> dict:
    """
    Calculates integer share quantities to buy per ticker.
    Returns detailed allocation table and uninvested cash.
    """
    allocation = []
    total_spent = 0.0

    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    for ticker, w in sorted_weights:
        if w > 0.0001 and ticker in last_prices:
             price = float(last_prices[ticker])
             target_dollars = budget * w
             shares = int(np.floor(target_dollars / price))

             cost = shares * price
             total_spent += cost

             allocation.append({
                 "ticker": ticker,
                 "weight_pct": round(w * 100, 2),
                 "shares": shares,
                 "price": round(price, 2),
                 "target_amount": round(target_dollars, 2),
                 "invested_amount": round(cost, 2)
             })

    remaining_cash = budget - total_spent
    return {
        "positions": allocation,
        "invested_cash": round(total_spent, 2),
        "remaining_cash": round(remaining_cash, 2),
        "total_budget": budget
    }


def calcular_acciones_y_efectivo(
    pesos_optimos: dict[str, float],
    precios_actuales: dict[str, float],
    presupuesto_total: float,
    comision_broker: float = 0.0025,
) -> dict:
    """
    Convierte pesos optimos en numero de acciones enteras y calcula efectivo restante.

    Args:
        pesos_optimos: {"AAPL": 0.25, "GOOGL": 0.30, ...}
        precios_actuales: {"AAPL": 150.00, "GOOGL": 120.00, ...}
        presupuesto_total: 50000.00
        comision_broker: 0.0025 (0.25%)

    Returns:
        {
            "asignacion": { ticker: { peso_teorico, peso_real, acciones, inversion,
                                      comision, precio_compra } },
            "efectivo_restante": float,
            "inversion_total": float,
            "comisiones_totales": float,
            "porcentaje_invertido": float,
            "desviacion_maxima_peso": float,
        }
    """
    presupuesto_disponible = presupuesto_total
    asignacion: dict[str, dict] = {}

    for ticker, peso in pesos_optimos.items():
        precio = precios_actuales.get(ticker, 0.0)
        if precio <= 0:
            logger.warning(f"Precio no disponible para {ticker}, omitido de asignacion real.")
            continue

        inversion_objetivo = peso * presupuesto_total
        num_acciones = int(inversion_objetivo / precio)
        inversion_real = num_acciones * precio
        comision = inversion_real * comision_broker
        costo_total = inversion_real + comision
        peso_real = inversion_real / presupuesto_total if presupuesto_total > 0 else 0.0

        asignacion[ticker] = {
            "peso_teorico": round(peso, 6),
            "peso_real": round(peso_real, 6),
            "acciones": num_acciones,
            "inversion": round(inversion_real, 2),
            "comision": round(comision, 2),
            "precio_compra": round(precio, 2),
        }
        presupuesto_disponible -= costo_total

    efectivo_restante = presupuesto_disponible
    inversion_total = sum(a["inversion"] for a in asignacion.values())
    comisiones_totales = sum(a["comision"] for a in asignacion.values())
    porcentaje_invertido = (inversion_total / presupuesto_total) * 100 if presupuesto_total > 0 else 0.0

    desviaciones = [abs(a["peso_real"] - a["peso_teorico"]) for a in asignacion.values()]
    desviacion_maxima = max(desviaciones) if desviaciones else 0.0

    return {
        "asignacion": asignacion,
        "efectivo_restante": round(efectivo_restante, 2),
        "inversion_total": round(inversion_total, 2),
        "comisiones_totales": round(comisiones_totales, 2),
        "porcentaje_invertido": round(porcentaje_invertido, 2),
        "desviacion_maxima_peso": round(desviacion_maxima, 6),
    }


def optimizar_efectivo_restante(
    asignacion: dict[str, dict],
    efectivo_restante: float,
    precios: dict[str, float],
    presupuesto_total: float,
    max_weight: float = 0.25,
    comision_broker: float = 0.0025,
) -> tuple[dict[str, dict], float]:
    """
    Intenta reducir el efectivo restante comprando 1 accion adicional por iteracion
    en el activo que mas reduzca el cash sin exceder max_weight.

    Solo actua si el efectivo restante > 5% del presupuesto total.
    Devuelve (asignacion_actualizada, efectivo_restante_actualizado).
    """
    umbral = 0.05 * presupuesto_total

    while efectivo_restante > umbral:
        mejor_ticker: str | None = None
        mejor_costo = 0.0

        for ticker, datos in asignacion.items():
            precio = precios.get(ticker, 0.0)
            if precio <= 0:
                continue

            costo_accion = precio * (1 + comision_broker)
            if costo_accion > efectivo_restante:
                continue

            nuevo_peso = (datos["inversion"] + precio) / presupuesto_total
            if nuevo_peso > max_weight:
                continue

            # Elegir el activo cuya accion extra sea la mas cara (max reduccion de cash)
            if precio > mejor_costo:
                mejor_costo = precio
                mejor_ticker = ticker

        if mejor_ticker is None:
            break

        precio_t = precios[mejor_ticker]
        comision_extra = precio_t * comision_broker
        asignacion[mejor_ticker]["acciones"] += 1
        asignacion[mejor_ticker]["inversion"] = round(
            asignacion[mejor_ticker]["inversion"] + precio_t, 2
        )
        asignacion[mejor_ticker]["comision"] = round(
            asignacion[mejor_ticker]["comision"] + comision_extra, 2
        )
        asignacion[mejor_ticker]["peso_real"] = round(
            asignacion[mejor_ticker]["inversion"] / presupuesto_total, 6
        )
        efectivo_restante = round(efectivo_restante - precio_t - comision_extra, 2)

    return asignacion, efectivo_restante


class MarkowitzOptimizer:
    """Optimizador Markowitz puro usando scipy (sin pypfopt).
    Recibe retornos logaritmicos y parametros de restriccion.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        tasa_libre_riesgo: float = 0.04,
        peso_maximo: float = 0.25,
        volatilidad_maxima: float | None = None,
    ):
        self.log_returns = log_returns
        self.tickers = list(log_returns.columns)
        self.n = len(self.tickers)
        self.rf = tasa_libre_riesgo
        self.peso_maximo = peso_maximo
        self.volatilidad_maxima = volatilidad_maxima

        # Retorno esperado anualizado (media geometrica de log-returns)
        self.mu = log_returns.mean() * 252
        # Matriz de covarianza anualizada
        self.cov = log_returns.cov() * 252

    def _portfolio_return(self, w: np.ndarray) -> float:
        return float(w @ self.mu.values)

    def _portfolio_volatility(self, w: np.ndarray) -> float:
        return float(np.sqrt(w @ self.cov.values @ w))

    def _neg_sharpe(self, w: np.ndarray) -> float:
        ret = self._portfolio_return(w)
        vol = self._portfolio_volatility(w)
        if vol < 1e-10:
            return 1e10
        return -(ret - self.rf) / vol

    def optimize(self) -> dict:
        """Encuentra el portafolio con maximo Sharpe Ratio.
        Si hay restriccion de volatilidad maxima, maximiza retorno sujeto a vol <= max.
        """
        w0 = np.ones(self.n) / self.n
        bounds = [(0.0, self.peso_maximo) for _ in range(self.n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if self.volatilidad_maxima is not None:
            # Maximizar retorno sujeto a vol <= max
            constraints.append({
                "type": "ineq",
                "fun": lambda w: self.volatilidad_maxima - self._portfolio_volatility(w),
            })
            objective = lambda w: -self._portfolio_return(w)
        else:
            objective = self._neg_sharpe

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if not result.success:
            raise OptimizerError(f"La optimizacion no convergio: {result.message}")

        w_opt = result.x
        # Limpiar pesos cercanos a cero
        w_opt[w_opt < 1e-4] = 0.0
        w_opt = w_opt / w_opt.sum()

        ret_opt = self._portfolio_return(w_opt)
        vol_opt = self._portfolio_volatility(w_opt)
        sharpe_opt = (ret_opt - self.rf) / vol_opt if vol_opt > 1e-10 else 0.0

        weights = {self.tickers[i]: round(float(w_opt[i]), 6) for i in range(self.n) if w_opt[i] > 1e-6}

        return {
            "weights": weights,
            "expected_return": round(ret_opt, 6),
            "volatility": round(vol_opt, 6),
            "sharpe_ratio": round(sharpe_opt, 4),
        }

    def efficient_frontier(self, n_points: int = 50) -> list[dict]:
        """Genera n_points puntos de la frontera eficiente
        variando el retorno objetivo entre el minimo y maximo alcanzable.
        """
        # Encontrar portafolio de minima varianza
        w0 = np.ones(self.n) / self.n
        bounds = [(0.0, self.peso_maximo) for _ in range(self.n)]
        constraints_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res_min = minimize(
            self._portfolio_volatility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_base,
            options={"maxiter": 1000},
        )
        min_ret = self._portfolio_return(res_min.x) if res_min.success else float(self.mu.min())

        # Encontrar portafolio de maximo retorno
        res_max = minimize(
            lambda w: -self._portfolio_return(w),
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_base,
            options={"maxiter": 1000},
        )
        max_ret = self._portfolio_return(res_max.x) if res_max.success else float(self.mu.max())

        if max_ret <= min_ret:
            max_ret = min_ret + 0.01

        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_points: list[dict] = []

        for target_ret in target_returns:
            cons = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, tr=target_ret: self._portfolio_return(w) - tr},
            ]
            res = minimize(
                self._portfolio_volatility,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 1000},
            )
            if res.success:
                vol = self._portfolio_volatility(res.x)
                ret = self._portfolio_return(res.x)
                sharpe = (ret - self.rf) / vol if vol > 1e-10 else 0.0
                frontier_points.append({
                    "expected_return": round(ret, 6),
                    "volatility": round(vol, 6),
                    "sharpe_ratio": round(sharpe, 4),
                })

        return frontier_points


class MonteCarloOptimizer:
    """Optimizador por simulacion Monte Carlo.
    Genera miles de portafolios aleatorios y selecciona el de maximo Sharpe.
    """

    def __init__(
        self,
        log_returns: pd.DataFrame,
        num_portfolios: int = 10000,
        tasa_libre_riesgo: float = 0.04,
        peso_maximo: float = 0.30,
    ):
        self.log_returns = log_returns
        self.tickers = list(log_returns.columns)
        self.n = len(self.tickers)
        self.num_portfolios = num_portfolios
        self.rf = tasa_libre_riesgo
        self.peso_maximo = peso_maximo

        self.mu = log_returns.mean().values * 252
        self.cov = log_returns.cov().values * 252

    def _generate_weights(self) -> np.ndarray:
        """Genera matriz (num_portfolios x n) de pesos aleatorios validos."""
        weights = np.random.random((self.num_portfolios, self.n))
        weights = np.minimum(weights, self.peso_maximo)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return weights

    def optimize(self) -> dict:
        """Ejecuta simulacion Monte Carlo y retorna portafolio optimo + nube."""
        weights = self._generate_weights()

        # Vectorizado: calcular metricas para todos los portafolios
        rets = weights @ self.mu
        vols = np.sqrt((weights @ self.cov * weights).sum(axis=1))
        sharpes = np.where(vols > 1e-10, (rets - self.rf) / vols, 0.0)

        max_sharpe_idx = int(np.argmax(sharpes))
        min_vol_idx = int(np.argmin(vols))

        # Portafolio optimo (max Sharpe)
        w_opt = weights[max_sharpe_idx]
        portafolio_optimo = {
            "weights": {
                self.tickers[i]: round(float(w_opt[i]), 6)
                for i in range(self.n) if w_opt[i] > 1e-6
            },
            "expected_return": round(float(rets[max_sharpe_idx]), 6),
            "volatility": round(float(vols[max_sharpe_idx]), 6),
            "sharpe_ratio": round(float(sharpes[max_sharpe_idx]), 4),
        }

        # Portafolio minima volatilidad
        w_min = weights[min_vol_idx]
        portafolio_min_vol = {
            "weights": {
                self.tickers[i]: round(float(w_min[i]), 6)
                for i in range(self.n) if w_min[i] > 1e-6
            },
            "expected_return": round(float(rets[min_vol_idx]), 6),
            "volatility": round(float(vols[min_vol_idx]), 6),
            "sharpe_ratio": round(float(sharpes[min_vol_idx]), 4),
        }

        # Nube de puntos (subsample para payload ligero, max 2000)
        max_cloud = min(2000, self.num_portfolios)
        subset = np.random.choice(
            self.num_portfolios, max_cloud, replace=False
        )
        # Asegurar que puntos clave estan incluidos
        subset = list(set(subset) | {max_sharpe_idx, min_vol_idx})

        cloud = [
            {
                "ret": round(float(rets[i]), 4),
                "vol": round(float(vols[i]), 4),
                "sharpe": round(float(sharpes[i]), 4),
            }
            for i in subset
        ]

        return {
            "portafolio_optimo": portafolio_optimo,
            "portafolio_min_vol": portafolio_min_vol,
            "simulacion": {
                "num_portfolios": self.num_portfolios,
                "cloud": cloud,
            },
        }
