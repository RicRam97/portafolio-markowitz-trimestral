import numpy as np
import pandas as pd
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
        
    # Calculate geometric mean return annualized correctly
    # Log returns to simple returns, then annualized
    cum_returns = np.exp(returns_df.sum()) - 1
    years = len(returns_df) / 252
    ann_returns = (1 + cum_returns) ** (1 / years) - 1 if years > 0 else np.zeros_like(cum_returns)
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
    
    # Use exponential moving average for returns to give more weight to recent data, 
    # but classic mean historical is robust. 
    mu = expected_returns.mean_historical_return(returns_df, returns_data=True, frequency=252)
    cov = risk_models.CovarianceShrinkage(returns_df, returns_data=True).ledoit_wolf()
    
    safe_max_weight = max(max_w, 1.0 / len(mu) + 0.05) if len(mu) > 0 else max_w
    
    ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
    
    # Optimization logic based on constraints
    try:
        if target_ret is not None and target_vol is not None:
             # Try to hit target return, if it hits volatility limits we might fail.
             # Alternatively we maximize sharpe. Let's do max_sharpe as primary if both are none,
             # if target_ret is set we use efficient_return, if target_vol is set we use efficient_risk
             
             # if both are set, we try to minimize risk for target return, then verify if vol <= target_vol
             ef.efficient_return(target_ret)
        elif target_ret is not None:
             ef.efficient_return(target_ret)
        elif target_vol is not None:
             ef.efficient_risk(target_vol)
        else:
             ef.max_sharpe(risk_free_rate=0.0)
    except Exception as e:
        logger.warning(f"Falla optimización con restricciones específicas, cayendo a max_sharpe: {e}")
        # Reset and fallback to max_sharpe
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, safe_max_weight))
        ef.max_sharpe(risk_free_rate=0.0)

    raw_weights = ef.clean_weights()
    w_series = pd.Series(raw_weights)
    
    ret_opt, vol_opt, sharpe_opt = ef.portfolio_performance(verbose=False, risk_free_rate=0.0)
    
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
            # Simple heuristic clipping and redistribution
            w_series = w_series.clip(upper=max_weight)
            w_series = w_series / w_series.sum()
        
        ret_hrp, vol_hrp, sharpe_hrp = hrp.portfolio_performance(risk_free_rate=0.0)
        
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
    sharpes = rets / vols
    
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
