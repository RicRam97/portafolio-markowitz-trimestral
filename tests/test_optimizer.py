import pytest
import pandas as pd
import numpy as np
from optimizer import sanity_filters, optimize_markowitz, optimize_hrp, run_monte_carlo, calculate_positions, OptimizerError

def test_sanity_filters():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=260)
    rets = pd.DataFrame({
        "A": np.random.normal(0.0001, 0.01, 260),
        "B": np.random.normal(0.0002, 0.015, 260)
    }, index=dates)
    
    res = sanity_filters(rets)
    assert res["valid"] is True
    assert not res["outlier_warning"]
    assert len(res["excluded_tickers"]) == 0
    assert "A" in res["filtered_returns"]
    assert "B" in res["filtered_returns"]

def test_sanity_filters_insufficient_assets():
    dates = pd.date_range("2020-01-01", periods=260)
    rets = pd.DataFrame({"A": np.random.normal(0, 0.01, 260)}, index=dates)
    with pytest.raises(OptimizerError, match="Mínimo 2 activos requeridos"):
        sanity_filters(rets)

def test_optimize_markowitz():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=260)
    rets = pd.DataFrame({
        "A": np.random.normal(0.0001, 0.01, 260),
        "B": np.random.normal(0.0002, 0.015, 260)
    }, index=dates)
    
    constraints = {"max_weight": 0.6, "broker_commission": 0.0}
    res = optimize_markowitz(rets, constraints)
    
    assert "weights" in res
    assert "expected_return_net" in res
    assert "expected_volatility" in res
    assert "sharpe_ratio" in res

def test_calculate_positions():
    weights = {"A": 0.6, "B": 0.4}
    prices = pd.Series({"A": 100.0, "B": 50.0})
    budget = 1000.0
    
    res = calculate_positions(weights, budget, prices)
    
    assert "positions" in res
    assert res["invested_cash"] <= budget
    
    pos_A = next(p for p in res["positions"] if p["ticker"] == "A")
    pos_B = next(p for p in res["positions"] if p["ticker"] == "B")
    
    assert pos_A["shares"] == 6
    assert pos_B["shares"] == 8
    
    assert res["invested_cash"] == 6*100 + 8*50 # 1000.0
