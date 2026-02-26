import pandas as pd
import numpy as np
from optimizer import estimar_parametros, optimizar_sharpe, frontera_eficiente, pesos_risk_parity

def create_synthetic_ret_panel():
    # 5 assets, 100 days
    np.random.seed(42)
    assets = ["A", "B", "C", "D", "E"]
    rets = np.random.normal(0.0005, 0.01, (100, 5))
    return pd.DataFrame(rets, columns=assets)

def test_estimar_parametros():
    df = create_synthetic_ret_panel()
    mu, cov = estimar_parametros(df, shrinkage_alpha=0.25)
    
    assert list(mu.index) == ["A", "B", "C", "D", "E"]
    assert cov.shape == (5, 5)
    # Check that covariance is symmetric
    assert np.allclose(cov.values, cov.values.T)

def test_optimizar_sharpe():
    df = create_synthetic_ret_panel()
    mu, cov = estimar_parametros(df, shrinkage_alpha=0.25)
    
    # Run optimizer with max weight of 0.4
    w = optimizar_sharpe(mu, cov, max_weight=0.4)
    
    # Assert weight sum to 1
    assert np.isclose(w.sum(), 1.0)
    # Assert bounds respected
    assert (w <= 0.4001).all()
    assert (w >= -0.0001).all()
    # Check length
    assert len(w) == 5

def test_frontera_eficiente():
    df = create_synthetic_ret_panel()
    mu, cov = estimar_parametros(df, shrinkage_alpha=0.25)
    
    # max_weight must be large enough to allow 5 assets to sum to 1.
    ef = frontera_eficiente(mu, cov, n_points=5, max_weight=1.0)
    assert not ef.empty
    assert list(ef.columns) == ["target", "ret", "vol", "sharpe"]
    # Usually vol increases as we demand higher returns
    if len(ef) >= 2:
        assert ef.iloc[-1]["ret"] > ef.iloc[0]["ret"]

def test_risk_parity():
    df = create_synthetic_ret_panel()
    _, cov = estimar_parametros(df)
    
    w_rp = pesos_risk_parity(cov)
    # Check weight sum to 1
    assert np.isclose(w_rp.sum(), 1.0)
    assert (w_rp >= -0.0001).all()
