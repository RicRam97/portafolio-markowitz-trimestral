import pandas as pd
import numpy as np
from datetime import datetime

from data import extraer_panel, limpiar_retornos, reporte_calidad

def test_extraer_panel_multiindex():
    # Setup mock MultiIndex df as yfinance returns
    dates = pd.date_range("2023-01-01", periods=3)
    tickers = ["AAPL", "MSFT"]
    fields = ["Close", "Volume"]
    
    # Create multiindex columns
    columns = pd.MultiIndex.from_product([tickers, fields])
    
    # Fill with dummy data
    data = np.arange(12).reshape(3, 4)
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    # Execute
    panel = extraer_panel(df, tickers, field="Close")
    
    # Assert
    assert panel.shape == (3, 2)
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert panel.loc["2023-01-01", "AAPL"] == 0 # AAPL Close col
    assert panel.loc["2023-01-01", "MSFT"] == 2 # MSFT Close col

def test_limpiar_retornos_outliers():
    dates = pd.date_range("2023-01-01", periods=10)
    # Create normal returns for AAPL, but one extreme outlier
    aapl_prices = [100, 101, 102, 103, 104, 200, 104, 105, 106, 107]
    df = pd.DataFrame({"AAPL": aapl_prices}, index=dates)
    
    # Execute with low sigma to force clipping
    rets = limpiar_retornos(df, sigma=1, cov_min=0)
    
    # Assert
    # The return at index 5 (200/104 - 1 = 0.92) should be clipped
    assert rets.shape == (9, 1)
    
def test_reporte_calidad():
    dates = pd.date_range("2023-01-01", periods=5)
    prices = pd.DataFrame({
        "A": [10, 11, np.nan, 12, 13],
        "B": [10, 11, 12, 13, 14]
    }, index=dates)
    
    rets = prices.pct_change().dropna()
    
    rep = reporte_calidad(prices, rets)
    assert len(rep) == 2
    assert "coverage_%" in rep.columns
    assert "gaps" in rep.columns
    assert rep.loc["A", "gaps"] == 1
    assert rep.loc["B", "gaps"] == 0
