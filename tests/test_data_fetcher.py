import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime
from data_fetcher import (
    get_historical_prices,
    filter_illiquid_tickers,
    clean_and_validate_data,
    get_benchmarks,
    DataFetchError
)

def test_invalid_period():
    with pytest.raises(ValueError, match="not supported"):
        get_historical_prices(["AAPL"], "10y")

def test_empty_tickers():
    with pytest.raises(DataFetchError, match="No tickers provided"):
        get_historical_prices([], "1y")

def test_clean_and_validate_success():
    dates = pd.date_range("2020-01-01", periods=260)
    prices = pd.DataFrame({"TEST1": np.random.rand(260) * 100 + 10}, index=dates)
    
    df, log_rets = clean_and_validate_data(prices)
    # The first row is dropped because shifted price is NaN for return
    assert len(df) == 259
    assert len(log_rets) == 259
    
def test_clean_and_validate_too_short():
    dates = pd.date_range("2020-01-01", periods=100)
    prices = pd.DataFrame({"TEST1": np.ones(100)}, index=dates)
    with pytest.raises(DataFetchError, match="Insufficient data"):
        clean_and_validate_data(prices)

def test_filter_illiquid_tickers_empty():
    res = filter_illiquid_tickers([])
    assert res == {"valid_tickers": [], "rejected_tickers": {}}

# E2E basic tests (these will hit yfinance)
def test_get_historical_prices_live():
    res = get_historical_prices(["AAPL"], "3y")
    assert "prices" in res
    assert "returns" in res
    assert "AAPL" in res["tickers"]
    assert "failed_tickers" in res
    
    # second call hits cache
    res2 = get_historical_prices(["AAPL"], "3y")
    assert "prices" in res2
    
def test_get_benchmarks_live():
    res = get_benchmarks("1y")
    assert "IPC_final" in res
    assert "S&P500_final" in res
    assert "CETES_final" in res
