import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache_yf_v2")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DURATION_HOURS = 4

class DataFetchError(Exception):
    def __init__(self, message, failed_tickers=None):
        super().__init__(message)
        self.failed_tickers = failed_tickers or []

def _get_cache_path(tickers: list[str], period: str) -> Path:
    """Generate a unique cache file path based on sorted tickers and period."""
    hash_input = str(sorted(tickers)) + period
    filename = hashlib.sha256(hash_input.encode('utf-8')).hexdigest() + ".json"
    return CACHE_DIR / filename

def _read_cache(cache_path: Path) -> dict | None:
    """Read data from cache if it exists and is less than CACHE_DURATION_HOURS old."""
    if not cache_path.exists():
        logger.info(f"Cache miss (not found): {cache_path.name}")
        return None
        
    try:
        mtime = cache_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        
        if age_hours > CACHE_DURATION_HOURS:
            logger.info(f"Cache miss (expired): {cache_path.name} (age: {age_hours:.1f}h)")
            return None
            
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"Cache hit: {cache_path.name}")
        return data
    except Exception as e:
        logger.warning(f"Error reading cache {cache_path}: {e}")
        return None

def _write_cache(cache_path: Path, data: dict):
    """Write data to cache."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Error writing to cache {cache_path}: {e}")

def get_historical_prices(tickers: list[str], period: str) -> dict:
    """
    Downloads historical prices for given tickers and period.
    Period must be one of: "1y", "3y", "5y".
    Returns cleaned data and failed tickers.
    """
    allowed_periods = {"1y", "3y", "5y"}
    if period not in allowed_periods:
        raise ValueError(f"Period '{period}' not supported. Use one of {allowed_periods}")

    if not tickers:
        raise DataFetchError("No tickers provided.")

    cache_path = _get_cache_path(tickers, period)
    cached_data = _read_cache(cache_path)
    
    if cached_data:
        # Convert prices back to DataFrame
        if "prices" in cached_data and cached_data["prices"]:
            cached_data["prices"] = pd.DataFrame.from_dict(cached_data["prices"])
            cached_data["prices"].index = pd.to_datetime(cached_data["prices"].index)
        if "returns" in cached_data and cached_data["returns"]:
            cached_data["returns"] = pd.DataFrame.from_dict(cached_data["returns"])
            cached_data["returns"].index = pd.to_datetime(cached_data["returns"].index)
        return cached_data

    logger.info(f"Downloading {len(tickers)} tickers for period {period}")
    failed_tickers = []
    
    try:
        raw_data = yf.download(tickers, period=period, group_by="ticker", auto_adjust=False, threads=True)
    except Exception as e:
        raise DataFetchError(f"API yfinance error: {e}")

    if raw_data.empty:
        raise DataFetchError("No data downloaded from yfinance.", failed_tickers=tickers)

    prices_df = pd.DataFrame()
    
    if len(tickers) == 1:
        ticker = tickers[0]
        # In single ticker, yfinance sometimes drops the top level MultiIndex
        if isinstance(raw_data.columns, pd.MultiIndex):
            t_data = raw_data[ticker]
            if "Adj Close" in t_data.columns and not t_data["Adj Close"].dropna().empty:
                prices_df[ticker] = t_data["Adj Close"].dropna()
            else:
                failed_tickers.append(ticker)
        else:
            if "Adj Close" in raw_data.columns and not raw_data["Adj Close"].dropna().empty:
                prices_df[ticker] = raw_data["Adj Close"].dropna()
            else:
                failed_tickers.append(ticker)
    else:
        for ticker in tickers:
            if isinstance(raw_data.columns, pd.MultiIndex) and ticker in raw_data.columns.levels[0]:
                t_data = raw_data[ticker]
                if "Adj Close" in t_data.columns and not t_data["Adj Close"].dropna().empty:
                    prices_df[ticker] = t_data["Adj Close"].dropna()
                else:
                    failed_tickers.append(ticker)
            elif not isinstance(raw_data.columns, pd.MultiIndex) and "Adj Close" in raw_data.columns:
                # Should not happen for multiple tickers unless yf changes API
                prices_df[ticker] = raw_data["Adj Close"]
            else:
                failed_tickers.append(ticker)

    if len(failed_tickers) == len(tickers):
        raise DataFetchError("No data found for any of the requested tickers.", failed_tickers=failed_tickers)
    
    # Process and Clean
    cleaned_prices, log_returns = clean_and_validate_data(prices_df)
    
    # Prepare result - convert to dicts with string keys for JSON serialization
    result = {
        "prices": {k: {timestamp.strftime("%Y-%m-%d"): v for timestamp, v in it.items()} for k, it in cleaned_prices.to_dict().items()},
        "returns": {k: {timestamp.strftime("%Y-%m-%d"): v for timestamp, v in it.items()} for k, it in log_returns.to_dict().items()},
        "tickers": list(cleaned_prices.columns),
        "period": period,
        "failed_tickers": failed_tickers
    }
    
    _write_cache(cache_path, result)
    
    # Return with DataFrames for local python usage
    return {
        "prices": cleaned_prices,
        "returns": log_returns,
        "tickers": list(cleaned_prices.columns),
        "period": period,
        "failed_tickers": failed_tickers
    }

def filter_illiquid_tickers(tickers: list[str], min_volume: float = 1_000_000) -> dict:
    """
    Downloads last 30 days of data and filters out tickers with average daily volume
    (Price * Volume) below min_volume. Targeted at BMV (.MX).
    Returns {"valid_tickers": [], "rejected_tickers": {"ticker": "reason"}}
    """
    if not tickers:
        return {"valid_tickers": [], "rejected_tickers": {}}
        
    try:
        data = yf.download(tickers, period="1mo", group_by="ticker", auto_adjust=False, threads=True)
    except Exception as e:
        logger.warning(f"Error downloading for liquidity check: {e}")
        return {"valid_tickers": tickers, "rejected_tickers": {}} 

    valid_tickers = []
    rejected_tickers = {}

    if len(tickers) == 1:
        ticker = tickers[0]
        if data.empty:
            rejected_tickers[ticker] = "No data returned"
        else:
            if isinstance(data.columns, pd.MultiIndex):
                t_data = data[ticker]
            else:
                t_data = data
                
            if "Volume" not in t_data.columns or "Close" not in t_data.columns:
                rejected_tickers[ticker] = "No volume/close data"
            else:
                adv = (t_data["Volume"] * t_data["Close"]).mean()
                if adv < min_volume or np.isnan(adv):
                    rejected_tickers[ticker] = f"ADV ${adv:,.2f} < ${min_volume:,.2f}"
                else:
                    valid_tickers.append(ticker)
    else:
        for ticker in tickers:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.levels[0]:
                    rejected_tickers[ticker] = "No data returned"
                    continue
                t_data = data[ticker]
            else:
                t_data = data
                
            if "Volume" not in t_data.columns or "Close" not in t_data.columns:
                rejected_tickers[ticker] = "No volume/close data"
                continue
                
            adv = (t_data["Volume"] * t_data["Close"]).mean()
            if adv < min_volume or pd.isna(adv):
                rejected_tickers[ticker] = f"ADV ${adv:,.2f} < ${min_volume:,.2f}" if not pd.isna(adv) else "ADV is NaN"
            else:
                valid_tickers.append(ticker)

    return {
        "valid_tickers": valid_tickers,
        "rejected_tickers": rejected_tickers
    }

def clean_and_validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans and validates price data:
    1. Forward fill NaN (max 5 days)
    2. Align dates: inner join of dates
    3. Validate minimum 252 days
    4. Detect outliers (diff > 50%)
    5. Calculate log returns
    """
    
    # 1. Forward fill (max 5 days)
    df = df.ffill(limit=5)
    
    # 2. Align dates and inner join (drop rows where any ticker has NA)
    df = df.dropna(how='any')
    
    # 3. Validate min 252 days
    if len(df) < 252:
        raise DataFetchError(f"Insufficient data: only {len(df)} days available, minimum 252 required.")
        
    # 4. Detect outliers
    pct_change = df.pct_change()
    outlier_mask = (pct_change > 0.5) | (pct_change < -0.5)
    if outlier_mask.any().any():
        outliers = outlier_mask.sum()
        for col in outliers[outliers > 0].index:
            logger.warning(f"Outlier detected for {col}: {outliers[col]} days with >50% variation.")
            
    # 5. Calculate logarithmic returns
    # shift(1) is previous day, log(price/prev_price)
    log_returns = np.log(df / df.shift(1)).dropna(how='all')
    
    # log returns still might have NA in the first row due to shift, drop them and align prices
    df = df.loc[log_returns.index]
    
    return df, log_returns

def get_benchmarks(period: str) -> dict:
    """
    Downloads ^MXX and ^GSPC for the period.
    Also returns CETES growth (hardcoded using CETES_RATE env var).
    Returns the growth of $10,000 in each benchmark.
    """
    INITIAL_AMOUNT = 10000.0
    
    # Fetch CETES rate
    cetes_rate_str = os.environ.get("CETES_RATE", "0.105") # 10.5% default
    try:
        cetes_rate = float(cetes_rate_str)
    except ValueError:
        cetes_rate = 0.105
        
    # Translate period to years
    if period == "1y": years = 1
    elif period == "3y": years = 3
    elif period == "5y": years = 5
    else: years = 1
        
    cetes_final = INITIAL_AMOUNT * ((1 + cetes_rate) ** years)
    
    benchmarks = {"^MXX": "IPC", "^GSPC": "S&P500"}
    bm_results = {}
    
    try:
        data = yf.download(list(benchmarks.keys()), period=period, group_by="ticker", auto_adjust=False, threads=True)
        for ticker, name in benchmarks.items():
            if len(benchmarks) == 1:
                t_data = data
            elif isinstance(data.columns, pd.MultiIndex):
                if ticker in data.columns.levels[0]:
                    t_data = data[ticker]
                else:
                    t_data = pd.DataFrame()
            else:
                t_data = data
                
            if "Adj Close" in t_data.columns and not t_data["Adj Close"].dropna().empty:
                prices = t_data["Adj Close"].dropna()
                first_price = prices.iloc[0]
                last_price = prices.iloc[-1]
                growth = (last_price / first_price)
                bm_results[name] = INITIAL_AMOUNT * growth
            else:
                bm_results[name] = None
    except Exception as e:
        logger.warning(f"Error fetching benchmarks: {e}")
        bm_results["IPC"] = None
        bm_results["S&P500"] = None
        
    return {
        "IPC_final": bm_results.get("IPC"),
        "S&P500_final": bm_results.get("S&P500"),
        "CETES_final": cetes_final,
        "initial_amount": INITIAL_AMOUNT,
        "period": period,
        "cetes_rate_used": cetes_rate
    }
