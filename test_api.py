import sys
from api import OptimizeRequest, optimize_portfolio
import traceback

try:
    req = OptimizeRequest(tickers=["AAPL", "MSFT", "GOOG"], budget=10000, start_date="2023-01-01", end_date="2026-01-01")
    res = optimize_portfolio(req)
    print("SUCCESS")
except Exception as e:
    traceback.print_exc()
