import time, json, traceback
from pydantic import BaseModel
from datetime import datetime, timedelta
start = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")
end = datetime.now().strftime("%Y-%m-%d")

from api import optimize_portfolio, OptimizeRequest
from config import cargar_tickers

tickers = cargar_tickers()[:40]

req = OptimizeRequest(
    tickers=tickers,
    budget=10000,
    start_date=start,
    end_date=end
)

print("--- FIRST RUN (Should Hit Cache or downloading) ---")
t0 = time.time()
try:
    res1 = optimize_portfolio(req)
    t1 = time.time()
    print(f"First run took {t1-t0:.2f}s")
except Exception as e:
    traceback.print_exc()

print("--- SECOND RUN (Should Hit Cache) ---")
t2 = time.time()
try:
    res2 = optimize_portfolio(req)
    t3 = time.time()
    print(f"Second run took {t3-t2:.2f}s")
except Exception as e:
    traceback.print_exc()

