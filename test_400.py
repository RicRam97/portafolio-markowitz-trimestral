import requests
import yaml

with open("tickers.yaml", "r") as f:
    tickers = yaml.safe_load(f)["tickers"]

payload = {
    "tickers": tickers,
    "budget": 10000,
    "start_date": "2023-01-01",
    "end_date": "2026-01-01"
}

r = requests.post("http://localhost:8000/api/optimize", json=payload)
print(f"Optimize: {r.status_code}")
if r.status_code != 200:
    print(r.json())

r2 = requests.post("http://localhost:8000/api/track", json=payload)
print(f"Track: {r2.status_code}")
if r2.status_code != 200:
    print(r2.json())
