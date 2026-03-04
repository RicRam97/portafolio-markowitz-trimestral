import yaml
from data import descargar_lotes, extraer_panel, limpiar_retornos

with open("tickers.yaml", "r") as f:
    tickers = yaml.safe_load(f)["tickers"]

raw, fallidos = descargar_lotes(tickers + ["^GSPC"], start="2023-01-01", end="2026-01-01")
prices = extraer_panel(raw, tickers + ["^GSPC"], "Close")

aapl = prices["AAPL"]
print(f"Total AAPL rows: {len(aapl)}")
print(f"Total AAPL NaNs: {aapl.isna().sum()}")
print("Dates where AAPL is NaN:")
print(aapl[aapl.isna()].index.tolist()[:30])
