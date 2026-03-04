import yaml
from data import descargar_lotes, extraer_panel, limpiar_retornos

with open("tickers.yaml", "r") as f:
    tickers = yaml.safe_load(f)["tickers"]

raw, fallidos = descargar_lotes(tickers + ["^GSPC"], start="2023-01-01", end="2026-01-01")
prices = extraer_panel(raw, tickers + ["^GSPC"], "Close")

returns_raw = prices.pct_change()
coverage = returns_raw.notna().mean()

print("\n--- Coverage Summary ---")
print(f"Mean Coverage: {coverage.mean():.4f}")
print(f"Min Coverage: {coverage.min():.4f}")
print(f"Max Coverage: {coverage.max():.4f}")

# Find total rows
print(f"Total rows: {len(prices)}")

# Print a ticker with worst coverage
worst_ticker = coverage.idxmin()
print(f"\nWorst ticker: {worst_ticker} ({coverage[worst_ticker]:.4f})")
null_counts = prices[worst_ticker].isna().sum()
print(f"Nulls in {worst_ticker}: {null_counts}")

print(prices[worst_ticker].head(10))
print(prices[worst_ticker].tail(10))
