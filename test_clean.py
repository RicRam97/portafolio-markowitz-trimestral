import yaml
from data import descargar_lotes, extraer_panel, limpiar_retornos
import traceback

try:
    with open("tickers.yaml", "r") as f:
        tickers = yaml.safe_load(f)["tickers"]
        
    print(f"Downloading {len(tickers)} tickers")
    raw, fallidos = descargar_lotes(tickers + ["^GSPC"], start="2023-01-01", end="2026-01-01")
    
    prices = extraer_panel(raw, tickers + ["^GSPC"], "Close")
    print("Prices shape:", prices.shape)
    
    rets_clean = limpiar_retornos(prices)
    print("Rets clean shape:", rets_clean.shape)

except Exception as e:
    traceback.print_exc()
