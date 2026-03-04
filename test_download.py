import yaml
from data import descargar_lotes
import traceback

try:
    with open("tickers.yaml", "r") as f:
        tickers = yaml.safe_load(f)["tickers"]
        
    print(f"Downloading {len(tickers)} tickers")
    raw, fallidos = descargar_lotes(tickers, start="2023-01-01", end="2026-01-01")
    print(raw.shape)
    print("Fallidos:", fallidos)
except Exception as e:
    traceback.print_exc()
