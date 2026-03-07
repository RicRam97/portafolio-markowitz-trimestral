import asyncio
import os
import sys

from config import get_supabase_client, FMP_API_KEY
from data_fetcher.fetcher import get_historical_prices, get_benchmarks

def test_fmp_access():
    print(f"[*] FMP_API_KEY Configurada: {'SI' if FMP_API_KEY else 'NO'}")
    
    try:
        supabase = get_supabase_client()
        print("[*] Supabase client: Conectado")
    except Exception as e:
        print(f"[!] Error Supabase client: {e}")
        return

    print("\n[*] Probando descarga de historical prices para AAPL y MSFT (1y)...")
    try:
        data = get_historical_prices(["AAPL", "MSFT"], "1y")
        print(f"  [+] Descarga exitosa!")
        print(f"  [+] Tickers procesados: {data['tickers']}")
        print(f"  [+] Fechas obtenidas: {len(data['prices'])} días")
        print(f"  [+] Tickers fallidos: {data['failed_tickers']}")
    except Exception as e:
        print(f"  [!] Exception: {e}")
        import traceback
        traceback.print_exc()

    print("\n[*] Probando descarga de Benchmarks (^MXX = IPC, ^GSPC = S&P500)...")
    try:
        benchmarks = get_benchmarks("1y")
        print(f"  [+] Descarga exitosa!")
        print(f"  [+] S&P500_final: {benchmarks['S&P500_final']}")
        print(f"  [+] IPC_final: {benchmarks['IPC_final']}")
    except Exception as e:
        print(f"  [!] Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fmp_access()
