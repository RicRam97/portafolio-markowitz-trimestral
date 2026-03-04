import requests
import pandas as pd
import json
from pathlib import Path
import io
import re

def get_us_tickers():
    """
    Obtiene la lista de emisoras del S&P 500 raspando Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) US_Scraper/1.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0] # La primera tabla es la de los componentes
        
        if 'Symbol' not in df.columns:
            raise ValueError("No se encontró la columna 'Symbol' en la tabla de Wikipedia.")
            
        records = df.to_dict('records')
        
        enriched_tickers = {}
        for r in records:
            # Yahoo Finance usa . en lugar de - para algunas acciones (como BRK.B en vez de BRK-B)
            # pero el scraper usualmente trae BRK.B o BRK.A de Wikipedia o con puntos/guiones.
            t = str(r.get('Symbol', '')).strip()
            # Yahoo usa '-' en sus URLs e IDs, pero Wikipedia a veces usa puntitos
            t_clean = t.replace('.', '-')
            
            if t_clean:
                name = str(r.get('Security', t_clean))
                sector = str(r.get('GICS Sector', 'N/A'))
                
                enriched_tickers[t_clean] = {
                    "symbol": t_clean,
                    "name": name,
                    "sector": sector,
                    "market": "US"
                }
                
        clean_tickers = [enriched_tickers[k] for k in sorted(enriched_tickers.keys())]
        return clean_tickers
        
    except Exception as e:
        err_msg = str(e)[:500]
        print(f"Error obteniendo emisoras de US: {err_msg}")
        return []

if __name__ == "__main__":
    print("Iniciando scraper de emisoras US (S&P 500)...")
    emisoras = get_us_tickers()
    
    if emisoras:
        print(f"Se obtuvieron {len(emisoras)} emisoras de US.")
        
        output_file = Path("us_tickers.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "source": "Wikipedia S&P 500",
                "count": len(emisoras),
                "tickers": emisoras
            }, f, indent=4)
            
        print(f"Tickers guardados exitosamente en {output_file.absolute()}")
        print(f"Ejemplo: {emisoras[:3]}")
    else:
        print("No se pudieron obtener emisoras.")
