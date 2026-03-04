import requests
import pandas as pd
import json
from pathlib import Path
import io
import re

def get_bmv_tickers():
    """
    Obtiene la lista de emisoras de la BMV raspando una fuente pública (Wikipedia).
    Esta es una forma legal y gratuita, ideal para no pagar endpoints premium como EODHD.
    """
    url = "https://es.wikipedia.org/wiki/%C3%8Dndice_de_Precios_y_Cotizaciones"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BMV_Scraper/1.0"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML tables to DataFrames
        tables = pd.read_html(io.StringIO(response.text))
        
        # En la pagina del IPC, la tabla de componentes tiene la columna 'Símbolo' o 'Clave'
        df = None
        for table in tables:
            if 'Símbolo' in table.columns or 'Clave de cotización' in table.columns or 'Ticker' in table.columns:
                df = table
                break
                
        if df is None:
            raise ValueError("No se encontró la tabla de constituyentes del IPC.")
            
        col_name = 'Símbolo' if 'Símbolo' in df.columns else ('Clave de cotización' if 'Clave de cotización' in df.columns else 'Ticker')
        name_col = 'Nombre' if 'Nombre' in df.columns else 'Empresa'
        sector_col = 'Sector' if 'Sector' in df.columns else None

        records = df.to_dict('records')
        
        # Limpiar y aniadir .MX (sufijo de Yahoo Finance para Mexico)
        enriched_tickers = {}
        
        for r in records:
            t = str(r.get(col_name, ''))
            # Remover notas de wikipedia como [2]
            t_base = re.sub(r'\[.*\]', '', t)
            # Limpiar caracteres invisibles, espacios, etc
            t_clean = ''.join(c for c in t_base if c.isalnum()).strip()
            
            if t_clean:
                # Yahoo usa sufijo .MX para mercado mexicano
                symbol = f"{t_clean}.MX"
                name = str(r.get(name_col, symbol)).replace('\xa0', ' ')
                sector = str(r.get(sector_col, 'N/A')) if sector_col else 'N/A'
                
                enriched_tickers[symbol] = {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "market": "MX"
                }
                
        # Sorted list of objects
        clean_tickers = [enriched_tickers[k] for k in sorted(enriched_tickers.keys())]
        
        return clean_tickers
        
    except Exception as e:
        err_msg = str(e)[:500]
        print(f"Error obteniendo emisoras: {err_msg}")
        return []

if __name__ == "__main__":
    print("Iniciando scraper de emisoras BMV...")
    emisoras = get_bmv_tickers()
    
    if emisoras:
        print(f"Se obtuvieron {len(emisoras)} emisoras de la BMV.")
        
        # Guardar en un JSON para consumo de frontend/backend
        output_file = Path("bmv_tickers.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "source": "Wikipedia BMV List",
                "count": len(emisoras),
                "tickers": emisoras
            }, f, indent=4)
            
        print(f"Tickers guardados exitosamente en {output_file.absolute()}")
        print(f"Ejemplo: {emisoras[:5]}")
    else:
        print("No se pudieron obtener emisoras.")
