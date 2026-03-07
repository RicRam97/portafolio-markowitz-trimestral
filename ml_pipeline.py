import os
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from supabase.client import create_client, Client
from sklearn.linear_model import Ridge
from config import (
    log, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
    FMP_API_KEY, FMP_BASE_URL, cargar_tickers,
)
from data_fetcher import get_historical_prices, DataFetchError


# Initialize Supabase Admin Client
# Requires SERVICE_ROLE_KEY to bypass RLS policies for background inserts
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    log.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set. ML Pipeline will fail if called.")
else:
    # We define it lazily within functions if testing locally, but here for convenience we declare the global
    pass


def get_supabase_admin() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY son necesarios para el pipeline ML.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def ingest_daily_prices():
    """
    Descarga los datos de los últimos 5 días de todos los activos base y los inserta
    en la tabla (hypertable) 'activos_precios' de Supabase.
    """
    log.info("Iniciando Ingesta de Precios Diarios hacia Supabase...")
    supabase = get_supabase_admin()
    
    # 1. Obtener la lista de tickers
    try:
        tickers = cargar_tickers()
    except Exception as e:
        log.error(f"Error cargando tickers: {e}")
        raise

    # 2. Descargar ultimos 5 dias (para asegurar solapamiento fines de semana/feriados)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=5)

    try:
        records_to_upsert = []

        for ticker in tickers:
            url = f"{FMP_BASE_URL}/historical-price-full/{ticker}"
            params = {
                "from": start_dt.isoformat(),
                "to": end_dt.isoformat(),
                "apikey": FMP_API_KEY,
            }
            try:
                resp = httpx.get(url, params=params, timeout=15)
                resp.raise_for_status()
                body = resp.json()
                historical = body.get("historical", [])
                for row in historical:
                    records_to_upsert.append({
                        "ticker": ticker,
                        "fecha": row["date"],
                        "precio_cierre": float(row.get("close", 0.0)),
                        "volumen": int(row.get("volume", 0)),
                    })
            except Exception as e:
                log.warning(f"FMP fallo para {ticker} en ingesta: {e}")

        # 3. Upsert a Supabase
        if not records_to_upsert:
            log.warning("No se encontraron registros válidos para ingestar.")
            return {"status": "warning", "message": "No data found for the date range."}
            
        log.info(f"Subiendo {len(records_to_upsert)} registros a Supabase 'activos_precios'...")
        
        # Upsert en lotes para no saturar REST API (max 1000 items por peticion)
        batch_size = 500
        for i in range(0, len(records_to_upsert), batch_size):
            batch = records_to_upsert[i:i+batch_size]
            res = supabase.table("activos_precios").upsert(batch).execute()
        
        log.info("Ingesta completada correctamente.")
        return {"status": "success", "upserted_records": len(records_to_upsert)}
        
    except Exception as e:
        log.error(f"Fallo crítico en ingest_daily_prices: {e}")
        raise


def train_and_predict():
    """
    Descarga el histórico de precios (o lo simula con get_historical_prices)
    y entrena un modelo predictivo base (Ridge Regression) para estimar
    el retorno mensual esperado.
    Guarda las proyecciones en 'predicciones_ml'.
    """
    log.info("Iniciando Entrenamiento ML y Proyecciones...")
    supabase = get_supabase_admin()
    
    try:
        tickers = cargar_tickers()
    except Exception as e:
        log.error(f"Error cargando tickers: {e}")
        raise
        
    # En un caso ideal consultaríamos 'activos_precios' con SQL a Supabase,
    # pero como el dataset no ha hecho backfill de 5 años, usaremos la funcion del backend `get_historical_prices`
    # para asegurar que siempre haya data de entrenamiento base mientras llenamos la BD.
    
    try:
        data = get_historical_prices(tickers, "2y")
    except DataFetchError as e:
        log.error(f"Falló la descarga de precios para ML: {e}")
        raise
        
    returns_df = data["returns"]
    prices_df = data["prices"]
    
    predictions = []
    fecha_hoy = datetime.now().strftime('%Y-%m-%d')
    
    for ticker in returns_df.columns:
        series = prices_df[ticker].dropna()
        if len(series) < 60:
            continue # Necesitamos al menos unos meses de data
            
        # Feature Engineering ultrabásico para la Regresión
        # X: Momentum a 5, 20, 60 días
        df_ml = pd.DataFrame(index=series.index)
        df_ml['price'] = series
        df_ml['mom_5'] = series.pct_change(5)
        df_ml['mom_20'] = series.pct_change(20)
        df_ml['mom_60'] = series.pct_change(60)
        df_ml['target'] = series.shift(-21) / series - 1 # Retorno futuro a 1 mes (21 dias de trading)
        
        df_ml = df_ml.dropna()
        
        if len(df_ml) < 20: 
            continue
            
        X = df_ml[['mom_5', 'mom_20', 'mom_60']].values
        Y = df_ml['target'].values
        
        # Train Ridge
        model = Ridge(alpha=1.0)
        model.fit(X, Y)
        
        # Predecir sobre hoy
        # Calculamos mom_5, 20, 60 de "hoy"
        mom_5 = series.pct_change(5).iloc[-1]
        mom_20 = series.pct_change(20).iloc[-1]
        mom_60 = series.pct_change(60).iloc[-1]
        
        if pd.isna(mom_5) or pd.isna(mom_20) or pd.isna(mom_60):
            continue
            
        x_pred = np.array([[mom_5, mom_20, mom_60]])
        y_pred = model.predict(x_pred)[0]
        
        # Calcular intervalo simple de confianza usando la dev std de los errores (MSE est.)
        rmse = np.sqrt(np.mean((Y - model.predict(X))**2))
        
        predictions.append({
            "modelo_id": "ridge_momentum_v1",
            "ticker": ticker,
            "fecha_prediccion": fecha_hoy,
            "retorno_estimado": float(y_pred),
            "intervalo_confianza_min": float(y_pred - (1.96 * rmse)),
            "intervalo_confianza_max": float(y_pred + (1.96 * rmse)),
        })
        
    if not predictions:
        log.warning("No se generaron predicciones.")
        return {"status": "warning", "message": "No predictions generated due to lack of data."}
        
    log.info(f"Subiendo {len(predictions)} proyecciones ML a Supabase...")
    
    # Upsert a Supabase
    try:
        # Upserting predicciones_ml
        # Tiene PK (modelo_id, ticker, fecha_prediccion)
        batch_size = 500
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i+batch_size]
            res = supabase.table("predicciones_ml").upsert(batch).execute()
        log.info("Proyecciones ML guardadas y finalizadas.")
    except Exception as e:
        log.error(f"Falla durante el upsert a predicciones_ml: {e}")
        raise
        
    return {"status": "success", "predicted_tickers": len(predictions), "model": "ridge_momentum_v1"}

if __name__ == "__main__":
    # Test rápido si se ejecuta el módulo directo
    log.info("Testing ml_pipeline directamente.")
    # ingest_daily_prices()
    # train_and_predict()
