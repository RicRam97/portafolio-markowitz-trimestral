import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from supabase import Client
import logging

log = logging.getLogger("ml-backend.predictor")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features avanzados basados en series de tiempo para ML Financiero.
    Asume que el DataFrame está ordenado por fecha ascendentemente.
    """
    df = df.copy()

    # Asegurar orden cronológico
    df.sort_values(by="fecha", inplace=True)

    # Retornos pasados (Lags) - Momentum de corto y mediano plazo
    df["ret_1d"] = df["retorno_diario"].shift(1)
    df["ret_5d"] = df["precio_cierre"].pct_change(5).shift(1)
    df["ret_21d"] = df["precio_cierre"].pct_change(21).shift(1)

    # Volatilidad móvil
    df["vol_21d"] = df["retorno_diario"].rolling(21).std() * np.sqrt(252)

    # Medias Móviles (Tendencia)
    df["sma_20"] = df["precio_cierre"].rolling(window=20).mean()
    df["sma_50"] = df["precio_cierre"].rolling(window=50).mean()

    # Distancia a la media (Reversión a la media)
    df["dist_sma_20"] = df["precio_cierre"] / df["sma_20"] - 1

    # Target: Queremos predecir el retorno acumulado de los próximos 21 días (1 mes hábil)
    df["target_21d_fwd"] = df["precio_cierre"].shift(-21) / df["precio_cierre"] - 1

    return df


def train_and_predict_models(supabase: Client, tickers: list[str]) -> dict:
    """
    Descarga la historia desde Supabase, entrena un ensamble (XGBoost + Ridge)
    para cada ticker, genera la predicción actual, y la guarda en predicciones_ml.
    """
    stats = {"success": 0, "failed": 0, "errors": []}

    fecha_hoy_str = datetime.now().strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            # 1. Traer datos de Supabase (últimos 3 años en días hábiles = aprox 750 registros)
            res = (
                supabase.table("activos_precios")
                .select("fecha,precio_cierre,retorno_diario")
                .eq("ticker", ticker)
                .order("fecha", desc=True)
                .limit(750)
                .execute()
            )

            data = res.data
            if len(data) < 100:
                log.warning(
                    f"Insuficientes datos para entrenar {ticker} ({len(data)} filas). Skipeando."
                )
                stats["failed"] += 1
                continue

            df = pd.DataFrame(data)
            df["fecha"] = pd.to_datetime(df["fecha"])
            df.sort_values(by="fecha", inplace=True, ignore_index=True)

            # 2. Feature Engineering
            df_feats = create_features(df)

            # Limpiar NaNs. Necesitamos features limpios y un target conocido para entrenar
            df_train = df_feats.dropna(
                subset=[
                    "ret_1d",
                    "ret_5d",
                    "ret_21d",
                    "vol_21d",
                    "dist_sma_20",
                    "target_21d_fwd",
                ]
            ).copy()

            if len(df_train) < 50:
                log.warning(
                    f"Muy pocos datos válidos tras feature eng para {ticker}. Skipeando."
                )
                stats["failed"] += 1
                continue

            # 3. Preparar variables
            features = ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "dist_sma_20"]
            X_train = df_train[features]
            y_train = df_train["target_21d_fwd"]

            # Obtener el último vector de caracteristicas disponibles (EL PRESENTE)
            # Para predecir el futuro, no usamos dropna en el target porque obviamente el target de hoy es NaN
            último_filtro = df_feats.dropna(subset=features).copy()
            if último_filtro.empty:
                continue

            X_latest = último_filtro[features].iloc[-1:]

            # 4. Escalamiento
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_latest_scaled = scaler.transform(X_latest)

            # 5. Entrenar Modelos
            # Modelo Lineal (Robusto a ruido financiero)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            pred_ridge = ridge.predict(X_latest_scaled)[0]

            # Modelo No-Lineal (XGBoost) - Parametrización conservadora para evitar overfitting en finanzas
            xgb_model = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            xgb_model.fit(X_train, y_train)  # XGBoost no requiere escalamiento
            pred_xgb = xgb_model.predict(X_latest)[0]

            # 6. Ensamble (Promedio simple para reducir varianza algorítmica)
            # En finanzas, los ensambles de modelos diferentes suelen generalizar mucho mejor
            pred_ensamble = (pred_ridge + pred_xgb) / 2

            # 7. Upsert en Supabase
            score_confianza = 0.5  # Placeholder, podríamos usar el métricas (R2 in-sample) como confianza

            record = {
                "ticker": ticker,
                "fecha": fecha_hoy_str,
                "modelo": "Ensemble_Ridge_XGboost_21d",
                "retorno_estimado": float(pred_ensamble),
                "horizonte": "21d",
                "score_confianza": float(score_confianza),
            }

            supabase.table("predicciones_ml").upsert([record]).execute()

            stats["success"] += 1
            log.info(
                f"[{ticker}] Predicción guardada: {pred_ensamble*100:.2f}% (21-días forward)."
            )

        except Exception as e:
            log.error(f"Error entrenando/prediciendo {ticker}: {e}")
            stats["failed"] += 1
            stats["errors"].append(f"[{ticker}] {str(e)}")

    return stats
