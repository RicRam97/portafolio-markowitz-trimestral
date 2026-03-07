# config.py — Configuración central, logging y variables de entorno
import os
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ── Cargar .env (solo afecta desarrollo local; Railway inyecta vars directo) ──
load_dotenv()

# ── Directorios ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Entorno ──────────────────────────────────────────────────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# ── Supabase ─────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
CRON_SECRET = os.getenv("CRON_SECRET", "dev_cron_secret_123")

# -- FMP (Financial Modeling Prep) ────────────────────────
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def get_supabase_client():
    """Retorna un cliente Supabase con service_role para operaciones del backend."""
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY deben estar configuradas.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ── CORS ─────────────────────────────────────────────────────
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:5174,https://kaudal.com.mx,https://www.kaudal.com.mx")
CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]

# ── Rate Limiting ────────────────────────────────────────────
RATE_LIMIT_AUTH = os.getenv("RATE_LIMIT_AUTH", "10/minute")
RATE_LIMIT_ANON = os.getenv("RATE_LIMIT_ANON", "3/minute")

# ── Parámetros del pipeline ──────────────────────────────────
LOOKBACK_YEARS = 3
BATCH_SIZE = 35
COVERAGE_THRESHOLD = 0.98
OUTLIER_SIGMA = 6
MAX_WEIGHT = 0.15
SHRINKAGE_ALPHA = 0.25          # intensidad de shrinkage sobre mu
LAMBDA_L2 = 1e-5                # regularización L2 en el optimizador
EF_POINTS = 25                  # puntos de la frontera eficiente
RF = 0.0                        # tasa libre de riesgo

# ── Tickers ──────────────────────────────────────────────────
def cargar_tickers(path: Path | None = None) -> list[str]:
    """Carga la lista de tickers desde tickers.yaml."""
    path = path or (PROJECT_ROOT / "tickers.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    tickers = data.get("tickers", [])
    if not tickers:
        raise ValueError(f"No se encontraron tickers en {path}")
    return tickers

# ── Logging ──────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configura y devuelve el logger del proyecto."""
    logger = logging.getLogger("portafolio")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

log = setup_logging()
