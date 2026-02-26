# config.py — Configuración central y logging
import logging
import yaml
from pathlib import Path

# ── Directorios ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"

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
