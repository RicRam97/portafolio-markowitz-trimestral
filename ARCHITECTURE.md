# Kaudal — Arquitectura del Repositorio

```
portafolio-markowitz-trimestral/
│
├── 🐍 BACKEND (Python · FastAPI · Railway)
│   ├── api.py                  # Servidor FastAPI — endpoints, CORS, rate limiting
│   ├── auth.py                 # Middleware JWT — valida tokens de Supabase Auth
│   ├── config.py               # Variables de entorno, parámetros del pipeline, logging
│   ├── data.py                 # Ingesta de datos: descarga yfinance, limpieza, filtros
│   ├── optimizer.py            # Optimización: Markowitz (LW), HRP, Monte Carlo, Smart Beta
│   ├── pipeline.py             # Pipeline orquestador (batch/scripts)
│   ├── report.py               # Generador de reportes PDF
│   ├── requirements.txt        # Dependencias Python
│   ├── build.sh                # Script de build para Railway
│   ├── render.yaml             # Configuración de despliegue (Render legacy)
│   │
│   ├── tickers.yaml            # Lista base de tickers
│   ├── bmv_tickers.json        # Catálogo enriquecido BMV (scraper)
│   ├── us_tickers.json         # Catálogo enriquecido S&P 500 (scraper)
│   ├── stats.json              # Contador de optimizaciones
│   │
│   ├── scrapers/               # Scripts de scraping de catálogos
│   │   ├── bmv_scraper.py      #   Scraper BMV → bmv_tickers.json
│   │   └── us_scraper.py       #   Scraper S&P 500 → us_tickers.json
│   │
│   └── tests/                  # Tests unitarios (pytest)
│       ├── test_data.py
│       ├── test_optimizer.py
│       └── test_pipeline.py
│
├── 🌐 FRONTEND (HTML/JS/CSS · Vite · Vercel)
│   └── frontend/
│       ├── index.html           # SPA principal (landing + dashboard + FAQ + About)
│       ├── main.js              # Lógica JS: navegación, optimización, gráficas
│       ├── style.css            # Estilos globales (dark theme fintech)
│       ├── kaudal-logo2.png     # Logo de marca
│       ├── vite.config.js       # Configuración Vite (dev server + build)
│       ├── package.json         # Dependencias Node (solo Vite)
│       │
│       ├── privacidad.html      # Aviso de Privacidad (LFPDPPP)
│       ├── terminos.html        # Términos y Condiciones
│       ├── cookies.html         # Política de Cookies
│       ├── disclaimer.html      # Disclaimer Financiero
│       └── cookie-banner.html   # Snippet del banner de cookies
│
├── 🗄️ SUPABASE (Base de datos + Auth)
│   └── supabase/
│       └── schema.sql           # DDL completo: tablas, RLS, triggers
│
├── 📁 CONFIGURACIÓN
│   ├── .env.example             # Plantilla de variables de entorno (sin secretos)
│   ├── .gitignore               # Archivos excluidos de Git
│   └── logo/                    # Logos de alta resolución
│       └── KAUDAL_LOGO2.PNG
│
└── 📁 GENERADOS (en .gitignore)
    ├── .cache_yf/               # Caché de llamadas a yfinance
    ├── reports/                  # Reportes PDF generados
    └── frontend/dist/           # Build de producción (Vite)
```

## Dominios en producción

| Servicio | Dominio | Hosting |
|----------|---------|---------|
| Frontend | kaudal.com.mx | Vercel |
| Backend  | api.kaudal.com.mx | Railway |
| Auth + DB | *.supabase.co | Supabase |

## Flujo de datos

```
Usuario → kaudal.com.mx (Vercel)
    │
    ├── Auth → Supabase Auth (JWT)
    │
    └── API calls (con JWT) → api.kaudal.com.mx (Railway)
                                    │
                                    ├── Valida JWT
                                    ├── Rate limit check
                                    ├── Descarga datos → Yahoo Finance
                                    ├── Optimiza → Markowitz / HRP
                                    └── Guarda resultados → Supabase DB
```
