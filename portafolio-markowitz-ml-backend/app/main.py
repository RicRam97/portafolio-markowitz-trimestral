import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

# Initialize FastAPI
app = FastAPI(
    title="Kaudal ML Backend",
    description="Backend de Machine Learning para optimización de portafolios",
    version="1.0.0"
)

# CORS configuration explicitly for Vercel
# Allows requests from kaudal.com.mx and its subdomains
origins = [
    "https://kaudal.com.mx",
    "https://www.kaudal.com.mx",
    "http://localhost:5173", # For local Vite development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Supabase Connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # We use service role for ML bypassing

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing Supabase credentials in environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Railway required Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ml-backend"}

# Basic Example Route using Supabase
@app.get("/api/ml/status")
def get_ml_status(supabase: Client = Depends(get_supabase)):
    try:
        # Example query just to confirm DB connection works
        res = supabase.table("activos_precios").select("ticker").limit(1).execute()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
        
    return {
        "status": "online",
        "database": db_status
    }

# Here you would include your routers:
# from app.routers import predictions, training
# app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
# app.include_router(training.router, prefix="/api/training", tags=["Training"])
