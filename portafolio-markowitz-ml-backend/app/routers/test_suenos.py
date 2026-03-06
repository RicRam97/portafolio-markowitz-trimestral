from enum import Enum
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from supabase import Client

from app.services.test_suenos import calcular_retorno_minimo

router = APIRouter()


class MetaTipo(str, Enum):
    casa = "casa"
    retiro = "retiro"
    educacion = "educacion"
    viaje = "viaje"
    libertad = "libertad"
    otra = "otra"


class Moneda(str, Enum):
    MXN = "MXN"
    USD = "USD"


class TestSuenosInput(BaseModel):
    meta_tipo: MetaTipo
    meta_descripcion: Optional[str] = None
    meta_dinero: float = Field(
        ..., gt=0, description="Monto objetivo en la moneda seleccionada"
    )
    capital_inicial: float = Field(..., ge=0)
    ahorro_mensual: float = Field(..., ge=0)
    anos_horizonte: int = Field(..., ge=1, le=50)
    moneda: Moneda = Moneda.MXN
    user_id: Optional[str] = None  # Se obtiene del token en producción; opcional aquí


class TestSuenosOutput(BaseModel):
    retorno_minimo_requerido: float
    retorno_porcentaje: float
    horizonte_anos: int
    factible: bool
    nivel: str
    mensaje: str


def get_supabase() -> Client:
    import os
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    return create_client(url, key)


@router.post("/", response_model=TestSuenosOutput)
def post_test_suenos(
    body: TestSuenosInput,
    supabase: Client = Depends(get_supabase),
):
    try:
        resultado = calcular_retorno_minimo(
            meta_dinero=body.meta_dinero,
            capital_inicial=body.capital_inicial,
            ahorro_mensual=body.ahorro_mensual,
            anos_horizonte=body.anos_horizonte,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Persist result in Supabase if user_id is provided
    if body.user_id:
        record = {
            "user_id": body.user_id,
            "meta_tipo": body.meta_tipo.value,
            "meta_dinero": body.meta_dinero,
            "capital_inicial": body.capital_inicial,
            "ahorro_mensual": body.ahorro_mensual,
            "anos_horizonte": body.anos_horizonte,
            "retorno_minimo_requerido": resultado["retorno_minimo_requerido"],
            "moneda": body.moneda.value,
        }
        supabase.table("test_suenos").upsert(record, on_conflict="user_id").execute()

    return TestSuenosOutput(**resultado)
