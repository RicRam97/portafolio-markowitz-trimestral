"""
Router: POST /api/ml/test-tolerancia
Recibe las 9 respuestas del test, calcula el perfil de riesgo,
persiste en Supabase y dispara el cálculo del perfil combinado.
"""
from __future__ import annotations

import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from supabase import Client, create_client

from app.services.test_tolerancia import calcular_tolerancia, calcular_perfil_combinado

router = APIRouter()


# ---------------------------------------------------------------------------
# Supabase dependency (mirrors the pattern in other routers)
# ---------------------------------------------------------------------------

def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("Missing Supabase credentials in environment variables.")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestToleranciaInput(BaseModel):
    # Dimensión 1: Situación financiera
    fondo_emergencia: bool
    necesita_dinero_2a: bool
    tiene_deudas: bool

    # Dimensión 2: Experiencia inversora
    ha_invertido: bool
    entiende_acciones: bool
    conoce_volatilidad: bool

    # Dimensión 3: Perfil emocional
    caida_15: Literal["vender", "esperar", "comprar"]
    certeza_vs_riesgo: Literal["certeza", "riesgo"]
    preocupacion: Literal["perder", "oportunidad"]

    # Auth — opcional; en producción se extrae del JWT
    user_id: Optional[str] = None


class PuntajePorDimension(BaseModel):
    situacion_financiera: int
    experiencia_inversora: int
    perfil_emocional: int


class TestToleranciaOutput(BaseModel):
    perfil: str
    volatilidad_maxima: float
    puntaje_total: int
    puntaje_por_dimension: PuntajePorDimension
    descripcion_perfil: str


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=TestToleranciaOutput)
def post_test_tolerancia(
    body: TestToleranciaInput,
    supabase: Client = Depends(get_supabase),
) -> TestToleranciaOutput:
    # 1. Calcular perfil
    try:
        resultado = calcular_tolerancia(
            fondo_emergencia=body.fondo_emergencia,
            necesita_dinero_2a=body.necesita_dinero_2a,
            tiene_deudas=body.tiene_deudas,
            ha_invertido=body.ha_invertido,
            entiende_acciones=body.entiende_acciones,
            conoce_volatilidad=body.conoce_volatilidad,
            caida_15=body.caida_15,
            certeza_vs_riesgo=body.certeza_vs_riesgo,
            preocupacion=body.preocupacion,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # 2. Persistir en Supabase (upsert por user_id)
    if body.user_id:
        record = {
            "user_id": body.user_id,
            # Raw inputs
            "resp_fondo_emergencia": body.fondo_emergencia,
            "resp_necesita_dinero_2a": body.necesita_dinero_2a,
            "resp_tiene_deudas": body.tiene_deudas,
            "resp_ha_invertido": body.ha_invertido,
            "resp_entiende_acciones": body.entiende_acciones,
            "resp_conoce_volatilidad": body.conoce_volatilidad,
            "resp_caida_15": body.caida_15,
            "resp_certeza_vs_riesgo": body.certeza_vs_riesgo,
            "resp_preocupacion": body.preocupacion,
            # Computed
            "perfil_resultado": resultado["perfil"],
            "volatilidad_maxima": resultado["volatilidad_maxima"],
            "puntaje_total": resultado["puntaje_total"],
            "puntaje_financiero": resultado["puntaje_por_dimension"]["situacion_financiera"],
            "puntaje_experiencia": resultado["puntaje_por_dimension"]["experiencia_inversora"],
            "puntaje_emocional": resultado["puntaje_por_dimension"]["perfil_emocional"],
        }
        try:
            supabase.table("test_tolerancia").upsert(
                record, on_conflict="user_id"
            ).execute()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Error guardando en base de datos: {exc}",
            )

        # 3. Calcular perfil combinado (sección 3.3) — no-blocking
        calcular_perfil_combinado(user_id=body.user_id, supabase=supabase)

    return TestToleranciaOutput(
        perfil=resultado["perfil"],
        volatilidad_maxima=resultado["volatilidad_maxima"],
        puntaje_total=resultado["puntaje_total"],
        puntaje_por_dimension=PuntajePorDimension(
            **resultado["puntaje_por_dimension"]
        ),
        descripcion_perfil=resultado["descripcion_perfil"],
    )
