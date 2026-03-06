"""
Servicio: Test de Tolerancia al Riesgo
Calcula el perfil de riesgo de un inversionista a partir de 9 preguntas
agrupadas en 3 dimensiones: situación financiera, experiencia y perfil emocional.
"""
from __future__ import annotations

from typing import Literal

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_CAIDA_15_SCORES: dict[str, int] = {
    "vender": 0,
    "esperar": 2,
    "comprar": 3,
}

_CERTEZA_SCORES: dict[str, int] = {
    "certeza": 0,
    "riesgo": 3,
}

_PREOCUPACION_SCORES: dict[str, int] = {
    "perder": 0,
    "oportunidad": 2,
}

# (label, volatilidad_maxima, descripcion)
_PERFIL_MAP: list[tuple[int, int, str, float, str]] = [
    # min, max, label, sigma, descripcion
    (0,  4,  "Conservador",  0.08, (
        "Priorizas la preservación de tu capital por encima del crecimiento. "
        "Un portafolio de renta fija y activos de bajo riesgo es ideal para ti."
    )),
    (5,  8,  "Moderado",     0.12, (
        "Buscas un balance entre seguridad y crecimiento. "
        "Un portafolio mixto con predominancia de renta fija se adapta a tu perfil."
    )),
    (9,  12, "Balanceado",   0.16, (
        "Aceptas cierta volatilidad a cambio de un mayor potencial de rendimiento. "
        "Una mezcla equilibrada entre renta fija y variable es tu punto óptimo."
    )),
    (13, 16, "Crecimiento",  0.22, (
        "Tu objetivo principal es el crecimiento del capital a largo plazo. "
        "Un portafolio con alta exposición a renta variable encaja con tus metas."
    )),
    (17, 20, "Agresivo",     0.30, (
        "Tienes alta tolerancia al riesgo y buscas maximizar el rendimiento. "
        "Un portafolio concentrado en renta variable y activos de alto crecimiento es tu terreno."
    )),
]


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def calcular_tolerancia(
    *,
    fondo_emergencia: bool,
    necesita_dinero_2a: bool,
    tiene_deudas: bool,
    ha_invertido: bool,
    entiende_acciones: bool,
    conoce_volatilidad: bool,
    caida_15: Literal["vender", "esperar", "comprar"],
    certeza_vs_riesgo: Literal["certeza", "riesgo"],
    preocupacion: Literal["perder", "oportunidad"],
) -> dict:
    """
    Retorna un dict con el perfil, volatilidad máxima, puntaje total,
    puntaje por dimensión y descripción del perfil.
    """
    dim1 = (
        (2 if fondo_emergencia else 0)
        + (2 if not necesita_dinero_2a else 0)
        + (2 if not tiene_deudas else 0)
    )
    dim2 = (
        (2 if ha_invertido else 0)
        + (2 if entiende_acciones else 0)
        + (2 if conoce_volatilidad else 0)
    )
    dim3 = (
        _CAIDA_15_SCORES[caida_15]
        + _CERTEZA_SCORES[certeza_vs_riesgo]
        + _PREOCUPACION_SCORES[preocupacion]
    )
    total = dim1 + dim2 + dim3

    perfil, sigma, descripcion = _mapear_perfil(total)

    return {
        "perfil": perfil,
        "volatilidad_maxima": sigma,
        "puntaje_total": total,
        "puntaje_por_dimension": {
            "situacion_financiera": dim1,
            "experiencia_inversora": dim2,
            "perfil_emocional": dim3,
        },
        "descripcion_perfil": descripcion,
    }


def _mapear_perfil(total: int) -> tuple[str, float, str]:
    for min_val, max_val, label, sigma, desc in _PERFIL_MAP:
        if min_val <= total <= max_val:
            return label, sigma, desc
    # Fallback — shouldn't happen with valid input (0–20)
    return "Agresivo", 0.30, _PERFIL_MAP[-1][4]


# ---------------------------------------------------------------------------
# Placeholder: perfil combinado (Sección 3.3)
# This function will merge tolerance + dreams test results into a final
# investor profile once section 3.3 is implemented.
# ---------------------------------------------------------------------------

def calcular_perfil_combinado(user_id: str, supabase) -> dict | None:
    """
    Combines test_tolerancia and test_suenos results to produce a unified
    investor profile. Stub — to be fully implemented in section 3.3.
    """
    try:
        tol = (
            supabase.table("test_tolerancia")
            .select("perfil_resultado, volatilidad_maxima, puntaje_total")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        suenos = (
            supabase.table("test_suenos")
            .select("retorno_minimo_requerido, nivel")
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )

        if not tol.data or not suenos.data:
            return None  # One of the tests is not yet completed

        # Section 3.3 will define the combination logic.
        # For now we simply persist both pieces so they are available.
        perfil_combinado = {
            "user_id": user_id,
            "perfil_riesgo": tol.data["perfil_resultado"],
            "volatilidad_maxima": tol.data["volatilidad_maxima"],
            "retorno_minimo": suenos.data["retorno_minimo_requerido"],
            "nivel_suenos": suenos.data["nivel"],
        }
        supabase.table("perfil_combinado").upsert(
            perfil_combinado, on_conflict="user_id"
        ).execute()
        return perfil_combinado

    except Exception:
        # Non-blocking: tolerance result is already saved; combined profile
        # will be recalculated the next time both tests are available.
        return None
