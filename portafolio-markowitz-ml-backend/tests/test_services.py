"""
Unit tests for pure-Python services (no DB/network required).
"""
import pytest
from app.services.test_suenos import calcular_retorno_minimo
from app.services.test_tolerancia import calcular_tolerancia


# ---------------------------------------------------------------------------
# test_suenos
# ---------------------------------------------------------------------------

def test_retorno_minimo_no_necesario():
    """Si el ahorro puro alcanza la meta, retorno = 0."""
    result = calcular_retorno_minimo(
        meta_dinero=100_000,
        capital_inicial=50_000,
        ahorro_mensual=5_000,
        anos_horizonte=10,
    )
    assert result["retorno_minimo_requerido"] == 0.0
    assert result["factible"] is True
    assert result["nivel"] == "conservador"


def test_retorno_minimo_requiere_rendimiento():
    """Meta que requiere rendimiento positivo (ahorro solo no alcanza)."""
    result = calcular_retorno_minimo(
        meta_dinero=5_000_000,
        capital_inicial=100_000,
        ahorro_mensual=5_000,
        anos_horizonte=15,
    )
    assert isinstance(result["retorno_minimo_requerido"], float)
    assert result["retorno_minimo_requerido"] > 0
    assert "horizonte_anos" in result
    assert result["horizonte_anos"] == 15


def test_retorno_minimo_impracticable():
    """Meta inalcanzable matemáticamente debe lanzar ValueError."""
    with pytest.raises(ValueError):
        calcular_retorno_minimo(
            meta_dinero=10_000_000_000,
            capital_inicial=0,
            ahorro_mensual=100,
            anos_horizonte=1,
        )


# ---------------------------------------------------------------------------
# test_tolerancia
# ---------------------------------------------------------------------------

PERFIL_CONSERVADOR = dict(
    fondo_emergencia=False,
    necesita_dinero_2a=True,
    tiene_deudas=True,
    ha_invertido=False,
    entiende_acciones=False,
    conoce_volatilidad=False,
    caida_15="vender",
    certeza_vs_riesgo="certeza",
    preocupacion="perder",
)

PERFIL_AGRESIVO = dict(
    fondo_emergencia=True,
    necesita_dinero_2a=False,
    tiene_deudas=False,
    ha_invertido=True,
    entiende_acciones=True,
    conoce_volatilidad=True,
    caida_15="comprar",
    certeza_vs_riesgo="riesgo",
    preocupacion="oportunidad",
)


def test_tolerancia_conservador():
    result = calcular_tolerancia(**PERFIL_CONSERVADOR)
    assert result["perfil"].lower() == "conservador"
    assert result["puntaje_total"] == 0
    assert result["volatilidad_maxima"] == 0.08


def test_tolerancia_agresivo():
    result = calcular_tolerancia(**PERFIL_AGRESIVO)
    assert result["perfil"].lower() == "agresivo"
    assert result["puntaje_total"] == 20
    assert result["volatilidad_maxima"] == 0.30


def test_tolerancia_estructura_respuesta():
    result = calcular_tolerancia(**PERFIL_AGRESIVO)
    assert "perfil" in result
    assert "volatilidad_maxima" in result
    assert "puntaje_total" in result
    assert "puntaje_por_dimension" in result
    assert "descripcion_perfil" in result
    dim = result["puntaje_por_dimension"]
    assert "situacion_financiera" in dim
    assert "experiencia_inversora" in dim
    assert "perfil_emocional" in dim
