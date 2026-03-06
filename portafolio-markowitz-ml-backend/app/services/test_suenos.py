from scipy.optimize import brentq


def calcular_retorno_minimo(
    meta_dinero: float,
    capital_inicial: float,
    ahorro_mensual: float,
    anos_horizonte: int,
) -> dict:
    pmt_anual = ahorro_mensual * 12
    ahorro_total = capital_inicial + pmt_anual * anos_horizonte

    # Si con puro ahorro ya se alcanza la meta, no se necesita rendimiento
    if ahorro_total >= meta_dinero:
        return {
            "retorno_minimo_requerido": 0.0,
            "retorno_porcentaje": 0.0,
            "horizonte_anos": anos_horizonte,
            "factible": True,
            "nivel": "conservador",
            "mensaje": "Tu meta es alcanzable solo ahorrando, sin necesitar rendimiento de inversión.",
        }

    def ecuacion_fv(r):
        if abs(r) < 1e-10:
            return capital_inicial + pmt_anual * anos_horizonte - meta_dinero
        fv = capital_inicial * (1 + r) ** anos_horizonte
        fv += pmt_anual * (((1 + r) ** anos_horizonte - 1) / r)
        return fv - meta_dinero

    try:
        r_optimo = brentq(ecuacion_fv, -0.5, 2.0)
    except ValueError:
        raise ValueError(
            "La meta no es alcanzable matemáticamente con los parámetros dados. "
            "Intenta aumentar el horizonte, las aportaciones o reducir la meta."
        )

    nivel = _clasificar_nivel(r_optimo)
    mensaje = _generar_mensaje(r_optimo)

    return {
        "retorno_minimo_requerido": round(r_optimo, 4),
        "retorno_porcentaje": round(r_optimo * 100, 2),
        "horizonte_anos": anos_horizonte,
        "factible": r_optimo <= 0.30,
        "nivel": nivel,
        "mensaje": mensaje,
    }


def _clasificar_nivel(r: float) -> str:
    if r <= 0.05:
        return "conservador"
    elif r <= 0.10:
        return "moderado"
    elif r <= 0.20:
        return "agresivo"
    else:
        return "muy_agresivo"


def _generar_mensaje(r: float) -> str:
    if r <= 0.05:
        return "Tu meta es muy conservadora. Un portafolio de renta fija podría alcanzarla."
    elif r <= 0.10:
        return "Tu meta es realista. Un portafolio moderado puede alcanzarla."
    elif r <= 0.20:
        return "Tu meta requiere un portafolio con alta exposición a renta variable."
    elif r <= 0.30:
        return "Tu meta es ambiciosa. Considera extender el horizonte o aumentar las aportaciones."
    else:
        return (
            "Tu meta requiere un rendimiento poco realista (>30% anual). "
            "Extiende el horizonte, aumenta las aportaciones o ajusta la meta."
        )
