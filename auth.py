# auth.py — Middleware JWT para validar tokens de Supabase Auth
import httpx
from fastapi import Request, HTTPException, Depends
from jose import jwt, JWTError, ExpiredSignatureError
from config import SUPABASE_JWT_SECRET, SUPABASE_URL, log
from error_codes import api_error

# Cache for JWKS keys (ES256)
_jwks_cache: dict | None = None


def _get_jwks() -> dict:
    """Fetch and cache JWKS from Supabase for ES256 verification."""
    global _jwks_cache
    if _jwks_cache is not None:
        return _jwks_cache

    jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    try:
        resp = httpx.get(jwks_url, timeout=10)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        return _jwks_cache
    except Exception as e:
        log.error(f"Error fetching JWKS from {jwks_url}: {e}")
        raise HTTPException(
            status_code=500,
            detail="No se pudo obtener las claves de autenticación.",
        )


def _get_signing_key(token: str) -> tuple[str | dict, str]:
    """
    Determine the signing key and algorithm from the token header.
    Returns (key, algorithm).
    Supports both ES256 (JWKS) and HS256 (legacy secret).
    """
    header = jwt.get_unverified_header(token)
    alg = header.get("alg", "HS256")

    if alg == "ES256":
        kid = header.get("kid")
        jwks = _get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key, "ES256"
        raise HTTPException(
            status_code=401,
            detail="Clave de firma no encontrada.",
        )

    # Fallback to HS256 with SUPABASE_JWT_SECRET
    if not SUPABASE_JWT_SECRET:
        log.warning("SUPABASE_JWT_SECRET no configurado — autenticación deshabilitada.")
        raise HTTPException(
            status_code=500,
            detail="Autenticación no configurada en el servidor.",
        )
    return SUPABASE_JWT_SECRET, "HS256"


def verify_supabase_token(token: str) -> dict:
    """
    Decodifica y valida un JWT emitido por Supabase Auth.
    Retorna los claims del token si es válido.
    Lanza HTTPException 401 si no lo es.
    """
    try:
        key, alg = _get_signing_key(token)
        payload = jwt.decode(
            token,
            key,
            algorithms=[alg],
            options={"verify_aud": False},
        )
        return payload
    except HTTPException:
        raise
    except ExpiredSignatureError:
        api_error(401, "AUTH_SESSION_EXPIRED", "Tu sesion expiro. Por favor, inicia sesion nuevamente.")
    except JWTError as e:
        log.warning(f"JWT inválido: {e}")
        api_error(401, "AUTH_REQUIRED", "Token de autenticacion invalido.")


def _extract_bearer(request: Request) -> str | None:
    """Extrae el token Bearer del header Authorization."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_current_user(request: Request) -> dict:
    """
    Dependency de FastAPI: requiere autenticación.
    Retorna dict con: sub (user_id), email, role, etc.
    Uso: user = Depends(get_current_user)
    """
    token = _extract_bearer(request)
    if not token:
        api_error(401, "AUTH_REQUIRED", "Se requiere autenticacion. Incluye el header Authorization: Bearer <token>.")

    payload = verify_supabase_token(token)

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email", ""),
        "role": payload.get("role", "authenticated"),
        "plan": payload.get("user_metadata", {}).get("plan", "free"),
    }


async def get_optional_user(request: Request) -> dict | None:
    """
    Dependency de FastAPI: autenticación opcional.
    Retorna el usuario si el token es válido, None si no hay token.
    Lanza 401 solo si el token existe pero es inválido.
    Uso: user = Depends(get_optional_user)
    """
    token = _extract_bearer(request)
    if not token:
        return None

    payload = verify_supabase_token(token)

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email", ""),
        "role": payload.get("role", "authenticated"),
        "plan": payload.get("user_metadata", {}).get("plan", "free"),
    }
