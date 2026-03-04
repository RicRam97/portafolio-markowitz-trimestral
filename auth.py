# auth.py — Middleware JWT para validar tokens de Supabase Auth
from fastapi import Request, HTTPException, Depends
from jose import jwt, JWTError, ExpiredSignatureError
from config import SUPABASE_JWT_SECRET, log


def verify_supabase_token(token: str) -> dict:
    """
    Decodifica y valida un JWT emitido por Supabase Auth.
    Retorna los claims del token si es válido.
    Lanza HTTPException 401 si no lo es.
    """
    if not SUPABASE_JWT_SECRET:
        log.warning("SUPABASE_JWT_SECRET no configurado — autenticación deshabilitada.")
        raise HTTPException(
            status_code=500,
            detail="Autenticación no configurada en el servidor.",
        )

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase no siempre incluye aud
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Tu sesión expiró. Por favor, inicia sesión nuevamente.",
        )
    except JWTError as e:
        log.warning(f"JWT inválido: {e}")
        raise HTTPException(
            status_code=401,
            detail="Token de autenticación inválido.",
        )


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
        raise HTTPException(
            status_code=401,
            detail="Se requiere autenticación. Incluye el header Authorization: Bearer <token>.",
        )

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
