"""
Centralized error codes for the Kaudal API.
Each error raises HTTPException with detail = {"code": ..., "message": ...}
so the frontend can map codes to user-friendly messages.
"""

from fastapi import HTTPException


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    """Raises an HTTPException with a structured detail payload."""
    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )
