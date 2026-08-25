import os
import logging
from typing import Optional, Dict, Any

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)
_security = HTTPBearer(auto_error=False)

SECRET_KEY = os.getenv("API_TOKEN_SECRET", "insecure-dev-secret-change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")


def _load_static_tokens() -> Dict[str, Dict[str, Any]]:
    """Parse STATIC_TOKENS env var into a token-keyed lookup dict.

    Format: token:role:user_id[:department_id[:linked_student_id]], comma-separated.
    Example: dev-admin:admin:admin_01:1,dev-parent:parent:p_01::42

    A token whose role is exactly "service" is a trusted backend calling on
    behalf of one of its own already-authenticated end users (see
    gateway/main.py's merge logic for /api/v1/ask) — not a normal per-user
    credential. Its own user_id here is just a label for audit logs.
    """
    tokens: Dict[str, Dict[str, Any]] = {}
    raw = os.getenv("STATIC_TOKENS", "")
    if not raw:
        return tokens
    for entry in raw.split(","):
        parts = [p.strip() for p in entry.split(":")]
        if len(parts) < 3:
            logger.warning("Malformed STATIC_TOKENS entry (need token:role:user_id): %s", entry)
            continue
        token_val, role, user_id = parts[0], parts[1], parts[2]
        tokens[token_val] = {
            "role": role,
            "user_id": user_id,
            "department_id": parts[3] if len(parts) > 3 and parts[3] else None,
            "linked_student_id": parts[4] if len(parts) > 4 and parts[4] else None,
            "auth_type": "static",
        }
    return tokens


# Loaded once at startup; reload the module to pick up env changes.
_STATIC_TOKENS: Dict[str, Dict[str, Any]] = _load_static_tokens()


def _decode_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = payload.get("role") or payload.get("scope")
    user_id = payload.get("sub")
    if not role or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing required claims: role, sub",
        )
    return {
        "role": role,
        "user_id": user_id,
        "department_id": payload.get("department_id"),
        "tenant_id": payload.get("tenant_id"),
        "linked_student_id": payload.get("linked_student_id"),
        "auth_type": "jwt",
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_security),
) -> Dict[str, Any]:
    """FastAPI dependency: validates Bearer token (static key or JWT) and returns user context."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials

    # Static API key checked first so service accounts don't pay JWT decode cost.
    if token in _STATIC_TOKENS:
        return _STATIC_TOKENS[token]

    return _decode_jwt(token)
