import hmac
import os

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

TRUSTED_PROXY = os.getenv("TRUSTED_PROXY", "false").lower() == "true"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


def get_proxied_remote_address(request: Request) -> str:
    # Only honor forwarded-for headers behind a trusted reverse proxy that
    # sets/overwrites them itself -- otherwise any client can spoof them to
    # dodge rate limiting.
    if TRUSTED_PROXY:
        x_forwarded_for = request.headers.get('X-Forwarded-For')
        real_ip = request.headers.get('X-Real-IP')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0].strip()
        elif real_ip:
            return real_ip
    return get_remote_address(request)


limiter = Limiter(key_func=get_proxied_remote_address)


def require_admin_key(request: Request) -> None:
    provided = request.headers.get("X-Admin-Key", "")
    # Fail closed: an unconfigured ADMIN_API_KEY must never be treated as
    # "no key required" (hmac.compare_digest("", "") is True).
    if not ADMIN_API_KEY or not hmac.compare_digest(provided, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
