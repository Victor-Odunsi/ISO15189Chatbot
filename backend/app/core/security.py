import hmac
import json
import os

from fastapi import HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import redis_url

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


def get_rate_limit_key(request: Request) -> str:
    # IP alone over-groups everyone behind shared NAT/CGNAT; session_id alone
    # is free for a client to rotate (unauthenticated, client-generated) and
    # would make the limit trivially bypassable. Keying on both together
    # keeps IP as the real abuse-resistant backstop while still giving
    # per-conversation granularity on top.
    ip = get_proxied_remote_address(request)
    session_id = getattr(request.state, "session_id", None)
    return f"{ip}:{session_id}" if session_id else ip


class SessionIdCaptureMiddleware:
    """
    Pure ASGI middleware (not Starlette's BaseHTTPMiddleware, which is known
    to interfere with streaming responses) that peeks at the session_id in
    POST /chat's JSON body and stashes it on request.state, so the
    synchronous slowapi key_func (get_rate_limit_key above) can read it --
    slowapi calls key_func(request) synchronously and only passes it the raw
    Request, so there's no way to await the body inside the key_func itself.
    Scoped to exactly the one route that needs it; every other request
    (including the multipart file upload on /admin/upload-doc/) passes
    through untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/chat" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        body = b"".join(body_chunks)

        session_id = None
        try:
            session_id = json.loads(body).get("session_id")
        except (json.JSONDecodeError, AttributeError):
            pass
        scope.setdefault("state", {})["session_id"] = session_id

        replayed = False

        async def receive_wrapper():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, receive_wrapper, send)


limiter = Limiter(key_func=get_rate_limit_key, storage_uri=redis_url)


def require_admin_key(request: Request) -> None:
    provided = request.headers.get("X-Admin-Key", "")
    # Fail closed: an unconfigured ADMIN_API_KEY must never be treated as
    # "no key required" (hmac.compare_digest("", "") is True).
    if not ADMIN_API_KEY or not hmac.compare_digest(provided, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")
