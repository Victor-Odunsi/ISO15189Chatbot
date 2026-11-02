import asyncio
import json

from starlette.requests import Request

from app.core.security import SessionIdCaptureMiddleware, get_rate_limit_key


def _make_request(session_id_state=None, ip="1.2.3.4"):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/chat",
        "headers": [],
        "client": (ip, 12345),
        "state": {"session_id": session_id_state} if session_id_state is not None else {},
    }
    return Request(scope)


def test_rate_limit_key_includes_session_id_when_present():
    request = _make_request(session_id_state="session-abc", ip="1.2.3.4")
    assert get_rate_limit_key(request) == "1.2.3.4:session-abc"


def test_rate_limit_key_falls_back_to_ip_without_session_id():
    request = _make_request(session_id_state=None, ip="1.2.3.4")
    assert get_rate_limit_key(request) == "1.2.3.4"


def test_rate_limit_key_differs_for_different_sessions_same_ip():
    key_a = get_rate_limit_key(_make_request(session_id_state="session-a", ip="9.9.9.9"))
    key_b = get_rate_limit_key(_make_request(session_id_state="session-b", ip="9.9.9.9"))
    assert key_a != key_b


def test_middleware_extracts_session_id_and_replays_body_to_chat():
    captured = {}

    async def downstream_app(scope, receive, send):
        captured["session_id"] = scope["state"].get("session_id")
        message = await receive()
        captured["body"] = message["body"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = SessionIdCaptureMiddleware(downstream_app)
    body = json.dumps({"question": "hi", "session_id": "session-xyz"}).encode()
    scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        pass

    asyncio.run(middleware(scope, receive, send))

    assert captured["session_id"] == "session-xyz"
    assert captured["body"] == body


def test_middleware_tolerates_malformed_body():
    captured = {}

    async def downstream_app(scope, receive, send):
        captured["session_id"] = scope["state"].get("session_id")
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = SessionIdCaptureMiddleware(downstream_app)
    scope = {"type": "http", "method": "POST", "path": "/chat", "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"not json", "more_body": False}

    async def send(message):
        pass

    asyncio.run(middleware(scope, receive, send))

    assert captured["session_id"] is None


def test_middleware_passes_through_untouched_for_other_routes():
    """Admin uploads (multipart, potentially large) must never be buffered
    by this middleware -- only POST /chat should be intercepted."""
    received_receive = None

    async def downstream_app(scope, receive, send):
        nonlocal received_receive
        received_receive = receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = SessionIdCaptureMiddleware(downstream_app)
    scope = {"type": "http", "method": "POST", "path": "/admin/upload-doc/", "headers": []}

    async def original_receive():
        return {"type": "http.request", "body": b"multipart-stuff", "more_body": False}

    async def send(message):
        pass

    asyncio.run(middleware(scope, original_receive, send))

    assert received_receive is original_receive
