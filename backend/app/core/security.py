from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def get_proxied_remote_address(request: Request) -> str:
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    real_ip = request.headers.get('X-Real-IP')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    elif real_ip:
        return real_ip
    else:
        return get_remote_address(request)


limiter = Limiter(key_func=get_proxied_remote_address)
