"""
Per-request stage timing, emitted as one structured JSON log line per request.

Usage:
    trace = start_trace(route="/chat")
    with stage("history"):
        ...
    log_trace(trace)

The trace lives in a ContextVar, so stage() works anywhere downstream of
start_trace() -- including code run via asyncio.to_thread (which copies the
current context into the worker thread). Outside a trace, stage() is a no-op.

On Lambda, the log line lands in CloudWatch; query it with Logs Insights:
    fields @timestamp, total_ms, ttft_ms, cold_start, intent, stages_ms.analyze_llm
    | filter msg = "request_trace"
    | sort @timestamp desc
"""
import json
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

_current: ContextVar[dict | None] = ContextVar("trace", default=None)

# Module state survives across requests within one container, so only the
# first request a container ever serves sees this as True -- which is what
# separates cold-start outliers from steady-state latency in the numbers.
_cold_start = True


def start_trace(**fields) -> dict:
    global _cold_start
    trace = {
        "msg": "request_trace",
        "request_id": str(uuid.uuid4()),
        "cold_start": _cold_start,
        "stages_ms": {},
        "_open": {},
        "_t0": time.perf_counter(),
        **fields,
    }
    _cold_start = False
    _current.set(trace)
    return trace


def current_trace() -> dict | None:
    return _current.get()


def elapsed_ms(trace: dict) -> float:
    return round((time.perf_counter() - trace["_t0"]) * 1000, 1)


@contextmanager
def stage(name: str):
    trace = _current.get()
    if trace is None:
        yield
        return
    t = time.perf_counter()
    trace["_open"][name] = t
    try:
        yield
    finally:
        trace["_open"].pop(name, None)
        trace["stages_ms"][name] = round((time.perf_counter() - t) * 1000, 1)


@contextmanager
def stack_dumps_while_slow(label: str, every_s: int = 20):
    """
    While the block runs, dump every thread's stack to stderr every `every_s`
    seconds (Python's built-in faulthandler). Stage timings say *where* time
    goes; these dumps say *why* -- successive dumps showing different import
    lines mean slow progress, the same line every time means stuck.

    faulthandler supports one pending timer per process, so don't nest these.
    """
    import faulthandler

    print(f"stack_dumps_while_slow: watching {label} (dumping every {every_s}s if still running)",
          file=sys.stderr, flush=True)
    faulthandler.dump_traceback_later(every_s, repeat=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def log_trace(trace: dict) -> None:
    trace["total_ms"] = elapsed_ms(trace)
    # Stages still running when the request ended -- e.g. a worker thread
    # stuck in a model download when Lambda's timeout cancelled the request.
    # Without this, a hang shows up only as a *missing* stage.
    now = time.perf_counter()
    trace["unfinished_stages_ms"] = {
        name: round((now - t) * 1000, 1) for name, t in list(trace["_open"].items())
    }
    # A bare JSON line on stdout, not logging (whose "INFO:name:" prefix
    # stops CloudWatch Logs Insights from auto-extracting the JSON fields).
    print(json.dumps({k: v for k, v in trace.items() if not k.startswith("_")}), file=sys.stdout, flush=True)
