import logging
from functools import lru_cache

import groq
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from app.core.config import (
    gemini_api_key,
    generation_models,
    groq_api_key,
    mistral_api_key,
    router_models,
)

logger = logging.getLogger(__name__)

# Gemini via its OpenAI-compatible endpoint: reuses langchain-openai (already
# a dependency) instead of pulling in the google-genai SDK and its gRPC stack.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Fixed judge model for evals/ -- deliberately NOT the production generation
# chain, so an eval never grades its own model's output (self-preference bias).
JUDGE_MODEL = "openai/gpt-oss-120b"

ROUTER_TIMEOUT_S = 10
GENERATION_TIMEOUT_S = 30

# Errors meaning "this provider can't serve the request right now" -- move on
# to the next model in the chain. Anything else (e.g. a 400 for a malformed
# request) is a bug that a different model won't fix, so it propagates.
# Mistral is only ever last in a chain, so its errors needn't be listed.
FALLBACK_ON = (
    groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError,
    openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError,
)

_API_KEYS = {"groq": groq_api_key, "gemini": gemini_api_key, "mistral": mistral_api_key}


def _build(provider: str, model: str, timeout: int, max_retries: int) -> BaseChatModel:
    common = {"temperature": 0.0, "streaming": True, "timeout": timeout, "max_retries": max_retries}
    if provider == "groq":
        extra = {}
        if model.startswith("openai/gpt-oss"):
            # Reasoning model: low effort + hidden reasoning keeps latency down
            # and keeps .content free of chain-of-thought text.
            extra = {"reasoning_effort": "low", "reasoning_format": "hidden"}
        return ChatGroq(model=model, api_key=groq_api_key, **common, **extra)
    if provider == "gemini":
        return ChatOpenAI(model=model, api_key=gemini_api_key, base_url=GEMINI_BASE_URL, **common)
    if provider == "mistral":
        return ChatMistralAI(model=model, api_key=mistral_api_key, **common)
    raise ValueError(f"Unknown LLM provider {provider!r}")


def _chain_models(specs: str, timeout: int) -> list[BaseChatModel]:
    """Parse "provider:model,provider:model" into models, skipping providers
    with no API key configured. Every model but the last gets max_retries=0:
    on a 429, falling through to the next provider immediately beats
    exponential-backoff retries against a bucket that's already empty."""
    parsed = []
    for spec in (s.strip() for s in specs.split(",") if s.strip()):
        provider, _, model = spec.partition(":")
        if not _API_KEYS.get(provider):
            logger.warning(f"Skipping {spec}: no API key configured for {provider}")
            continue
        parsed.append((provider, model))
    if not parsed:
        raise ValueError(f"No usable models in {specs!r}")
    return [
        _build(provider, model, timeout, max_retries=1 if i == len(parsed) - 1 else 0)
        for i, (provider, model) in enumerate(parsed)
    ]


def with_fallbacks(runnables: list[Runnable]) -> Runnable:
    first, rest = runnables[0], runnables[1:]
    return first.with_fallbacks(rest, exceptions_to_handle=FALLBACK_ON) if rest else first


@lru_cache(maxsize=1)
def get_generation_llm() -> Runnable:
    return with_fallbacks(_chain_models(generation_models, GENERATION_TIMEOUT_S))


@lru_cache(maxsize=1)
def get_router_models() -> list[BaseChatModel]:
    return _chain_models(router_models, ROUTER_TIMEOUT_S)


def get_structured_router(schema) -> Runnable:
    # with_structured_output is per-model, so fallbacks wrap the structured
    # runnables rather than the raw models.
    return with_fallbacks([m.with_structured_output(schema) for m in get_router_models()])


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Judge/helper model for evals/ only -- see JUDGE_MODEL."""
    return _build("groq", JUDGE_MODEL, GENERATION_TIMEOUT_S, max_retries=2)
