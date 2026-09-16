import logging
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from app.core.config import groq_api_key

logger = logging.getLogger(__name__)

# Both model names below were verified live against each provider's
# current /v1/models catalog -- the originals this file used
# (llama-3.1-8b-instant on Groq, mistral-small-3.1 on Mistral) have
# since been retired by their providers and now 404.
GROQ_MODEL = "openai/gpt-oss-20b"
MISTRAL_MODEL = "mistral-small-latest"


@lru_cache(maxsize=1)
def get_llm():
    try:
        return ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=groq_api_key,
            temperature=0.0,
            streaming=True,
            # gpt-oss is a reasoning model; low effort + hidden reasoning
            # keeps latency/cost down and keeps .content free of
            # chain-of-thought text for this grounded Q&A use case.
            reasoning_effort="low",
            reasoning_format="hidden",
        )
    except Exception as e:
        logger.error(f'Groq Initialization failed: {e}')
        return ChatMistralAI(
            model=MISTRAL_MODEL,
            temperature=0.0,
            streaming=True
        )
