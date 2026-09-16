import logging

from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from app.core.config import groq_api_key

logger = logging.getLogger(__name__)


def get_llm():
    try:
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0,
            streaming=True
        )
    except Exception as e:
        logger.error(f'Groq Initialization failed: {e}')
        return ChatMistralAI(
            model="mistral-small-3.1",
            temperature=0.0,
            streaming=True
        )


def get_streaming_llm(callbacks=None):
    try:
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.0,
            streaming=True,
            max_retries=1,
            request_timeout=100,
            callbacks=callbacks or []
        )
    except Exception as e:
        logger.error(f'Groq Initialization failed: {e}')
        return ChatMistralAI(
            model="mistral-small-3.1",
            temperature=0.0,
            streaming=True,
            max_retries=1,
            request_timeout=100,
            callbacks=callbacks or []
        )
