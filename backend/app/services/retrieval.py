import os
import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"}
    )


@lru_cache(maxsize=1)
def get_chroma():
    logger.info(f'Loading ChromaDB from {PERSIST_DIR}')
    try:
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
        logger.info('Successfully loaded and cached ChromaDB')
        return db
    except Exception as e:
        logger.error(f"Error in Loading Chroma DB: {e}")
