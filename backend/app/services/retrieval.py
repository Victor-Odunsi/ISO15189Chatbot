import os
import logging
import pickle
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
BM25_CORPUS_PATH = Path(PERSIST_DIR) / "bm25_corpus.pkl"

DENSE_K = 15
SPARSE_K = 15
RRF_K = 60
RERANK_TOP_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


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


@lru_cache(maxsize=1)
def get_bm25_retriever():
    if not BM25_CORPUS_PATH.exists():
        logger.warning(
            f"No BM25 corpus at {BM25_CORPUS_PATH}; sparse retrieval will return "
            "nothing until ingestion has run."
        )
        return None
    with open(BM25_CORPUS_PATH, "rb") as f:
        corpus: dict[str, Document] = pickle.load(f)
    retriever = BM25Retriever.from_documents(list(corpus.values()))
    retriever.k = SPARSE_K
    return retriever


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = RRF_K
) -> list[Document]:
    """
    Merge multiple ranked document lists into one, using Reciprocal Rank
    Fusion: score(d) = sum(1 / (k + rank_in_list)). Documents are matched
    and deduped across lists by their `chunk_id` metadata (assigned at
    ingestion time), so the same chunk retrieved by both dense and sparse
    search is counted once with a combined score.
    """
    scores: dict[str, float] = {}
    doc_by_id: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            doc_by_id.setdefault(chunk_id, doc)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [doc_by_id[chunk_id] for chunk_id in ranked_ids]


def rerank(query: str, documents: list[Document], top_k: int = RERANK_TOP_K) -> list[Document]:
    if not documents:
        return []
    pairs = [(query, doc.page_content) for doc in documents]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def hybrid_retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[Document]:
    """
    Dense (Chroma) + sparse (BM25) retrieval, merged with Reciprocal Rank
    Fusion, then reranked with a local cross-encoder for the final top_k.
    Blocking/CPU-bound end to end; call via asyncio.to_thread from async code.
    """
    dense_docs = get_chroma().similarity_search(query, k=DENSE_K)

    bm25 = get_bm25_retriever()
    sparse_docs = bm25.invoke(query) if bm25 is not None else []

    fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
    return rerank(query, fused, top_k=top_k)
