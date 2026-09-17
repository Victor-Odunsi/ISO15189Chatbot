import logging
import pickle
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from sqlalchemy import select

from app.core.config import aws_region, s3_bucket
from app.db.models import Chunk
from app.db.session import SyncSessionLocal

logger = logging.getLogger(__name__)

BM25_S3_KEY = "bm25_corpus.pkl"

DENSE_K = 15
SPARSE_K = 15
RRF_K = 60
RERANK_TOP_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

s3_client = boto3.client("s3", region_name=aws_region)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"}
    )


def dense_search(query: str, k: int = DENSE_K) -> list[Document]:
    query_vector = get_embeddings().embed_query(query)
    with SyncSessionLocal() as session:
        rows = (
            session.execute(
                select(Chunk).order_by(Chunk.embedding.cosine_distance(query_vector)).limit(k)
            )
            .scalars()
            .all()
        )
    return [
        Document(
            page_content=row.content,
            metadata={"chunk_id": row.id, "source": row.source, "page": row.page},
        )
        for row in rows
    ]


@lru_cache(maxsize=1)
def get_bm25_retriever():
    if not s3_bucket:
        logger.warning("S3_BUCKET not configured; sparse retrieval disabled.")
        return None
    try:
        obj = s3_client.get_object(Bucket=s3_bucket, Key=BM25_S3_KEY)
        corpus: dict[str, Document] = pickle.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.warning(
                f"No BM25 corpus at s3://{s3_bucket}/{BM25_S3_KEY}; sparse retrieval "
                "will return nothing until ingestion has run."
            )
        else:
            logger.error(f"Error loading BM25 corpus from S3: {e}")
        return None
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
    Dense (pgvector) + sparse (BM25) retrieval, merged with Reciprocal Rank
    Fusion, then reranked with a local cross-encoder for the final top_k.
    Blocking/CPU-bound end to end; call via asyncio.to_thread from async code.
    """
    dense_docs = dense_search(query, k=DENSE_K)

    bm25 = get_bm25_retriever()
    sparse_docs = bm25.invoke(query) if bm25 is not None else []

    fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
    return rerank(query, fused, top_k=top_k)
