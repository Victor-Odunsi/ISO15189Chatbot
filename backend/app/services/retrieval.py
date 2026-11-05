import logging
import os
import pickle
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sqlalchemy import select

from app.core.config import aws_region, s3_bucket
from app.core.tracing import stack_dumps_while_slow, stage
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


# The Docker image pre-downloads both models into HF_CACHE_DIR at build time
# (see backend/Dockerfile) and sets HF_HUB_OFFLINE=1, so a container never
# touches the network to load them. Loading them at request time from HF Hub
# into /tmp is a suspect in retrieval hanging until Lambda's 120s timeout.
# /tmp remains the default for local runs outside Docker, where the first
# call still downloads.
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/tmp/hf_cache")


@lru_cache(maxsize=1)
def get_embeddings():
    # Deferred: langchain_huggingface pulls in torch/sentence_transformers,
    # a genuinely slow import. At module level, it blocked uvicorn from
    # starting at all on a cold container -- even a bare CORS preflight
    # had to wait for it, since nothing can respond until the app finishes
    # importing. Deferring it here means only requests that actually need
    # embeddings (not chitchat, not session/message endpoints) pay that cost.
    #
    # On Lambda this load ran >118s (vs ~5s locally under the same CPU/memory
    # limits), so it's wrapped in periodic stack dumps to show what it's doing.
    # torch and sentence_transformers are imported explicitly, one stage each:
    # HuggingFaceEmbeddings imports them lazily inside its constructor, which
    # previously hid the import cost inside load_embedder.
    with stack_dumps_while_slow("embedder load"):
        with stage("import_torch"):
            import torch  # noqa: F401
        with stage("import_sentence_transformers"):
            import sentence_transformers  # noqa: F401
            from langchain_huggingface import HuggingFaceEmbeddings

        with stage("load_embedder"):
            return HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en",
                model_kwargs={"device": "cpu"},
                cache_folder=HF_CACHE_DIR,
            )


def dense_search(query: str, k: int = DENSE_K) -> list[Document]:
    embeddings = get_embeddings()
    with stage("embed_query"):
        query_vector = embeddings.embed_query(query)
    with stage("dense_sql"), SyncSessionLocal() as session:
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
        with stage("load_bm25_s3"):
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
    with stage("build_bm25_index"):
        retriever = BM25Retriever.from_documents(list(corpus.values()))
    retriever.k = SPARSE_K
    return retriever


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":
    from sentence_transformers import CrossEncoder  # see get_embeddings() above

    with stack_dumps_while_slow("reranker load"), stage("load_reranker"):
        return CrossEncoder(RERANKER_MODEL, cache_folder=HF_CACHE_DIR)


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
    reranker = get_reranker()
    with stage("rerank"):
        scores = reranker.predict(pairs)
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
    with stage("bm25_search"):
        sparse_docs = bm25.invoke(query) if bm25 is not None else []

    with stage("fusion"):
        fused = reciprocal_rank_fusion([dense_docs, sparse_docs])
    return rerank(query, fused, top_k=top_k)
