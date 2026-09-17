import hashlib
import logging
import pickle

from botocore.exceptions import ClientError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from sqlalchemy import delete

from app.core.config import s3_bucket
from app.db.models import Chunk
from app.db.session import SyncSessionLocal
from app.services.retrieval import BM25_S3_KEY, get_bm25_retriever, get_embeddings, s3_client

logger = logging.getLogger(__name__)


def _stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{content}".encode("utf-8"))
    return digest.hexdigest()


def _assign_chunk_ids(docs: list) -> list[str]:
    """Stable, source-relative chunk IDs, used as both the `chunks.id`
    primary key and the BM25 corpus key so hybrid retrieval can match/dedupe
    the same chunk across dense and sparse result lists."""
    per_source_index: dict[str, int] = {}
    chunk_ids = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        idx = per_source_index.get(source, 0)
        per_source_index[source] = idx + 1
        chunk_id = _stable_chunk_id(source, idx, doc.page_content)
        doc.metadata["chunk_id"] = chunk_id
        chunk_ids.append(chunk_id)
    return chunk_ids


def _load_bm25_corpus() -> dict:
    try:
        obj = s3_client.get_object(Bucket=s3_bucket, Key=BM25_S3_KEY)
        return pickle.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return {}
        raise


def _save_bm25_corpus(corpus: dict) -> None:
    s3_client.put_object(Bucket=s3_bucket, Key=BM25_S3_KEY, Body=pickle.dumps(corpus))


def _upsert_chunks(docs: list, chunk_ids: list[str]) -> None:
    embeddings = get_embeddings().embed_documents([doc.page_content for doc in docs])
    with SyncSessionLocal() as session:
        for chunk_id, doc, embedding in zip(chunk_ids, docs, embeddings):
            session.merge(
                Chunk(
                    id=chunk_id,
                    source=doc.metadata.get("source", ""),
                    page=doc.metadata.get("page"),
                    content=doc.page_content,
                    embedding=embedding,
                )
            )
        session.commit()


def _delete_source(source: str) -> None:
    with SyncSessionLocal() as session:
        session.execute(delete(Chunk).where(Chunk.source == source))
        session.commit()


def _reset_chunks() -> None:
    with SyncSessionLocal() as session:
        session.execute(delete(Chunk))
        session.commit()


def run_ingestion(data_path: str = "./data"):
    """
    Full rebuild: loads every PDF under data_path, splits, embeds, and
    persists into both the dense (pgvector) and sparse (BM25, S3) indexes.
    Resets both first, so this is safe to re-run from scratch.
    """
    logger.info("Loading documents...")
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(docs)} chunks.")

    chunk_ids = _assign_chunk_ids(docs)

    _reset_chunks()
    _upsert_chunks(docs, chunk_ids)
    _save_bm25_corpus(dict(zip(chunk_ids, docs)))
    get_bm25_retriever.cache_clear()

    logger.info(f"Stored {len(docs)} chunks (dense + sparse indexes).")


def ingest_file(local_path: str, source: str | None = None):
    """
    Incrementally (re-)ingest a single file: replaces any existing chunks
    from this source in both indexes first, so uploading the same file
    twice -- or a revised version of it with a different chunk count --
    doesn't leave duplicate or stale content behind.

    `source` is the stable identifier stored as metadata (e.g. an S3 object
    key) -- kept separate from `local_path` since on Lambda the file only
    exists at a transient /tmp path that isn't a meaningful identifier
    across invocations. Defaults to local_path for local/dev use.
    """
    source = source or local_path
    loader = PyPDFLoader(local_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = source

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    chunk_ids = _assign_chunk_ids(docs)

    _delete_source(source)
    _upsert_chunks(docs, chunk_ids)

    corpus = _load_bm25_corpus()
    corpus = {cid: doc for cid, doc in corpus.items() if doc.metadata.get("source") != source}
    corpus.update(dict(zip(chunk_ids, docs)))
    _save_bm25_corpus(corpus)

    # BM25 is lru_cache'd for the process lifetime; without this, chat
    # requests would keep serving the pre-upload sparse index until the
    # container recycled. Dense (pgvector) reads live data on every query,
    # so it needs no equivalent cache invalidation.
    get_bm25_retriever.cache_clear()

    logger.info(f"Ingested {len(docs)} chunks from {source} (dense + sparse indexes refreshed).")
