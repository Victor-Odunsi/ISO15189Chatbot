import hashlib
import logging
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from app.services.retrieval import (
    BM25_CORPUS_PATH,
    PERSIST_DIR,
    get_bm25_retriever,
    get_chroma,
    get_embeddings,
)

logger = logging.getLogger(__name__)


def _stable_chunk_id(source: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{chunk_index}:{content}".encode("utf-8"))
    return digest.hexdigest()


def _assign_chunk_ids(docs: list) -> list[str]:
    """Stable, source-relative chunk IDs, used as both the Chroma document
    ID and the BM25 corpus key so hybrid retrieval can match/dedupe the
    same chunk across dense and sparse result lists."""
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
    if not BM25_CORPUS_PATH.exists():
        return {}
    with open(BM25_CORPUS_PATH, "rb") as f:
        return pickle.load(f)


def _save_bm25_corpus(corpus: dict) -> None:
    BM25_CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(corpus, f)


def _reset_indexes() -> None:
    try:
        Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings()).delete_collection()
    except Exception as e:
        logger.info(f"No existing Chroma collection to reset ({e})")
    if BM25_CORPUS_PATH.exists():
        BM25_CORPUS_PATH.unlink()


def run_ingestion(data_path: str = "./data"):
    """
    Full rebuild: loads every PDF under data_path, splits, embeds, and
    persists into both the dense (Chroma) and sparse (BM25) indexes.
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

    _reset_indexes()
    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=PERSIST_DIR, ids=chunk_ids)
    _save_bm25_corpus(dict(zip(chunk_ids, docs)))

    logger.info(f"Stored {len(docs)} chunks in {PERSIST_DIR} (dense + sparse indexes).")
    return db


def ingest_file(file_path: str):
    """
    Incrementally (re-)ingest a single file: replaces any existing chunks
    from this source in both indexes first, so uploading the same file
    twice -- or a revised version of it with a different chunk count --
    doesn't leave duplicate or stale content behind.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    chunk_ids = _assign_chunk_ids(docs)

    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())
    db.delete(where={"source": file_path})
    db.add_documents(docs, ids=chunk_ids)

    corpus = _load_bm25_corpus()
    corpus = {cid: doc for cid, doc in corpus.items() if doc.metadata.get("source") != file_path}
    corpus.update(dict(zip(chunk_ids, docs)))
    _save_bm25_corpus(corpus)

    # Both retrievers are lru_cache'd for the process lifetime; without
    # this, chat requests would keep serving the pre-upload index until
    # the server restarted.
    get_chroma.cache_clear()
    get_bm25_retriever.cache_clear()

    logger.info(f"Ingested {len(docs)} chunks from {file_path} (dense + sparse indexes refreshed).")
    return db
