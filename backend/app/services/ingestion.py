import hashlib
import logging
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from app.services.retrieval import BM25_CORPUS_PATH, get_embeddings, PERSIST_DIR

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
