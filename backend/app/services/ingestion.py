import logging

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from app.services.retrieval import get_embeddings, PERSIST_DIR

logger = logging.getLogger(__name__)


def run_ingestion(data_path: str = "./data"):
    """
    Loads documents, splits, embeds, and persists them into ChromaDB.
    """
    logger.info("Loading documents...")
    loader = DirectoryLoader(data_path, glob="**/*.pdf")  # or .docx, .txt etc.
    documents = loader.load()

    logger.info(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    logger.info(f"Split into {len(docs)} chunks.")

    db = Chroma.from_documents(docs, get_embeddings(), persist_directory=PERSIST_DIR)
    db.persist()

    logger.info(f"Stored embeddings in {PERSIST_DIR}")
    return db
