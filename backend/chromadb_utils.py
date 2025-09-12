import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

@lru_cache(maxsize=1)
def get_chroma():
    logger.info(f'Loading ChromaDB from {PERSIST_DIR}')
    try:
        embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"}
        )
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        logger.info(f'Successfully loaded and cached ChromaDB')
        return db
    except Exception as e:
        logger.error(f"Error in Loading Chroma DB: {e}")

def run_ingestion(data_path: str = "./data"):
    """
    Loads documents, splits, embeds, and persists them into ChromaDB.
    """
    print("Loading documents...")
    loader = DirectoryLoader(data_path, glob="**/*.pdf")  # or .docx, .txt etc.
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print(f"Split into {len(docs)} chunks.")

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"}
    )

    db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    db.persist()

    print(f"Stored embeddings in {PERSIST_DIR}")
