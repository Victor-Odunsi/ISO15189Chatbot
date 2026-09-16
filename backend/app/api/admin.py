import os
import shutil
import logging

from fastapi import APIRouter, UploadFile, File

from app.services.ingestion import run_ingestion

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = "./data"


@router.post("/admin/upload-doc/")
async def upload_doc(file: UploadFile = File(...)):
    """
    Admin endpoint to upload a new document and re-run ingestion.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_ingestion(DATA_DIR)
    logger.info(f"✅ {file.filename} uploaded and ingested.")
