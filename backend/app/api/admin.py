import logging
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

from app.core.security import limiter, require_admin_key
from app.services.ingestion import ingest_file

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = Path("./data")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
FILENAME_PATTERN = re.compile(r"^[\w\-. ]+\.pdf$", re.IGNORECASE)


@router.post("/admin/upload-doc/", dependencies=[Depends(require_admin_key)])
@limiter.limit("3/hour")
async def upload_doc(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Admin endpoint to upload a new document and incrementally re-ingest it.
    """
    filename = Path(file.filename or "").name
    if not FILENAME_PATTERN.match(filename):
        raise HTTPException(status_code=400, detail="Only .pdf filenames with safe characters are accepted")

    DATA_DIR.mkdir(exist_ok=True)
    file_path = DATA_DIR / filename

    size = 0
    too_large = False
    with open(file_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                too_large = True
                break
            buffer.write(chunk)

    if too_large:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    background_tasks.add_task(ingest_file, str(file_path))
    logger.info(f"Accepted {filename} for ingestion ({size} bytes).")
    return {"status": "accepted", "filename": filename}
