import json
import logging
import re
from pathlib import Path

import boto3
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from app.core.config import aws_region, s3_bucket, sqs_queue_url
from app.core.security import limiter, require_admin_key

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
FILENAME_PATTERN = re.compile(r"^[\w\-. ]+\.pdf$", re.IGNORECASE)

s3_client = boto3.client("s3", region_name=aws_region)
sqs_client = boto3.client("sqs", region_name=aws_region)


@router.post("/admin/upload-doc/", dependencies=[Depends(require_admin_key)])
@limiter.limit("3/hour")
async def upload_doc(request: Request, file: UploadFile = File(...)):
    """
    Admin endpoint to upload a new document for ingestion. Uploads
    straight to S3 and queues an SQS message for the ingestion worker
    Lambda to pick up -- ingestion doesn't happen in this request, since
    FastAPI's BackgroundTasks aren't reliable once this handler's own
    Lambda execution environment is frozen after the response is sent.
    """
    filename = Path(file.filename or "").name
    if not FILENAME_PATTERN.match(filename):
        raise HTTPException(status_code=400, detail="Only .pdf filenames with safe characters are accepted")

    contents = bytearray()
    while chunk := await file.read(1024 * 1024):
        contents.extend(chunk)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    key = f"uploads/{filename}"
    s3_client.put_object(Bucket=s3_bucket, Key=key, Body=bytes(contents))
    sqs_client.send_message(
        QueueUrl=sqs_queue_url,
        MessageBody=json.dumps({"bucket": s3_bucket, "key": key}),
    )

    logger.info(f"Queued {filename} for ingestion ({len(contents)} bytes, s3 key={key}).")
    return {"status": "queued", "key": key}
