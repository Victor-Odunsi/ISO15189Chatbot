"""
SQS-triggered Lambda handler for background document ingestion.

Deployed separately from the API (see backend/Dockerfile.worker and the
IngestWorkerFunction in template.yaml) -- this is a plain event-driven
Lambda handler, not an HTTP server, so it needs neither FastAPI nor the
Lambda Web Adapter used by the API function.
"""
import json
import logging

import boto3

from app.core.config import aws_region
from app.services.ingestion import ingest_file

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", region_name=aws_region)


def handler(event, context):
    for record in event["Records"]:
        body = json.loads(record["body"])
        bucket, key = body["bucket"], body["key"]

        local_path = f"/tmp/{key.split('/')[-1]}"
        s3_client.download_file(bucket, key, local_path)

        logger.info(f"Ingesting s3://{bucket}/{key}")
        ingest_file(local_path, source=key)
