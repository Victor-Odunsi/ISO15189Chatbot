import json
from unittest.mock import patch

import boto3
from moto import mock_aws


def test_ingest_handler_downloads_from_s3_and_ingests():
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        s3.put_object(Bucket="test-bucket", Key="uploads/test-doc.pdf", Body=b"%PDF-1.4 fake content")

        from app.worker import ingest_handler

        event = {
            "Records": [
                {"body": json.dumps({"bucket": "test-bucket", "key": "uploads/test-doc.pdf"})}
            ]
        }

        with patch("app.worker.ingest_handler.ingest_file") as mock_ingest:
            ingest_handler.handler(event, None)

        mock_ingest.assert_called_once_with("/tmp/test-doc.pdf", source="uploads/test-doc.pdf")

        with open("/tmp/test-doc.pdf", "rb") as f:
            assert f.read() == b"%PDF-1.4 fake content"
