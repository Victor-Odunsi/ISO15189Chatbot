import json
from unittest.mock import patch

import boto3
from fastapi.testclient import TestClient
from moto import mock_aws


def _client_with_bucket_and_queue():
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")
    sqs = boto3.client("sqs", region_name="us-east-1")
    queue_url = sqs.create_queue(QueueName="test-queue")["QueueUrl"]
    return s3, sqs, queue_url


def test_upload_doc_queues_to_s3_and_sqs():
    with mock_aws():
        s3, sqs, queue_url = _client_with_bucket_and_queue()

        with patch("app.api.admin.sqs_queue_url", queue_url):
            from app.main import app

            client = TestClient(app)
            response = client.post(
                "/admin/upload-doc/",
                headers={"X-Admin-Key": "test-admin-key"},
                files={"file": ("test-doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )

        assert response.status_code == 200
        assert response.json() == {"status": "queued", "key": "uploads/test-doc.pdf"}

        obj = s3.get_object(Bucket="test-bucket", Key="uploads/test-doc.pdf")
        assert obj["Body"].read() == b"%PDF-1.4 fake content"

        messages = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1)
        body = json.loads(messages["Messages"][0]["Body"])
        assert body == {"bucket": "test-bucket", "key": "uploads/test-doc.pdf"}


def test_upload_doc_requires_admin_key():
    with mock_aws():
        _client_with_bucket_and_queue()

        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/admin/upload-doc/",
            files={"file": ("test-doc.pdf", b"content", "application/pdf")},
        )
        assert response.status_code == 401


def test_upload_doc_rejects_non_pdf_filename():
    with mock_aws():
        _client_with_bucket_and_queue()

        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/admin/upload-doc/",
            headers={"X-Admin-Key": "test-admin-key"},
            files={"file": ("notes.txt", b"content", "text/plain")},
        )
        assert response.status_code == 400
