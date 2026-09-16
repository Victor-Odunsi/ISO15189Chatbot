from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


class FakeAnalysis:
    def __init__(self, standalone_question: str, intent: str):
        self.standalone_question = standalone_question
        self.intent = intent


class FakeGenerationChain:
    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, _inputs):
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def test_chat_streams_expected_event_sequence(client):
    with patch("app.api.chat.get_chat_history", new=AsyncMock(return_value=[])), patch(
        "app.api.chat.insert_message", new=AsyncMock()
    ), patch(
        "app.api.chat.analyze_query",
        new=AsyncMock(return_value=FakeAnalysis("What is clause 4.1?", "general")),
    ), patch(
        "app.api.chat.retrieve_context", return_value=[]
    ), patch(
        "app.api.chat.build_citations", return_value=[]
    ), patch(
        "app.api.chat.get_generation_chain",
        return_value=FakeGenerationChain(["Clause ", "4.1 ", "covers..."]),
    ):
        response = client.post("/chat", json={"question": "What is clause 4.1?", "session_id": "test-session"})

        assert response.status_code == 200
        body = response.text
        assert "event: session" in body
        assert "event: citations" in body
        assert body.count("event: token") == 3
        assert "event: end" in body


def test_chat_emits_error_event_on_failure(client):
    with patch("app.api.chat.get_chat_history", new=AsyncMock(return_value=[])), patch(
        "app.api.chat.insert_message", new=AsyncMock()
    ), patch(
        "app.api.chat.analyze_query", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        response = client.post("/chat", json={"question": "anything", "session_id": "test-session"})

        assert response.status_code == 200
        assert "event: error" in response.text
        assert "event: end" not in response.text
