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


def _request_traces(capsys) -> list[dict]:
    import json

    lines = capsys.readouterr().out.splitlines()
    return [json.loads(line) for line in lines if '"request_trace"' in line]


def test_chat_logs_per_stage_trace(client, capsys):
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
        return_value=FakeGenerationChain(["Clause ", "4.1"]),
    ):
        client.post("/chat", json={"question": "What is clause 4.1?", "session_id": "test-session"})

    [trace] = _request_traces(capsys)
    assert trace["intent"] == "general"
    assert set(trace["stages_ms"]) >= {"history", "analyze_llm", "retrieve_total", "generate_llm", "persist"}
    assert "ttft_ms" in trace and trace["ttft_ms"] <= trace["total_ms"]
    assert "error" not in trace


def test_chat_logs_trace_even_when_persist_fails(client, capsys):
    with patch("app.api.chat.get_chat_history", new=AsyncMock(return_value=[])), patch(
        "app.api.chat.insert_message", new=AsyncMock(side_effect=RuntimeError("db down"))
    ), patch(
        "app.api.chat.analyze_query", new=AsyncMock(side_effect=RuntimeError("boom"))
    ):
        try:
            client.post("/chat", json={"question": "anything", "session_id": "test-session"})
        except RuntimeError:
            pass

    [trace] = _request_traces(capsys)
    assert trace["error"] == "RuntimeError"


def test_trace_reports_stage_still_running_when_request_ends(capsys):
    import json

    from app.core.tracing import log_trace, stage, start_trace

    trace = start_trace(route="/chat")
    with stage("retrieve_total"):
        # Simulates Lambda cancelling the request while a worker thread is
        # still inside a stage (the production model-load hang).
        trace["_open"]["load_embedder"] = trace["_t0"]
    log_trace(trace)

    [logged] = [json.loads(l) for l in capsys.readouterr().out.splitlines() if "request_trace" in l]
    assert "load_embedder" in logged["unfinished_stages_ms"]
    assert "retrieve_total" in logged["stages_ms"]
