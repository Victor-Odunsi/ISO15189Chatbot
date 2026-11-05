import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.security import limiter
from app.core.tracing import elapsed_ms, log_trace, stage, start_trace
from app.db.repository import delete_session, get_chat_history, insert_message
from app.models.schemas import QueryInput
from app.services.chain import (
    analyze_query,
    build_citations,
    format_context,
    get_generation_chain,
    retrieve_context,
)
from app.services.memory import get_chat_context, maybe_update_summary

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


@router.post('/chat')
@limiter.limit("15/minute")
async def chat(request: Request, query: QueryInput):
    session_id = query.session_id or str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}, User question: {query.question}")

    async def event_stream():
        full_answer = ""
        trace = start_trace(route="/chat", session_id=session_id)
        try:
            yield _sse("session", {"session_id": session_id})

            with stage("history"):
                # Bounded: running summary + recent turns, not the full transcript.
                chat_history = await get_chat_context(session_id)
            trace["history_msgs"] = len(chat_history)

            with stage("analyze_llm"):
                analysis = await analyze_query(query.question, chat_history)
            trace["intent"] = analysis.intent

            if analysis.intent == "chitchat":
                # Greetings/small talk don't need retrieval -- running it
                # anyway fed GENERAL_PROMPT's "say so if context is
                # insufficient" instruction an irrelevant query, so a plain
                # "hi" got treated as an unanswerable content question.
                citations = []
                payload = {
                    "standalone_question": analysis.standalone_question,
                    "chat_history": chat_history,
                }
            else:
                with stage("retrieve_total"):
                    documents = await asyncio.to_thread(retrieve_context, analysis.standalone_question)
                citations = build_citations(documents)
                payload = {
                    "context": format_context(documents),
                    "standalone_question": analysis.standalone_question,
                    "chat_history": chat_history,
                }
                trace["context_chars"] = len(payload["context"])

            yield _sse("citations", {"citations": citations})

            generation_chain = get_generation_chain(analysis.intent)
            with stage("generate_llm"):
                async for chunk in generation_chain.astream(payload):
                    if not full_answer:
                        # Time-to-first-token from request start: the latency the user feels.
                        trace["ttft_ms"] = elapsed_ms(trace)
                    full_answer += chunk
                    yield _sse("token", {"content": chunk})
            trace["answer_chars"] = len(full_answer)

            yield _sse("end", {})

        except Exception as e:
            logger.exception("Chat error")
            trace["error"] = type(e).__name__
            yield _sse("error", {"message": "Something went wrong generating a response. Please try again."})
        finally:
            try:
                with stage("persist"):
                    await insert_message(session_id, "user", query.question)
                    if full_answer:
                        await insert_message(session_id, "assistant", full_answer)
                # After the user already has their answer (the "end" event went
                # out above), so its LLM call never adds to perceived latency.
                with stage("summarize"):
                    trace["summary_updated"] = await maybe_update_summary(session_id)
            finally:
                # Always emit the trace -- a failing persist (e.g. DB down) is
                # exactly the request whose timings we most need to see.
                log_trace(trace)

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get('/sessions/{session_id}/messages')
async def list_session_messages(session_id: str):
    return await get_chat_history(session_id)


@router.delete('/sessions/{session_id}')
async def remove_session(session_id: str):
    await delete_session(session_id)
    return {"status": "deleted"}
