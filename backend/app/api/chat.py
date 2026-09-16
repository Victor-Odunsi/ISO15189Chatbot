import json
import re
import uuid
import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.core.security import limiter
from app.db.repository import get_chat_history, insert_application_logs
from app.models.schemas import QueryInput
from app.services.chain import get_chat_agent, DummyHandler

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post('/chat')
@limiter.limit("15/minute")
async def chat(request: Request, query: QueryInput):
    session_id = query.session_id or str(uuid.uuid4())
    logger.info(f"'Session ID': {session_id}, User question: {query.question}")

    chat_history = get_chat_history(session_id)
    handler = DummyHandler()
    chat_agent = get_chat_agent(session_id, handler)

    async def token_generator():
        full_answer = ""
        try:
            yield (json.dumps({"type": "session", "session_id": session_id}) + "\n").encode("utf-8")

            try:
                result = await chat_agent.ainvoke({
                    'input': query.question,
                    'chat_history': chat_history
                })

                # Try multiple ways to get the answer
                full_response = ""

                # Method 1: Check if handler captured tool output
                if hasattr(handler, 'found_answer') and handler.found_answer:
                    full_response = handler.final_answer

                # Method 2: Extract from result
                elif isinstance(result, dict):
                    full_response = result.get('output', '') or result.get('answer', '')

                # Method 3: Extract from captured output
                elif hasattr(handler, 'captured_output') and handler.captured_output:
                    if "Final Answer:" in handler.captured_output:
                        full_response = handler.captured_output.split("Final Answer:")[-1].strip()
                    elif "'output':" in handler.captured_output:
                        # Extract from the observation directly
                        match = re.search(r"'output':\s*\"([^\"]+)\"", handler.captured_output)
                        if match:
                            full_response = match.group(1)

                # Fallback
                if not full_response:
                    full_response = "I found information but couldn't format the response properly. Please try again."

            except Exception as e:
                full_response = "I encountered an error processing your request. Please try again."
                logger.error(f"Agent error: {e}")

            # Stream the response
            if full_response:
                # Split by lines to preserve paragraphs and bullet points
                for line in full_response.splitlines(keepends=True):
                    stripped_line = line.strip()
                    if stripped_line:
                        # Stream word by word for pseudo-streaming
                        for word in stripped_line.split():
                            full_answer += word + " "
                            yield (json.dumps({"type": "token", "content": word + " "}) + "\n").encode("utf-8")
                            await asyncio.sleep(0.03)
                    # Send newline after each line to preserve paragraphs/bullets
                    full_answer += "\n"

        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            insert_application_logs(session_id, query.question, full_answer)
            yield (json.dumps({"type": "end"}) + "\n").encode("utf-8")

    return StreamingResponse(token_generator(), media_type='application/x-ndjson')
