from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import os, json, re
import uuid
import logging
from pydantic_utils import QueryInput
from chromadb_utils import get_chroma
from chromadb_utils import run_ingestion
from langchain_utils import get_chat_agent
from sqldb_utils import get_chat_history
from sqldb_utils import create_application_logs
from sqldb_utils import insert_application_logs
from contextlib import asynccontextmanager
import asyncio
from langchain_utils import DummyHandler

logging.basicConfig(filename = 'app.log', level = logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        chromadb_instance = get_chroma()
        if chromadb_instance is None:
            logging.info(f'Chroma DB not loaded')
        create_application_logs()
        logging.info(f'Application Initialization complete')
    except Exception as e:
        logging.error(f'Error initializing Application: {e}')
    yield
    logging.info(f'Application Shutdown')


def get_proxied_remote_address(request: Request):
    x_forwarded_for = request.headers.get('x-Forwarded-For')
    real_ip = request.headers.get('X-Real_IP')
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    elif real_ip:
        return real_ip
    else:
        return get_remote_address(request)

limiter = Limiter(key_func = get_proxied_remote_address)

app = FastAPI(lifespan = lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:8501')

app.add_middleware(
    CORSMiddleware,
    allow_origins = [frontend_origin],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

@app.post('/chat')
@limiter.limit("15/minute")
async def chat(request: Request, query: QueryInput):
    session_id = query.session_id or str(uuid.uuid4())
    logging.info(f"'Session ID': {session_id}, User question: {query.question}")

    chat_history = get_chat_history(session_id)
    queue = asyncio.Queue()
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
                logging.error(f"Agent error: {e}")

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
            logging.error(f"Error: {e}")
        finally:
            insert_application_logs(session_id, query.question, full_answer)
            yield (json.dumps({"type": "end"}) + "\n").encode("utf-8")

    return StreamingResponse(token_generator(), media_type='application/x-ndjson')


DATA_DIR = "./data"

@app.post("/admin/upload-doc/")
async def upload_doc(file: UploadFile = File(...)):
    """
    Admin endpoint to upload a new document and re-run ingestion.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    run_ingestion(DATA_DIR)
    logging.info(f"✅ {file.filename} uploaded and ingested.")
    