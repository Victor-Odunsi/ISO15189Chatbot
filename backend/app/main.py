import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware

from app.core.security import limiter
from app.services.retrieval import get_chroma
from app.db.repository import create_application_logs
from app.api.chat import router as chat_router
from app.api.admin import router as admin_router

logging.basicConfig(filename='app.log', level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        chromadb_instance = get_chroma()
        if chromadb_instance is None:
            logging.info('Chroma DB not loaded')
        create_application_logs()
        logging.info('Application Initialization complete')
    except Exception as e:
        logging.error(f'Error initializing Application: {e}')
    yield
    logging.info('Application Shutdown')


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:8501')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)
