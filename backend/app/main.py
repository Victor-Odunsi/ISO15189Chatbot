import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware

from app.core.security import limiter
from app.api.chat import router as chat_router
from app.api.admin import router as admin_router

logging.basicConfig(filename='app.log', level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema (including pgvector's `chunks` table) is managed by Alembic
    # migrations (`alembic upgrade head`), not created here — see
    # backend/alembic/. Dense/sparse retrieval connect lazily on first use,
    # so there's nothing to eagerly initialize at startup.
    logging.info('Application Initialization complete')
    yield
    logging.info('Application Shutdown')


app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)
