from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import database_url, sync_database_url

engine = create_async_engine(database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Dense (pgvector) retrieval runs off the request/response async path (see
# services/retrieval.py, called via asyncio.to_thread), so it uses a plain
# sync engine rather than juggling a second event loop.
sync_engine = create_engine(sync_database_url, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False, class_=Session)
