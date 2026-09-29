from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import database_url, sync_database_url

# Disables asyncpg's server-side prepared-statement cache. Harmless against
# a direct Postgres connection, but required against a transaction-mode
# pooler (e.g. Supabase's Supavisor, or PgBouncer) -- in that mode, each
# query can be routed to a different underlying connection, and a
# prepared statement created on one won't exist on another, surfacing as
# "prepared statement ... does not exist" errors. Necessary the moment
# DATABASE_URL points at a pooled endpoint (the norm on Lambda, where many
# concurrent invocations would otherwise exhaust Postgres's connection
# limit -- see the pooling note in README).
engine = create_async_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"statement_cache_size": 0, "timeout": 10},
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Dense (pgvector) retrieval runs off the request/response async path (see
# services/retrieval.py, called via asyncio.to_thread), so it uses a plain
# sync engine rather than juggling a second event loop. prepare_threshold
# disables psycopg's own server-side prepared statements -- same pooler
# incompatibility as asyncpg's statement_cache_size above, since this
# points at the same (potentially pooled) database.
sync_engine = create_engine(
    sync_database_url,
    pool_pre_ping=True,
    connect_args={"prepare_threshold": None},
)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False, class_=Session)
