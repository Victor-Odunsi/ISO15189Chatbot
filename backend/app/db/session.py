from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import database_url, sync_database_url

# statement_cache_size=0 only disables asyncpg's own client-side statement
# cache -- it does NOT stop SQLAlchemy's asyncpg dialect from calling the
# driver's connection.prepare(), which auto-names every prepared statement
# from a small per-process counter (__asyncpg_stmt_1__, _2__, ...) when no
# explicit name is given. Against a transaction-mode pooler (Supabase's
# Supavisor, or PgBouncer), two different Lambda execution environments can
# get routed to the same physical backend connection and collide on that
# same counter-based name -- this hit live as DuplicatePreparedStatementError
# on the very first query of a fresh container. prepared_statement_name_func
# makes every name unique instead; NullPool is required alongside it (see
# SQLAlchemy's own "Prepared Statement Name with PGBouncer" docs) since
# unique names are never reused, so pooling connections for reuse would just
# accumulate unused prepared statements server-side.
engine = create_async_engine(
    database_url,
    poolclass=NullPool,
    connect_args={
        "statement_cache_size": 0,
        "timeout": 10,
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
    },
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
