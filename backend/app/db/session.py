from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import async_database_url, sync_database_url

# Both engines use psycopg 3 with prepare_threshold=None, which never creates
# named server-side prepared statements. That's what makes a real client-side
# pool safe against Supabase's transaction-mode pooler (Supavisor, port 6543):
# successive transactions on one pooled client connection can land on
# different Postgres backends, so any named prepared statement would collide
# or go missing (this hit live as DuplicatePreparedStatementError under
# asyncpg, whose SQLAlchemy dialect always prepares statements).
#
# The pool is the point: without one, every query paid a fresh TCP + TLS +
# SCRAM-auth handshake to the pooler -- ~2s per query from the Lambda region
# to the database region, measured as `history` taking ~2s to return 0 rows.
#
# Sizing: a Lambda container serves one request at a time, so it needs very
# few connections; small pools also keep N warm containers x pool size well
# under the pooler's client-connection limit.
#   pool_pre_ping: Lambda freezes idle containers, and the pooler drops idle
#     connections meanwhile -- a 1-round-trip liveness check on checkout beats
#     a failed query on a dead connection.
#   pool_recycle: proactively replace connections older than the pooler's
#     idle timeout window.
_POOL = {"pool_size": 2, "max_overflow": 2, "pool_pre_ping": True, "pool_recycle": 300}
_PSYCOPG_ARGS = {"prepare_threshold": None, "connect_timeout": 10}

engine = create_async_engine(async_database_url, connect_args=_PSYCOPG_ARGS, **_POOL)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Dense (pgvector) retrieval runs off the request/response async path (see
# services/retrieval.py, called via asyncio.to_thread), so it uses a plain
# sync engine rather than juggling a second event loop.
sync_engine = create_engine(sync_database_url, connect_args=_PSYCOPG_ARGS, **_POOL)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False, class_=Session)
