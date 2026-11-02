from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import sync_database_url
from app.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=sync_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # DDL runs sequentially regardless -- migrations gain nothing from an
    # async engine, so plain psycopg avoids needing greenlet/asyncpg here.
    connectable = create_engine(
        sync_database_url, poolclass=pool.NullPool, connect_args={"prepare_threshold": None}
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
