"""Alembic runs this for every command (upgrade, downgrade, revision…).

It connects to DATABASE_URL and hands Alembic our tables' description
(Base.metadata), so `alembic revision --autogenerate` can compare the
models in app/db.py with the real database and write the difference.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from app.db import Base, database_url  # importing app also loads .env

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)  # the [loggers] sections of alembic.ini

target_metadata = Base.metadata


def include_name(name, type_, parent_names) -> bool:
    """Autogenerate only looks at our own tables. Without this, it would see the
    checkpointer's tables (created by its setup(), not by us) and write a
    migration that drops them.
    """
    if type_ == "table":
        return name in target_metadata.tables
    return True


def run_migrations_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, include_name=include_name)
    with context.begin_transaction():  # each upgrade runs in one transaction
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(database_url())
    async with engine.connect() as connection:
        # Alembic itself is synchronous; run_sync gives it a sync-style
        # connection that drives our async one underneath.
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
