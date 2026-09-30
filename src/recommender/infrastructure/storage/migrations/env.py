"""Окружение Alembic.

Два режима:
- приложение передаёт готовое соединение через config.attributes["connection"]
  (так работает init_db внутри уже запущенного event loop);
- CLI `alembic upgrade head` — создаём async-движок сами по sqlalchemy.url
  или settings.db_url.
"""

import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from recommender.config import settings
from recommender.infrastructure.storage.postgres import Base

# Произвольный ключ: сервисы, стартующие одновременно, мигрируют по очереди
MIGRATION_LOCK_ID = 720_260_930


def run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        # SQLite не умеет ALTER для большинства изменений — batch-режим пересоздаёт таблицу
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        if connection.dialect.name == "postgresql":
            connection.exec_driver_sql(f"SELECT pg_advisory_xact_lock({MIGRATION_LOCK_ID})")
        context.run_migrations()


async def run_with_own_engine() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or settings.db_url
    engine = create_async_engine(url, poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(run_migrations)
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("Offline (--sql) mode is not supported")

shared_connection = context.config.attributes.get("connection")
if shared_connection is not None:
    run_migrations(shared_connection)
else:
    asyncio.run(run_with_own_engine())
