"""Перенос данных между базами, например из локальной SQLite в Postgres.

    python scripts/copy_db.py --source sqlite+aiosqlite:///data/recommender.db \\
        --target postgresql+asyncpg://recommender:recommender@localhost:5432/recommender

Схема целевой базы создаётся миграциями. Целевая база должна быть пустой:
скрипт не сливает данные. id сохраняются, счётчики id в Postgres
выставляются после вставки.
"""

import argparse
import asyncio

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from recommender.infrastructure.storage.postgres import Base, init_db

# Порядок важен: likes и track_embeddings ссылаются на tracks
TABLES = ["tracks", "track_embeddings", "likes", "automl_runs"]
SERIAL_TABLES = ["likes", "automl_runs"]


async def copy(source: AsyncEngine, target: AsyncEngine) -> dict[str, int]:
    await init_db(source)
    await init_db(target)

    async with target.connect() as conn:
        for name in TABLES:
            if await conn.scalar(select(func.count()).select_from(Base.metadata.tables[name])):
                raise SystemExit(f"Target table '{name}' is not empty, refusing to copy")

    copied = {}
    async with source.connect() as src, target.begin() as dst:
        for name in TABLES:
            table = Base.metadata.tables[name]
            rows = [dict(r) for r in (await src.execute(select(table))).mappings()]
            if rows:
                await dst.execute(table.insert(), rows)
            copied[name] = len(rows)

        if dst.dialect.name == "postgresql":
            for name in SERIAL_TABLES:
                await dst.execute(
                    text(
                        f"SELECT setval(pg_get_serial_sequence('{name}', 'id'), "
                        f"COALESCE((SELECT max(id) FROM {name}), 0) + 1, false)"
                    )
                )
    return copied


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="SQLAlchemy async URL to copy from")
    parser.add_argument("--target", required=True, help="SQLAlchemy async URL to copy into")
    args = parser.parse_args()

    source, target = create_async_engine(args.source), create_async_engine(args.target)
    try:
        copied = await copy(source, target)
    finally:
        await source.dispose()
        await target.dispose()
    print("copied: " + ", ".join(f"{name}={n}" for name, n in copied.items()))


if __name__ == "__main__":
    asyncio.run(main())
