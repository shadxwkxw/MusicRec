"""Миграции Alembic: схема совпадает с моделями, накат идемпотентен и т.д."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text

from recommender.infrastructure.storage.postgres import (
    BASELINE_REVISION,
    MIGRATIONS,
    Base,
    init_db,
)


async def _tables(db_engine) -> set[str]:
    async with db_engine.connect() as conn:
        return await conn.run_sync(lambda c: set(inspect(c).get_table_names()))


async def _revision(db_engine) -> str | None:
    async with db_engine.connect() as conn:
        return await conn.run_sync(lambda c: MigrationContext.configure(c).get_current_revision())


async def test_migrations_match_models(fresh_db):
    await init_db(fresh_db)

    async with fresh_db.connect() as conn:
        diff = await conn.run_sync(
            lambda c: compare_metadata(MigrationContext.configure(c), Base.metadata)
        )

    # Непустой diff = модели изменили, а миграцию не написали (make migration)
    assert diff == []


async def test_init_db_is_idempotent(fresh_db):
    await init_db(fresh_db)
    await init_db(fresh_db)

    assert {"tracks", "likes", "automl_runs", "alembic_version"} <= await _tables(fresh_db)


def _alembic(connection, action: str, revision: str) -> None:
    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS)
    cfg.attributes["connection"] = connection
    getattr(command, action)(cfg, revision)


async def test_legacy_create_all_database_is_adopted(fresh_db):
    # База из времён до миграций: схема как у базовой ревизии, но без alembic_version
    async with fresh_db.begin() as conn:
        await conn.run_sync(_alembic, "upgrade", BASELINE_REVISION)
        await conn.execute(text("DROP TABLE alembic_version"))
        await conn.execute(
            text("INSERT INTO tracks (id, title, filename) VALUES ('t1', 'Old', 'old.mp3')")
        )

    await init_db(fresh_db)

    head = ScriptDirectory.from_config(_config()).get_current_head()
    assert await _revision(fresh_db) == head
    async with fresh_db.connect() as conn:
        title = await conn.scalar(text("SELECT title FROM tracks WHERE id = 't1'"))
    assert title == "Old"


async def test_downgrade_to_base_and_back(fresh_db):
    await init_db(fresh_db)

    async with fresh_db.begin() as conn:
        await conn.run_sync(_alembic, "downgrade", "base")
    assert await _tables(fresh_db) <= {"alembic_version"}

    async with fresh_db.begin() as conn:
        await conn.run_sync(_alembic, "upgrade", "head")
    assert {"tracks", "likes", "automl_runs"} <= await _tables(fresh_db)


def _config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS)
    return cfg


def test_baseline_revision_exists():
    assert ScriptDirectory.from_config(_config()).get_revision(BASELINE_REVISION) is not None
