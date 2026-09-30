"""SQLAlchemy-хранилище: ORM-модели, engine, session factory.

Доменные сущности живут в `recommender.domain.models`; этот модуль отвечает
только за персистентность.
"""

import datetime
from collections.abc import AsyncIterator

from sqlalchemy import (
    Column,
    Connection,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    inspect,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from recommender.config import settings


def utcnow() -> datetime.datetime:
    """Наивное UTC-время: в БД уже хранятся значения без таймзоны."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class TrackORM(Base):
    __tablename__ = "tracks"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    artist = Column(String, default="Unknown")
    filename = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    feature_vector = Column(LargeBinary, nullable=True)  # numpy bytes
    created_at = Column(DateTime, default=utcnow)

    likes = relationship("LikeORM", back_populates="track")


class LikeORM(Base):
    __tablename__ = "likes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    track_id = Column(String, ForeignKey("tracks.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=utcnow)

    track = relationship("TrackORM", back_populates="likes")


class AutoMLRunORM(Base):
    __tablename__ = "automl_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, default="pending")  # pending | running | completed | failed
    best_score = Column(Float, nullable=True)
    best_params = Column(String, nullable=True)  # JSON-string
    n_trials = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


engine = create_async_engine(settings.db_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


MIGRATIONS = "recommender.infrastructure.storage:migrations"
BASELINE_REVISION = "0001"


def _upgrade_to_head(connection: Connection) -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", MIGRATIONS)
    cfg.attributes["connection"] = connection

    tables = set(inspect(connection).get_table_names())
    if "tracks" in tables and "alembic_version" not in tables:
        # База создана через create_all до появления миграций: её схема
        # совпадает с базовой ревизией, помечаем и мигрируем дальше как обычно.
        command.stamp(cfg, BASELINE_REVISION)
    command.upgrade(cfg, "head")


async def init_db(db_engine: AsyncEngine | None = None) -> None:
    """Довести схему БД до последней миграции (alembic upgrade head)."""
    async with (db_engine or engine).begin() as conn:
        await conn.run_sync(_upgrade_to_head)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session
