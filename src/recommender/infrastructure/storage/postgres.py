"""SQLAlchemy-хранилище: ORM-модели, engine, session factory.

Доменные сущности живут в `recommender.domain.models`; этот модуль отвечает
только за персистентность.
"""

import datetime
from collections.abc import AsyncIterator

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
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


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session
