"""Доменные сущности рекомендера.

Чистые Pydantic-модели, не зависящие от SQLAlchemy, FAISS или FastAPI.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Track(BaseModel):
    """Аудио-трек в каталоге."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    artist: str = "Unknown"
    filename: str
    duration: float | None = None
    created_at: datetime | None = None


class TrackFeatures(BaseModel):
    """Извлечённый вектор аудио-признаков трека."""

    track_id: str
    dimension: int
    features: list[float]


class Like(BaseModel):
    """Лайк пользователя на трек."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    user_id: str
    track_id: str
    created_at: datetime | None = None


class Recommendation(BaseModel):
    """Одна рекомендация с релевантностью."""

    track_id: str
    score: float


class AutoMLRun(BaseModel):
    """Запуск гиперпараметрической оптимизации рекомендера."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    status: str = "pending"  # pending | running | completed | failed
    best_score: float | None = None
    best_params: dict | None = None
    n_trials: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None
