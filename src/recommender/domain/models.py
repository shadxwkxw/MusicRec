"""Доменные сущности рекомендера.

Чистые Pydantic-модели, не зависящие от SQLAlchemy, FAISS или FastAPI.
"""

from pydantic import BaseModel


class Recommendation(BaseModel):
    """Одна рекомендация с релевантностью."""

    track_id: str
    score: float
