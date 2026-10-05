"""Pydantic-DTO для REST API."""

from datetime import datetime

from pydantic import BaseModel, Field


class TrackResponse(BaseModel):
    id: str
    title: str
    artist: str
    genre: str | None = None
    duration: float | None = None
    created_at: datetime
    indexed: bool = True
    """False — трек в БД, но в поисковом индексе ещё нет. Вызови /index/rebuild."""


class TrackUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    artist: str | None = Field(default=None, min_length=1)
    genre: str | None = Field(default=None, min_length=1)


class TrackFeaturesResponse(BaseModel):
    track_id: str
    dimension: int
    features: list[float]


class RecommendationItem(BaseModel):
    track_id: str
    title: str
    artist: str
    genre: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[RecommendationItem]


class RecommendationResponse(BaseModel):
    source_track_id: str
    recommendations: list[RecommendationItem]
    strategy: str | None = None
    """Для пользователя: interests — по его лайкам, popular — лайков пока нет."""


class LikeRequest(BaseModel):
    user_id: str
    track_id: str


class LikeResponse(BaseModel):
    status: str  # ok — лайк добавлен, exists — уже был
    user_id: str
    track_id: str


class AutoMLStatusResponse(BaseModel):
    id: int
    status: str
    best_score: float | None = None
    best_params: dict | None = None
    metrics: dict | None = None
    n_trials: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
