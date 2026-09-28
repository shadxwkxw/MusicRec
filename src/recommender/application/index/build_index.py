"""Use case: полная пересборка FAISS-индекса из БД.

Берёт все треки с фичами, заново фитит нормализатор и строит новый
FAISS-индекс. Метод нормализации, веса признаков и метрика берутся из
сохранённых артефактов (результат тюнинга); если их нет — standard + cosine.
Артефакты сохраняются на диск.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import TrackORM


class NoTracksError(RuntimeError):
    """В БД нет треков с извлечёнными фичами."""


@dataclass
class BuildIndexResult:
    engine: FaissRecommender
    normalizer: FeatureNormalizer
    tracks_indexed: int
    feature_dim: int


def _saved_params() -> tuple[str, np.ndarray | None, str, float]:
    """Параметры из сохранённых артефактов, иначе дефолты."""
    try:
        prev_norm = FeatureNormalizer.load()
        method, weights = prev_norm.method, prev_norm.weights
    except FileNotFoundError:
        method, weights = "standard", None
    try:
        prev_engine = FaissRecommender.load()
        metric, boost_weight = prev_engine.metric, prev_engine.boost_weight
    except FileNotFoundError:
        metric, boost_weight = "cosine", 0.3
    return method, weights, metric, boost_weight


async def rebuild_index(db: AsyncSession) -> BuildIndexResult:
    """Пересобрать индекс и нормализатор из всех треков БД.

    Returns:
        Свежий движок/нормализатор и статистика. Оба уже сохранены на диск.
    """
    result = await db.execute(select(TrackORM).where(TrackORM.feature_vector.isnot(None)))
    tracks: Sequence[TrackORM] = result.scalars().all()

    if not tracks:
        raise NoTracksError("No tracks with features in database")

    track_ids = [t.id for t in tracks]
    features = np.array([bytes_to_features(t.feature_vector) for t in tracks])

    method, weights, metric, boost_weight = _saved_params()

    normalizer = FeatureNormalizer(method=method, weights=weights)
    normalized = normalizer.fit_transform(features)
    normalizer.save()

    engine = FaissRecommender(
        dimension=normalized.shape[1], metric=metric, boost_weight=boost_weight
    )
    engine.rebuild(track_ids, normalized)
    engine.save()

    return BuildIndexResult(
        engine=engine,
        normalizer=normalizer,
        tracks_indexed=len(track_ids),
        feature_dim=normalized.shape[1],
    )
