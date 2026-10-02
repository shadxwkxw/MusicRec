"""Use case: полная пересборка FAISS-индекса из БД.

Берёт все треки с фичами, заново фитит нормализатор и строит новый
FAISS-индекс. Метод нормализации, веса признаков, метрика и вес буста берутся
из сохранённых артефактов (результат тюнинга); если их нет — из конфига
(recommendation.default_*).
Артефакты сохраняются на диск.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.features import load_vectors, uses_embeddings
from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.artifacts import load_current, publish
from recommender.infrastructure.storage.faiss_index import FaissRecommender


class NoTracksError(RuntimeError):
    """В БД нет треков с извлечёнными фичами."""


@dataclass
class BuildIndexResult:
    engine: FaissRecommender
    normalizer: FeatureNormalizer
    tracks_indexed: int
    feature_dim: int


def _saved_params(dim: int) -> tuple[str, np.ndarray | None, str, float]:
    """Параметры из сохранённых артефактов, иначе дефолты.

    Если сохранённый индекс другой размерности (сменился источник признаков),
    его параметры к новым векторам не подходят — берутся дефолты из конфига.
    """
    try:
        previous = load_current()
    except FileNotFoundError:
        previous = None
    if previous is None or previous.engine.dimension != dim:
        return (
            settings.default_norm_method,
            None,
            settings.default_metric,
            settings.default_boost_weight,
        )
    norm, engine = previous.normalizer, previous.engine
    return norm.method, norm.weights, engine.metric, engine.boost_weight


async def rebuild_index(db: AsyncSession) -> BuildIndexResult:
    """Пересобрать индекс и нормализатор из всех треков с векторами в текущем источнике.

    Returns:
        Свежий движок/нормализатор и статистика; опубликованы новой версией на диске.
    """
    vectors = await load_vectors(db)
    if not vectors:
        source = settings.feature_source
        hint = " (run batch embed)" if uses_embeddings() else ""
        raise NoTracksError(f"No tracks with {source} features in database{hint}")

    track_ids = list(vectors)
    features = np.stack([vectors[t] for t in track_ids])

    method, weights, metric, boost_weight = _saved_params(features.shape[1])

    normalizer = FeatureNormalizer(method=method, weights=weights)
    normalized = normalizer.fit_transform(features)

    engine = FaissRecommender(
        dimension=normalized.shape[1], metric=metric, boost_weight=boost_weight
    )
    engine.rebuild(track_ids, normalized)
    publish(engine, normalizer)

    return BuildIndexResult(
        engine=engine,
        normalizer=normalizer,
        tracks_indexed=len(track_ids),
        feature_dim=normalized.shape[1],
    )
