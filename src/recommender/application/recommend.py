"""Use case: получение рекомендаций.

Две стратегии:
- по треку (content-based + collaborative boost)
- по пользователю (усреднение векторов его лайков + collaborative boost)
"""

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import (
    compute_like_boost,
    compute_user_like_boost,
)
from recommender.application.features import load_vectors
from recommender.config import settings
from recommender.domain.models import Recommendation
from recommender.domain.recommender import Recommender
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.postgres import LikeORM, TrackORM


class TrackNotFoundError(LookupError):
    """Трека нет в БД."""


class FeaturesNotReadyError(LookupError):
    """Трек есть, но векторов в текущем источнике признаков для него ещё нет."""


class NoLikedTracksError(LookupError):
    """У пользователя нет лайкнутых треков с фичами."""


async def recommend_by_track(
    track_id: str,
    db: AsyncSession,
    engine: Recommender,
    normalizer: FeatureNormalizer,
    limit: int = settings.default_rec_limit,
    use_likes: bool = True,
) -> list[Recommendation]:
    """Рекомендации по треку: контентное сходство + коллаборативный бустинг."""
    if await db.get(TrackORM, track_id) is None:
        raise TrackNotFoundError(track_id)
    vectors = await load_vectors(db, [track_id])
    if track_id not in vectors:
        raise FeaturesNotReadyError(track_id)

    features = vectors[track_id]
    if normalizer.is_fitted:
        features = normalizer.transform(features).flatten()

    like_boost = await compute_like_boost(track_id, db) if use_likes else None

    return engine.recommend(
        features,
        limit=limit,
        exclude_ids={track_id},
        like_boost=like_boost,
    )


async def recommend_for_user(
    user_id: str,
    db: AsyncSession,
    engine: Recommender,
    normalizer: FeatureNormalizer,
    limit: int = settings.default_rec_limit,
    use_likes: bool = True,
) -> list[Recommendation]:
    """Персональные рекомендации: усреднение векторов лайков + коллаборативный бустинг."""
    result = await db.execute(select(LikeORM.track_id).where(LikeORM.user_id == user_id))
    liked_ids = [row[0] for row in result.fetchall()]

    if not liked_ids:
        raise NoLikedTracksError(user_id)

    vectors = list((await load_vectors(db, liked_ids)).values())
    if not vectors:
        raise NoLikedTracksError(user_id)

    avg_features = np.mean(vectors, axis=0)
    if normalizer.is_fitted:
        avg_features = normalizer.transform(avg_features).flatten()

    like_boost = await compute_user_like_boost(user_id, db) if use_likes else None

    return engine.recommend(
        avg_features,
        limit=limit,
        exclude_ids=set(liked_ids),
        like_boost=like_boost or None,
    )
