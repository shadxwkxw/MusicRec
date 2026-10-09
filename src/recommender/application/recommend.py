"""Use case: получение рекомендаций.

Две стратегии:
- по треку (content-based + collaborative boost)
- по пользователю (интересы из последних лайков + collaborative boost,
  см. user_profile.py; без лайков — популярные треки)

Треки скрытых источников (visibility.py) в выдачу не попадают.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import (
    compute_like_boost,
    compute_user_like_boost,
)
from recommender.application.features import load_vectors
from recommender.application.user_profile import blend, interest_candidates
from recommender.application.visibility import hidden_track_ids
from recommender.config import settings
from recommender.domain.artists import ArtistCap, artist_names, names_of
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

    hidden = await hidden_track_ids(db, engine)
    return engine.recommend(
        features,
        limit=limit,
        exclude_ids={track_id},
        like_boost=like_boost,
        hidden_ids=set(hidden),
    )


@dataclass
class UserRecommendations:
    items: list[Recommendation]
    strategy: str  # interests — по лайкам; popular — холодный старт


async def recommend_for_user(
    user_id: str,
    db: AsyncSession,
    engine: Recommender,
    normalizer: FeatureNormalizer,
    limit: int = settings.default_rec_limit,
    use_likes: bool = True,
) -> UserRecommendations:
    """Персональные рекомендации по интересам из последних лайков + коллаборативный бустинг.

    Без лайков (или пока у лайкнутых треков нет векторов) — популярные и новые
    треки, если user_recommendation.cold_start, иначе NoLikedTracksError.
    """
    result = await db.execute(
        select(LikeORM.track_id)
        .where(LikeORM.user_id == user_id)
        .order_by(LikeORM.created_at.desc(), LikeORM.id.desc())
    )
    all_liked = list(dict.fromkeys(row[0] for row in result.all()))
    recent = all_liked[: settings.user_max_likes]

    hidden = await hidden_track_ids(db, engine)
    vectors_by_id = await load_vectors(db, recent)
    rows = [vectors_by_id[t] for t in recent if t in vectors_by_id]
    if not rows:
        if not settings.user_cold_start:
            raise NoLikedTracksError(user_id)
        return UserRecommendations(
            await popular_tracks(db, engine, limit, exclude_ids=set(all_liked) | hidden), "popular"
        )

    vectors = np.stack(rows)
    if normalizer.is_fitted:
        vectors = normalizer.transform(vectors)
    like_boost = await compute_user_like_boost(user_id, db) if use_likes else None

    candidates = interest_candidates(
        engine,
        vectors,
        limit,
        exclude_ids=set(all_liked),
        like_boost=like_boost or None,
        hidden_ids=set(hidden),
    )
    artists = await _artists(db, candidates.track_ids | set(recent))
    liked_artists = names_of(artists, recent)
    return UserRecommendations(
        blend(candidates, limit, artists, liked_artists=liked_artists), "interests"
    )


async def _artists(db: AsyncSession, track_ids: set[str]) -> dict[str, str | None]:
    if not track_ids or not settings.user_max_per_artist:
        return {}
    result = await db.execute(
        select(TrackORM.id, TrackORM.artist).where(TrackORM.id.in_(list(track_ids)))
    )
    return {track_id: artist for track_id, artist in result.all()}


async def popular_tracks(
    db: AsyncSession, engine: Recommender, limit: int, exclude_ids: set[str] | None = None
) -> list[Recommendation]:
    """Самые лайкаемые треки из индекса, при равенстве — новые. Скор — число лайков."""
    exclude_ids = exclude_ids or set()
    indexed = set(engine.track_ids)
    likes = func.count(LikeORM.id)
    result = await db.execute(
        select(TrackORM.id, TrackORM.artist, likes)
        .outerjoin(LikeORM, LikeORM.track_id == TrackORM.id)
        .group_by(TrackORM.id, TrackORM.artist)
        .order_by(likes.desc(), TrackORM.created_at.desc(), TrackORM.id)
    )
    picked: list[Recommendation] = []
    skipped: list[Recommendation] = []  # из-за лимита по артисту — добор, если не хватит
    limiter = ArtistCap(settings.user_max_per_artist)
    for track_id, artist, count in result.all():
        if track_id not in indexed or track_id in exclude_ids:
            continue
        rec = Recommendation(track_id=track_id, score=float(count))
        names = artist_names(artist)
        if not limiter.allows(names):
            skipped.append(rec)
            continue
        limiter.add(names)
        picked.append(rec)
        if len(picked) >= limit:
            break
    if len(picked) < limit:
        picked = sorted(picked + skipped[: limit - len(picked)], key=lambda r: -r.score)
    return picked
