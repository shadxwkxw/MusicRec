"""Коллаборативный сигнал из пользовательских лайков.

Строит co-like сигнал: если пользователи, которые лайкнули трек A, также
лайкали трек B, то B получает бустинг при рекомендации от A. Здесь считается
только сила сигнала в [0, 1]; вес и знак применяет Recommender.
"""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.storage.postgres import LikeORM


def _scale_to_unit(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    top = max(scores.values())
    return {tid: score / top for tid, score in scores.items()}


def co_like_strength(track_id: str, user_likes: dict[str, set[str]]) -> dict[str, float]:
    """{track_id: сила} — как часто трек лайкали вместе с track_id, в [0, 1]."""
    counts: dict[str, float] = defaultdict(float)
    for liked in user_likes.values():
        if track_id not in liked:
            continue
        for tid in liked:
            if tid != track_id:
                counts[tid] += 1
    return _scale_to_unit(counts)


def user_co_like_strength(liked: set[str], user_likes: dict[str, set[str]]) -> dict[str, float]:
    """Co-like сигнал для пользователя: сумма сигналов от всех его лайков, в [0, 1].

    Сами лайки пользователя в результат не входят — их и так не рекомендуют.
    """
    totals: dict[str, float] = defaultdict(float)
    for track_id in liked:
        for tid, strength in co_like_strength(track_id, user_likes).items():
            if tid not in liked:
                totals[tid] += strength
    return _scale_to_unit(totals)


def _group_by_user(rows) -> dict[str, set[str]]:
    user_likes: dict[str, set[str]] = defaultdict(set)
    for user_id, track_id in rows:
        user_likes[user_id].add(track_id)
    return user_likes


async def load_user_likes(db: AsyncSession) -> dict[str, set[str]]:
    """Все лайки из БД: {user_id: {track_id, ...}}."""
    result = await db.execute(select(LikeORM.user_id, LikeORM.track_id))
    return _group_by_user(result.all())


async def compute_like_boost(track_id: str, db: AsyncSession) -> dict[str, float]:
    """Co-like сигнал для рекомендаций от track_id по лайкам из БД."""
    fans = select(LikeORM.user_id).where(LikeORM.track_id == track_id)
    result = await db.execute(
        select(LikeORM.user_id, LikeORM.track_id).where(LikeORM.user_id.in_(fans))
    )
    return co_like_strength(track_id, _group_by_user(result.all()))


async def compute_user_like_boost(user_id: str, db: AsyncSession) -> dict[str, float]:
    """Co-like сигнал для персональных рекомендаций пользователя."""
    liked = select(LikeORM.track_id).where(LikeORM.user_id == user_id)
    neighbours = select(LikeORM.user_id).where(LikeORM.track_id.in_(liked))
    result = await db.execute(
        select(LikeORM.user_id, LikeORM.track_id).where(LikeORM.user_id.in_(neighbours))
    )
    user_likes = _group_by_user(result.all())
    return user_co_like_strength(user_likes.get(user_id, set()), user_likes)
