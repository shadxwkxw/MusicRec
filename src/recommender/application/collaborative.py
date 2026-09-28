"""Коллаборативный сигнал из пользовательских лайков.

Строит co-like сигнал: если пользователи, которые лайкнули трек A, также
лайкали трек B, то B получает бустинг при рекомендации от A. Здесь считается
только сила сигнала в [0, 1]; вес и знак применяет Recommender.
"""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.storage.postgres import LikeORM


def co_like_strength(track_id: str, user_likes: dict[str, set[str]]) -> dict[str, float]:
    """{track_id: сила} — как часто трек лайкали вместе с track_id, в [0, 1]."""
    counts: dict[str, int] = defaultdict(int)
    for liked in user_likes.values():
        if track_id not in liked:
            continue
        for tid in liked:
            if tid != track_id:
                counts[tid] += 1

    if not counts:
        return {}
    max_count = max(counts.values())
    return {tid: count / max_count for tid, count in counts.items()}


async def compute_like_boost(track_id: str, db: AsyncSession) -> dict[str, float]:
    """Co-like сигнал для рекомендаций от track_id по лайкам из БД."""
    fans = select(LikeORM.user_id).where(LikeORM.track_id == track_id)
    result = await db.execute(
        select(LikeORM.user_id, LikeORM.track_id).where(LikeORM.user_id.in_(fans))
    )

    user_likes: dict[str, set[str]] = defaultdict(set)
    for user_id, tid in result.all():
        user_likes[user_id].add(tid)
    return co_like_strength(track_id, user_likes)
