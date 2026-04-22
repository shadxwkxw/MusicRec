"""Коллаборативный сигнал из пользовательских лайков.

Строит co-like сигнал: если пользователи, которые лайкнули трек A, также
лайкали трек B, то B получает бустинг при рекомендации от A.
"""

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.storage.postgres import LikeORM


async def compute_like_boost(
    track_id: str,
    db: AsyncSession,
    boost_weight: float = 0.3,
) -> dict[str, float]:
    """Посчитать бустинг-скоры для рекомендаций от заданного трека.

    Args:
        track_id: исходный трек
        db: сессия БД
        boost_weight: вес коллаборативного сигнала относительно контентного

    Returns:
        {track_id: boost_score} нормализованные в [0, boost_weight]
    """
    # Шаг 1: пользователи, лайкнувшие этот трек
    result = await db.execute(
        select(LikeORM.user_id).where(LikeORM.track_id == track_id)
    )
    users = [row[0] for row in result.fetchall()]

    if not users:
        return {}

    # Шаг 2: все треки, которые они лайкали (кроме исходного)
    result = await db.execute(
        select(LikeORM.track_id)
        .where(LikeORM.user_id.in_(users))
        .where(LikeORM.track_id != track_id)
    )
    co_liked = [row[0] for row in result.fetchall()]

    if not co_liked:
        return {}

    # Шаг 3: счётчик совпадений
    counts: dict[str, int] = defaultdict(int)
    for tid in co_liked:
        counts[tid] += 1

    # Шаг 4: нормализация в [0, boost_weight]
    max_count = max(counts.values())
    return {
        tid: (count / max_count) * boost_weight
        for tid, count in counts.items()
    }
