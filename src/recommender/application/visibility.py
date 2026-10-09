"""Какие треки не показывать в выдаче: источники из recommendation.hidden_sources.

Например, датасет FMA нужен для тюнинга и оценки (у него размеченные жанры),
но для приложения с русской музыкой его англоязычные любительские треки в
рекомендациях — шум: на каталоге из 508 своих треков они занимали треть
выдачи, а без них доля того же жанра не изменилась (0.71).

Скрытые треки остаются в индексе и в базе. Их можно использовать как запрос
(похожие на трек FMA), но в результатах их нет. Тюнинг и оценка их видят.

Список кэшируется по версии индекса: скрытые треки появляются только через
импорт, после которого индекс пересобирается в новую версию.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.config import settings
from recommender.domain.recommender import Recommender
from recommender.infrastructure.storage.postgres import TrackORM

_cache: dict[tuple[str, tuple[str, ...]], frozenset[str]] = {}


async def hidden_track_ids(db: AsyncSession, engine: Recommender) -> frozenset[str]:
    sources = tuple(sorted(settings.hidden_sources))
    if not sources:
        return frozenset()
    version = getattr(engine, "version", None)
    key = (version or "", sources)
    if version is not None and key in _cache:
        return _cache[key]
    result = await db.execute(select(TrackORM.id).where(TrackORM.source.in_(sources)))
    ids = frozenset(result.scalars().all())
    if version is not None:
        _cache.clear()  # нужна только текущая версия
        _cache[key] = ids
    return ids
