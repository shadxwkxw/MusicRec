"""Use case: обложки каталога из тегов файлов — в S3 под storage.s3_cover_prefix.

У трека появляется cover_path (s3://bucket/catalog-covers/<id>.jpg) или "" —
проверено, обложки в файле нет; повторный запуск такие треки не перечитывает.
Обложками владеет сервис: при удалении трека удаляется и его обложка.
"""

from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.features import audio_path_of
from recommender.config import settings
from recommender.infrastructure.data_processing.covers import (
    Cover,
    cover_from_id3,
    id3_tag_size,
    read_cover,
)
from recommender.infrastructure.storage.audio_store import (
    is_s3,
    parse_s3_uri,
    s3_client,
    s3_uri,
    store_for,
)
from recommender.infrastructure.storage.postgres import TrackORM

MAX_TAG_BYTES = 20 * 1024 * 1024  # больший тег — читаем файл целиком


@dataclass
class CoverResult:
    found: int = 0
    without_cover: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


def _range(bucket: str, key: str, end: int) -> bytes:
    response = s3_client().get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{end}")
    return response["Body"].read()


def fetch_cover(location: str) -> Cover | None:
    """Обложка трека: для mp3 в S3 — по заголовку и тегу ID3, иначе из файла целиком."""
    if is_s3(location) and location.lower().endswith(".mp3"):
        bucket, key = parse_s3_uri(location)
        size = id3_tag_size(_range(bucket, key, 9))
        if size is None:
            return None  # mp3 без тега ID3v2 — обложке негде быть
        if size <= MAX_TAG_BYTES:
            return cover_from_id3(_range(bucket, key, size - 1))
    with store_for(location).local_copy(location) as path:
        return read_cover(path)


def _cover_key(track_id: str, cover: Cover) -> str:
    return f"{settings.s3_cover_prefix}{track_id}{cover.extension}"


def is_own_cover(location: str | None) -> bool:
    if not location or not is_s3(location):
        return False
    bucket, key = parse_s3_uri(location)
    return bucket == settings.s3_bucket and key.startswith(settings.s3_cover_prefix)


def delete_cover(location: str | None) -> None:
    if is_own_cover(location):
        bucket, key = parse_s3_uri(location or "")
        s3_client().delete_object(Bucket=bucket, Key=key)


async def extract_covers(
    db: AsyncSession,
    sources: Collection[str] | None = None,
    workers: int = 8,
    chunk: int = 50,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CoverResult:
    """Найти обложки треков без cover_path (по умолчанию — не скрытых источников)."""
    if settings.storage_backend != "s3":
        raise ValueError("Covers are stored in S3: set AUDIO_STORAGE=s3")
    query = select(TrackORM).where(TrackORM.cover_path.is_(None)).order_by(TrackORM.id)
    if sources:
        query = query.where(TrackORM.source.in_(list(sources)))
    elif settings.hidden_sources:
        query = query.where(TrackORM.source.not_in(settings.hidden_sources))
    if limit:
        query = query.limit(limit)
    tracks = (await db.execute(query)).scalars().all()
    result = CoverResult()

    def process(track: TrackORM) -> tuple[str | None, str | None]:
        """(cover_path, ошибка)."""
        try:
            cover = fetch_cover(audio_path_of(track))
            if cover is None:
                return "", None
            key = _cover_key(track.id, cover)
            s3_client().put_object(
                Bucket=settings.s3_bucket, Key=key, Body=cover.data, ContentType=cover.mime
            )
            return s3_uri(key), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(tracks), chunk):
            batch = tracks[start : start + chunk]
            for track, (path, error) in zip(batch, pool.map(process, batch), strict=True):
                if error:
                    result.failed.append((track.id, error))
                    continue
                track.cover_path = path
                if path:
                    result.found += 1
                else:
                    result.without_cover += 1
            await db.commit()
            if progress:
                progress(min(start + chunk, len(tracks)), len(tracks))
    return result
