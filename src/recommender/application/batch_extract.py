"""Use case: массовый импорт аудио — извлечение признаков и запись треков в БД.

run_batch_import принимает готовый список треков с метаданными (например из
датасета), run_batch_extract и s3_import_items строят его по директории или
префиксу S3. Жанр, если не задан, берётся из тегов файла (tags.py). Уже
загруженные треки пропускаются по полю filename. Признаки
можно считать в нескольких процессах (workers): это самая долгая часть
импорта. Файлы из S3 скачиваются во временную папку и удаляются после расчёта.
"""

import uuid
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.features import audio_path_of
from recommender.infrastructure.data_processing.audio import get_duration
from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.data_processing.tags import read_tags
from recommender.infrastructure.storage.audio_store import (
    AUDIO_EXTENSIONS,
    configured_s3_store,
    local_copies,
    store_for,
)
from recommender.infrastructure.storage.postgres import TrackORM, utcnow


@dataclass(frozen=True)
class ImportItem:
    path: Path | str  # локальный путь или s3://bucket/key
    filename: str  # уникальный ключ трека: по нему повторный импорт пропускается
    title: str
    artist: str
    genre: str | None = None
    source: str = "import"  # fma для датасета, см. TrackORM.source


@dataclass
class BatchExtractResult:
    processed: int = 0
    skipped: int = 0
    paths_filled: int = 0  # уже импортированным трекам дописан путь к аудио
    failed: list[tuple[str, str]] = field(default_factory=list)  # (filename, error)


Extracted = tuple[bytes, float, str | None]  # признаки, длительность, жанр из тегов


def _extract(location: str) -> Extracted | str:
    """Признаки, длительность и жанр, либо текст ошибки (исключения из процессов не тащим)."""
    try:
        with store_for(location).local_copy(location) as path:
            features = extract_features(path)
            return features_to_bytes(features), get_duration(path), read_tags(path).genre
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _extract_all(paths: list[str], workers: int) -> Iterator[Extracted | str]:
    if workers <= 1:
        yield from map(_extract, paths)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(_extract, paths, chunksize=8)


async def run_batch_import(
    items: list[ImportItem],
    db: AsyncSession,
    workers: int = 1,
    commit_every: int = 100,
    progress: Callable[[int, int], None] | None = None,
) -> BatchExtractResult:
    """Извлечь признаки и сохранить треки, которых ещё нет в БД."""
    rows = (await db.execute(select(TrackORM.filename, TrackORM.audio_path))).all()
    existing: dict[str, str | None] = {filename: path for filename, path in rows}
    todo = [item for item in items if item.filename not in existing]
    stats = BatchExtractResult(skipped=len(items) - len(todo))

    # Треки, импортированные до появления audio_path: путь дописываем без
    # повторного извлечения признаков — он нужен для batch embed
    for item in items:
        if item.filename in existing and not existing[item.filename]:
            await db.execute(
                update(TrackORM)
                .where(TrackORM.filename == item.filename)
                .values(audio_path=str(item.path))
            )
            stats.paths_filled += 1
    if stats.paths_filled:
        await db.commit()

    pending = 0
    for done, (item, result) in enumerate(
        zip(todo, _extract_all([str(i.path) for i in todo], workers), strict=True), start=1
    ):
        if isinstance(result, str):
            stats.failed.append((item.filename, result))
        else:
            vector, duration, tag_genre = result
            db.add(
                TrackORM(
                    id=str(uuid.uuid4()),
                    title=item.title,
                    artist=item.artist,
                    genre=item.genre or tag_genre,
                    source=item.source,
                    filename=item.filename,
                    audio_path=str(item.path),
                    duration=duration,
                    feature_vector=vector,
                    created_at=utcnow(),
                )
            )
            stats.processed += 1
            pending += 1
        if pending >= commit_every:
            await db.commit()
            pending = 0
        if progress:
            progress(done, len(todo))
    await db.commit()
    return stats


def artist_and_title(stem: str, default_artist: str) -> tuple[str, str]:
    """«Артист - Название» из имени файла; без разделителя — default_artist и имя целиком.

    Делится по первому « - » с пробелами вокруг: дефис внутри слов (A-ha,
    Jay-Z) и в названии после разделителя не мешают.
    """
    artist, sep, title = stem.partition(" - ")
    if not sep or not artist.strip() or not title.strip():
        return default_artist, stem
    return artist.strip(), title.strip()


async def run_batch_extract(
    input_dir: Path,
    db: AsyncSession,
    default_artist: str = "Unknown",
    workers: int = 1,
) -> BatchExtractResult:
    """Импортировать все аудиофайлы директории: «Артист - Название.mp3» или default_artist."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    items = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            artist, title = artist_and_title(path.stem, default_artist)
            items.append(ImportItem(path, path.name, title, artist))
    return await run_batch_import(items, db, workers=workers)


def s3_import_items(prefix: str, default_artist: str = "Unknown") -> list[ImportItem]:
    """Аудио под префиксом бакета storage.s3_bucket (кроме загрузок сервиса).

    filename — ключ объекта: он уникален в бакете и по нему повторный импорт
    пропускает уже загруженное.
    """
    store = configured_s3_store()
    items = []
    for key in store.list_audio(prefix):
        artist, title = artist_and_title(PurePosixPath(key).stem, default_artist)
        items.append(ImportItem(f"s3://{store.bucket}/{key}", key, title, artist))
    return items


@dataclass
class GenreFillResult:
    filled: int = 0
    no_tag: int = 0  # у файла нет тега жанра
    missing: list[str] = field(default_factory=list)  # аудио не найдено


async def fill_genres_from_tags(
    db: AsyncSession,
    sources: Collection[str] | None = None,
    chunk: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> GenreFillResult:
    """Дописать жанр из тегов файла трекам, у которых его нет (загружены до чтения тегов).

    sources ограничивает источники (upload, import, fma). Аудио из S3
    скачивается пачками во временную папку.
    """
    query = select(TrackORM).where(TrackORM.genre.is_(None)).order_by(TrackORM.id)
    if sources:
        query = query.where(TrackORM.source.in_(list(sources)))
    tracks = (await db.execute(query)).scalars().all()
    result = GenreFillResult()
    for start in range(0, len(tracks), chunk):
        batch = tracks[start : start + chunk]
        locations = [audio_path_of(track) for track in batch]
        with local_copies(locations) as copies:
            for track, location, copy in zip(batch, locations, copies, strict=True):
                if isinstance(copy, Exception):
                    result.missing.append(location)
                    continue
                genre = read_tags(copy).genre
                if genre:
                    track.genre = genre
                    result.filled += 1
                else:
                    result.no_tag += 1
        await db.commit()
        if progress:
            progress(min(start + chunk, len(tracks)), len(tracks))
    return result
