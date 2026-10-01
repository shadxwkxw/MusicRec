"""Use case: массовый импорт аудио — извлечение признаков и запись треков в БД.

run_batch_import принимает готовый список треков с метаданными (например из
датасета), run_batch_extract строит его по содержимому директории. Уже
загруженные треки пропускаются по полю filename. Признаки можно считать в
нескольких процессах (workers): это самая долгая часть импорта.
"""

import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import librosa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.storage.postgres import TrackORM, utcnow

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class ImportItem:
    path: Path
    filename: str  # уникальный ключ трека: по нему повторный импорт пропускается
    title: str
    artist: str
    genre: str | None = None


@dataclass
class BatchExtractResult:
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (filename, error)


def _extract(path: Path) -> tuple[bytes, float] | str:
    """Признаки и длительность, либо текст ошибки (исключения из процессов не тащим)."""
    try:
        features = extract_features(path)
        return features_to_bytes(features), librosa.get_duration(path=str(path))
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _extract_all(paths: list[Path], workers: int) -> Iterator[tuple[bytes, float] | str]:
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
    existing: set[str] = set((await db.execute(select(TrackORM.filename))).scalars().all())
    todo = [item for item in items if item.filename not in existing]
    stats = BatchExtractResult(skipped=len(items) - len(todo))

    pending = 0
    for done, (item, result) in enumerate(
        zip(todo, _extract_all([i.path for i in todo], workers), strict=True), start=1
    ):
        if isinstance(result, str):
            stats.failed.append((item.filename, result))
        else:
            vector, duration = result
            db.add(
                TrackORM(
                    id=str(uuid.uuid4()),
                    title=item.title,
                    artist=item.artist,
                    genre=item.genre,
                    filename=item.filename,
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


async def run_batch_extract(
    input_dir: Path,
    db: AsyncSession,
    default_artist: str = "Unknown",
    workers: int = 1,
) -> BatchExtractResult:
    """Импортировать все аудиофайлы директории: title — имя файла, artist — default_artist."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    items = [
        ImportItem(path=path, filename=path.name, title=path.stem, artist=default_artist)
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return await run_batch_import(items, db, workers=workers)
