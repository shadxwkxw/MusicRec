"""Use case: массовое извлечение фич из директории с аудио.

Сканирует директорию, извлекает фичи для каждого поддерживаемого файла,
сохраняет трек и вектор в БД. Повторная обработка уже загруженных файлов
пропускается (по имени файла).
"""

import datetime
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import librosa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.infrastructure.data_processing.extract import (
    extract_features,
    features_to_bytes,
)
from recommender.infrastructure.storage.postgres import TrackORM


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


@dataclass
class BatchExtractResult:
    processed: int = 0
    skipped: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)  # (filename, error)


async def run_batch_extract(
    input_dir: Path,
    db: AsyncSession,
    default_artist: str = "Unknown",
) -> BatchExtractResult:
    """Извлечь фичи из всех аудио-файлов директории.

    Args:
        input_dir: директория со звуковыми файлами
        db: сессия БД
        default_artist: значение artist по умолчанию (из названия файла взять title)

    Returns:
        Статистика: сколько обработано, пропущено, упало.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    # Уже загруженные файлы — пропускаем по полю filename
    result = await db.execute(select(TrackORM.filename))
    existing = {row[0] for row in result.fetchall()}

    stats = BatchExtractResult()

    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        if path.name in existing:
            stats.skipped += 1
            continue

        try:
            features = extract_features(path)
            duration = librosa.get_duration(filename=str(path))

            track = TrackORM(
                id=str(uuid.uuid4()),
                title=path.stem,
                artist=default_artist,
                filename=path.name,
                duration=duration,
                feature_vector=features_to_bytes(features),
                created_at=datetime.datetime.utcnow(),
            )
            db.add(track)
            await db.commit()
            stats.processed += 1
        except Exception as e:
            await db.rollback()
            stats.failed.append((path.name, str(e)))

    return stats
