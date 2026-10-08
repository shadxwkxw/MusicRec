"""Use case: перенос локальных данных в S3 после переключения на storage.backend=s3.

- аудио треков из папки загрузок (paths.audio_dir) уходит под
  storage.s3_upload_prefix, audio_path в БД переписывается на s3://. Это те
  же «собственные загрузки» сервиса: удаление трека через API удаляет и файл.
  Треки из других мест (например датасет FMA) остаются на диске.
- текущая версия индекса копируется в бакет (дальше это делает каждая публикация);
- выгрузки batch recommend (.parquet, .csv) — под storage.s3_artifacts_prefix;
- папка с новой музыкой (upload_directory) — под префикс каталога со структурой
  подпапок; треки, которые уже есть в бакете под тем же именем файла, пропускаются.

Повторный запуск безопасен: перенесённые треки уже ссылаются на s3:// и
пропускаются. Локальные копии аудио удаляются только с delete_local и только
после того, как новая ссылка сохранена в БД.
"""

import re
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.features import audio_path_of
from recommender.config import settings
from recommender.infrastructure.storage import artifacts
from recommender.infrastructure.storage.audio_store import (
    AUDIO_EXTENSIONS,
    S3_SCHEME,
    configured_s3_store,
    s3_uri,
    upload_file,
)
from recommender.infrastructure.storage.postgres import TrackORM

ARTIFACT_SUFFIXES = {".parquet", ".csv"}


@dataclass
class AudioMigrationResult:
    moved: int = 0
    deleted_local: int = 0
    outside_audio_dir: int = 0  # аудио не из папки загрузок — не переносится
    missing: list[str] = field(default_factory=list)  # нет файла на диске
    failed: list[tuple[str, str]] = field(default_factory=list)  # (путь, ошибка)


def _require_s3() -> None:
    if settings.storage_backend != "s3":
        raise ValueError("Set AUDIO_STORAGE=s3 (and S3_BUCKET) before migrating to S3")


async def migrate_audio(
    db: AsyncSession,
    delete_local: bool = False,
    workers: int = 8,
    chunk: int = 50,
    progress: Callable[[int, int], None] | None = None,
) -> AudioMigrationResult:
    """Перенести аудио треков из paths.audio_dir в S3 и переписать audio_path."""
    _require_s3()
    local_audio = or_(TrackORM.audio_path.is_(None), ~TrackORM.audio_path.startswith(S3_SCHEME))
    tracks: Sequence[TrackORM] = (
        (await db.execute(select(TrackORM).where(local_audio))).scalars().all()
    )
    audio_dir = settings.audio_dir.resolve()
    result = AudioMigrationResult()
    todo: list[tuple[TrackORM, Path]] = []
    for track in tracks:
        path = Path(audio_path_of(track))
        if not path.resolve().is_relative_to(audio_dir):
            result.outside_audio_dir += 1
        elif not path.is_file():
            result.missing.append(str(path))
        else:
            todo.append((track, path))

    def upload(item: tuple[TrackORM, Path]) -> str | Exception:
        _, path = item
        location = s3_uri(f"{settings.s3_upload_prefix}{path.name}")
        try:
            upload_file(path, location)
        except Exception as e:
            return e
        return location

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), chunk):
            batch = todo[start : start + chunk]
            uploaded: list[Path] = []
            for (track, path), outcome in zip(batch, pool.map(upload, batch), strict=True):
                if isinstance(outcome, Exception):
                    result.failed.append((str(path), f"{type(outcome).__name__}: {outcome}"))
                    continue
                track.audio_path = outcome
                uploaded.append(path)
            await db.commit()  # ссылки на S3 сохранены — локальные копии больше не нужны
            result.moved += len(uploaded)
            if delete_local:
                for path in uploaded:
                    path.unlink(missing_ok=True)
                    result.deleted_local += 1
            if progress:
                progress(min(start + chunk, len(todo)), len(todo))
    return result


def migrate_index() -> str | None:
    """Скопировать текущую версию индекса в бакет; вернуть версию в бакете."""
    _require_s3()
    return artifacts.sync()


def migrate_artifacts(directory: Path) -> list[str]:
    """Выгрузить файлы batch recommend из directory; вернуть ссылки s3://."""
    _require_s3()
    if not directory.is_dir():
        return []
    locations = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in ARTIFACT_SUFFIXES:
            location = s3_uri(f"{settings.s3_artifacts_prefix}{path.name}")
            upload_file(path, location)
            locations.append(location)
    return locations


_UPLOAD_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_")


def _nfc(name: str) -> str:
    # macOS хранит имена в разложенной форме (й = и + ˘): без NFC «одинаковые»
    # названия не совпадут ни при сравнении, ни в ключах S3
    return unicodedata.normalize("NFC", name)


def _same_track_key(name: str) -> str:
    """Ключ сравнения имён: без формы Unicode и регистра («дора - втюрилась» = «Дора - Втюрилась»)."""
    return _nfc(name).casefold()


def bucket_track_names() -> set[str]:
    """Ключи сравнения имён аудиофайлов в бакете (у загрузок через API — без префикса id)."""
    store = configured_s3_store()
    names = set()
    for page in store.client.get_paginator("list_objects_v2").paginate(Bucket=store.bucket):
        for obj in page.get("Contents", []):
            name = PurePosixPath(obj["Key"]).name
            if PurePosixPath(name).suffix.lower() in AUDIO_EXTENSIONS:
                names.add(_same_track_key(_UPLOAD_ID.sub("", name)))
    return names


@dataclass
class DirectoryUploadResult:
    uploaded: list[str] = field(default_factory=list)  # ключи s3
    duplicates: list[str] = field(default_factory=list)  # локальные пути: уже есть в бакете
    failed: list[tuple[str, str]] = field(default_factory=list)


def upload_directory(directory: Path, prefix: str, workers: int = 8) -> DirectoryUploadResult:
    """Залить аудио из папки (рекурсивно) под prefix, пропуская уже известные имена файлов."""
    _require_s3()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    known = bucket_track_names()
    result = DirectoryUploadResult()
    todo: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        name = _same_track_key(path.name)
        if name in known:
            result.duplicates.append(str(path))
            continue
        known.add(name)  # одинаковые файлы в разных подпапках — один трек
        todo.append((path, prefix + _nfc(path.relative_to(directory).as_posix())))

    def upload(item: tuple[Path, str]) -> Exception | None:
        try:
            upload_file(item[0], s3_uri(item[1]))
        except Exception as e:
            return e
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (path, key), error in zip(todo, pool.map(upload, todo), strict=True):
            if error is None:
                result.uploaded.append(key)
            else:
                result.failed.append((str(path), f"{type(error).__name__}: {error}"))
    return result
