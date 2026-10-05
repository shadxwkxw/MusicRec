"""Версии индекса на диске.

Раскладка в paths.index_dir:
    versions/<id>/faiss.index, meta.joblib, normalizer.joblib — неизменяемая сборка
    CURRENT — id текущей версии
    .lock   — файловая блокировка записи

Версия пишется целиком во временную папку и переименовывается, указатель
заменяется атомарно, поэтому читатель всегда видит согласованную тройку
индекс + метаданные + нормализатор. Вся запись (публикация batch-сборки и
инкрементальные изменения online-сервиса) идёт под блокировкой: изменение
сервиса применяется к актуальной версии, а не перетирает более новую.

При storage.backend=s3 версии копируются в бакет (index_mirror): новая версия
выгружается после записи на диск, а перед чтением и изменением локальная
копия догоняет бакет — так batch и сервис на разных машинах видят одну и ту
же текущую версию. Если S3 недоступен, работа идёт с локальной копией, а
невыгруженная версия уйдёт в бакет при следующей синхронизации.

Индекс в старом формате (файлы прямо в index_dir и models_dir) читается как
версия "legacy" и удаляется после первой публикации.
"""

import datetime
import fcntl
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage import index_mirror
from recommender.infrastructure.storage.faiss_index import FaissRecommender

logger = logging.getLogger(__name__)

POINTER = "CURRENT"
VERSIONS = "versions"
LOCK = ".lock"
LEGACY = "legacy"
TMP_PREFIX = ".tmp-"
NORMALIZER = "normalizer.joblib"


@dataclass
class IndexArtifacts:
    engine: FaissRecommender  # engine.version — id версии на диске
    normalizer: FeatureNormalizer


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else settings.index_dir


def _legacy_files(root: Path) -> list[Path]:
    return [root / "faiss.index", root / "meta.joblib", settings.models_dir / NORMALIZER]


@contextmanager
def locked(root: Path | None = None) -> Iterator[None]:
    """Эксклюзивная блокировка записи (между процессами и потоками)."""
    path = _root(root)
    path.mkdir(parents=True, exist_ok=True)
    with (path / LOCK).open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def current_version(root: Path | None = None) -> str | None:
    path = _root(root)
    pointer = path / POINTER
    if pointer.exists():
        return pointer.read_text().strip() or None
    if all(p.exists() for p in _legacy_files(path)):
        return LEGACY
    return None


def list_versions(root: Path | None = None) -> list[str]:
    """Опубликованные версии от старых к новым."""
    versions_dir = _root(root) / VERSIONS
    if not versions_dir.is_dir():
        return []
    return sorted(
        p.name for p in versions_dir.iterdir() if p.is_dir() and not p.name.startswith(TMP_PREFIX)
    )


def load_version(version: str, root: Path | None = None) -> IndexArtifacts:
    path = _root(root)
    if version == LEGACY:
        engine = FaissRecommender.load(path)
        normalizer = FeatureNormalizer.load(settings.models_dir / NORMALIZER)
    else:
        folder = path / VERSIONS / version
        engine = FaissRecommender.load(folder)
        normalizer = FeatureNormalizer.load(folder / NORMALIZER)
    engine.version = version
    return IndexArtifacts(engine, normalizer)


def load_current(root: Path | None = None) -> IndexArtifacts:
    """Текущая версия; FileNotFoundError, если индекс ещё ни разу не собирался."""
    path = _root(root)
    if index_mirror.configured_mirror() is not None:
        with locked(path):
            _sync_quietly(path)
    version = current_version(path)
    if version is None:
        raise FileNotFoundError(f"No saved index in {path}")
    return load_version(version, path)


def _new_version_id() -> str:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def _write(artifacts: IndexArtifacts, root: Path) -> str:
    """Записать новую версию и сделать её текущей. Вызывается под блокировкой."""
    version = _new_version_id()
    versions_dir = root / VERSIONS
    tmp = versions_dir / f"{TMP_PREFIX}{version}"
    tmp.mkdir(parents=True)
    artifacts.engine.save(tmp)
    artifacts.normalizer.save(tmp / NORMALIZER)
    os.rename(tmp, versions_dir / version)
    _set_pointer(root, version)
    artifacts.engine.version = version

    _prune(root, keep=settings.index_keep_versions, current=version)
    mirror = index_mirror.configured_mirror()
    if mirror is not None:
        try:
            _push(mirror, root, version)
        except Exception:
            logger.warning("Index version %s not copied to S3, will retry on next sync", version)
    return version


def _set_pointer(root: Path, version: str) -> None:
    pointer_tmp = root / f"{POINTER}.tmp"
    pointer_tmp.write_text(version)
    os.replace(pointer_tmp, root / POINTER)


def _push(mirror: index_mirror.S3IndexMirror, root: Path, version: str) -> None:
    mirror.upload(version, root / VERSIONS / version)
    mirror.set_current(version)  # после файлов: по указателю всегда полная версия
    mirror.prune(keep=settings.index_keep_versions, current=version)


def _sync(root: Path) -> str | None:
    """Свести локальную копию и бакет к более новой версии. Вызывается под блокировкой.

    Возвращает текущую версию в бакете после синхронизации.
    """
    mirror = index_mirror.configured_mirror()
    if mirror is None:
        return None
    local, remote = current_version(root), mirror.current()
    if remote is not None and (local in (None, LEGACY) or remote > local):
        if remote not in list_versions(root):
            tmp = root / VERSIONS / f"{TMP_PREFIX}{remote}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            mirror.download(remote, tmp)
            os.rename(tmp, root / VERSIONS / remote)
        _set_pointer(root, remote)
        _prune(root, keep=settings.index_keep_versions, current=remote)
        return remote
    if local is not None and local != LEGACY and local != remote:
        _push(mirror, root, local)  # локальная новее: прошлая выгрузка не удалась
        return local
    return remote


def _sync_quietly(root: Path) -> None:
    try:
        _sync(root)
    except Exception:
        logger.warning("S3 index sync failed, using the local copy", exc_info=True)


def sync(root: Path | None = None) -> str | None:
    """Синхронизировать локальную копию с бакетом; ошибки S3 не глушатся."""
    path = _root(root)
    with locked(path):
        return _sync(path)


def _prune(root: Path, keep: int, current: str) -> None:
    versions_dir = root / VERSIONS
    for leftover in versions_dir.glob(f"{TMP_PREFIX}*"):  # от прерванных записей
        shutil.rmtree(leftover, ignore_errors=True)
    for old in list_versions(root)[:-keep]:
        if old != current:
            shutil.rmtree(versions_dir / old, ignore_errors=True)
    for legacy in _legacy_files(root):
        legacy.unlink(missing_ok=True)


def publish(
    engine: FaissRecommender, normalizer: FeatureNormalizer, root: Path | None = None
) -> str:
    """Опубликовать новую сборку (rebuild, tune) как текущую версию."""
    path = _root(root)
    with locked(path):
        return _write(IndexArtifacts(engine, normalizer), path)


def update_current(
    base: IndexArtifacts,
    change: Callable[[IndexArtifacts], bool],
    root: Path | None = None,
) -> IndexArtifacts:
    """Применить изменение к актуальной версии и опубликовать результат.

    Если с момента загрузки base на диске появилась более новая версия (её
    опубликовал batch rebuild), изменение применяется к ней. change возвращает
    False, если менять нечего, — тогда новая версия не пишется. Возвращает
    артефакты, с которыми дальше должен работать вызывающий.
    """
    path = _root(root)
    with locked(path):
        _sync_quietly(path)
        version = current_version(path)
        artifacts = (
            base
            if version is None or version == base.engine.version
            else load_version(version, path)
        )
        if change(artifacts):
            _write(artifacts, path)
        return artifacts
