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

Индекс в старом формате (файлы прямо в index_dir и models_dir) читается как
версия "legacy" и удаляется после первой публикации.
"""

import datetime
import fcntl
import os
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender

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
    version = current_version(root)
    if version is None:
        raise FileNotFoundError(f"No saved index in {_root(root)}")
    return load_version(version, root)


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

    pointer_tmp = root / f"{POINTER}.tmp"
    pointer_tmp.write_text(version)
    os.replace(pointer_tmp, root / POINTER)
    artifacts.engine.version = version

    _prune(root, keep=settings.index_keep_versions, current=version)
    return version


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
        version = current_version(path)
        artifacts = (
            base
            if version is None or version == base.engine.version
            else load_version(version, path)
        )
        if change(artifacts):
            _write(artifacts, path)
        return artifacts
