"""Хранилище аудиофайлов: локальная папка или S3-совместимое хранилище.

У трека в БД лежит audio_path — локальный путь или ссылка s3://bucket/key.
Хранилище выбирается по самой ссылке (store_for), поэтому треки, загруженные
до переключения storage.backend, продолжают работать. Новые загрузки через API
идут в хранилище из конфига (upload_store).

Удаляется только то, что сервис загрузил сам (папка загрузок или префикс
storage.s3_upload_prefix): импортированный каталог не трогаем.
"""

import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from recommender.config import settings

S3_SCHEME = "s3://"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def is_s3(location: str) -> bool:
    return location.startswith(S3_SCHEME)


def parse_s3_uri(location: str) -> tuple[str, str]:
    bucket, _, key = location[len(S3_SCHEME) :].partition("/")
    if not bucket or not key:
        raise ValueError(f"Not an s3://bucket/key URI: {location}")
    return bucket, key


class AudioStore(ABC):
    @abstractmethod
    def save_upload(self, local_file: Path, filename: str) -> str:
        """Сохранить загруженный файл (локальный файл перемещается), вернуть его audio_path."""

    @abstractmethod
    @contextmanager
    def local_copy(self, location: str) -> Iterator[Path]:
        """Локальный файл на время работы с ним; FileNotFoundError, если его нет."""

    @abstractmethod
    def delete_upload(self, location: str) -> bool:
        """Удалить, если это собственная загрузка сервиса. True — удалено."""


class LocalAudioStore(AudioStore):
    @property
    def root(self) -> Path:
        return settings.audio_dir

    def save_upload(self, local_file: Path, filename: str) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / filename
        shutil.move(str(local_file), destination)
        return str(destination)

    @contextmanager
    def local_copy(self, location: str) -> Iterator[Path]:
        path = Path(location)
        if not path.is_file():
            raise FileNotFoundError(f"audio not found: {location}")
        yield path

    def delete_upload(self, location: str) -> bool:
        path = Path(location).resolve()
        if not path.is_relative_to(self.root.resolve()) or not path.is_file():
            return False
        path.unlink()
        return True


@lru_cache(maxsize=4)
def _s3_client(endpoint_url: str | None, region: str) -> Any:
    import boto3

    return boto3.client("s3", endpoint_url=endpoint_url, region_name=region)


def s3_client() -> Any:
    return _s3_client(settings.s3_endpoint_url, settings.s3_region)


class S3AudioStore(AudioStore):
    def __init__(self, bucket: str, upload_prefix: str = "", client: Any = None):
        self.bucket = bucket
        self.upload_prefix = upload_prefix
        self._client = client

    @property
    def client(self) -> Any:
        return self._client or s3_client()

    def save_upload(self, local_file: Path, filename: str) -> str:
        key = f"{self.upload_prefix}{filename}"
        self.client.upload_file(str(local_file), self.bucket, key)
        local_file.unlink(missing_ok=True)
        return f"{S3_SCHEME}{self.bucket}/{key}"

    @contextmanager
    def local_copy(self, location: str) -> Iterator[Path]:
        from botocore.exceptions import ClientError

        bucket, key = parse_s3_uri(location)
        folder = Path(tempfile.mkdtemp(prefix="recommender-s3-"))
        try:
            path = folder / PurePosixPath(key).name  # расширение нужно декодерам аудио
            try:
                self.client.download_file(bucket, key, str(path))
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                    raise FileNotFoundError(f"audio not found: {location}") from e
                raise
            yield path
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def delete_upload(self, location: str) -> bool:
        bucket, key = parse_s3_uri(location)
        if (
            bucket != self.bucket
            or not self.upload_prefix
            or not key.startswith(self.upload_prefix)
        ):
            return False
        self.client.delete_object(Bucket=bucket, Key=key)
        return True

    def list_audio(self, prefix: str) -> list[str]:
        """Ключи аудиофайлов под префиксом, кроме собственных загрузок сервиса."""
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.upload_prefix and key.startswith(self.upload_prefix):
                    continue
                if PurePosixPath(key).suffix.lower() in AUDIO_EXTENSIONS:
                    keys.append(key)
        return sorted(keys)


def s3_uri(key: str) -> str:
    """Ссылка на объект в бакете storage.s3_bucket."""
    return f"{S3_SCHEME}{settings.s3_bucket}/{key}"


def upload_file(local_file: Path, location: str) -> None:
    """Выгрузить локальный файл по ссылке s3://bucket/key (файл остаётся на месте)."""
    bucket, key = parse_s3_uri(location)
    s3_client().upload_file(str(local_file), bucket, key)


def configured_s3_store() -> S3AudioStore:
    if not settings.s3_bucket:
        raise ValueError("S3_BUCKET is not set")
    return S3AudioStore(settings.s3_bucket, settings.s3_upload_prefix)


def upload_store() -> AudioStore:
    """Куда класть новые загрузки — по storage.backend."""
    return configured_s3_store() if settings.storage_backend == "s3" else LocalAudioStore()


def store_for(location: str) -> AudioStore:
    """Хранилище, где лежит конкретный файл, — по его ссылке."""
    if not is_s3(location):
        return LocalAudioStore()
    bucket, _ = parse_s3_uri(location)
    own = settings.s3_bucket == bucket
    return S3AudioStore(bucket, settings.s3_upload_prefix if own else "")


@contextmanager
def local_copies(locations: Sequence[str], workers: int = 8) -> Iterator[list[Path | Exception]]:
    """Локальные копии пачки файлов (скачиваются параллельно), в порядке locations.

    Вместо пути — исключение, если файл получить не удалось.
    """
    managers = [store_for(location).local_copy(location) for location in locations]

    def enter(manager: Any) -> Path | Exception:
        try:
            return manager.__enter__()
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(enter, managers))
    try:
        yield results
    finally:
        for manager, result in zip(managers, results, strict=True):
            if not isinstance(result, Exception):
                manager.__exit__(None, None, None)
