"""Копия версий индекса в S3 (storage.backend=s3).

Раскладка под storage.s3_index_prefix — как на диске:
    versions/<id>/faiss.index, meta.joblib, normalizer.joblib
    CURRENT — id текущей версии

Локальная index_dir остаётся рабочей копией: сервис и batch читают и пишут
её под файловой блокировкой, а бакет — общая точка между машинами
(artifacts.py выгружает новую версию и догоняет бакет перед чтением).
Указатель пишется после файлов версии, поэтому по CURRENT всегда лежит
полная версия. Общей блокировки между машинами нет: при одновременной
публикации побеждает последняя.
"""

from pathlib import Path, PurePosixPath
from typing import Any

from recommender.config import settings
from recommender.infrastructure.storage.audio_store import s3_client

POINTER = "CURRENT"
VERSIONS = "versions/"


class S3IndexMirror:
    def __init__(self, bucket: str, prefix: str, client: Any = None):
        self.bucket = bucket
        self.prefix = prefix
        self._client = client

    @property
    def client(self) -> Any:
        return self._client or s3_client()

    def _version_prefix(self, version: str) -> str:
        return f"{self.prefix}{VERSIONS}{version}/"

    def _keys(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        return [
            obj["Key"]
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for obj in page.get("Contents", [])
        ]

    def current(self) -> str | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self.prefix + POINTER)["Body"]
        except self.client.exceptions.NoSuchKey:
            return None
        return body.read().decode().strip() or None

    def set_current(self, version: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.prefix + POINTER, Body=version.encode())

    def versions(self) -> list[str]:
        """Версии в бакете от старых к новым."""
        paginator = self.client.get_paginator("list_objects_v2")
        prefix = self.prefix + VERSIONS
        return sorted(
            p["Prefix"][len(prefix) :].rstrip("/")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/")
            for p in page.get("CommonPrefixes", [])
        )

    def upload(self, version: str, folder: Path) -> None:
        for path in sorted(folder.iterdir()):
            self.client.upload_file(
                str(path), self.bucket, self._version_prefix(version) + path.name
            )

    def download(self, version: str, folder: Path) -> None:
        keys = self._keys(self._version_prefix(version))
        if not keys:
            raise FileNotFoundError(f"s3://{self.bucket}/{self._version_prefix(version)} is empty")
        for key in keys:
            self.client.download_file(self.bucket, key, str(folder / PurePosixPath(key).name))

    def prune(self, keep: int, current: str) -> None:
        for old in self.versions()[:-keep]:
            if old == current:
                continue
            keys = self._keys(self._version_prefix(old))
            if keys:
                self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in keys]}
                )


def configured_mirror() -> S3IndexMirror | None:
    """Копия в бакете storage.s3_bucket, если хранилище — S3."""
    if settings.storage_backend != "s3" or not settings.s3_bucket:
        return None
    return S3IndexMirror(settings.s3_bucket, settings.s3_index_prefix)
