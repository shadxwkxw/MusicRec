"""Доступ к API по ключу в заголовке X-API-Key.

- изменяющие запросы (загрузка и удаление треков, лайки, индекс, тюнинг) —
  всегда по ключу;
- чтение (рекомендации, поиск, списки) — по ключу, если api.protect_reads.

Сервис доверяет user_id из запроса, поэтому лайки ставит бэкенд приложения
со своим ключом, а не браузер пользователя напрямую.

Без api.key проверки нет — это режим локальной разработки, при старте
сервис об этом предупреждает.
"""

import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from recommender.config import settings

API_KEY_HEADER = "X-API-Key"
_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _check(key: str | None) -> None:
    expected = settings.api_key
    if expected is None:
        return
    if key is None or not secrets.compare_digest(
        key.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(
            401, f"Missing or invalid {API_KEY_HEADER}", headers={"WWW-Authenticate": "ApiKey"}
        )


def require_api_key(key: str | None = Security(_header)) -> None:
    """Зависимость для изменяющих запросов."""
    _check(key)


def require_read_access(key: str | None = Security(_header)) -> None:
    """Зависимость для чтения: ключ нужен, только если api.protect_reads."""
    if settings.api_protect_reads:
        _check(key)
