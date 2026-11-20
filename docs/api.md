# REST API

Полный машиночитаемый контракт — [`openapi.json`](openapi.json). Для работающего сервиса интерактивная документация доступна по `/docs`, схема JSON — `/openapi.json`.

## Основные маршруты

| Область | Маршруты | Назначение |
|---|---|---|
| Треки | `POST /tracks/upload`, `POST /tracks/register`, `GET /tracks`, `PATCH /tracks/{id}`, `DELETE /tracks/{id}` | Загрузка и управление каталогом |
| Признаки и поиск | `GET /tracks/{id}/features`, `GET /search` | Просмотр признаков и текстовый поиск (для embedding-режима) |
| Рекомендации | `GET /recommendations/{track_id}`, `GET /recommendations/user/{user_id}` | Похожие треки и персональная выдача |
| Лайки | `POST /likes`, `DELETE /likes/{user_id}/{track_id}`, `GET /users/{user_id}/likes` | Управление пользовательскими сигналами |
| Прослушивания | `POST /events` | Запись показов и событий плеера |
| Индекс | `POST /index/rebuild`, `POST /index/reload` | Пересборка и обновление индекса |
| Тюнинг | `POST /automl/train`, `GET /automl/status` | Фоновый подбор параметров и состояние запусков |
| Мониторинг | `GET /health/live`, `GET /health/ready`, `GET /metrics` | Проверки и метрики сервиса |

## Доступ

При заданном `API_KEY` изменяющие запросы требуют заголовок `X-API-Key`. Чтение по умолчанию открыто; `API_PROTECT_READS=true` включает для него ту же проверку. Ключ следует хранить на серверной стороне приложения. CORS задаётся через `CORS_ORIGINS`.

Ограничения размеров, параметры запросов и схемы ответов смотрите в OpenAPI-контракте и в корневом [README](../README.md#api).
