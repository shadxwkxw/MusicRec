# Music Recommender

Контентный рекомендатель музыки: из аудио извлекаются 82 признака (librosa),
похожие треки ищутся в FAISS-индексе, а лайки пользователей добавляют
коллаборативный буст. Параметры (нормализация, метрика, веса признаков, вес
буста) подбираются Optuna по лайкам. Есть REST API (FastAPI) и batch CLI.

## Как это работает

```
аудио ──► librosa: 82 признака ──► БД (сырые признаки)
                                     │
                                     ▼
                      нормализатор: скейлер + веса групп
                                     │
                                     ▼
          FAISS-индекс (cosine или L2) ──► топ-N похожих ──► + co-like буст ──► ответ
                                                                  ▲
                                                             лайки из БД
```

- **Признаки** хранятся в БД сырыми. Нормализатор (`standard` / `minmax` /
  `robust` плюс веса групп признаков) и индекс сохраняются на диск в
  `data/models/` и `data/index/` и подгружаются при старте сервиса.
- **Co-like буст:** если пользователи, лайкнувшие трек A, лайкали и B, то B
  поднимается в выдаче от A. Для персональных рекомендаций сигнал суммируется
  по всем лайкам пользователя.
- **Тюнинг** подбирает параметры по лайкам и пересобирает индекс; обычный
  rebuild сохраняет подобранные параметры.

## Быстрый старт

Нужны Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) и `libsndfile`
(для mp3 ещё `ffmpeg`).

```bash
make install   # .venv с точными версиями из uv.lock + dev-зависимости
make run       # API на http://localhost:8000, документация на /docs
```

Загрузить треки:

```bash
curl -X POST http://localhost:8000/tracks/upload \
  -F "file=@song.mp3" -F "title=My Song" -F "artist=Artist"
```

До первой сборки индекса треки только сохраняются в БД (`"indexed": false`):
нормализатор ещё не обучен, а сырые признаки в индекс класть нельзя. Когда
загружено несколько треков, собери индекс:

```bash
curl -X POST http://localhost:8000/index/rebuild
```

После этого новые загрузки сразу попадают в индекс (`"indexed": true`).

```bash
# похожие треки
curl "http://localhost:8000/recommendations/<track_id>?limit=10"

# лайк и персональные рекомендации
curl -X POST http://localhost:8000/likes -H "Content-Type: application/json" \
  -d '{"user_id": "u1", "track_id": "<track_id>"}'
curl "http://localhost:8000/recommendations/user/u1"

# подбор параметров в фоне и его статус
curl -X POST http://localhost:8000/automl/train
curl http://localhost:8000/automl/status
```

## API

| Метод | Путь | Описание |
|---|---|---|
| POST | `/tracks/upload` | Загрузить аудио (`file`, `title`, `artist`), извлечь признаки |
| GET | `/tracks` | Список треков (`limit` ≤ 200, `offset`) с флагом `indexed` |
| PATCH | `/tracks/{id}` | Исправить `title` / `artist` |
| DELETE | `/tracks/{id}` | Удалить трек, его лайки, аудиофайл и запись в индексе |
| GET | `/tracks/{id}/features` | Сырой вектор признаков |
| GET | `/recommendations/{track_id}` | Похожие треки (`limit`, `use_likes`) |
| GET | `/recommendations/user/{user_id}` | Персональные рекомендации (`limit`, `use_likes`) |
| POST | `/likes` | Лайк `{"user_id", "track_id"}` |
| POST | `/automl/train` | Запустить тюнинг в фоне |
| GET | `/automl/status` | Запуски тюнинга: статус, `best_score`, `best_params` |
| POST | `/index/rebuild` | Пересобрать индекс из всех треков БД |

Скоры в выдаче зависят от метрики индекса: для `cosine` больше значит
ближе, для `euclidean` меньше значит ближе (это квадрат расстояния).
`use_likes` по умолчанию `true`.

### Жизненный цикл индекса

- `upload` и `delete` сразу сохраняют индекс на диск, рестарт их не теряет.
- `/index/rebuild` берёт метод нормализации, веса, метрику и вес буста из
  сохранённых артефактов. Если их нет, используются `standard` + `cosine`.
- `/automl/train` подбирает параметры, пересобирает и подменяет индекс в
  работающем сервисе.
- После обновления scikit-learn (`make upgrade`) сделай `/index/rebuild`:
  нормализатор сохранён через joblib и привязан к версии sklearn.

## Признаки

| Группа | Размерность | Что описывает |
|---|---|---|
| MFCC (13 коэф., mean + std) | 26 | тембр |
| Chroma (12 полутонов, mean + std) | 24 | гармония, тональность |
| Spectral contrast (6 полос + 1, mean + std) | 14 | яркость, текстура |
| Tonnetz (6, mean + std) | 12 | тональные отношения |
| Tempo | 1 | BPM |
| RMS | 1 | громкость |
| Zero crossing rate | 1 | шумность, перкуссивность |
| Spectral centroid / bandwidth / rolloff | 3 | частотные характеристики |

Итого **82** признака. Анализируются первые `duration_limit` секунд трека.

## Тюнинг (Optuna)

Подбирается:
- метод нормализации: `standard`, `minmax`, `robust`;
- метрика: `cosine`, `euclidean`;
- веса 8 групп признаков, от 0 до 3 (применяются после скейлера, иначе
  сокращаются);
- вес co-like буста, от 0 до 3.

**Оценка:** leave-one-out по лайкам. Для каждого пользователя по очереди
прячется один лайк и проверяется его позиция в выдаче, построенной по
остальным лайкам, на обоих путях:
- *по треку:* запрос от каждого другого лайка, с бустом;
- *по пользователю:* запрос — среднее остальных лайков, с бустом.

Спрятанный лайк убирается и из буста, иначе он подсказывает ответ. Для
каждого пути считаются hit@10 и MRR@10; `best_score` — среднее MRR@10 по двум
путям.

Нужны пользователи минимум с двумя лайками. Чтобы тюнинг мог оценить буст,
лайки разных пользователей должны пересекаться.

## Batch CLI

```bash
# импорт всех аудио из папки (уже импортированные по имени файла пропускаются)
make batch-extract INPUT_DIR=data/audio_inbox

# top-N похожих для каждого трека в .parquet или .csv
make batch-recommend OUTPUT=artifacts/recs.parquet TOP_N=10

# то же с co-like бустом
.venv/bin/python services/batch/main.py recommend --output artifacts/recs.parquet --top-n 10 --use-likes
```

Колонки выгрузки: `source_track_id`, `rank`, `target_track_id`, `score`.

## Конфигурация

Настройки лежат в [`configs/config.yaml`](configs/config.yaml): пути к данным,
URL базы, параметры извлечения признаков и тюнинга. Поддерживаются
подстановки `${VAR}` и `${VAR:-default}`, например URL базы берётся из
`DB_URL`.

Конфиг ищется так: `$CONFIG_PATH` → `./configs/config.yaml` → копия,
встроенная в пакет (`src/recommender/default_config.yaml`).

`n_contrast_bands` не может быть больше 6 при `sample_rate: 22050`: верхняя
полоса иначе выходит за частоту Найквиста.

## Разработка

| Команда | Что делает |
|---|---|
| `make install` | окружение строго по `uv.lock` |
| `make lock` | обновить `uv.lock` после правки зависимостей в `pyproject.toml` |
| `make upgrade` | поднять зависимости до свежих версий и переустановить |
| `make test` / `test-unit` / `test-integration` | тесты |
| `make coverage` | тесты с покрытием, падает ниже 80% |
| `make smoke` | настоящий uvicorn: загрузка, rebuild, лайки, тюнинг, удаление, рестарт, batch CLI |
| `make lint` / `format` | ruff: проверка / автоисправление и форматирование |
| `make typecheck` | mypy |
| `make audit` | известные уязвимости в версиях из `uv.lock` |
| `make build` | собрать wheel |

Интеграционные тесты и `make smoke` работают во временных папках со своей
SQLite и не трогают `data/`.

### CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) на push
в `main` и на pull request:

| Джоб | Проверка |
|---|---|
| lint | актуальность `uv.lock`, ruff той же версии, что в lock |
| typecheck | mypy |
| audit | уязвимости в зависимостях |
| unit | юнит-тесты на Python 3.11–3.14 |
| integration | все тесты + покрытие (отчёт в summary и артефактах) |
| smoke | `make smoke` |
| build | wheel ставится в чистое окружение и запускается вне репозитория |
| docker | сборка образов online и batch после прохождения тестов |

## Docker

```bash
make docker-build && make docker-up      # online-сервис на :8000
make docker-batch-extract INPUT_DIR=data/audio
make docker-batch-recommend OUTPUT=artifacts/recs.parquet TOP_N=10
```

Образы ставят зависимости из `uv.lock`. `docker-compose.yml` монтирует
`data/`, `configs/` и исходники, поэтому online-сервис перезапускается при
правках кода.

## Структура

```
src/recommender/
  domain/                  сущности и порт Recommender
  application/             use cases: рекомендации, co-like буст, rebuild,
                           тюнинг, batch extract / recommend
  infrastructure/
    data_processing/       извлечение признаков (librosa), нормализатор
    storage/               SQLAlchemy-модели и сессии, FAISS-индекс
  interfaces/online/       FastAPI: роуты, схемы, приложение
  default_config.yaml      конфиг по умолчанию внутри пакета
services/
  online/                  точка входа uvicorn + Dockerfile
  batch/                   CLI (extract / recommend) + Dockerfile
configs/config.yaml        основной конфиг
scripts/smoke_test.py      end-to-end проверка
tests/unit/                юнит-тесты
tests/integration/         тесты API на временной SQLite
```

## Известные ограничения

- Хранилище — SQLite через `aiosqlite`, хотя модуль называется `postgres.py`.
  Схема создаётся через `create_all`, миграций нет: изменение моделей
  требует ручной правки базы.
- Индекс `IndexFlat` — точный полный перебор; каждая загрузка и удаление
  перезаписывают индекс на диске целиком. Для десятков тысяч треков нужен
  приближённый индекс и периодическое сохранение.
- `recommendation.default_limit`, `recommendation.faiss_nprobe` и секция
  `api` в конфиге пока не используются: лимит по умолчанию (10) и порт (8000)
  заданы в коде.
