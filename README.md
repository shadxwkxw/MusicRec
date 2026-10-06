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
  по всем лайкам пользователя. Вес буста задаётся в долях разрыва между
  ближайшим к запросу треком и медианным: вес 1.0 при полной силе сигнала
  поднимает типичный трек до уровня ближайшего. Поэтому он одинаково работает
  с любой метрикой и нормализацией (для L2 скоры — квадраты расстояний,
  десятки и сотни, абсолютная прибавка там бы терялась).
- **Тюнинг** подбирает параметры по лайкам и пересобирает индекс; обычный
  rebuild сохраняет подобранные параметры.

## Быстрый старт

Нужны Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) и `libsndfile`
(для mp3 ещё `ffmpeg`).

```bash
make install   # .venv с точными версиями из uv.lock + dev-зависимости
make env       # .env из .env.example: источник признаков, БД, пароль Postgres
make run       # API на http://localhost:8000, документация на /docs
```

Примеры ниже — без ключа, как с пустым `API_KEY` в `.env`. Если ключ задан,
добавь к изменяющим запросам `-H "X-API-Key: $API_KEY"` (см. «Доступ к API»).

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
| POST | `/tracks/upload` | Загрузить аудио (`file`, `title`, `artist`, `genre`), извлечь признаки |
| GET | `/tracks` | Список треков (`limit` ≤ 200, `offset`) с флагом `indexed` |
| PATCH | `/tracks/{id}` | Исправить `title` / `artist` / `genre` |
| DELETE | `/tracks/{id}` | Удалить трек, его лайки, аудиофайл и запись в индексе |
| GET | `/tracks/{id}/features` | Сырой вектор признаков |
| GET | `/recommendations/{track_id}` | Похожие треки (`limit`, `use_likes`) |
| GET | `/recommendations/user/{user_id}` | Персональные рекомендации (`limit`, `use_likes`); `strategy`: `interests` или `popular` |
| GET | `/search` | Поиск по описанию на английском (`q`, `limit` ≤ 100), режим `embedding` |
| POST | `/likes` | Лайк `{"user_id", "track_id"}`; повторный — `status: exists` |
| DELETE | `/likes/{user_id}/{track_id}` | Убрать лайк |
| GET | `/users/{user_id}/likes` | Лайкнутые треки, сначала последние (`limit`, `offset`) |
| POST | `/automl/train` | Запустить тюнинг в фоне |
| GET | `/automl/status` | Запуски тюнинга: статус, `best_score`, `best_params` |
| POST | `/index/rebuild` | Пересобрать индекс из всех треков БД |
| POST | `/index/reload` | Подхватить индекс с диска, например после batch `rebuild` / `tune` (409, если индекса ещё нет) |

### Доступ к API

Изменяющие запросы (`POST`, `PATCH`, `DELETE`: загрузка и удаление треков, лайки,
индекс, тюнинг) требуют заголовок `X-API-Key` со значением `API_KEY`. Чтение
(рекомендации, поиск, списки) открыто, а с `API_PROTECT_READS=true` тоже идёт по
ключу. Без `API_KEY` проверки нет, при старте сервис об этом предупреждает:
это режим только для локальной разработки.

```bash
curl -X POST http://localhost:8000/likes -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" -d '{"user_id": "u1", "track_id": "<id>"}'
```

Как подключать к веб-приложению: запросы идут **с бэкенда приложения**, ключ
хранится только там. Сервис доверяет `user_id` из запроса, поэтому браузер не
должен ставить лайки напрямую, иначе любой поставит лайк от чужого имени. Если
браузер всё же ходит в API сам (например, за рекомендациями), его домены
перечисляются в `CORS_ORIGINS`.

Ограничения: `limit` в рекомендациях — до `api.max_rec_limit` (100), файл при
загрузке — до `api.max_upload_mb` (50 МБ, иначе 413). `make index-reload` и
задача Airflow `reload_online_index` передают ключ сами.

Скоры в выдаче зависят от метрики индекса: для `cosine` больше значит
ближе, для `euclidean` меньше значит ближе (это квадрат расстояния).
`use_likes` по умолчанию `true`.

### Персональные рекомендации

Профиль строится по последним `user_recommendation.max_likes` лайкам. Лайки
делятся на интересы (до `max_interests`, группа меньше
`min_likes_per_interest` присоединяется к ближайшей): одно среднее у человека,
который слушает и рэп, и фолк, попадает между ними и не подходит ни под то, ни
под другое. Выдача: доля `mean_share` — ближайшие к общему среднему, остальное
делится между интересами пропорционально числу их лайков. Не больше
`max_per_artist` треков одного артиста, кроме артистов, которых пользователь
уже лайкал. Без лайков — популярные (по числу лайков) и новые треки,
`strategy: popular`.

Параметры подобраны на синтетических пользователях из жанров FMA (по 300 на сценарий):

| Пользователь | Покрыты все интересы, среднее → интересы | Доля его жанров в топ-10 |
|---|---|---|
| лайки из 2 жанров | 0.56 → 0.87 | 0.81 → 0.80 |
| лайки из 3 жанров | 0.18 → 0.55 | 0.80 → 0.80 |
| лайки из 1 жанра | — | 0.79 → 0.77 |

На реальных отложенных лайках (`batch evaluate`, путь `user`) результат тот же,
что у среднего. `evaluate` и `tune` считают путь `user` той же стратегией, что и API.

### Мониторинг

| Путь | Что | Ключ |
|---|---|---|
| `GET /health/live` | процесс отвечает | не нужен |
| `GET /health/ready` | база и хранилище (S3: `head_bucket`) доступны, состояние индекса; 503, если база или хранилище недоступны | не нужен |
| `GET /metrics` | метрики Prometheus | если `API_PROTECT_READS` |

`/health/ready` возвращает `ok`, `degraded` (индекс пуст или собран из другого
источника признаков: сервис принимает загрузки, но рекомендации пустые) или
`fail` с кодом 503 и подробностями по каждой проверке. На нём же стоит
`HEALTHCHECK` Docker-образа. Каждая проверка ограничена
`observability.health_timeout` (3 с).

Метрики: `recommender_http_requests_total` и
`recommender_http_request_duration_seconds` по методу, **шаблону маршрута**
(`/recommendations/{track_id}`, а не по каждому id) и статусу;
`recommender_recommendations_total` по виду и стратегии (`interests`,
`popular`); `recommender_index_tracks` и `recommender_index_info` (версия,
источник, метрика); стандартные метрики процесса. Счётчики живут в процессе:
сервис запускается одним воркером uvicorn.

Логи: одна строка на запрос — метод, путь, шаблон маршрута, статус, время и
`request_id`. `X-Request-ID` из запроса (например, от веб-приложения)
сохраняется и возвращается в ответе, иначе генерируется. `LOG_JSON=true`
пишет JSON-строки для сборщиков логов (Loki, ELK, CloudWatch). Пробы
`/health/*` и `/metrics` логируются только при `LOG_LEVEL=DEBUG`.

### Жизненный цикл индекса

Каждая сборка индекса — отдельная неизменяемая версия в
`data/index/versions/<id>/` (индекс, метаданные и нормализатор вместе), а
текущая указана в `data/index/CURRENT`. Версия пишется целиком во временную
папку и переименовывается, указатель заменяется атомарно, поэтому сервис и
batch никогда не читают недописанный или «смешанный» индекс. Хранятся последние
`index.keep_versions` версий.

- `upload` и `delete` сразу публикуют новую версию, рестарт их не теряет. Если
  к этому моменту batch уже опубликовал более новую сборку, изменение
  применяется к ней, а не перетирает её устаревшим индексом из памяти сервиса.
  Запись идёт под файловой блокировкой `data/index/.lock`.
- `/index/rebuild` берёт метод нормализации, веса, метрику и вес буста из
  сохранённых артефактов. Если их нет, используются `standard` + `cosine`.
- `/automl/train` подбирает параметры, пересобирает и подменяет индекс в
  работающем сервисе.
- Batch-команды `rebuild` и `tune` пишут индекс на диск, но работающий сервис
  продолжает отвечать по старому, пока не вызвать `POST /index/reload`
  (`make index-reload`).
- После обновления scikit-learn (`make upgrade`) сделай `/index/rebuild`:
  нормализатор сохранён через joblib и привязан к версии sklearn.

## Признаки

Источник векторов задаётся `features.source` (переменная `FEATURE_SOURCE`):

- `librosa` (по умолчанию) — 82 признака из аудио, считаются при загрузке трека;
- `embedding` — эмбеддинги предобученной модели `features.embedding_model`
  (по умолчанию `laion/clap-htsat-unfused`, CLAP, вектор 512). На FMA с
  artist filter доля того же жанра среди 10 соседей **0.53 против 0.34** у
  librosa, и выше на каждом из 8 жанров.

### Эмбеддинги

```bash
make install-embeddings                    # torch + transformers, веса ~1.2 ГБ скачаются при первом запуске
make batch-embed                           # досчитать эмбеддинги треков без них (GPU Apple / CUDA, если есть)
# FEATURE_SOURCE=embedding в .env (или в окружении), затем
make batch-rebuild
make run
```

- Модель видит не больше 10 секунд, поэтому трек режется на окна
  (`embedding_window_seconds`, до `embedding_max_windows` окон), эмбеддинги
  окон усредняются. Это детерминированно: по умолчанию модель вырезала бы
  случайный кусок.
- Online-сервис модель не грузит. Трек, загруженный через API в режиме
  `embedding`, сохраняется с `indexed: false`, а в рекомендациях появляется
  после `batch embed` и `rebuild` (удобно ставить в расписание). Запрос
  рекомендаций для трека без эмбеддинга возвращает 409.
- Эмбеддинги хранятся в таблице `track_embeddings` по одному на модель, так
  что смена модели не стирает старые.
- Индекс помнит, из какого источника собран. Если запустить сервис с другим
  `FEATURE_SOURCE`, он стартует с пустым индексом и предупреждением, а
  `/index/reload` и batch-команды откажутся работать — нужен `rebuild`.
- Тюнинг для эмбеддингов подбирает нормализацию, метрику и вес буста;
  групповые веса есть только у librosa-признаков.
- `laion/larger_clap_music` не подходит: опубликованная для `transformers`
  версия выдаёт одинаковый вектор для любого входа (обсуждение #2 на её
  странице Hugging Face).

### Поиск по тексту

```bash
curl "http://localhost:8000/search?q=calm+acoustic+folk+with+female+vocals&limit=5"
```

CLAP переводит текст в то же пространство, что и аудио: запрос получает
эмбеддинг текстовой частью модели и ищется в том же индексе, что и
рекомендации. Нужны режим `embedding` и `torch` (`make install-embeddings`,
для Docker — `ONLINE_EXTRAS=embeddings`). Без `torch` ответ 501, на индексе из
librosa-признаков — 409.

- **Шаблоны формулировок.** CLAP чувствительна к формулировке, поэтому запрос
  подставляется в `search.prompt_templates` («{} music», «a {} song», «this is
  {} music»), и эмбеддинги усредняются. На FMA, запрос «<жанр>»: доля нужного
  жанра в топ-100 **0.37** с шаблонами против 0.22 для запроса как есть
  (случайный уровень 0.125).
- **Запросы на английском**: текстовая часть модели обучена на английских
  описаниях.
- **Отдельный процесс.** Модель живёт в дочернем процессе
  (`text_worker.py`): на macOS faiss и torch везут каждый свою копию OpenMP, и
  в одном процессе это приводит к падению. Процесс запускается при первом
  поиске (на Mac ~7 с, дальше ~30 мс на запрос), так что без поиска сервис
  остаётся лёгким. Грузится только текстовая часть модели (~125M параметров).

### librosa-признаки

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

**Цель подбора** (`tuning.objective`):
- `genre` — доля того же жанра среди 10 ближайших с artist filter. Артисты
  делятся на подбор и проверку, индекс в каждой попытке строится только из
  треков подбора. Вес буста жанрами не оценить, поэтому он подбирается вторым
  шагом по лайкам (или берётся из конфига, если лайков нет);
- `likes` — leave-one-out по лайкам, как описано ниже, буст подбирается вместе
  с остальным;
- `auto` (по умолчанию) — `genre`, если в каталоге хотя бы
  `tuning.min_genre_tracks` треков с жанром, иначе `likes`.

**Оценка.** Лайки каждого пользователя делятся на обучающие и отложенные
(`tuning.test_fraction`, по умолчанию 20%, фиксированный `tuning.seed`).

- *Подбор* идёт по leave-one-out на обучающих лайках: по очереди прячется один
  лайк и проверяется его позиция в выдаче, построенной по остальным, на обоих
  путях — *по треку* (запрос от каждого другого лайка) и *по пользователю*
  (среднее остальных лайков). Спрятанный лайк убирается и из буста.
  `best_score` — среднее MRR@10 по двум путям.
- *Итоговая оценка* — на отложенных лайках, которых тюнинг не видел, вместе с
  бейзлайнами на тех же запросах: `content_only` (система без буста),
  `same_artist`, `popularity` и `random` (точное матожидание).

Обе оценки сохраняются в запуске и видны в `/automl/status` (`metrics.train`,
`metrics.holdout`). `make batch-evaluate` считает ту же таблицу для текущего
индекса без тюнинга. Цифры имеют смысл, только если отложенных лайков хотя бы
несколько десятков: команда показывает их число.

Нужны пользователи минимум с двумя лайками. Чтобы тюнинг мог оценить буст,
лайки разных пользователей должны пересекаться.

## Batch CLI

```bash
# импорт всех аудио из папки (уже импортированные по имени файла пропускаются)
make batch-extract INPUT_DIR=data/audio_inbox
.venv/bin/python services/batch/main.py extract --input-dir data/audio_inbox --workers 8

# досчитать эмбеддинги (режим embedding, нужен make install-embeddings)
make batch-embed

# пересобрать индекс / подобрать параметры по лайкам, затем переключить сервис
make batch-rebuild      # или make batch-tune
make batch-evaluate     # качество текущего индекса против бейзлайнов
make index-reload       # API_URL=http://... для другого адреса

# top-N похожих для каждого трека в .parquet или .csv
make batch-recommend OUTPUT=artifacts/recs.parquet TOP_N=10

# то же с co-like бустом
.venv/bin/python services/batch/main.py recommend --output artifacts/recs.parquet --top-n 10 --use-likes
```

Колонки выгрузки: `source_track_id`, `rank`, `target_track_id`, `score`.

При ошибке команды завершаются с ненулевым кодом и понятным сообщением, а
неудавшийся запуск тюнинга помечается `failed` в `/automl/status`: это
удобно для планировщика вроде Airflow.

## Airflow

Расписание batch-шагов — два DAG'а в [`airflow/dags/recommender.py`](airflow/dags/recommender.py).
Каждая задача — одна команда batch CLI в отдельном контейнере образа
`music-recommender-batch` (`DockerOperator`), так что логика остаётся в CLI, а
зависимости Airflow и проекта не смешиваются.

| DAG | Когда | Шаги |
|---|---|---|
| `recommender_daily` | каждый день в 03:00 UTC | `extract` из `data/inbox` → `embed` (только при `FEATURE_SOURCE=embedding`) → `rebuild` → `/index/reload` → `recommend --use-likes` в `artifacts/recs_<дата>.parquet` |
| `recommender_weekly_tuning` | по воскресеньям в 04:00 UTC | `tune` → `/index/reload` → `evaluate` |

```bash
make airflow-up      # собирает batch-образ, поднимает Postgres, online-сервис и Airflow
make airflow-check   # DAG'и импортируются без ошибок (в чистом образе Airflow)
make airflow-logs
make airflow-down
```

- UI — http://localhost:8080, логин и пароль — `AIRFLOW_USER` / `AIRFLOW_PASSWORD`
  из `.env` (по умолчанию `airflow` / `airflow`). Новые DAG'и создаются на
  паузе. Если снять DAG с паузы, Airflow сразу запустит прогон за последний
  интервал.
- Компоненты: своя база метаданных, api-server, scheduler (`LocalExecutor`) и
  dag-processor. Docker API задачам даёт `docker-socket-proxy` (только
  контейнеры и образы), а не прямой доступ к `docker.sock`.
- Batch-контейнеры подключаются к сети `recommender-net` (Postgres и
  online-сервис) и монтируют `data/`, `artifacts/`, `configs/` проекта. Веса
  моделей для `embed` кешируются в volume `recommender-hf-cache`.
- Код batch запечён в образ: после правок пересобери его
  (`docker compose --profile batch build recommender-batch`). В режиме
  `embedding` нужен `BATCH_EXTRAS=embeddings` в `.env`, чтобы в образе был
  `torch` (CPU-сборка, образ ~3.5 ГБ).
- Стек в Docker работает с Postgres, а локальная разработка по умолчанию с
  SQLite: перед первым запуском перенеси данные `make db-copy`. Если порт
  5432 занят локальным Postgres, задай `POSTGRES_HOST_PORT` в `.env`.

## Датасет FMA

Для оценки на большом каталоге с метками используется
[Free Music Archive](https://github.com/mdeff/fma): `fma_small` — 8000 треков по
30 секунд, 8 жанров по 1000, у каждого есть название, артист и жанр, лицензии
Creative Commons. Архивы (`fma_metadata.zip` ~342 МБ, `fma_small.zip` ~7.2 ГБ)
скачиваются по ссылкам из README проекта в `data/fma/`.

```bash
make fma-unpack                 # архивы сжаты bzip2: распаковка через Python, не unzip
make fma-import WORKERS=8       # признаки в 8 процессов, повторный запуск пропускает готовые
make batch-rebuild && make batch-evaluate
```

`batch evaluate` выводит вторую таблицу: какая доля из 10 ближайших треков того
же жанра (`system`) против случайного уровня (`random`), всего и по жанрам.
`filtered` — то же с artist filter: треки того же артиста исключены из соседей,
иначе метрика отчасти меряет «нашёл других треков артиста». Лайки для неё не
нужны.

Треки FMA — 30-секундные фрагменты, а свои треки анализируются по первым
`duration_limit` (120) секундам, поэтому признаки у них из немного разных
распределений.

## Хранилище аудио (локально или S3)

`storage.backend` (переменная `AUDIO_STORAGE`): `local` — папка `data/audio`,
`s3` — любое S3-совместимое хранилище (AWS, Yandex Object Storage, Selectel и
т.д.; для не-AWS задаётся `S3_ENDPOINT_URL`). Ключи — стандартные
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, в конфиг они не попадают.

- **Загрузка через API**: файл сначала проверяется (извлекаются признаки из
  временной копии), потом кладётся в `s3://<S3_BUCKET>/<S3_UPLOAD_PREFIX><id>_<имя>`.
  От имени файла клиента берётся только само имя, без каталогов.
- **Импорт каталога**: `make batch-import-s3 S3_IMPORT_PREFIX=music/ WORKERS=8`
  обходит аудио под префиксом (кроме собственных загрузок сервиса), каждый
  процесс скачивает файл во временную папку и удаляет его после расчёта.
  Название — имя файла, ключ объекта — уникальный идентификатор для повторных
  запусков.
- **`batch embed`** скачивает аудио из S3 пачками во временные файлы.
- **Удаление трека** удаляет объект, только если это загрузка сервиса (под
  `S3_UPLOAD_PREFIX` в своём бакете). Импортированный каталог не трогается.
- У трека в базе хранится ссылка `s3://...` или локальный путь, и хранилище
  выбирается по ней: треки, загруженные до переключения на S3, продолжают
  работать.
- **Индекс** (`faiss.index`, `meta.joblib`, `normalizer.joblib`): каждая
  опубликованная версия копируется в `s3://<S3_BUCKET>/<S3_INDEX_PREFIX>versions/<id>/`,
  потом обновляется указатель `CURRENT`. Локальная `data/index` остаётся рабочей
  копией: перед загрузкой и изменением индекса она догоняет бакет, поэтому batch
  и сервис на разных машинах видят одну версию. Если S3 недоступен, сервис
  работает с локальной копией, а невыгруженная версия уйдёт в бакет при
  следующей синхронизации. Старые версии в бакете чистятся так же
  (`index.keep_versions`).
- **Рекомендации batch recommend** пишутся и в S3: `--output s3://music/artifacts/recs.parquet`.
  Ежедневный DAG при `AUDIO_STORAGE=s3` кладёт их в `<S3_ARTIFACTS_PREFIX>recs_<дата>.parquet`.

**Перенос существующих данных** после переключения на S3:

```bash
make batch-migrate-s3                 # копирует, локальные файлы остаются
make batch-migrate-s3 DELETE_LOCAL=1  # и удаляет аудио с диска после переноса
```

Аудио из `data/audio` (загрузки сервиса) уходит под `S3_UPLOAD_PREFIX`, ссылки
в базе переписываются на `s3://`; треки из других мест (датасет FMA) остаются
на диске. Текущая версия индекса и `artifacts/*.parquet|csv` копируются в
бакет. Повторный запуск пропускает уже перенесённое.

Локально S3 поднимается в профиле `s3` — это [RustFS](https://rustfs.com),
S3-совместимый сервер с веб-консолью. Данные хранятся в томе `s3-data`,
бакет `S3_BUCKET` создаётся при старте. В `.env`:

```bash
AUDIO_STORAGE=s3
S3_BUCKET=music
S3_ENDPOINT_URL=http://localhost:5050   # для make run и batch CLI с хоста
S3_CONTAINER_ENDPOINT_URL=http://s3:9000  # для контейнеров (online, batch, Airflow)
AWS_ACCESS_KEY_ID=rustfsadmin           # логин консоли
AWS_SECRET_ACCESS_KEY=придумайте-пароль # пароль консоли
```

```bash
docker compose --profile s3 up -d s3
```

Консоль — http://localhost:9001/rustfs/console/: там видны, загружаются и
скачиваются файлы. Порты меняются через `S3_HOST_PORT` / `S3_CONSOLE_PORT`.

## База данных

По умолчанию — локальная SQLite (`data/recommender.db`), для этого ничего
настраивать не нужно. Postgres включается через `DB_URL`, например
`postgresql+asyncpg://user:pass@host:5432/db`; в `docker-compose` он уже
настроен.

Схема ведётся миграциями Alembic (`src/recommender/infrastructure/storage/migrations/`).
Online-сервис и batch CLI применяют их сами при старте. В Postgres миграции
берут advisory lock, поэтому сервисы, стартующие одновременно, мигрируют по
очереди. База, созданная до появления миграций, подхватывается
автоматически, данные не меняются.

```bash
make migrate                               # применить миграции явно
make migration MSG="add genre to tracks"   # новая миграция по изменениям моделей
make db-copy                               # перенести данные из SQLite в Postgres из docker-compose
```

`make db-copy` пишет только в пустую базу и сохраняет id. Источник и цель
задаются через `COPY_FROM` и `COPY_TO`.

## Конфигурация

Настройки лежат в [`configs/config.yaml`](configs/config.yaml). Поддерживаются
подстановки `${VAR}` и `${VAR:-default}`, например URL базы берётся из
`DB_URL`.

| Секция | Что настраивает |
|---|---|
| `paths` | папки для аудио, индекса и моделей |
| `database` | URL базы |
| `audio` | частота дискретизации, длительность анализа, число MFCC, chroma и полос контраста |
| `recommendation` | лимит выдачи по умолчанию (API и batch `--top-n`), запас кандидатов для FAISS и параметры до первого тюнинга: метрика, нормализация, вес буста |
| `tuning` | число попыток и таймаут Optuna, k для hit@k / MRR@k, верхние границы весов признаков и буста, какие методы нормализации и метрики перебирать |
| `api` | хост и порт для `recommender-online`, размер страницы `GET /tracks` и его максимум |

Значения проверяются при старте: неизвестная метрика или метод нормализации,
отрицательный вес и т.п. остановят сервис с понятной ошибкой.

Конфиг ищется так: `$CONFIG_PATH` → `./configs/config.yaml` → копия,
встроенная в пакет (`src/recommender/default_config.yaml`).

### Переменные окружения и `.env`

Перед чтением конфига приложение подгружает `.env` из рабочей папки. Шаблон —
[`.env.example`](.env.example), создать `.env` из него: `make env`. Сам `.env`
в git не попадает. Переменные, заданные в окружении явно, важнее файла;
`ENV_FILE=путь` читает другой файл, `ENV_FILE=` (пусто) отключает чтение.

| Переменная | Что задаёт |
|---|---|
| `FEATURE_SOURCE` | `librosa` или `embedding` |
| `DB_URL` | база приложения, по умолчанию локальная SQLite |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Postgres в `docker-compose` и адрес для `make db-copy` |
| `CONFIG_PATH` | другой файл конфига |
| `API_KEY`, `API_PROTECT_READS`, `CORS_ORIGINS` | доступ к API, см. «Доступ к API» |
| `LOG_LEVEL`, `LOG_JSON` | уровень и формат логов, см. «Мониторинг» |

Пароль Postgres применяется при первом создании volume `postgres-data`;
чтобы сменить его у существующей базы, volume придётся пересоздать.

Тесты и `make smoke` `.env` не читают и от `FEATURE_SOURCE` в окружении не
зависят. В продакшене те же переменные передаются через секреты окружения
(например, GitHub Secrets), а `.env` в образ не кладётся.

`n_contrast_bands` не может быть больше 6 при `sample_rate: 22050`: верхняя
полоса иначе выходит за частоту Найквиста.

`src/recommender/default_config.yaml` должен совпадать с `configs/config.yaml`
(это проверяет тест): при правке конфига меняй оба файла.

## Разработка

| Команда | Что делает |
|---|---|
| `make install` | окружение строго по `uv.lock` |
| `make migrate` / `migration MSG=...` / `db-copy` | миграции и перенос данных, см. «База данных» |
| `make lock` | обновить `uv.lock` после правки зависимостей в `pyproject.toml` |
| `make upgrade` | поднять зависимости до свежих версий и переустановить |
| `make test` / `test-unit` / `test-integration` | тесты |
| `make coverage` | тесты с покрытием, падает ниже 80% |
| `make smoke` | настоящий uvicorn: загрузка, rebuild, лайки, тюнинг, удаление, рестарт, batch CLI |
| `make lint` / `format` | ruff: проверка / автоисправление и форматирование |
| `make typecheck` | mypy |
| `make audit` | известные уязвимости в версиях из `uv.lock` |
| `make build` | собрать wheel |
| `make e2e-compose` | E2E в docker compose: собранный образ, Postgres, API-ключ, рестарт (свои тома, `.env` и `data/` не трогает) |
| `make openapi` / `openapi-check` | обновить / проверить контракт API `docs/openapi.json` |

Интеграционные тесты и `make smoke` работают во временных папках со своей
SQLite и не трогают `data/`. Схема в тестах создаётся миграциями. Чтобы
прогнать интеграционные тесты на Postgres, задай `TEST_DATABASE_URL`: схема
`public` в этой базе пересоздаётся перед каждым тестом.

### CI

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) на push
в `main` и на pull request:

| Джоб | Проверка |
|---|---|
| lint | актуальность `uv.lock`, ruff той же версии, что в lock |
| secrets | gitleaks по всей истории коммитов (ложные срабатывания — `.gitleaksignore`) |
| api-contract | `docs/openapi.json` совпадает с кодом: изменения API видны в PR, по схеме можно генерировать клиент |
| typecheck | mypy |
| audit | уязвимости в зависимостях |
| unit | юнит-тесты на Python 3.11–3.14 |
| integration | все тесты + покрытие (отчёт в summary и артефактах) |
| postgres | интеграционные тесты на Postgres 17 и `alembic check`: модели совпадают с миграциями |
| s3-server | тесты S3 на настоящем сервере RustFS (локально — на moto), см. `S3_TEST_ENDPOINT` |
| smoke | `make smoke` |
| e2e-compose | `make e2e-compose`: образ + Postgres + миграции, healthcheck, загрузка, лайки, рекомендации, 401 без ключа, метрики, рестарт |
| airflow-dags | DAG'и импортируются в образе Airflow (`make airflow-check`) |
| build | wheel ставится в чистое окружение и запускается вне репозитория |
| docker | сборка образов online и batch после тестов и Trivy: падает на исправимых уязвимостях HIGH/CRITICAL |

Обновления зависимостей, actions и базовых образов раз в неделю присылает
Dependabot ([`.github/dependabot.yml`](.github/dependabot.yml)); каждый PR
проходит весь CI. Trivy и gitleaks запускаются из образов с закреплённой
версией и digest, а не через сторонние actions.

## Docker

```bash
make docker-build && make docker-up      # Postgres + online-сервис на :8000
make db-copy                             # один раз: перенести локальные данные в Postgres
make docker-batch-extract INPUT_DIR=data/audio
make docker-batch-rebuild && make index-reload
make docker-batch-recommend OUTPUT=artifacts/recs.parquet TOP_N=10
```

Образы ставят зависимости из `uv.lock`. Postgres хранит данные в volume
`postgres-data` и доступен с хоста на `localhost:5432`. `docker-compose.yml` монтирует
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
    storage/               SQLAlchemy-модели и сессии, миграции Alembic, FAISS-индекс,
                           версии индекса на диске (artifacts.py)
  interfaces/online/       FastAPI: роуты, схемы, приложение
  default_config.yaml      конфиг по умолчанию внутри пакета
services/
  online/                  точка входа uvicorn + Dockerfile
  batch/                   CLI (extract / recommend) + Dockerfile
configs/config.yaml        основной конфиг
scripts/smoke_test.py      end-to-end проверка
scripts/copy_db.py         перенос данных между базами
scripts/check_dags.py      проверка импорта DAG'ов в образе Airflow
airflow/dags/              DAG'и Airflow (ежедневный конвейер и еженедельный тюнинг)
alembic.ini                конфиг CLI Alembic
tests/unit/                юнит-тесты
tests/integration/         тесты API на временной SQLite
```

## Известные ограничения

- Блокировка записи индекса — `fcntl.flock`: работает на Linux и macOS и между
  контейнерами с общим томом на одной машине, но не на Windows и не на сетевых
  файловых системах вроде NFS.
- Индекс `IndexFlat` — точный полный перебор; каждая загрузка и удаление
  пишут новую версию индекса целиком. Для десятков тысяч треков нужен
  приближённый индекс и периодическое сохранение.
- Docker-образы не включают `torch`: `batch embed` запускается локально или в
  отдельном образе с `make install-embeddings`.
- `make run` и `docker-compose` запускают uvicorn напрямую на порту 8000;
  `api.host` и `api.port` действуют только на точку входа `recommender-online`.
