# 🎵 Music Recommender — AutoML Content-Based Recommendation System

## Архитектура

```
┌─────────────────┐       ┌──────────────────────┐       ┌─────────────────┐
│   Node.js App   │──────▶│  FastAPI ML Service   │──────▶│   FAISS Index   │
│  (основной API) │  HTTP │                      │       │  (vector search)│
└─────────────────┘       │  ┌─────────────────┐ │       └─────────────────┘
                          │  │ Feature Extractor│ │
                          │  │  (librosa)       │ │
                          │  └────────┬────────┘ │
                          │  ┌────────▼────────┐ │
                          │  │  AutoML Pipeline │ │
                          │  │  (optuna)        │ │
                          │  └────────┬────────┘ │
                          │  ┌────────▼────────┐ │
                          │  │  Recommender     │ │
                          │  │  Engine          │ │
                          │  └─────────────────┘ │
                          └──────────────────────┘
```

## Структура репозитория

```
src/recommender/
  domain/               # сущности, абстрактные порты
  application/          # use cases (recommend, collaborative, index, training, batch)
  infrastructure/       # адаптеры: postgres, faiss, librosa, sklearn
  interfaces/online/    # FastAPI routes + main
services/
  online/               # Dockerfile + uvicorn entrypoint (REST)
  batch/                # Dockerfile + CLI (extract / recommend)
configs/                # конфиги
deploy/k8s/             # k8s-манифесты
tests/unit/             # unit-тесты
```

## Batch-сервис

```bash
# Массовое извлечение фич из директории
make batch-extract INPUT_DIR=data/audio_inbox

# Precompute top-10 похожих для всех треков
make batch-recommend OUTPUT=artifacts/recs.parquet TOP_N=10

# Через docker compose
make docker-batch-extract INPUT_DIR=data/audio
```

## Конфигурация

Все настройки живут в [`configs/config.yaml`](configs/config.yaml). Поддерживаются
подстановки `${ENV_VAR}` и `${ENV_VAR:-default}`. Путь к конфигу можно
переопределить через `CONFIG_PATH`.

## Быстрый старт

```bash
# Всё в одну команду (создаёт .venv, ставит проект в editable + dev-зависимости)
make install

# Запустить online-сервис
make run

# Прогнать тесты
make test

# Или вручную:
pip install -e ".[dev]"
uvicorn recommender.interfaces.online.main:app --reload --port 8000

# 3. Загрузить треки
curl -X POST http://localhost:8000/tracks/upload \
  -F "file=@song.mp3" \
  -F "title=My Song" \
  -F "artist=Artist"

# 4. Получить рекомендации
curl http://localhost:8000/recommendations/track123?limit=10
```

## API Endpoints

| Метод  | Путь                              | Описание                          |
|--------|-----------------------------------|-----------------------------------|
| POST   | `/tracks/upload`                  | Загрузить трек + извлечь фичи     |
| GET    | `/tracks/{track_id}/features`     | Получить фичи трека               |
| GET    | `/recommendations/{track_id}`     | Рекомендации по треку              |
| POST   | `/likes`                          | Записать лайк пользователя        |
| GET    | `/recommendations/user/{user_id}` | Персональные рекомендации          |
| POST   | `/automl/train`                   | Запустить AutoML оптимизацию       |
| GET    | `/automl/status`                  | Статус обучения                    |
| POST   | `/index/rebuild`                  | Пересобрать FAISS-индекс           |

## Извлекаемые аудио-фичи

- **MFCC** (13 коэффициентов) — тембр
- **Chroma** (12 полутонов) — гармония и тональность
- **Spectral Contrast** (7 полос) — яркость/текстура звука
- **Tonnetz** (6 измерений) — тональные отношения
- **Tempo** — BPM
- **RMS Energy** — громкость/динамика
- **Zero Crossing Rate** — шумность/перкуссивность
- **Spectral Centroid/Bandwidth/Rolloff** — частотные характеристики

Итого: **58-мерный вектор** на трек.

## Hyperparameter tuning (Optuna)

Автоматически оптимизируются:
- Веса фич (какие важнее для похожести)
- Метрика расстояния (cosine / euclidean / manhattan)
- Число кластеров (для предварительной группировки)
- Баланс content vs collaborative signal
- Алгоритм нормализации фич
