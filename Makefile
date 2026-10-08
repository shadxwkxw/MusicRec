.PHONY: env install install-embeddings lock upgrade run test test-unit test-integration coverage smoke audit build \
        lint format typecheck clean e2e-compose openapi openapi-check \
        docker-build docker-up docker-down docker-logs \
        batch-extract batch-import-s3 batch-migrate-s3 add-music batch-embed batch-rebuild batch-tune batch-evaluate batch-recommend index-reload \
        docker-batch-extract docker-batch-embed docker-batch-rebuild docker-batch-tune docker-batch-recommend \
        migrate migration db-copy fma-unpack fma-import \
        airflow-up airflow-down airflow-logs airflow-check

# .env (если есть) — те же переменные, что читает приложение; см. .env.example
-include .env

VENV := .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
export UV_PROJECT_ENVIRONMENT := $(abspath $(VENV))

# ── Local development ────────────────────────────────────────────
# Создать .env из шаблона (существующий не трогает)
env:
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo "created .env from .env.example")

# Точные версии из uv.lock; падает, если lock не соответствует pyproject.toml
install:
	uv sync --locked --extra dev

# То же плюс torch и transformers для предобученных аудиоэмбеддингов
install-embeddings:
	uv sync --locked --extra dev --extra embeddings

# Обновить uv.lock после правки зависимостей в pyproject.toml
lock:
	uv lock

# Поднять все зависимости до свежих версий в рамках ограничений pyproject.toml
upgrade:
	uv lock --upgrade
	uv sync --locked --extra dev

run:
	$(VENV)/bin/uvicorn recommender.interfaces.online.main:app --reload --port 8000

# ── Database ─────────────────────────────────────────────────────
# БД выбирается через DB_URL (по умолчанию локальная SQLite из configs/config.yaml).
# Сервисы применяют миграции сами при старте; migrate — чтобы сделать это явно.
migrate:
	$(VENV)/bin/alembic upgrade head

# Новая миграция по изменениям моделей: make migration MSG="add genre to tracks"
migration:
	@test -n "$(MSG)" || (echo 'usage: make migration MSG="..."' && exit 1)
	$(VENV)/bin/alembic revision --autogenerate -m "$(MSG)"

# Перенос данных в пустую базу, по умолчанию из локальной SQLite в Postgres из docker-compose
COPY_FROM ?= sqlite+aiosqlite:///data/recommender.db
POSTGRES_USER     ?= recommender
POSTGRES_PASSWORD ?= recommender
POSTGRES_DB       ?= recommender
POSTGRES_HOST_PORT ?= 5432
COPY_TO   ?= postgresql+asyncpg://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_HOST_PORT)/$(POSTGRES_DB)
# @ — не печатать команду: в адресах базы пароль из .env
db-copy:
	@$(PY) scripts/copy_db.py --source "$(COPY_FROM)" --target "$(COPY_TO)"

test:
	$(VENV)/bin/pytest tests/ -v

test-unit:
	$(VENV)/bin/pytest tests/unit -v

test-integration:
	$(VENV)/bin/pytest tests/integration -v

# Покрытие по всем тестам; падает, если ниже порога
coverage:
	$(VENV)/bin/pytest tests/ -q --cov=recommender --cov-report=term-missing \
		--cov-report=html --cov-fail-under=80

# Настоящий uvicorn + рестарт + batch CLI во временной папке
smoke:
	$(PY) scripts/smoke_test.py

# Контракт API: docs/openapi.json (make openapi после изменения эндпоинтов)
openapi:
	$(PY) scripts/export_openapi.py

openapi-check:
	$(PY) scripts/export_openapi.py --check

# E2E в docker compose: собранный образ, Postgres, API-ключ, рестарт сервиса.
# Отдельный проект и тома, .env не читается — рабочие данные не затрагиваются.
E2E_PORT    ?= 18000
E2E_API_KEY ?= e2e-local-key-0123456789abcdef
E2E_COMPOSE := API_KEY=$(E2E_API_KEY) ONLINE_HOST_PORT=$(E2E_PORT) LOG_JSON=true \
	docker compose -p recommender-e2e --env-file /dev/null -f docker-compose.yml -f docker-compose.e2e.yml
E2E_RUN     := API_KEY=$(E2E_API_KEY) API_URL=http://localhost:$(E2E_PORT) python3 scripts/e2e_compose.py

e2e-compose:
	$(E2E_COMPOSE) up -d --build --wait postgres recommender-online
	$(E2E_RUN) seed && $(E2E_COMPOSE) restart recommender-online && $(E2E_RUN) verify; \
	status=$$?; [ $$status -eq 0 ] || $(E2E_COMPOSE) logs recommender-online | tail -50; \
	$(E2E_COMPOSE) down -v; exit $$status

# Проверяет зафиксированные в uv.lock версии (для всех версий Python), а не .venv
audit:
	uv export --locked --extra dev --no-emit-project -o $(VENV)/audit-requirements.txt
	$(VENV)/bin/pip-audit --disable-pip --require-hashes -r $(VENV)/audit-requirements.txt

build:
	rm -rf dist
	$(PY) -m build --wheel

lint:
	$(VENV)/bin/ruff check src services tests scripts airflow
	$(VENV)/bin/ruff format --check src services tests scripts airflow

format:
	$(VENV)/bin/ruff format src services tests scripts airflow
	$(VENV)/bin/ruff check --fix src services tests scripts airflow

typecheck:
	$(VENV)/bin/mypy src

# ── Batch service (local) ────────────────────────────────────────
# Usage: make batch-extract INPUT_DIR=data/audio_inbox
INPUT_DIR ?= data/audio
OUTPUT    ?= artifacts/recs.parquet
TOP_N     ?= 10
API_URL   ?= http://localhost:8000

batch-extract:
	$(PY) services/batch/main.py extract --input-dir $(INPUT_DIR)

# Импорт аудио из S3 (storage.s3_bucket): make batch-import-s3 S3_IMPORT_PREFIX=music/ WORKERS=8
S3_IMPORT_PREFIX ?=
batch-import-s3:
	$(PY) services/batch/main.py import-s3 --prefix "$(S3_IMPORT_PREFIX)" --workers $(WORKERS)

# Новая музыка одной командой: папка → S3 (без дублей) → импорт → эмбеддинги → индекс.
# make add-music DIR=musicnew [MUSIC_PREFIX=music/]. Файлы — «Артист - Название.mp3»,
# жанр берётся из тегов. Reload — если сервис запущен (иначе подхватит при старте).
MUSIC_PREFIX ?= music/
add-music:
	@test -n "$(DIR)" || (echo "usage: make add-music DIR=path/to/folder" && exit 1)
	$(PY) services/batch/main.py upload-s3 --dir "$(DIR)" --prefix "$(MUSIC_PREFIX)"
	$(PY) services/batch/main.py import-s3 --prefix "$(MUSIC_PREFIX)" --workers $(WORKERS)
	$(PY) services/batch/main.py embed
	$(PY) services/batch/main.py rebuild
	@$(MAKE) --no-print-directory index-reload || echo "service is not running: it will load the new index on start"

# Перенос в S3 (AUDIO_STORAGE=s3): аудио из data/audio, индекс, artifacts/*.parquet.
# make batch-migrate-s3 DELETE_LOCAL=1 — удалить локальные копии аудио после переноса
DELETE_LOCAL ?=
batch-migrate-s3:
	$(PY) services/batch/main.py migrate-s3 $(if $(DELETE_LOCAL),--delete-local)

# Эмбеддинги модели features.embedding_model для треков без них (нужен make install-embeddings)
batch-embed:
	$(PY) services/batch/main.py embed

# Индекс пишется на диск; запущенный сервис подхватит его после make index-reload
batch-rebuild:
	$(PY) services/batch/main.py rebuild

batch-tune:
	$(PY) services/batch/main.py tune

# Текущий индекс на отложенных лайках против бейзлайнов
batch-evaluate:
	$(PY) services/batch/main.py evaluate

# @ — команда не печатается: в ней может быть API_KEY из .env
index-reload:
	@curl -fsS -X POST $(if $(API_KEY),-H "X-API-Key: $(API_KEY)") $(API_URL)/index/reload

batch-recommend:
	$(PY) services/batch/main.py recommend --output $(OUTPUT) --top-n $(TOP_N)

# ── Free Music Archive ───────────────────────────────────────────
# Архивы: https://github.com/mdeff/fma (fma_metadata.zip, fma_small.zip) в $(FMA_ROOT)
FMA_ROOT   ?= data/fma
FMA_SUBSET ?= small
WORKERS    ?= 4

fma-unpack:
	# архивы сжаты bzip2, который не умеет unzip на macOS; zipfile из Python умеет
	cd $(FMA_ROOT) && $(abspath $(PY)) -m zipfile -e fma_metadata.zip . \
		&& $(abspath $(PY)) -m zipfile -e fma_$(FMA_SUBSET).zip .

fma-import:
	$(PY) services/batch/main.py import-fma --root $(FMA_ROOT) --subset $(FMA_SUBSET) --workers $(WORKERS)

# ── Docker ───────────────────────────────────────────────────────
docker-build:
	docker compose build

# --build: образ пересобирается, если поменялись зависимости (иначе слои из кэша)
docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f recommender-online

docker-batch-extract:
	docker compose --profile batch run --rm recommender-batch \
		extract --input-dir /app/$(INPUT_DIR)

docker-batch-embed:
	docker compose --profile batch run --rm recommender-batch embed

docker-batch-rebuild:
	docker compose --profile batch run --rm recommender-batch rebuild

docker-batch-tune:
	docker compose --profile batch run --rm recommender-batch tune

docker-batch-recommend:
	docker compose --profile batch run --rm recommender-batch \
		recommend --output /app/$(OUTPUT) --top-n $(TOP_N)

# ── Airflow (профиль airflow в docker-compose) ───────────────────
# UI на http://localhost:8080 (логин и пароль — AIRFLOW_USER / AIRFLOW_PASSWORD из .env).
# DAG'и запускают образ music-recommender-batch, поэтому он собирается заранее;
# после правок кода batch его нужно пересобрать (make docker-build).
airflow-up:
	mkdir -p data/inbox artifacts airflow/logs
	docker compose --profile batch build recommender-batch
	docker compose --profile airflow up -d --build

airflow-down:
	docker compose --profile airflow down

airflow-logs:
	docker compose --profile airflow logs -f airflow-scheduler airflow-dag-processor

# Импортируются ли DAG'и без ошибок — в чистом образе Airflow, без запущенного стека
AIRFLOW_VERSION ?= 3.3.2
airflow-check:
	docker run --rm -e AIRFLOW__CORE__LOAD_EXAMPLES=false \
		-v "$(CURDIR)/airflow/dags:/opt/airflow/dags:ro" \
		-v "$(CURDIR)/scripts/check_dags.py:/check_dags.py:ro" \
		apache/airflow:$(AIRFLOW_VERSION) python /check_dags.py recommender_daily recommender_weekly_tuning

# ── Cleanup ──────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc'     -not -path './$(VENV)/*' -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist
