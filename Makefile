.PHONY: install lock upgrade run test test-unit test-integration coverage smoke audit build \
        lint format typecheck clean \
        docker-build docker-up docker-down docker-logs \
        batch-extract batch-recommend \
        docker-batch-extract docker-batch-recommend \
        migrate migration db-copy

VENV := .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)
export UV_PROJECT_ENVIRONMENT := $(abspath $(VENV))

# ── Local development ────────────────────────────────────────────
# Точные версии из uv.lock; падает, если lock не соответствует pyproject.toml
install:
	uv sync --locked --extra dev

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
COPY_TO   ?= postgresql+asyncpg://recommender:recommender@localhost:5432/recommender
db-copy:
	$(PY) scripts/copy_db.py --source "$(COPY_FROM)" --target "$(COPY_TO)"

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

# Проверяет зафиксированные в uv.lock версии (для всех версий Python), а не .venv
audit:
	uv export --locked --extra dev --no-emit-project -o $(VENV)/audit-requirements.txt
	$(VENV)/bin/pip-audit --disable-pip --require-hashes -r $(VENV)/audit-requirements.txt

build:
	rm -rf dist
	$(PY) -m build --wheel

lint:
	$(VENV)/bin/ruff check src services tests scripts
	$(VENV)/bin/ruff format --check src services tests scripts

format:
	$(VENV)/bin/ruff format src services tests scripts
	$(VENV)/bin/ruff check --fix src services tests scripts

typecheck:
	$(VENV)/bin/mypy src

# ── Batch service (local) ────────────────────────────────────────
# Usage: make batch-extract INPUT_DIR=data/audio_inbox
INPUT_DIR ?= data/audio
OUTPUT    ?= artifacts/recs.parquet
TOP_N     ?= 10

batch-extract:
	$(PY) services/batch/main.py extract --input-dir $(INPUT_DIR)

batch-recommend:
	$(PY) services/batch/main.py recommend --output $(OUTPUT) --top-n $(TOP_N)

# ── Docker ───────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f recommender-online

docker-batch-extract:
	docker compose --profile batch run --rm recommender-batch \
		extract --input-dir /app/$(INPUT_DIR)

docker-batch-recommend:
	docker compose --profile batch run --rm recommender-batch \
		recommend --output /app/$(OUTPUT) --top-n $(TOP_N)

# ── Cleanup ──────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc'     -not -path './$(VENV)/*' -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist
