.PHONY: install run test test-unit test-integration coverage smoke audit build \
        lint format typecheck clean \
        docker-build docker-up docker-down docker-logs \
        batch-extract batch-recommend \
        docker-batch-extract docker-batch-recommend

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(PY) -m pip
export PYTHONPATH := $(CURDIR)/src:$(CURDIR)

# ── Local development ────────────────────────────────────────────
install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

run:
	$(VENV)/bin/uvicorn recommender.interfaces.online.main:app --reload --port 8000

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

audit:
	$(VENV)/bin/pip-audit --skip-editable

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
