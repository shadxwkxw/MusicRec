.PHONY: install run test lint format typecheck clean \
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

lint:
	$(VENV)/bin/ruff check src services tests
	$(VENV)/bin/ruff format --check src services tests

format:
	$(VENV)/bin/ruff format src services tests
	$(VENV)/bin/ruff check --fix src services tests

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
	rm -rf .pytest_cache .ruff_cache
