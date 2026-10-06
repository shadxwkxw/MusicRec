FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# upgrade — исправления безопасности поверх базового образа. Компилятор не нужен:
# у всех зависимостей есть готовые wheels
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv подключается только на время сборки (RUN --mount) и в образ не попадает.
# Ставим в системный Python образа; --inexact не трогает pip и прочие базовые пакеты
ENV UV_PROJECT_ENVIRONMENT=/usr/local UV_LINK_MODE=copy UV_NO_CACHE=1

# EXTRAS=embeddings добавляет torch и transformers для GET /search
ARG EXTRAS=""

# Зависимости отдельным слоем: правки в src/ не пересобирают тяжёлые пакеты
COPY pyproject.toml uv.lock README.md ./
RUN --mount=from=ghcr.io/astral-sh/uv:0.10.11,source=/uv,target=/bin/uv \
    uv sync --locked --inexact --no-install-project ${EXTRAS:+--extra $EXTRAS}

COPY src/ src/
COPY services/ services/
RUN --mount=from=ghcr.io/astral-sh/uv:0.10.11,source=/uv,target=/bin/uv \
    uv sync --locked --inexact --no-editable ${EXTRAS:+--extra $EXTRAS}

COPY configs/ configs/
RUN mkdir -p data/audio data/index data/models

EXPOSE 8000

CMD ["uvicorn", "recommender.interfaces.online.main:app", "--host", "0.0.0.0", "--port", "8000"]
