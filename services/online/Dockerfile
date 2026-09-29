FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /bin/uv
# Ставим в системный Python образа; --inexact не трогает pip и прочие базовые пакеты
ENV UV_PROJECT_ENVIRONMENT=/usr/local UV_LINK_MODE=copy UV_NO_CACHE=1

# Зависимости отдельным слоем: правки в src/ не пересобирают тяжёлые пакеты
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --inexact --no-install-project

COPY src/ src/
COPY services/ services/
RUN uv sync --locked --inexact --no-editable

COPY configs/ configs/
RUN mkdir -p data/audio data/index data/models

EXPOSE 8000

CMD ["uvicorn", "recommender.interfaces.online.main:app", "--host", "0.0.0.0", "--port", "8000"]
