FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY services/ services/

RUN pip install --no-cache-dir .

COPY configs/ configs/
RUN mkdir -p data/audio data/index data/models

ENTRYPOINT ["python", "services/batch/main.py"]
