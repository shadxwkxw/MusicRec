"""Проверки состояния для Docker, балансировщика и Airflow. Ключ API не нужен.

- /health/live  — процесс отвечает (зависимости не проверяются);
- /health/ready — база и хранилище доступны (иначе 503) и в каком состоянии
  индекс. Пустой индекс или индекс другого источника признаков — degraded:
  сервис принимает загрузки, но рекомендации будут пустыми.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.config import settings
from recommender.infrastructure.storage.audio_store import s3_client
from recommender.infrastructure.storage.postgres import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def live() -> dict:
    return {"status": "ok"}


async def _timed(check: Callable[[], Awaitable[dict]]) -> dict:
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(check(), timeout=settings.health_timeout)
        result.setdefault("status", "ok")
    except TimeoutError:
        result = {"status": "fail", "error": f"timeout after {settings.health_timeout:g}s"}
    except Exception as e:
        result = {"status": "fail", "error": f"{type(e).__name__}: {e}"}
    result["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


async def _database(db: AsyncSession) -> dict:
    await db.execute(text("SELECT 1"))
    return {}


async def _storage() -> dict:
    if settings.storage_backend == "s3":
        bucket = settings.s3_bucket
        await run_in_threadpool(lambda: s3_client().head_bucket(Bucket=bucket))
        return {"backend": "s3", "bucket": bucket}
    audio_dir = settings.audio_dir
    if not audio_dir.is_dir() or not os.access(audio_dir, os.W_OK):
        raise RuntimeError(f"{audio_dir} is not a writable directory")
    return {"backend": "local"}


def _index(request: Request) -> dict:
    engine = request.app.state.engine
    tracks = int(engine.index.ntotal)
    compatible = engine.source == settings.feature_source_id
    status = "ok" if tracks and compatible else "degraded"
    return {"status": status, "tracks": tracks, "version": engine.version, "source": engine.source}


@router.get("/ready")
async def ready(request: Request, db: AsyncSession = Depends(get_db)) -> JSONResponse:
    database, storage = await asyncio.gather(_timed(lambda: _database(db)), _timed(_storage))
    checks = {"database": database, "storage": storage, "index": _index(request)}
    if any(checks[name]["status"] == "fail" for name in ("database", "storage")):
        status, code = "fail", 503
    elif checks["index"]["status"] != "ok":
        status, code = "degraded", 200
    else:
        status, code = "ok", 200
    return JSONResponse({"status": status, "checks": checks}, status_code=code)
