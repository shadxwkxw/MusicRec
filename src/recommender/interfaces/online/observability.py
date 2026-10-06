"""Логи, request id и метрики Prometheus online-сервиса.

- Логи: одна строка на запрос (метод, маршрут, статус, время, request id),
  текстом или JSON (observability.log_json) — для сборщиков логов. Access-лог
  uvicorn выключен: он дублировал бы эти строки без request id.
- X-Request-ID: берётся из запроса (его может передать веб-приложение) или
  создаётся; возвращается в ответе и попадает во все логи запроса.
- Метрики: запросы и задержки по шаблону маршрута (/recommendations/{track_id},
  а не по каждому id), рекомендации по стратегиям, размер и версия индекса.
  Свой реестр, а не глобальный: тесты и повторный импорт не конфликтуют.
  Счётчики живут в процессе — сервис запускается одним воркером uvicorn.
"""

import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    PlatformCollector,
    ProcessCollector,
)
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from prometheus_client.registry import Collector

from recommender.config import settings

REQUEST_ID_HEADER = "X-Request-ID"
QUIET_ROUTES = {"/health/live", "/health/ready", "/metrics"}  # пробы и скрейпы — в DEBUG
# Служебные INFO библиотек (плагины alembic, источник ключей boto) — только при DEBUG
NOISY_LOGGERS = ("alembic", "botocore", "boto3", "s3transfer", "urllib3", "httpx")
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("recommender.requests")

# ── Логи ─────────────────────────────────────────────────────────


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Запись лога — один JSON-объект в строку."""

    EXTRA = ("request_id", "method", "route", "path", "status", "duration_ms")

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in self.EXTRA:
            value = getattr(record, key, None)
            if value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        request_id = getattr(record, "request_id", None)
        return f"{line} [{request_id}]" if request_id else line


def configure_logging() -> None:
    """Корневой логгер в stderr; логи uvicorn — в том же формате, его access-лог выключен."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if settings.log_json else TextFormatter())
    handler.addFilter(_RequestIdFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    if settings.log_level != "DEBUG":
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


# ── Метрики ──────────────────────────────────────────────────────


class _IndexCollector(Collector):
    """Размер и версия индекса — читаются из состояния сервиса в момент скрейпа."""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    def collect(self) -> Iterator[GaugeMetricFamily | InfoMetricFamily]:
        engine = getattr(self.app.state, "engine", None)
        if engine is None:
            return
        tracks = GaugeMetricFamily("recommender_index_tracks", "Tracks in the loaded index")
        tracks.add_metric([], float(engine.index.ntotal))
        yield tracks
        info = InfoMetricFamily("recommender_index", "Loaded index version")
        info.add_metric(
            [],
            {
                "version": engine.version or "none",
                "source": engine.source,
                "metric": engine.metric,
            },
        )
        yield info


class Metrics:
    def __init__(self, app: FastAPI) -> None:
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        self.registry.register(_IndexCollector(app))
        self.requests = Counter(
            "recommender_http_requests",
            "HTTP requests by route template and status",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "recommender_http_request_duration_seconds",
            "HTTP request latency by route template",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )
        self.recommendations = Counter(
            "recommender_recommendations",
            "Served recommendation lists",
            ["kind", "strategy"],
            registry=self.registry,
        )


def metrics_of(app: FastAPI) -> Metrics:
    return app.state.metrics


def count_recommendations(request: Request, kind: str, strategy: str) -> None:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.recommendations.labels(kind=kind, strategy=strategy).inc()


# ── Middleware ───────────────────────────────────────────────────


def _route_template(request: Request) -> str:
    """Шаблон маршрута вместо пути: id не раздувают число рядов метрик."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "unmatched"


def install(app: FastAPI) -> None:
    """Подключить метрики и middleware запросов к приложению."""
    app.state.metrics = Metrics(app)

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        except Exception:
            logger.exception("Unhandled error in %s %s", request.method, request.url.path)
            raise
        finally:
            elapsed = time.perf_counter() - started
            route = _route_template(request)
            metrics = metrics_of(request.app)
            metrics.requests.labels(request.method, route, str(status)).inc()
            metrics.latency.labels(request.method, route).observe(elapsed)
            quiet = route in QUIET_ROUTES and status < 400
            logger.log(
                logging.DEBUG if quiet else logging.WARNING if status >= 500 else logging.INFO,
                "%s %s %s %.1fms",
                request.method,
                request.url.path,
                status,
                elapsed * 1000,
                extra={
                    "method": request.method,
                    "route": route,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
            request_id_var.reset(token)
