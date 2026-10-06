"""Health-проверки, метрики Prometheus, request id и формат логов."""

import asyncio
import json
import logging

from recommender.config import settings
from recommender.infrastructure.storage.postgres import get_db
from recommender.interfaces.online import health, observability
from recommender.interfaces.online.main import app
from tests.integration.test_s3 import s3  # noqa: F401 — фикстура S3 (moto)


def _sample(name: str, **labels: str) -> float:
    value = app.state.metrics.registry.get_sample_value(name, labels)
    return value or 0.0


# ── Health ───────────────────────────────────────────────────────


async def test_live_is_always_ok(api):
    assert (await api.client.get("/health/live")).json() == {"status": "ok"}


async def test_ready_ok_with_database_storage_and_index(api):
    await api.seed(3)

    resp = await api.client.get("/health/ready")

    body = resp.json()
    assert resp.status_code == 200, body
    assert body["status"] == "ok"
    assert body["checks"]["database"]["status"] == "ok"
    assert (body["checks"]["storage"]["status"], body["checks"]["storage"]["backend"]) == (
        "ok",
        "local",
    )
    assert body["checks"]["index"]["tracks"] == 3


async def test_empty_index_is_degraded_but_ready(api):
    resp = await api.client.get("/health/ready")

    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
    assert resp.json()["checks"]["index"]["status"] == "degraded"


async def test_database_down_is_not_ready(api):
    class Broken:
        async def execute(self, *args, **kwargs):
            raise ConnectionError("database is down")

    async def broken_db():
        yield Broken()

    app.dependency_overrides[get_db] = broken_db

    resp = await api.client.get("/health/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["database"] == {
        "status": "fail",
        "error": "ConnectionError: database is down",
        "ms": resp.json()["checks"]["database"]["ms"],
    }


async def test_s3_bucket_is_checked(api, s3, monkeypatch):  # noqa: F811
    assert (await api.client.get("/health/ready")).json()["checks"]["storage"]["status"] == "ok"

    monkeypatch.setattr(settings, "s3_bucket", "no-such-bucket")
    resp = await api.client.get("/health/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["storage"]["status"] == "fail"


async def test_slow_check_times_out(api, monkeypatch):
    async def slow():
        await asyncio.sleep(5)
        return {}

    monkeypatch.setattr(settings, "health_timeout", 0.05)
    monkeypatch.setattr(health, "_storage", slow)

    resp = await api.client.get("/health/ready")

    assert resp.status_code == 503
    assert resp.json()["checks"]["storage"]["error"] == "timeout after 0.05s"


# ── Метрики ──────────────────────────────────────────────────────


async def test_metrics_use_route_templates_and_count_recommendations(api):
    ids = await api.seed(3)
    route = "/recommendations/{track_id}"
    before = _sample("recommender_http_requests_total", method="GET", route=route, status="200")
    popular = _sample("recommender_recommendations_total", kind="user", strategy="popular")

    for _ in range(2):
        await api.client.get(f"/recommendations/{ids[0]}")
    await api.client.get("/recommendations/user/newcomer")
    await api.client.get("/no/such/path")
    text = (await api.client.get("/metrics")).text

    after = _sample("recommender_http_requests_total", method="GET", route=route, status="200")
    assert after - before == 2
    assert _sample("recommender_recommendations_total", kind="user", strategy="popular") == (
        popular + 1
    )
    assert _sample("recommender_http_requests_total", method="GET", route="unmatched", status="404")
    assert ids[0] not in text  # id треков не попадают в метки
    assert "recommender_index_tracks 3.0" in text
    assert 'recommender_index_info{metric="cosine"' in text
    assert "recommender_http_request_duration_seconds_bucket" in text


# ── Request id и логи ────────────────────────────────────────────


async def test_request_id_is_generated_or_taken_from_client(api):
    generated = (await api.client.get("/tracks")).headers["X-Request-ID"]
    echoed = await api.client.get("/tracks", headers={"X-Request-ID": "web-42.a_b"})
    unsafe = await api.client.get("/tracks", headers={"X-Request-ID": "bad id; drop"})

    assert len(generated) == 32
    assert echoed.headers["X-Request-ID"] == "web-42.a_b"
    assert unsafe.headers["X-Request-ID"] != "bad id; drop"


async def test_each_request_is_logged_once_with_its_id(api, caplog):
    caplog.set_level(logging.INFO, logger="recommender.requests")

    await api.client.get("/tracks", headers={"X-Request-ID": "req-1"})
    await api.client.get("/health/live")  # пробы — только в DEBUG

    records = [r for r in caplog.records if r.name == "recommender.requests"]
    assert len(records) == 1
    record = records[0]
    assert (record.route, record.status, record.method) == ("/tracks", 200, "GET")
    entry = json.loads(observability.JsonFormatter().format(record))
    assert entry["path"] == "/tracks"
    assert entry["level"] == "INFO"
    assert entry["duration_ms"] >= 0


def test_json_formatter_includes_request_id_and_errors():
    formatter = observability.JsonFormatter()
    token = observability.request_id_var.set("abc")
    try:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom %s", ("!",), None)
        observability._RequestIdFilter().filter(record)
        try:
            raise ValueError("bad")
        except ValueError:
            import sys

            record.exc_info = sys.exc_info()
    finally:
        observability.request_id_var.reset(token)

    entry = json.loads(formatter.format(record))
    assert (entry["message"], entry["request_id"], entry["level"]) == ("boom !", "abc", "ERROR")
    assert "ValueError: bad" in entry["exc_info"]
