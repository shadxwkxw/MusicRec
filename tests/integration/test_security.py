"""Доступ к API: ключ для изменений (и для чтения при protect_reads), лимиты, CORS."""

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from recommender.config import settings
from recommender.interfaces.online.main import app, create_app

KEY = "test-key-0123456789abcdef"  # gitleaks:allow (ненастоящий ключ для тестов)


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setattr(settings, "api_key", SecretStr(KEY))


def _routes(methods: set[str]) -> list[tuple[str, str]]:
    """Все эндпоинты API по OpenAPI-схеме (включая подключённые роутеры)."""
    found = []
    for path, operations in app.openapi()["paths"].items():
        concrete = path.replace("{track_id}", "t").replace("{user_id}", "u")
        found += [(m.upper(), concrete) for m in operations if m.upper() in methods]
    return found


async def test_every_write_route_needs_the_key(api, keyed):
    writes = _routes({"POST", "PATCH", "DELETE", "PUT"})
    assert len(writes) >= 8

    for method, path in writes:
        resp = await api.client.request(method, path)
        assert resp.status_code == 401, (method, path)
        resp = await api.client.request(method, path, headers={"X-API-Key": "wrong" * 5})
        assert resp.status_code == 401, (method, path)


async def test_reads_are_open_unless_protected(api, keyed, monkeypatch):
    # health открыт всегда: Docker и балансировщик ключ не передают
    reads = [(m, p) for m, p in _routes({"GET"}) if not p.startswith("/health/")]
    assert len(reads) >= 6
    for method, path in [*reads, ("GET", "/metrics")]:
        assert (await api.client.request(method, path)).status_code != 401, path

    monkeypatch.setattr(settings, "api_protect_reads", True)
    for method, path in [*reads, ("GET", "/metrics")]:
        assert (await api.client.request(method, path)).status_code == 401, path
    resp = await api.client.get("/tracks", headers={"X-API-Key": KEY})
    assert resp.status_code == 200
    for path in ("/health/live", "/health/ready"):
        assert (await api.client.get(path)).status_code == 200, path


async def test_backend_with_key_can_upload_like_and_reload(api, keyed, audio_files):
    headers = {"X-API-Key": KEY}
    with audio_files[0].open("rb") as f:
        resp = await api.client.post(
            "/tracks/upload",
            files={"file": ("a.wav", f, "audio/wav")},
            data={"title": "A"},
            headers=headers,
        )
    assert resp.status_code == 200, resp.text
    track_id = resp.json()["id"]

    like = {"user_id": "u1", "track_id": track_id}
    assert (await api.client.post("/likes", json=like)).status_code == 401
    assert (await api.client.post("/likes", json=like, headers=headers)).status_code == 200
    assert (await api.client.post("/index/rebuild", headers=headers)).status_code == 200


async def test_too_large_upload_is_rejected_and_not_stored(api, audio_files, monkeypatch):
    monkeypatch.setattr(settings, "api_max_upload_mb", 0.01)  # ~10 KB, тестовый wav больше

    with audio_files[0].open("rb") as f:
        resp = await api.client.post(
            "/tracks/upload", files={"file": ("big.wav", f, "audio/wav")}, data={"title": "Big"}
        )

    assert resp.status_code == 413
    assert list(settings.audio_dir.iterdir()) == []
    assert (await api.client.get("/tracks")).json() == []


async def test_recommendation_limit_is_bounded(api):
    (track_id,) = await api.seed(1)
    too_many = settings.api_max_rec_limit + 1

    for path in (f"/recommendations/{track_id}", "/recommendations/user/u1"):
        assert (await api.client.get(path, params={"limit": too_many})).status_code == 422
        assert (await api.client.get(path, params={"limit": 0})).status_code == 422


def test_short_key_is_rejected():
    from pathlib import Path

    import yaml

    from recommender.config import _build_settings

    raw = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/config.yaml").read_text())
    raw["database"]["url"] = "sqlite+aiosqlite://"
    raw["features"]["source"] = "librosa"
    raw["storage"]["backend"] = "local"
    raw["api"]["key"] = "short"

    with pytest.raises(ValidationError, match="API_KEY is too short"):
        _build_settings(raw)


async def test_cors_allows_only_configured_origins(monkeypatch):
    monkeypatch.setattr(settings, "api_cors_origins", ["https://app.example.com"])
    transport = httpx.ASGITransport(app=create_app())

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def preflight(origin: str) -> httpx.Response:
            return await client.options(
                "/likes",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-API-Key",
                },
            )

        allowed = await preflight("https://app.example.com")
        denied = await preflight("https://evil.example.com")

    assert allowed.headers["access-control-allow-origin"] == "https://app.example.com"
    assert "access-control-allow-origin" not in denied.headers


def test_upload_copy_stops_past_the_limit():
    """Потоковая проверка — для загрузок, размер которых заранее неизвестен."""
    import io

    from recommender.interfaces.online.routes import _copy_limited

    target = io.BytesIO()
    assert _copy_limited(io.BytesIO(b"x" * 100), target, max_bytes=100) == 100
    assert _copy_limited(io.BytesIO(b"x" * 101), io.BytesIO(), max_bytes=100) is None
    assert target.getvalue() == b"x" * 100
