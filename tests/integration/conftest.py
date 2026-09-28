"""Фикстуры интеграционных тестов API.

Каждый тест получает свою SQLite, свои папки audio/index/models и пустое
состояние приложения — рабочая data/ не затрагивается. Lifespan не
запускается (ASGITransport его не вызывает), поэтому state задаётся вручную.
"""

from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import Base, get_db
from recommender.interfaces.online import routes
from recommender.interfaces.online.main import app

SR = 22050


@pytest.fixture(scope="session")
def audio_files(tmp_path_factory) -> list[Path]:
    """8 синтетических треков с разной тональностью и темпом."""
    out = tmp_path_factory.mktemp("audio_src")
    rng = np.random.default_rng(42)
    t = np.arange(SR * 3) / SR
    paths = []
    for i in range(8):
        freq = 110 * (1 + i * 0.35)
        y = 0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * freq * 1.5 * t)
        clicks = np.zeros_like(t)
        clicks[:: int(SR * 60 / (70 + i * 12))] = 1.0
        y += 0.5 * np.convolve(clicks, np.hanning(200), mode="same")
        y += 0.02 * rng.standard_normal(len(t))
        path = out / f"track_{i}.wav"
        sf.write(path, y.astype(np.float32), SR)
        paths.append(path)
    return paths


@pytest.fixture
async def api(tmp_path, monkeypatch, audio_files):
    for name in ("audio_dir", "index_dir", "models_dir"):
        folder = tmp_path / name
        folder.mkdir()
        monkeypatch.setattr(settings, name, folder)
    monkeypatch.setattr(settings, "automl_n_trials", 4)

    db_engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    async with db_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with sessions() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    monkeypatch.setattr(routes, "async_session", sessions)
    app.state.engine = FaissRecommender()
    app.state.normalizer = FeatureNormalizer()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def upload(i: int, title: str | None = None, artist: str = "Artist") -> dict:
            path = audio_files[i]
            with path.open("rb") as f:
                resp = await client.post(
                    "/tracks/upload",
                    files={"file": (path.name, f, "audio/wav")},
                    data={"title": title or f"Track {i}", "artist": artist},
                )
            assert resp.status_code == 200, resp.text
            return resp.json()

        async def seed(n: int) -> list[str]:
            """Загрузить n треков и собрать индекс."""
            ids = [(await upload(i))["id"] for i in range(n)]
            resp = await client.post("/index/rebuild")
            assert resp.status_code == 200, resp.text
            return ids

        async def like(user_id: str, track_id: str) -> None:
            resp = await client.post("/likes", json={"user_id": user_id, "track_id": track_id})
            assert resp.status_code == 200, resp.text

        yield SimpleNamespace(client=client, sessions=sessions, upload=upload, seed=seed, like=like)

    app.dependency_overrides.clear()
    await db_engine.dispose()
