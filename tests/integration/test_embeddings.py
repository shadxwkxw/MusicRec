"""Режим features.source=embedding с поддельной моделью (без torch и весов)."""

import zlib
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from recommender.application.batch_embed import run_batch_embed
from recommender.application.batch_extract import ImportItem, run_batch_import
from recommender.config import settings
from recommender.infrastructure.storage.artifacts import load_current, publish
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import TrackEmbeddingORM, TrackORM
from recommender.interfaces.online.main import app

DIM = 16


class FakeEmbedder:
    """Вектор детерминированно зависит от содержимого файла."""

    model_name = "fake/model"

    def __init__(self, broken: set[str] = frozenset()):
        self.broken = broken
        self.calls = 0

    def embed_files(self, paths):
        for i, path in enumerate(paths):
            self.calls += 1
            if Path(path).name in self.broken:
                yield i, "RuntimeError: cannot decode"
                continue
            seed = zlib.crc32(Path(path).read_bytes())
            yield i, np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


@pytest.fixture
def embedding_mode(monkeypatch):
    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)


async def _embed(api, embedder=None):
    async with api.sessions() as db:
        return await run_batch_embed(db, embedder or FakeEmbedder())


async def _count_embeddings(api, track_id=None) -> int:
    query = select(func.count()).select_from(TrackEmbeddingORM)
    if track_id:
        query = query.where(TrackEmbeddingORM.track_id == track_id)
    async with api.sessions() as db:
        return await db.scalar(query)


async def test_batch_embed_stores_vectors_skips_done_and_reports_problems(api, audio_files):
    tracks = [await api.upload(i) for i in range(4)]
    async with api.sessions() as db:
        # трек, у которого аудио пропало с диска
        lost = await db.get(TrackORM, tracks[3]["id"])
        Path(lost.audio_path).unlink()

    broken_name = Path((await _track(api, tracks[2]["id"])).audio_path).name
    stats = await _embed(api, FakeEmbedder(broken={broken_name}))

    assert stats.processed == 2
    assert sorted(err.split(":")[0] for _, err in stats.failed) == [
        "RuntimeError",
        "audio not found",
    ]
    assert await _count_embeddings(api) == 2

    again = FakeEmbedder()
    stats = await _embed(api, again)
    assert (stats.processed, stats.skipped) == (1, 2)  # досчитан только ранее упавший
    assert again.calls == 1


async def _track(api, track_id) -> TrackORM:
    async with api.sessions() as db:
        return await db.get(TrackORM, track_id)


async def test_embedding_mode_end_to_end(api, embedding_mode):
    ids = [(await api.upload(i))["id"] for i in range(5)]

    # До подсчёта эмбеддингов индексировать нечего
    assert (await api.client.post("/index/rebuild")).status_code == 400
    resp = await api.client.get(f"/recommendations/{ids[0]}")
    assert resp.status_code == 409

    await _embed(api)
    rebuilt = (await api.client.post("/index/rebuild")).json()

    assert (rebuilt["tracks_indexed"], rebuilt["feature_dim"]) == (5, DIM)
    recs = (await api.client.get(f"/recommendations/{ids[0]}")).json()["recommendations"]
    assert {r["track_id"] for r in recs} == set(ids[1:])

    # Загрузка в режиме embedding не попадает в индекс сразу, даже при обученном нормализаторе
    late = await api.upload(5)
    assert late["indexed"] is False
    assert late["id"] not in app.state.engine.track_ids
    await _embed(api)
    await api.client.post("/index/rebuild")
    assert late["id"] in load_current().engine.track_ids


async def test_switching_to_embeddings_drops_librosa_tuned_params(api, monkeypatch):
    await api.seed(4)
    # Как после тюнинга на librosa: у нормализатора сохранены 82 веса групп
    current = load_current()
    current.normalizer.weights = np.full(settings.feature_dim, 2.0, dtype=np.float32)
    publish(current.engine, current.normalizer)

    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)
    await _embed(api)
    resp = await api.client.post("/index/rebuild")

    assert resp.status_code == 200, resp.text
    assert resp.json()["feature_dim"] == DIM
    assert app.state.normalizer.weights is None


async def test_delete_removes_embeddings(api, embedding_mode):
    ids = [(await api.upload(i))["id"] for i in range(3)]
    await _embed(api)

    assert (await api.client.delete(f"/tracks/{ids[0]}")).status_code == 204

    assert await _count_embeddings(api, ids[0]) == 0
    assert await _count_embeddings(api) == 2


async def test_tuning_in_embedding_mode_has_no_feature_group_weights(api, embedding_mode):
    ids = [(await api.upload(i))["id"] for i in range(6)]
    for tid in ids[0:3]:
        await api.like("u1", tid)
    for tid in ids[1:4]:
        await api.like("u2", tid)
    await _embed(api)

    await api.client.post("/automl/train")
    run = (await api.client.get("/automl/status")).json()[0]

    assert run["status"] == "completed", run
    assert not [k for k in run["best_params"] if k.startswith("w_")]
    assert app.state.engine.dimension == DIM


async def test_batch_recommend_in_embedding_mode(api, embedding_mode, tmp_path):
    import pandas as pd

    from recommender.application.batch_recommend import run_batch_recommend

    for i in range(4):
        await api.upload(i)
    await _embed(api)
    await api.client.post("/index/rebuild")

    async with api.sessions() as db:
        result = await run_batch_recommend(db, tmp_path / "recs.parquet", top_n=2)

    assert result.tracks_scored == 4
    assert len(pd.read_parquet(tmp_path / "recs.parquet")) == 8


async def test_upload_and_import_store_audio_path(api, audio_files, tmp_path):
    uploaded = await api.upload(0)
    assert Path((await _track(api, uploaded["id"])).audio_path).exists()

    item = ImportItem(audio_files[1], "legacy.wav", "Legacy", "A")
    async with api.sessions() as db:
        await run_batch_import([item], db)
        track = (
            await db.execute(select(TrackORM).where(TrackORM.filename == "legacy.wav"))
        ).scalar_one()
        assert track.audio_path == str(audio_files[1])
        # Трек из времён до audio_path: повторный импорт дописывает путь без пересчёта
        track.audio_path = None
        await db.commit()
        stats = await run_batch_import([item], db)

    assert (stats.processed, stats.skipped, stats.paths_filled) == (0, 1, 1)
    async with api.sessions() as db:
        track = (
            await db.execute(select(TrackORM).where(TrackORM.filename == "legacy.wav"))
        ).scalar_one()
    assert track.audio_path == str(audio_files[1])


async def test_only_configured_model_embeddings_are_used(api, embedding_mode):
    from recommender.application.features import load_vectors

    ids = [(await api.upload(i))["id"] for i in range(3)]
    await _embed(api)
    async with api.sessions() as db:
        # эмбеддинги другой модели другой размерности для тех же треков
        for tid in ids:
            db.add(
                TrackEmbeddingORM(
                    track_id=tid, model="other/model", vector=np.ones(8, np.float32).tobytes()
                )
            )
        await db.commit()
        vectors = await load_vectors(db)

    assert set(vectors) == set(ids)
    assert {v.shape for v in vectors.values()} == {(DIM,)}


async def test_index_from_other_source_is_refused(api, monkeypatch, tmp_path):
    from recommender.application.batch_recommend import run_batch_recommend
    from recommender.application.features import IndexSourceMismatchError

    await api.seed(3)  # индекс на librosa
    assert load_current().engine.source == "librosa"

    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)

    resp = await api.client.post("/index/reload")
    assert resp.status_code == 409
    assert "librosa" in resp.json()["detail"] and "rebuild" in resp.json()["detail"]
    with pytest.raises(IndexSourceMismatchError):
        async with api.sessions() as db:
            await run_batch_recommend(db, tmp_path / "recs.csv")


def test_index_saved_before_source_tracking_counts_as_librosa(tmp_path):
    from joblib import dump, load

    engine = FaissRecommender(dimension=4, metric="cosine")
    engine.add_tracks(["a"], np.ones((1, 4), np.float32))
    engine.save(tmp_path)
    meta = load(tmp_path / "meta.joblib")
    del meta["source"]
    dump(meta, tmp_path / "meta.joblib")

    assert FaissRecommender.load(tmp_path).source == "librosa"


async def test_count_missing_embeddings(api):
    from recommender.application.batch_embed import count_missing_embeddings

    for i in range(3):
        await api.upload(i)
    async with api.sessions() as db:
        assert await count_missing_embeddings(db, FakeEmbedder.model_name) == 3
    await _embed(api)
    async with api.sessions() as db:
        assert await count_missing_embeddings(db, FakeEmbedder.model_name) == 0
        assert await count_missing_embeddings(db, "other/model") == 3
