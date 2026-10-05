"""Интеграционные тесты REST API: реальные роуты, librosa, FAISS и SQLite."""

import numpy as np
import pytest
from sqlalchemy import func, select

from recommender.config import settings
from recommender.infrastructure.storage.artifacts import load_current
from recommender.infrastructure.storage.postgres import LikeORM
from recommender.interfaces.online.main import app


async def _tracks(api) -> dict[str, dict]:
    resp = await api.client.get("/tracks", params={"limit": 200})
    assert resp.status_code == 200
    return {t["id"]: t for t in resp.json()}


def _expected_bonus(candidate_scores, strength: float = 1.0) -> float:
    """Прибавка буста (cosine): вес × сила × разрыв между ближайшим и медианным кандидатом.

    Работает, когда в выдаче без буста все кандидаты, — так в этих тестах.
    """
    scores = list(candidate_scores)
    return app.state.engine.boost_weight * strength * (max(scores) - float(np.median(scores)))


async def _recs(api, track_id: str, **params) -> list[dict]:
    resp = await api.client.get(f"/recommendations/{track_id}", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()["recommendations"]


# ── Upload и индекс ─────────────────────────────────────────────


async def test_upload_before_rebuild_is_not_indexed(api):
    track = await api.upload(0)

    assert track["indexed"] is False
    assert (await _tracks(api))[track["id"]]["indexed"] is False
    assert app.state.engine.index.ntotal == 0


async def test_upload_invalid_audio_is_rejected_without_leftovers(api):
    resp = await api.client.post(
        "/tracks/upload",
        files={"file": ("broken.wav", b"not audio at all", "audio/wav")},
        data={"title": "Broken"},
    )

    assert resp.status_code == 400
    assert list(settings.audio_dir.iterdir()) == []
    assert await _tracks(api) == {}


async def test_rebuild_without_tracks_returns_400(api):
    resp = await api.client.post("/index/rebuild")
    assert resp.status_code == 400


async def test_first_rebuild_uses_config_defaults(api, monkeypatch):
    monkeypatch.setattr(settings, "default_metric", "euclidean")
    monkeypatch.setattr(settings, "default_norm_method", "minmax")
    monkeypatch.setattr(settings, "default_boost_weight", 0.9)

    await api.seed(3)

    engine = load_current().engine
    assert (engine.metric, engine.boost_weight) == ("euclidean", 0.9)
    assert app.state.normalizer.method == "minmax"


async def test_tracks_page_limits_come_from_config(api):
    too_big = settings.api_tracks_page_max + 1
    assert (await api.client.get("/tracks", params={"limit": too_big})).status_code == 422
    ok = await api.client.get("/tracks", params={"limit": settings.api_tracks_page_max})
    assert ok.status_code == 200


async def test_rebuild_indexes_everything_and_persists(api):
    ids = [(await api.upload(i))["id"] for i in range(3)]

    resp = await api.client.post("/index/rebuild")

    assert resp.json() == {
        "status": "ok",
        "tracks_indexed": 3,
        "feature_dim": settings.feature_dim,
    }
    assert all(t["indexed"] for t in (await _tracks(api)).values())
    assert set(load_current().engine.track_ids) == set(ids)


async def test_upload_after_rebuild_is_indexed_and_saved(api):
    await api.seed(3)

    track = await api.upload(3)

    assert track["indexed"] is True
    assert track["id"] in load_current().engine.track_ids


async def test_reload_picks_up_index_rebuilt_outside_the_service(api, audio_files):
    from recommender.application.batch_extract import run_batch_extract
    from recommender.application.index.build_index import rebuild_index

    await api.seed(2)

    # Как DAG: batch-процесс импортирует треки и пересобирает индекс на диске
    async with api.sessions() as db:
        await run_batch_extract(audio_files[0].parent, db)
        await rebuild_index(db)

    stale = await _tracks(api)
    assert sum(t["indexed"] for t in stale.values()) == 2  # сервис ещё на старом индексе

    resp = await api.client.post("/index/reload")

    assert resp.status_code == 200
    assert resp.json()["tracks_indexed"] == len(stale) == 2 + len(audio_files)
    assert all(t["indexed"] for t in (await _tracks(api)).values())


async def test_reload_without_saved_index_returns_409(api):
    assert (await api.client.post("/index/reload")).status_code == 409


async def test_features_endpoint(api):
    track = await api.upload(0)

    resp = await api.client.get(f"/tracks/{track['id']}/features")

    assert resp.status_code == 200
    assert resp.json()["dimension"] == settings.feature_dim
    assert (await api.client.get("/tracks/missing/features")).status_code == 404


# ── Рекомендации и лайки ────────────────────────────────────────


async def test_recommendations_exclude_source_and_are_sorted(api):
    ids = await api.seed(5)

    recs = await _recs(api, ids[0], limit=10, use_likes=False)

    rec_ids = [r["track_id"] for r in recs]
    scores = [r["score"] for r in recs]
    assert len(recs) == 4
    assert ids[0] not in rec_ids
    assert len(set(rec_ids)) == len(rec_ids)
    assert scores == sorted(scores, reverse=True)  # cosine: больше = ближе


async def test_recommendations_for_unknown_track_404(api):
    await api.seed(2)
    resp = await api.client.get("/recommendations/missing")
    assert resp.status_code == 404


async def test_like_boost_raises_only_co_liked_track(api):
    a, b, c, d = await api.seed(4)
    await api.like("u1", a)
    await api.like("u1", c)

    plain = {r["track_id"]: r["score"] for r in await _recs(api, a, use_likes=False)}
    boosted = {r["track_id"]: r["score"] for r in await _recs(api, a, use_likes=True)}

    bonus = _expected_bonus(plain.values())
    assert boosted[c] == pytest.approx(plain[c] + bonus, abs=1e-3)
    assert boosted[b] == pytest.approx(plain[b], abs=1e-3)
    assert boosted[d] == pytest.approx(plain[d], abs=1e-3)


async def test_like_unknown_track_404(api):
    resp = await api.client.post("/likes", json={"user_id": "u1", "track_id": "missing"})
    assert resp.status_code == 404


async def test_user_recommendations_exclude_liked(api):
    ids = await api.seed(5)
    await api.like("u1", ids[0])
    await api.like("u1", ids[1])

    resp = await api.client.get("/recommendations/user/u1")

    rec_ids = {r["track_id"] for r in resp.json()["recommendations"]}
    assert rec_ids == set(ids[2:])
    assert resp.json()["strategy"] == "interests"


async def test_user_like_boost_raises_only_neighbour_liked_track(api):
    a, b, c, d, e = await api.seed(5)
    await api.like("u1", a)
    await api.like("u1", b)
    await api.like("u2", a)
    await api.like("u2", c)  # у u2 общий с u1 лайк a → c получает буст для u1

    async def scores(use_likes: bool) -> dict[str, float]:
        resp = await api.client.get("/recommendations/user/u1", params={"use_likes": use_likes})
        return {r["track_id"]: r["score"] for r in resp.json()["recommendations"]}

    plain, boosted = await scores(False), await scores(True)

    bonus = _expected_bonus(plain.values())
    assert boosted[c] == pytest.approx(plain[c] + bonus, abs=1e-3)
    assert boosted[d] == pytest.approx(plain[d], abs=1e-3)
    assert boosted[e] == pytest.approx(plain[e], abs=1e-3)


async def test_batch_recommend_with_and_without_likes(api, tmp_path):
    import pandas as pd

    from recommender.application.batch_recommend import run_batch_recommend

    a, b, c, d = await api.seed(4)
    await api.like("u1", a)
    await api.like("u1", c)

    async def batch(use_likes: bool) -> pd.DataFrame:
        out = tmp_path / f"recs_{use_likes}.parquet"
        async with api.sessions() as db:
            result = await run_batch_recommend(db, out, top_n=3, use_likes=use_likes)
        assert result.tracks_scored == 4
        return pd.read_parquet(out)

    plain, boosted = await batch(False), await batch(True)

    assert len(plain) == len(boosted) == 4 * 3
    assert (plain.source_track_id != plain.target_track_id).all()

    def score(df: pd.DataFrame, src: str, dst: str) -> float:
        row = df[(df.source_track_id == src) & (df.target_track_id == dst)]
        return float(row.score.iloc[0])

    bonus = _expected_bonus(plain[plain.source_track_id == a].score)
    assert score(boosted, a, c) == pytest.approx(score(plain, a, c) + bonus, abs=1e-3)
    assert score(boosted, a, b) == pytest.approx(score(plain, a, b), abs=1e-3)


# ── Редактирование и удаление ───────────────────────────────────


async def test_patch_updates_only_given_fields(api):
    track = await api.upload(0, title="Old", artist="Keep")

    resp = await api.client.patch(f"/tracks/{track['id']}", json={"title": "New"})

    assert resp.status_code == 200
    assert (resp.json()["title"], resp.json()["artist"]) == ("New", "Keep")
    assert (await _tracks(api))[track["id"]]["title"] == "New"


async def test_genre_is_stored_returned_and_editable(api, audio_files):
    with audio_files[0].open("rb") as f:
        resp = await api.client.post(
            "/tracks/upload",
            files={"file": ("a.wav", f, "audio/wav")},
            data={"title": "T", "genre": "Rock"},
        )
    track = resp.json()
    assert track["genre"] == "Rock"
    assert (await api.upload(1))["genre"] is None

    resp = await api.client.patch(f"/tracks/{track['id']}", json={"genre": "Jazz"})

    assert resp.json()["genre"] == "Jazz"
    assert (await _tracks(api))[track["id"]]["genre"] == "Jazz"


async def test_patch_validation(api):
    track = await api.upload(0)

    empty = await api.client.patch(f"/tracks/{track['id']}", json={"title": ""})
    missing = await api.client.patch("/tracks/missing", json={"title": "x"})

    assert empty.status_code == 422
    assert missing.status_code == 404


async def test_delete_removes_track_everywhere(api):
    ids = await api.seed(4)
    victim = ids[1]
    await api.like("u1", victim)
    await api.like("u1", ids[2])

    resp = await api.client.delete(f"/tracks/{victim}")

    assert resp.status_code == 204
    assert victim not in await _tracks(api)
    assert list(settings.audio_dir.glob(f"{victim}_*")) == []
    assert victim not in app.state.engine.track_ids
    assert victim not in load_current().engine.track_ids
    async with api.sessions() as s:
        likes_left = await s.scalar(
            select(func.count()).select_from(LikeORM).where(LikeORM.track_id == victim)
        )
    assert likes_left == 0
    assert victim not in {r["track_id"] for r in await _recs(api, ids[0])}
    assert (await api.client.delete(f"/tracks/{victim}")).status_code == 404


# ── Тюнинг ──────────────────────────────────────────────────────


async def _run_tuning(api) -> dict:
    resp = await api.client.post("/automl/train")
    assert resp.json()["status"] == "started"
    # BackgroundTasks выполняются до завершения запроса в ASGITransport
    status = (await api.client.get("/automl/status")).json()
    return status[0]


async def test_tuning_without_likes_fails_gracefully(api):
    await api.seed(5)

    run = await _run_tuning(api)

    assert run["status"] == "failed"
    assert "liked" in run["best_params"]["error"]


async def test_tuning_params_reach_queries_and_survive_rebuild(api):
    ids = await api.seed(6)
    for tid in ids[0:3]:
        await api.like("u1", tid)
    for tid in ids[1:4]:
        await api.like("u2", tid)

    run = await _run_tuning(api)

    assert run["status"] == "completed"
    # По 3 лайка у пользователя: 1 в тест, 2 на подбор
    assert run["metrics"]["test_likes"] == 2
    assert set(run["metrics"]["holdout"]) == {
        "system",
        "content_only",
        "same_artist",
        "popularity",
        "random",
    }
    assert "track_mrr@10" in run["metrics"]["train"]
    best = run["best_params"]
    engine, normalizer = app.state.engine, app.state.normalizer
    assert engine.metric == best["metric"]
    assert engine.boost_weight == pytest.approx(best["boost_weight"])
    assert normalizer.method == best["norm_method"]
    assert normalizer.weights is not None

    # Запрос из сырых фич через normalizer должен попасть ровно в свой вектор
    # индекса — иначе веса тюнинга не доходят до запросов.
    self_score = 1.0 if engine.metric == "cosine" else 0.0
    for tid in ids:
        raw = np.array(
            (await api.client.get(f"/tracks/{tid}/features")).json()["features"],
            dtype=np.float32,
        )
        top = engine.recommend(normalizer.transform(raw).flatten(), limit=1)[0]
        assert top.track_id == tid
        assert top.score == pytest.approx(self_score, abs=1e-3)

    weights_before = normalizer.weights.copy()
    await api.client.post("/index/rebuild")

    assert app.state.engine.metric == best["metric"]
    assert app.state.engine.boost_weight == pytest.approx(best["boost_weight"])
    assert app.state.normalizer.method == best["norm_method"]
    np.testing.assert_allclose(app.state.normalizer.weights, weights_before)


async def test_service_changes_land_on_top_of_newer_batch_index(api, audio_files, tmp_path):
    import shutil

    from recommender.application.batch_extract import ImportItem, run_batch_import
    from recommender.application.index.build_index import rebuild_index

    ids = await api.seed(2)
    stale_version = app.state.engine.version

    # batch-процесс импортирует 2 трека и публикует новую сборку; сервис о ней не знает
    copies = []
    for i in (5, 6):
        copy = tmp_path / f"batch_{i}.wav"
        shutil.copy(audio_files[i], copy)
        copies.append(ImportItem(copy, copy.name, copy.stem, "Batch"))
    async with api.sessions() as db:
        await run_batch_import(copies, db)
        await rebuild_index(db)
    batch_version = load_current().engine.version
    assert app.state.engine.version == stale_version != batch_version

    uploaded = await api.upload(3)
    assert (await api.client.delete(f"/tracks/{ids[0]}")).status_code == 204

    on_disk = load_current().engine
    # загрузка и удаление применены поверх сборки batch, а не поверх устаревшего индекса
    assert len(on_disk.track_ids) == 2 + 2 + 1 - 1
    assert uploaded["id"] in on_disk.track_ids and ids[0] not in on_disk.track_ids
    assert app.state.engine.version == on_disk.version
    assert set(app.state.engine.track_ids) == set(on_disk.track_ids)


# ── Лайки и персональные рекомендации ────────────────────────────


async def test_repeated_like_is_stored_once(api):
    (track_id,) = await api.seed(1)

    first = await api.client.post("/likes", json={"user_id": "u1", "track_id": track_id})
    again = await api.client.post("/likes", json={"user_id": "u1", "track_id": track_id})

    assert (first.json()["status"], again.json()["status"]) == ("ok", "exists")
    async with api.sessions() as db:
        assert await db.scalar(select(func.count()).select_from(LikeORM)) == 1


async def test_unlike_and_list_likes(api):
    ids = await api.seed(3)
    for track_id in ids:
        await api.like("u1", track_id)

    listed = (await api.client.get("/users/u1/likes")).json()
    assert [t["id"] for t in listed] == ids[::-1]  # сначала последние

    assert (await api.client.delete(f"/likes/u1/{ids[1]}")).status_code == 204
    assert (await api.client.delete(f"/likes/u1/{ids[1]}")).status_code == 404
    assert [t["id"] for t in (await api.client.get("/users/u1/likes")).json()] == [ids[2], ids[0]]
    assert (await api.client.get("/users/nobody/likes")).json() == []


async def test_new_user_gets_popular_tracks(api):
    ids = await api.seed(4)
    for user, liked in {"a": ids[:3], "b": ids[1:3], "c": ids[2:3]}.items():
        for track_id in liked:
            await api.like(user, track_id)

    resp = await api.client.get("/recommendations/user/newcomer", params={"limit": 3})

    body = resp.json()
    assert resp.status_code == 200
    assert body["strategy"] == "popular"
    assert [r["track_id"] for r in body["recommendations"]] == [ids[2], ids[1], ids[0]]
    assert [r["score"] for r in body["recommendations"]] == [3.0, 2.0, 1.0]


async def test_cold_start_can_be_disabled(api, monkeypatch):
    await api.seed(2)
    monkeypatch.setattr(settings, "user_cold_start", False)

    assert (await api.client.get("/recommendations/user/newcomer")).status_code == 404


async def test_user_profile_uses_only_recent_likes(api, monkeypatch):
    ids = await api.seed(6)
    monkeypatch.setattr(settings, "user_max_likes", 1)
    await api.like("u1", ids[0])
    await api.like("u1", ids[5])  # профиль — только последний лайк

    params = {"limit": 10, "use_likes": False}  # больше, чем треков: в выдаче весь каталог
    recs = (await api.client.get("/recommendations/user/u1", params=params)).json()
    by_last = (await api.client.get(f"/recommendations/{ids[5]}", params=params)).json()

    rec_ids = [r["track_id"] for r in recs["recommendations"]]
    assert sorted(rec_ids) == sorted(ids[1:5])  # старый лайк по-прежнему не рекомендуется
    assert rec_ids == [r["track_id"] for r in by_last["recommendations"] if r["track_id"] != ids[0]]
