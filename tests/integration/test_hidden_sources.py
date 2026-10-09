"""Скрытые источники (recommendation.hidden_sources): не в выдаче, но в индексе."""

import pytest
from sqlalchemy import select, update

from recommender.application.batch_recommend import run_batch_recommend
from recommender.config import settings
from recommender.infrastructure.storage.postgres import TrackORM
from recommender.interfaces.online.main import app
from tests.integration.test_embeddings import FakeEmbedder, _embed
from tests.integration.test_search import FakeTextEncoder


async def _mark_fma(api, ids):
    async with api.sessions() as db:
        await db.execute(update(TrackORM).where(TrackORM.id.in_(ids)).values(source="fma"))
        await db.commit()


def _ids(resp) -> set[str]:
    assert resp.status_code == 200, resp.text
    return {r["track_id"] for r in resp.json()["recommendations"]}


async def test_hidden_tracks_are_not_recommended_but_stay_queryable(api, monkeypatch):
    ids = await api.seed(6)
    await _mark_fma(api, ids[3:])

    shown = _ids(await api.client.get(f"/recommendations/{ids[0]}", params={"limit": 10}))
    from_hidden = _ids(await api.client.get(f"/recommendations/{ids[4]}", params={"limit": 10}))

    assert shown == {ids[1], ids[2]}
    assert from_hidden == set(ids[:3])  # похожие на скрытый трек — только видимые

    monkeypatch.setattr(settings, "hidden_sources", [])
    everything = _ids(await api.client.get(f"/recommendations/{ids[0]}", params={"limit": 10}))
    assert everything == set(ids[1:])


async def test_user_and_cold_start_skip_hidden(api):
    ids = await api.seed(6)
    await _mark_fma(api, ids[3:])
    for track_id in ids[3:]:  # скрытые треки самые популярные
        for user in ("a", "b"):
            await api.like(user, track_id)
    await api.like("u1", ids[0])

    user = await api.client.get("/recommendations/user/u1", params={"limit": 10})
    newcomer = await api.client.get("/recommendations/user/newcomer", params={"limit": 10})

    assert _ids(user) == {ids[1], ids[2]}
    assert _ids(newcomer) == set(ids[:3])
    assert newcomer.json()["strategy"] == "popular"


async def test_search_skips_hidden(api, monkeypatch):
    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)
    ids = [(await api.upload(i))["id"] for i in range(4)]
    await _mark_fma(api, ids[:2])
    await _embed(api)
    assert (await api.client.post("/index/rebuild")).status_code == 200
    from recommender.application.features import load_vectors

    async with api.sessions() as db:
        target = (await load_vectors(db, [ids[0]]))[ids[0]]
    monkeypatch.setattr(app.state, "text_encoder", FakeTextEncoder(target), raising=False)

    resp = await api.client.get("/search", params={"q": "rock", "limit": 10})

    assert {r["track_id"] for r in resp.json()["results"]} == set(ids[2:])


async def test_batch_recommend_skips_hidden_rows_and_targets(api, tmp_path):
    ids = await api.seed(5)
    await _mark_fma(api, ids[3:])

    async with api.sessions() as db:
        result = await run_batch_recommend(db, tmp_path / "recs.csv", top_n=10)

    lines = (tmp_path / "recs.csv").read_text().splitlines()[1:]
    sources = {line.split(",")[0] for line in lines}
    targets = {line.split(",")[2] for line in lines}
    assert sources == targets == set(ids[:3])
    assert result.tracks_scored == 3


async def test_track_source_is_recorded(api, audio_files, tmp_path):
    from recommender.application.batch_extract import run_batch_extract

    uploaded = await api.upload(0)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "A - B.wav").write_bytes(audio_files[1].read_bytes())
    async with api.sessions() as db:
        await run_batch_extract(inbox, db)
        sources = dict((await db.execute(select(TrackORM.title, TrackORM.source))).all())

    assert uploaded["source"] == "upload"
    assert sources == {"Track 0": "upload", "B": "import"}
    listed = (await api.client.get("/tracks")).json()
    assert {t["source"] for t in listed} == {"upload", "import"}


@pytest.mark.parametrize("hidden", ["fma", "fma, upload"])
def test_hidden_sources_from_config(hidden):
    from recommender.config import _build_settings, _load_config_dict

    raw = _load_config_dict()  # с подстановкой ${...} из окружения
    raw["recommendation"]["hidden_sources"] = hidden

    assert _build_settings(raw).hidden_sources == [s.strip() for s in hidden.split(",")]


async def test_hidden_list_follows_index_version(api):
    from types import SimpleNamespace

    from recommender.application.visibility import hidden_track_ids

    ids = await api.seed(4)
    assert _ids(await api.client.get(f"/recommendations/{ids[0]}")) == set(ids[1:])  # в кэше пусто

    await _mark_fma(api, [ids[3]])
    assert (await api.client.post("/index/rebuild")).status_code == 200  # новая версия индекса
    assert _ids(await api.client.get(f"/recommendations/{ids[0]}")) == {ids[1], ids[2]}

    unversioned = SimpleNamespace(version=None)  # индекс в памяти: без кэша
    async with api.sessions() as db:
        assert await hidden_track_ids(db, unversioned) == {ids[3]}
        await _mark_fma(api, [ids[2]])
        assert await hidden_track_ids(db, unversioned) == {ids[2], ids[3]}
