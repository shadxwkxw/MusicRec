"""GET /search с поддельными моделями: без torch и весов CLAP."""

import numpy as np
import pytest

from recommender.application.features import load_vectors
from recommender.config import settings
from recommender.interfaces.online import routes
from recommender.interfaces.online.main import app
from tests.integration.test_embeddings import FakeEmbedder, _embed


class FakeTextEncoder:
    """Любой текст → один и тот же вектор (эмбеддинг выбранного трека)."""

    model_name = FakeEmbedder.model_name

    def __init__(self, vector: np.ndarray):
        self.vector = vector
        self.seen: list[str] = []

    def encode(self, texts):
        self.seen += list(texts)
        return np.tile(self.vector, (len(texts), 1))


@pytest.fixture
def embedding_mode(monkeypatch):
    monkeypatch.setattr(settings, "feature_source", "embedding")
    monkeypatch.setattr(settings, "embedding_model", FakeEmbedder.model_name)
    monkeypatch.setattr(app.state, "text_encoder", None, raising=False)


async def _indexed_tracks(api, n: int) -> list[str]:
    ids = [(await api.upload(i, artist=f"Artist {i}", title=f"Song {i}"))["id"] for i in range(n)]
    await _embed(api)
    assert (await api.client.post("/index/rebuild")).status_code == 200
    return ids


async def test_search_finds_track_matching_the_text_embedding(api, embedding_mode):
    ids = await _indexed_tracks(api, 5)
    async with api.sessions() as db:
        target = (await load_vectors(db, [ids[3]]))[ids[3]]
    encoder = FakeTextEncoder(target)
    app.state.text_encoder = encoder

    resp = await api.client.get("/search", params={"q": "calm acoustic folk", "limit": 3})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "calm acoustic folk"
    assert len(body["results"]) == 3
    top = body["results"][0]
    assert (top["track_id"], top["title"], top["artist"]) == (ids[3], "Song 3", "Artist 3")
    assert top["score"] == pytest.approx(1.0, abs=1e-4)
    assert encoder.seen == [
        t.format("calm acoustic folk") for t in settings.search_prompt_templates
    ]


async def test_text_model_is_loaded_once_on_first_search(api, embedding_mode, monkeypatch):
    ids = await _indexed_tracks(api, 3)
    async with api.sessions() as db:
        target = (await load_vectors(db, [ids[0]]))[ids[0]]
    loads = []
    monkeypatch.setattr(
        routes, "_load_text_encoder", lambda: loads.append(1) or FakeTextEncoder(target)
    )

    for _ in range(2):
        assert (await api.client.get("/search", params={"q": "rock"})).status_code == 200

    assert len(loads) == 1


async def test_search_without_torch_returns_501(api, embedding_mode, monkeypatch):
    await _indexed_tracks(api, 2)

    def missing():
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(routes, "_load_text_encoder", missing)

    resp = await api.client.get("/search", params={"q": "rock"})

    assert resp.status_code == 501
    assert "install-embeddings" in resp.json()["detail"]


async def test_search_on_librosa_index_returns_409(api, monkeypatch):
    await api.seed(3)
    monkeypatch.setattr(app.state, "text_encoder", FakeTextEncoder(np.ones(4)), raising=False)

    resp = await api.client.get("/search", params={"q": "rock"})

    assert resp.status_code == 409
    assert "FEATURE_SOURCE=embedding" in resp.json()["detail"]


async def test_search_validates_query(api, embedding_mode):
    assert (await api.client.get("/search", params={"q": ""})).status_code == 422
    assert (await api.client.get("/search")).status_code == 422
    assert (await api.client.get("/search", params={"q": "x", "limit": 0})).status_code == 422


def test_query_vector_averages_prompt_templates(monkeypatch):
    from recommender.application.search import query_vector

    monkeypatch.setattr(settings, "search_prompt_templates", ["{} music", "a {} song"])

    class PerText:
        model_name = "m"

        def encode(self, texts):
            table = {"rock music": [1.0, 0.0], "a rock song": [0.0, 1.0]}
            return np.array([table[t] for t in texts], dtype=np.float32)

    np.testing.assert_allclose(query_vector(PerText(), "rock"), [2**-0.5, 2**-0.5], atol=1e-6)


def test_prompt_templates_must_have_one_placeholder():
    from pathlib import Path

    import yaml
    from pydantic import ValidationError

    from recommender.config import _build_settings

    raw = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/config.yaml").read_text())
    raw["database"]["url"] = "sqlite+aiosqlite://"
    raw["features"]["source"] = "librosa"
    raw["search"]["prompt_templates"] = ["music without placeholder"]

    with pytest.raises(ValidationError, match="prompt template"):
        _build_settings(raw)
