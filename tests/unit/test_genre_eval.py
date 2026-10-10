"""Жанровая оценка: каталог (видимый или весь), macro против micro, кандидаты."""

from types import SimpleNamespace

import numpy as np
import pytest

from recommender.application.training.evaluation import genre_catalog, genre_report, genre_score
from recommender.config import settings
from recommender.infrastructure.storage.faiss_index import FaissRecommender


def _track(track_id, genre, source="import"):
    return SimpleNamespace(id=track_id, genre=genre, source=source)


@pytest.fixture
def small_minimum(monkeypatch):
    monkeypatch.setattr(settings, "tuning_min_genre_tracks", 2)
    monkeypatch.setattr(settings, "hidden_sources", ["fma"])


def test_visible_catalog_skips_hidden_sources(small_minimum):
    tracks = [
        _track("a", "Rock"),
        _track("b", "Pop"),
        _track("c", None),
        _track("f", "Rock", "fma"),
    ]

    genres, candidates, catalog = genre_catalog(tracks)

    assert (genres, candidates, catalog) == ({"a": "Rock", "b": "Pop"}, {"a", "b", "c"}, "visible")


def test_falls_back_to_all_when_visible_catalog_is_too_small(small_minimum, monkeypatch):
    tracks = [_track("a", "Rock"), _track("f1", "Rock", "fma"), _track("f2", "Pop", "fma")]

    assert genre_catalog(tracks)[2] == "all"
    monkeypatch.setattr(settings, "tuning_genre_catalog", "all")
    two_visible = [_track("a", "Rock"), _track("b", "Pop"), _track("f", "Rock", "fma")]
    assert genre_catalog(two_visible) == (
        {"a": "Rock", "b": "Pop", "f": "Rock"},
        {"a", "b", "f"},
        "all",
    )


def test_macro_and_micro_scores(monkeypatch):
    report = {"all": {"filtered@10": 0.7}, "macro": {"filtered@10": 0.3}}

    assert genre_score(report, 10) == 0.3
    monkeypatch.setattr(settings, "tuning_genre_average", "micro")
    assert genre_score(report, 10) == 0.7


def _engine(vectors: dict[str, list[float]]):
    ids = list(vectors)
    X = np.array([vectors[t] for t in ids], dtype=np.float32)
    engine = FaissRecommender(dimension=2, metric="cosine")
    engine.add_tracks(ids, X.copy())
    return engine, X, {t: i for i, t in enumerate(ids)}


def test_neighbours_and_random_level_come_from_candidates_only():
    # q и r1 — рок рядом друг с другом; h — тоже рок рядом, но скрыт; p — поп далеко
    engine, X, idx = _engine({"q": [1, 0], "r1": [0.95, 0.3], "h": [0.99, 0.1], "p": [0, 1]})
    genres = {"q": "Rock", "r1": "Rock", "h": "Rock", "p": "Pop"}
    artists = {t: t for t in idx}

    report = genre_report(
        engine, X, idx, genres, artists, k=1, query_ids=["q"], candidates={"q", "r1", "p"}
    )

    assert report["Rock"]["system@1"] == 1.0  # нашёл r1, а не скрытый h
    assert report["Rock"]["random@1"] == pytest.approx(1 / 2)  # 1 рок из 2 других кандидатов


def test_macro_weighs_genres_equally():
    engine, X, idx = _engine(
        {
            "r1": [1, 0],
            "r2": [0.99, 0.1],
            "r3": [0.98, 0.2],
            "p1": [0.97, 0.25],
            "p2": [0, 1],
            "p3": [0.1, 0.99],
        }
    )
    genres = {"r1": "Rock", "r2": "Rock", "r3": "Rock", "p1": "Pop", "p2": "Pop", "p3": "Pop"}

    report = genre_report(engine, X, idx, genres, {t: t for t in idx}, k=1)

    rock, pop = report["Rock"]["system@1"], report["Pop"]["system@1"]
    assert report["macro"]["system@1"] == pytest.approx((rock + pop) / 2)
    assert report["macro"]["tracks"] == 6


def test_artist_split_keeps_every_genre_in_test():
    from recommender.application.training.evaluation import split_by_artist

    plan = {"Hip-Hop": 8, "Rock": 2, "Pop": 3, "Jazz": 1}
    artists, genres = {}, {}
    for genre, n_artists in plan.items():
        for a in range(n_artists):
            for i in range(2):
                track = f"{genre}-{a}-{i}"
                artists[track], genres[track] = f"{genre} artist {a}", genre

    tune, test = split_by_artist(list(artists), artists, 0.2, seed=1, genres=genres)

    assert {genres[t] for t in test} == {"Hip-Hop", "Rock", "Pop"}  # у Jazz один артист — в подборе
    assert {genres[t] for t in tune} == set(plan)
    assert not {artists[t] for t in tune} & {artists[t] for t in test}
    assert split_by_artist(list(artists), artists, 0.2, seed=1, genres=genres) == (tune, test)
