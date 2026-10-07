"""Профиль пользователя: интересы из лайков и сборка выдачи."""

import numpy as np

from recommender.application.user_profile import (
    InterestCandidates,
    blend,
    interest_candidates,
    interest_groups,
)
from recommender.config import settings
from recommender.domain.models import Recommendation
from recommender.infrastructure.storage.faiss_index import FaissRecommender

DIM = 8


def _around(axis: int, n: int, seed: int, noise: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.zeros(DIM)
    base[axis] = 1.0
    return base + noise * rng.standard_normal((n, DIM))


def _recs(*ids: str) -> list[Recommendation]:
    return [Recommendation(track_id=t, score=1.0 - i / 100) for i, t in enumerate(ids)]


def test_two_tastes_become_two_interests():
    vectors = np.vstack([_around(0, 4, 1), _around(1, 4, 2)])

    groups = interest_groups(vectors, max_interests=3, min_likes=3, merge_similarity=0.3)

    assert sorted(map(sorted, groups)) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_one_taste_stays_one_interest():
    groups = interest_groups(_around(0, 10, 3), max_interests=3, min_likes=3, merge_similarity=0.0)

    assert len(groups) == 1


def test_single_stray_like_is_not_an_interest():
    vectors = np.vstack([_around(0, 5, 4), _around(5, 1, 5)])

    groups = interest_groups(vectors, max_interests=3, min_likes=3, merge_similarity=0.0)

    assert [len(g) for g in groups] == [6]


def test_number_of_interests_is_capped():
    vectors = np.vstack([_around(axis, 3, axis) for axis in range(5)])

    groups = interest_groups(vectors, max_interests=3, min_likes=1, merge_similarity=0.0)

    assert len(groups) == 3
    assert sorted(i for g in groups for i in g) == list(range(15))


def test_blend_gives_mean_share_then_splits_by_interest_size():
    candidates = InterestCandidates(
        overall=_recs("m1", "m2", "m3", "m4"),
        by_interest=[_recs("a1", "a2", "a3", "a4", "a5"), _recs("b1", "b2", "b3")],
        interest_sizes=[6, 3],
    )

    ids = [r.track_id for r in blend(candidates, 8, max_per_artist=0, mean_share=0.25)]

    assert ids[:2] == ["m1", "m2"]  # 25% мест — общему среднему
    rest = ids[2:]
    assert sum(t.startswith("a") for t in rest) == 4  # интерес из 6 лайков — 2/3 остатка
    assert sum(t.startswith("b") for t in rest) == 2


def test_blend_caps_unknown_artist_but_not_liked_one():
    candidates = InterestCandidates(_recs("x1", "x2", "x3", "y1", "y2", "y3", "z1"), [], [5])
    artists = {"x1": "X", "x2": "X", "x3": "X", "y1": "Y", "y2": "Y", "y3": "Y", "z1": "Z"}

    ids = [r.track_id for r in blend(candidates, 6, artists, max_per_artist=2, liked_artists={"y"})]

    assert ids == ["x1", "x2", "y1", "y2", "y3", "z1"]


def test_blend_fills_without_cap_when_catalog_is_one_artist():
    candidates = InterestCandidates(_recs("x1", "x2", "x3", "x4"), [], [3])

    ids = [
        r.track_id for r in blend(candidates, 4, dict.fromkeys(["x1", "x2", "x3", "x4"], "X"), 2)
    ]

    assert ids == ["x1", "x2", "x3", "x4"]


def test_interest_candidates_search_each_interest_and_exclude_likes(monkeypatch):
    monkeypatch.setattr(settings, "user_merge_similarity", 0.3)
    likes = np.vstack([_around(0, 3, 6, 0.05), _around(1, 3, 7, 0.05)]).astype(np.float32)
    catalog = np.vstack([_around(0, 5, 8, 0.05), _around(1, 5, 9, 0.05)]).astype(np.float32)
    engine = FaissRecommender(dimension=DIM, metric="cosine")
    engine.add_tracks([f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)], catalog.copy())

    found = interest_candidates(engine, likes, limit=4, exclude_ids={"a0"})

    assert found.interest_sizes == [3, 3]
    tops = {lst[0].track_id[0] for lst in found.by_interest}
    assert tops == {"a", "b"}  # у каждого интереса свои ближайшие
    assert "a0" not in found.track_ids
