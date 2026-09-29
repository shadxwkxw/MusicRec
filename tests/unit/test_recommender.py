"""Unit-тесты рекомендера.

Запуск: pytest tests/ -v
"""

import numpy as np
import pytest

from recommender.domain.models import Recommendation
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender


class TestFeatureNormalizer:
    def test_standard_normalization(self):
        data = np.random.randn(20, 58).astype(np.float32)
        norm = FeatureNormalizer("standard")
        result = norm.fit_transform(data)

        assert result.shape == data.shape
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(result.std(axis=0), 1, atol=1e-6)

    def test_minmax_normalization(self):
        data = np.random.randn(20, 58).astype(np.float32)
        norm = FeatureNormalizer("minmax")
        result = norm.fit_transform(data)

        # float32 MinMax может давать 1.0000001 из-за округления — допускаем epsilon
        assert result.min() >= -1e-5
        assert result.max() <= 1.0 + 1e-5

    def test_robust_normalization(self):
        data = np.random.randn(20, 58).astype(np.float32)
        norm = FeatureNormalizer("robust")
        result = norm.fit_transform(data)
        assert result.shape == data.shape

    def test_single_vector_transform(self):
        data = np.random.randn(20, 58).astype(np.float32)
        norm = FeatureNormalizer("standard")
        norm.fit(data)

        single = data[0]
        result = norm.transform(single)
        assert result.shape == (1, 58)

    def test_unfitted_raises(self):
        norm = FeatureNormalizer("standard")
        with pytest.raises(RuntimeError):
            norm.transform(np.zeros(58))

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            FeatureNormalizer("invalid")

    def test_save_load(self, tmp_path):
        data = np.random.randn(20, 58).astype(np.float32)
        norm = FeatureNormalizer("standard")
        norm.fit(data)

        path = tmp_path / "normalizer.joblib"
        norm.save(path)
        loaded = FeatureNormalizer.load(path)

        original = norm.transform(data[0])
        restored = loaded.transform(data[0])
        np.testing.assert_allclose(original, restored)


class TestWeightedNormalizer:
    def test_weights_survive_save_load(self, tmp_path):
        data = np.random.randn(20, 82).astype(np.float32)
        weights = np.linspace(0.5, 3.0, 82, dtype=np.float32)
        norm = FeatureNormalizer("minmax", weights=weights).fit(data)

        path = tmp_path / "normalizer.joblib"
        norm.save(path)
        loaded = FeatureNormalizer.load(path)

        np.testing.assert_allclose(loaded.weights, weights)
        np.testing.assert_allclose(loaded.transform(data[0]), norm.transform(data[0]))

    @pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
    def test_weights_actually_scale_output(self, method):
        data = np.random.default_rng(0).standard_normal((20, 82)).astype(np.float32)
        weights = np.linspace(0.5, 3.0, 82, dtype=np.float32)

        plain = FeatureNormalizer(method).fit_transform(data)
        weighted = FeatureNormalizer(method, weights=weights).fit_transform(data)

        np.testing.assert_allclose(weighted, plain * weights, rtol=1e-5, atol=1e-5)

    def test_raw_query_lands_on_itself(self):
        data = np.random.randn(30, 82).astype(np.float32)
        weights = np.linspace(0.5, 3.0, 82, dtype=np.float32)
        norm = FeatureNormalizer("minmax", weights=weights)
        normalized = norm.fit_transform(data)

        engine = FaissRecommender(dimension=82, metric="euclidean")
        engine.add_tracks([f"t{i}" for i in range(30)], normalized.copy())

        recs = engine.recommend(norm.transform(data[5]).flatten(), limit=1)
        assert recs[0].track_id == "t5"
        assert recs[0].score == pytest.approx(0.0, abs=1e-4)


class TestFaissRecommender:
    def _make_engine(self, n_tracks=50, dim=58):
        engine = FaissRecommender(dimension=dim, metric="cosine")
        ids = [f"track_{i}" for i in range(n_tracks)]
        features = np.random.randn(n_tracks, dim).astype(np.float32)
        engine.add_tracks(ids, features)
        return engine, ids, features

    def test_add_and_search(self):
        engine, _ids, features = self._make_engine()
        assert engine.index.ntotal == 50

        recs = engine.recommend(features[0], limit=5, exclude_ids={"track_0"})
        assert len(recs) == 5
        assert all(isinstance(r, Recommendation) for r in recs)
        assert all(r.track_id != "track_0" for r in recs)

    def test_self_is_most_similar(self):
        engine, _ids, features = self._make_engine()
        recs = engine.recommend(features[0], limit=1)
        assert recs[0].track_id == "track_0"

    def test_exclude_ids(self):
        engine, _ids, features = self._make_engine()
        exclude = {"track_0", "track_1", "track_2"}
        recs = engine.recommend(features[0], limit=5, exclude_ids=exclude)
        rec_ids = {r.track_id for r in recs}
        assert rec_ids.isdisjoint(exclude)

    def test_like_boost(self):
        engine, _ids, features = self._make_engine()
        boost = {"track_49": 100.0}
        recs = engine.recommend(features[0], limit=5, exclude_ids={"track_0"}, like_boost=boost)
        assert recs[0].track_id == "track_49"

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_boost_moves_track_up(self, metric):
        # Вес заведомо больше любого разрыва между скорами (L2 в FAISS —
        # квадрат расстояния, разрывы в единицы): при верном знаке трек
        # уходит на первое место, при неверном — на последнее.
        engine = FaissRecommender(dimension=58, metric=metric, boost_weight=1e6)
        ids = [f"track_{i}" for i in range(30)]
        features = np.random.default_rng(0).standard_normal((30, 58)).astype(np.float32)
        engine.add_tracks(ids, features.copy())

        plain = engine.recommend(features[0], limit=29, exclude_ids={"track_0"})
        target = plain[10].track_id
        boosted = engine.recommend(
            features[0], limit=29, exclude_ids={"track_0"}, like_boost={target: 1.0}
        )

        assert boosted[0].track_id == target

    def test_boost_weight_survives_save_load(self, tmp_path):
        engine = FaissRecommender(dimension=58, metric="euclidean", boost_weight=0.17)
        engine.add_tracks(["a"], np.random.randn(1, 58).astype(np.float32))
        engine.save(tmp_path)
        assert FaissRecommender.load(tmp_path).boost_weight == pytest.approx(0.17)

    def test_remove_tracks_keeps_positions_aligned(self):
        engine, ids, features = self._make_engine(n_tracks=20)
        removed = engine.remove_tracks({"track_3", "track_10", "missing"})

        assert removed == 2
        assert engine.index.ntotal == 18
        assert "track_3" not in engine.track_ids
        for i in (0, 4, 11, 19):
            assert engine.recommend(features[i], limit=1)[0].track_id == ids[i]

    def test_empty_index(self):
        engine = FaissRecommender(dimension=58)
        recs = engine.recommend(np.zeros(58), limit=5)
        assert recs == []

    def test_rebuild(self):
        engine, _ids, _features = self._make_engine(n_tracks=20)
        assert engine.index.ntotal == 20

        new_ids = [f"new_{i}" for i in range(10)]
        new_features = np.random.randn(10, 58).astype(np.float32)
        engine.rebuild(new_ids, new_features)
        assert engine.index.ntotal == 10
        assert len(engine.track_ids) == 10

    def test_save_load(self, tmp_path):
        engine, _ids, features = self._make_engine()
        engine.save(tmp_path)
        loaded = FaissRecommender.load(tmp_path)

        assert loaded.index.ntotal == engine.index.ntotal
        assert loaded.track_ids == engine.track_ids

        recs_original = engine.recommend(features[0], limit=5)
        recs_loaded = loaded.recommend(features[0], limit=5)
        assert recs_original == recs_loaded

    def test_euclidean_metric(self):
        engine = FaissRecommender(dimension=58, metric="euclidean")
        ids = [f"track_{i}" for i in range(20)]
        features = np.random.randn(20, 58).astype(np.float32)
        engine.add_tracks(ids, features)

        recs = engine.recommend(features[0], limit=5)
        assert len(recs) == 5
        # Для L2 меньше дистанция = ближе, сортировка по возрастанию
        scores = [r.score for r in recs]
        assert scores == sorted(scores)


class TestCoLikeStrength:
    def test_counts_shared_likes(self):
        from recommender.application.collaborative import co_like_strength

        likes = {"u1": {"a", "b", "c"}, "u2": {"a", "b"}, "u3": {"c", "d"}}
        assert co_like_strength("a", likes) == {"b": 1.0, "c": 0.5}

    def test_no_fans_gives_empty(self):
        from recommender.application.collaborative import co_like_strength

        assert co_like_strength("x", {"u1": {"a", "b"}}) == {}

    def test_user_strength_sums_over_likes_and_skips_own(self):
        from recommender.application.collaborative import user_co_like_strength

        likes = {"me": {"a", "b"}, "u1": {"a", "c"}, "u2": {"a", "b", "d"}, "u3": {"e"}}

        # от a: b=1, c=0.5, d=0.5; от b: a=1, d=0.5 → без своих: c=0.5, d=1.0
        assert user_co_like_strength({"a", "b"}, likes) == {"c": 0.5, "d": 1.0}


class TestEvaluate:
    @staticmethod
    def _setup(points: dict[str, list[float]], boost_weight: float = 0.0):
        ids = list(points)
        vectors = np.array([points[t] for t in ids], dtype=np.float32)
        engine = FaissRecommender(dimension=3, metric="euclidean", boost_weight=boost_weight)
        engine.add_tracks(ids, vectors.copy())
        return engine, vectors, {t: i for i, t in enumerate(ids)}

    def test_ranks_for_both_paths(self):
        from recommender.application.training.tune_recommender import evaluate

        engine, vectors, id_to_idx = self._setup(
            {
                "a1": [0, 0, 0],
                "a2": [0.1, 0, 0],
                "a3": [0.2, 0, 0],
                "b1": [5, 5, 5],
                "b2": [6, 6, 6],
            }
        )
        likes = {
            "u": {"a1", "a3"},
            "single": {"b1"},  # один лайк — оценить нельзя
            "ghost": {"a1", "missing"},  # трека нет в индексе
        }

        metrics = evaluate(engine, vectors, id_to_idx, likes, k=3)

        # По треку: из a1 первым идёт a2, a3 — вторым (и наоборот)
        assert metrics["track_hit@3"] == 1.0
        assert metrics["track_mrr@3"] == pytest.approx(0.5)
        # По пользователю: запрос = один оставшийся лайк, он исключён, так же a2 первым
        assert metrics["user_mrr@3"] == pytest.approx(0.5)

    def test_hidden_like_does_not_leak_into_boost(self):
        from recommender.application.training.tune_recommender import evaluate

        engine, vectors, id_to_idx = self._setup(
            {"a": [1, 0, 0], "b": [0.9, 0.1, 0], "x": [0, 0, 1], "c": [0, 0.1, 1]},
            boost_weight=1e6,
        )

        metrics = evaluate(engine, vectors, id_to_idx, {"u": {"a", "x"}}, k=1)

        # Без утечки ближайшими остаются b и c; с утечкой буст поднял бы спрятанный лайк
        assert metrics["track_hit@1"] == 0.0


class TestFeatureGroups:
    def test_apply_weights(self):
        from recommender.application.training.tune_recommender import (
            FEATURE_GROUPS,
            apply_feature_weights,
        )

        features = np.ones((5, 82), dtype=np.float32)
        weights = {name: 2.0 for name in FEATURE_GROUPS}
        weighted = apply_feature_weights(features, weights)
        np.testing.assert_allclose(weighted, 2.0)

    def test_zero_weight_kills_group(self):
        from recommender.application.training.tune_recommender import (
            FEATURE_GROUPS,
            apply_feature_weights,
        )

        features = np.ones((5, 82), dtype=np.float32)
        weights = {name: 1.0 for name in FEATURE_GROUPS}
        weights["mfcc"] = 0.0
        weighted = apply_feature_weights(features, weights)

        np.testing.assert_allclose(weighted[:, 0:26], 0.0)
        np.testing.assert_allclose(weighted[:, 26:], 1.0)
