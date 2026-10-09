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

    def test_many_excluded_neighbours_still_fill_the_limit(self):
        engine, _ids, features = self._make_engine(n_tracks=200)
        nearest = {r.track_id for r in engine.recommend(features[0], limit=60)}

        recs = engine.recommend(features[0], limit=10, exclude_ids=nearest)

        assert len(recs) == 10
        assert {r.track_id for r in recs}.isdisjoint(nearest)

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

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_boost_effect_does_not_depend_on_feature_scale(self, metric):
        # Растяжение пространства (другая нормализация, веса признаков) меняет
        # масштаб скоров, но не должно менять то, насколько буст двигает треки
        features = np.random.default_rng(1).standard_normal((60, 16)).astype(np.float32)
        boost = {f"t{i}": 1.0 for i in range(30, 40)}

        def ranking(scale: float) -> list[str]:
            engine = FaissRecommender(dimension=16, metric=metric, boost_weight=0.5)
            engine.add_tracks([f"t{i}" for i in range(60)], features * scale)
            recs = engine.recommend(features[0] * scale, limit=20, like_boost=boost)
            return [r.track_id for r in recs]

        assert ranking(1.0) == ranking(10.0)
        # и буст при этом действительно что-то меняет
        engine = FaissRecommender(dimension=16, metric=metric, boost_weight=0.5)
        engine.add_tracks([f"t{i}" for i in range(60)], features.copy())
        plain = [r.track_id for r in engine.recommend(features[0], limit=20)]
        assert ranking(1.0) != plain

    def test_full_boost_lifts_typical_track_to_nearest_level(self):
        # Вес 1.0 при силе 1.0 сдвигает трек на разрыв между ближайшим и медианным
        features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        engine = FaissRecommender(dimension=1, metric="euclidean", boost_weight=1.0)
        engine.add_tracks(["q", "a", "b", "c", "d"], features)

        recs = engine.recommend(np.array([0.0]), limit=4, exclude_ids={"q"}, like_boost={"c": 1.0})

        # Кандидаты a..d: L2² = 1, 4, 9, 16; ближайший 1, медиана 6.5 → c: 9 − 5.5 = 3.5
        scores = {r.track_id: r.score for r in recs}
        assert scores["c"] == pytest.approx(9 - (6.5 - 1))
        assert [r.track_id for r in recs][:2] == ["a", "c"]

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
        from recommender.application.training.evaluation import evaluate

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
        from recommender.application.training.evaluation import evaluate

        engine, vectors, id_to_idx = self._setup(
            {"a": [1, 0, 0], "b": [0.9, 0.1, 0], "x": [0, 0, 1], "c": [0, 0.1, 1]},
            boost_weight=1e6,
        )

        metrics = evaluate(engine, vectors, id_to_idx, {"u": {"a", "x"}}, k=1)

        # Без утечки ближайшими остаются b и c; с утечкой буст поднял бы спрятанный лайк
        assert metrics["track_hit@1"] == 0.0


class TestHoldout:
    def test_split_by_artist_never_shares_an_artist(self):
        from recommender.application.training.evaluation import split_by_artist

        artists = {f"t{i}": f"artist{i % 10}" for i in range(50)}

        tune, test = split_by_artist(list(artists), artists, test_fraction=0.2, seed=3)

        assert len({artists[t] for t in test}) == 2
        assert {artists[t] for t in tune}.isdisjoint({artists[t] for t in test})
        assert sorted(tune + test) == sorted(artists)
        assert split_by_artist(list(artists), artists, 0.2, seed=3) == (tune, test)

    def test_split_keeps_train_for_everyone_and_is_deterministic(self):
        from recommender.application.training.evaluation import split_likes

        likes = {
            "big": {f"t{i}" for i in range(10)},
            "pair": {"a", "b"},
            "single": {"s"},
        }

        train, test = split_likes(likes, test_fraction=0.2, seed=1)

        assert len(test["big"]) == 2 and len(train["big"]) == 8
        assert len(test["pair"]) == 1 and len(train["pair"]) == 1
        assert "single" not in test and train["single"] == {"s"}
        for user, liked in likes.items():
            assert train[user] | test.get(user, set()) == liked
            assert not train[user] & test.get(user, set())
        assert split_likes(likes, 0.2, seed=1) == (train, test)

    def test_expected_random_matches_formula(self):
        from recommender.application.training.evaluation import Query, expected_random

        query = Query("user", ("q",), frozenset({"q"}), "x", {})

        metrics = expected_random([query], n_tracks=11, k=10)

        # 10 кандидатов, k=10: цель точно в выдаче, MRR = H_10 / 10
        assert metrics["user_hit@10"] == pytest.approx(1.0)
        assert metrics["user_mrr@10"] == pytest.approx(sum(1 / r for r in range(1, 11)) / 10)

    def test_baselines_rank_as_expected(self):
        from recommender.application.training.evaluation import (
            Query,
            popularity_ranker,
            same_artist_ranker,
        )

        ids = ["a1", "a2", "b1", "b2"]
        artists = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
        train = {"u1": {"b2"}, "u2": {"b2", "a2"}}
        query = Query("track", ("a1",), frozenset({"a1"}), "a2", train)

        assert popularity_ranker(ids, train)(query, 3) == ["b2", "a2", "b1"]
        assert same_artist_ranker(ids, artists, train)(query, 3) == ["a2", "b2", "b1"]

    def test_holdout_report_hides_test_likes_from_boost(self):
        from recommender.application.training.evaluation import holdout_report

        ids = ["a", "b", "x", "c"]
        vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1], [0, 0.1, 1]], dtype=np.float32)
        engine = FaissRecommender(dimension=3, metric="euclidean", boost_weight=1e6)
        engine.add_tracks(ids, vectors.copy())

        report = holdout_report(
            engine,
            vectors,
            {t: i for i, t in enumerate(ids)},
            {t: "same" for t in ids},
            train={"u": {"a"}},
            test={"u": {"x"}},
            k=1,
        )

        assert set(report) == {"system", "content_only", "same_artist", "popularity", "random"}
        # x далеко от a, и буст не знает о спрятанном лайке → в топ-1 не попадает
        assert report["system"]["track_hit@1"] == 0.0
        assert report["system"] == report["content_only"]

    def test_holdout_report_is_empty_without_test_likes(self):
        from recommender.application.training.evaluation import holdout_report

        engine = FaissRecommender(dimension=3)
        vectors = np.eye(3, dtype=np.float32)
        engine.add_tracks(["a", "b", "c"], vectors.copy())

        report = holdout_report(
            engine, vectors, {"a": 0, "b": 1, "c": 2}, {}, train={"u": {"a"}}, test={}
        )

        assert report == {}


class TestGenreReport:
    def _setup(self):
        # Два жанра в разных углах пространства; у rock два артиста
        points = {
            "r1": [1, 0],
            "r2": [1, 0.1],
            "r3": [1, 0.2],
            "j1": [0, 1],
            "j2": [0.1, 1],
            "x": [0.5, 0.5],
        }
        ids = list(points)
        vectors = np.array([points[t] for t in ids], dtype=np.float32)
        engine = FaissRecommender(dimension=2, metric="euclidean", boost_weight=0.0)
        engine.add_tracks(ids, vectors.copy())
        genres = {"r1": "rock", "r2": "rock", "r3": "rock", "j1": "jazz", "j2": "jazz", "x": None}
        artists = {"r1": "A", "r2": "A", "r3": "B", "j1": "C", "j2": "D", "x": "E"}
        return engine, vectors, {t: i for i, t in enumerate(ids)}, genres, artists

    def test_same_genre_share_and_random_level(self):
        from recommender.application.training.evaluation import genre_report

        report = genre_report(*self._setup(), k=1)

        assert report["all"]["tracks"] == 5  # трек без жанра не оценивается
        assert report["all"]["system@1"] == 1.0
        # rock: 2 других rock из 5 других треков; jazz: 1 из 5
        assert report["rock"]["random@1"] == pytest.approx(2 / 5)
        assert report["jazz"]["random@1"] == pytest.approx(1 / 5)

    def test_artist_filter_excludes_same_artist_neighbours(self):
        from recommender.application.training.evaluation import genre_report

        report = genre_report(*self._setup(), k=1)

        # r1 и r2 одного артиста: без фильтра находят друг друга, с фильтром —
        # ближайший трек другого артиста, r3 (тот же жанр)
        assert report["rock"]["filtered@1"] == 1.0
        # r1: из 4 треков не артиста A rock только r3
        engine, vectors, id_to_idx, genres, artists = self._setup()
        artists["r3"] = "A"  # теперь у rock все треки одного артиста
        assert (
            genre_report(engine, vectors, id_to_idx, genres, artists, k=1)["rock"]["filtered@1"]
            == 0.0
        )


class TestConfig:
    def test_packaged_default_config_matches_repo_config(self):
        from pathlib import Path

        import recommender

        root = Path(__file__).resolve().parents[2]
        packaged = Path(recommender.__file__).parent / "default_config.yaml"
        # Встроенная копия используется вне репозитория (wheel, Docker) и не должна отставать
        assert packaged.read_text() == (root / "configs" / "config.yaml").read_text()

    def test_invalid_values_are_rejected(self):
        from pathlib import Path

        import yaml
        from pydantic import ValidationError

        from recommender.config import _build_settings

        root = Path(__file__).resolve().parents[2]
        raw = yaml.safe_load((root / "configs" / "config.yaml").read_text())
        raw["database"]["url"] = "sqlite+aiosqlite://"
        raw["recommendation"]["default_metric"] = "manhattan"

        with pytest.raises(ValidationError, match="default_metric"):
            _build_settings(raw)

    def test_defaults_come_from_config(self, monkeypatch):
        from recommender.config import settings

        monkeypatch.setattr(settings, "default_metric", "euclidean")
        monkeypatch.setattr(settings, "default_norm_method", "robust")
        monkeypatch.setattr(settings, "default_boost_weight", 0.7)

        engine = FaissRecommender(dimension=4)
        assert (engine.metric, engine.boost_weight) == ("euclidean", 0.7)
        assert FeatureNormalizer().method == "robust"
        # Явно переданные значения важнее конфига, включая нулевой вес буста
        assert FaissRecommender(dimension=4, boost_weight=0.0).boost_weight == 0.0

    def test_eval_k_comes_from_config(self, monkeypatch):
        from recommender.application.training.evaluation import evaluate
        from recommender.config import settings

        monkeypatch.setattr(settings, "tuning_eval_k", 3)
        engine = FaissRecommender(dimension=3, metric="euclidean", boost_weight=0.0)
        vectors = np.eye(3, dtype=np.float32)
        engine.add_tracks(["a", "b", "c"], vectors.copy())

        metrics = evaluate(engine, vectors, {"a": 0, "b": 1, "c": 2}, {"u": {"a", "b"}})

        assert set(metrics) == {"track_hit@3", "track_mrr@3", "user_hit@3", "user_mrr@3"}


class TestFeatureGroups:
    def test_groups_cover_the_whole_feature_vector(self):
        from recommender.application.training.tune_recommender import FEATURE_GROUPS
        from recommender.config import settings

        bounds = sorted(FEATURE_GROUPS.values())
        assert bounds[0][0] == 0 and bounds[-1][1] == settings.feature_dim
        assert all(prev[1] == nxt[0] for prev, nxt in zip(bounds, bounds[1:], strict=False))

    def test_weight_vector_expands_group_weights(self):
        from recommender.application.training.tune_recommender import (
            FEATURE_GROUPS,
            feature_weight_vector,
        )

        weights = {name: 1.0 for name in FEATURE_GROUPS}
        weights["mfcc"] = 0.0
        weights["tempo"] = 2.5

        vector = feature_weight_vector(weights)

        np.testing.assert_allclose(vector[0:26], 0.0)
        np.testing.assert_allclose(vector[76:77], 2.5)
        np.testing.assert_allclose(np.delete(vector, [*range(26), 76]), 1.0)


class TestEmbeddingWindows:
    def test_short_track_is_one_window(self):
        from recommender.infrastructure.data_processing.embeddings import split_windows

        windows = split_windows(np.arange(5), window=10, max_windows=6)

        assert [len(w) for w in windows] == [5]

    def test_long_track_is_cut_into_full_windows_with_tail(self):
        from recommender.infrastructure.data_processing.embeddings import (
            SAMPLE_RATE,
            split_windows,
        )

        window = 10 * SAMPLE_RATE
        y = np.arange(window * 3 + 5 * SAMPLE_RATE)  # 35 с: хвост 5 с длиннее порога 3 с

        windows = split_windows(y, window, max_windows=6)

        assert len(windows) == 4 and all(len(w) == window for w in windows)
        assert windows[-1][-1] == y[-1]  # последнее окно прижато к концу трека

    def test_short_tail_is_dropped_and_windows_are_spread_evenly(self):
        from recommender.infrastructure.data_processing.embeddings import (
            SAMPLE_RATE,
            split_windows,
        )

        window = 10 * SAMPLE_RATE
        y = np.arange(window * 12 + SAMPLE_RATE)  # 121 с: хвост 1 с отбрасывается

        windows = split_windows(y, window, max_windows=3)

        assert [int(w[0]) for w in windows] == [0, window * 6, window * 11]


class TestEnvFile:
    @pytest.fixture
    def env_file(self, tmp_path, monkeypatch):
        path = tmp_path / "custom.env"
        path.write_text("RECOMMENDER_TEST_A=from-file\nRECOMMENDER_TEST_B=from-file\n")
        # регистрируем ключи, чтобы monkeypatch вернул окружение после теста
        monkeypatch.delenv("RECOMMENDER_TEST_A", raising=False)
        monkeypatch.delenv("RECOMMENDER_TEST_B", raising=False)
        return path

    def test_file_values_load_but_environment_wins(self, env_file, monkeypatch):
        import os

        from recommender.config import load_env_file

        monkeypatch.setenv("ENV_FILE", str(env_file))
        monkeypatch.setenv("RECOMMENDER_TEST_B", "from-env")

        assert load_env_file() == env_file
        assert os.environ["RECOMMENDER_TEST_A"] == "from-file"
        assert os.environ["RECOMMENDER_TEST_B"] == "from-env"

    def test_empty_env_file_disables_loading(self, env_file, monkeypatch):
        import os

        from recommender.config import load_env_file

        monkeypatch.setenv("ENV_FILE", "")

        assert load_env_file() is None
        assert "RECOMMENDER_TEST_A" not in os.environ

    def test_missing_file_is_ignored(self, tmp_path, monkeypatch):
        from recommender.config import load_env_file

        monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))

        assert load_env_file() is None


def test_boost_strength_does_not_depend_on_excluded_tracks():
    """Скрытые далёкие треки (FMA) не должны ослаблять буст лайков у оставшихся."""
    import numpy as np

    from recommender.infrastructure.storage.faiss_index import FaissRecommender

    rng = np.random.default_rng(0)
    near = np.array([[1.0, 0.05], [0.9, 0.4]], dtype=np.float32)  # a ближе, b чуть дальше
    far = np.column_stack([-np.ones(200), rng.normal(0, 0.3, 200)]).astype(np.float32)
    engine = FaissRecommender(dimension=2, metric="cosine", boost_weight=0.5)
    engine.add_tracks(["a", "b"] + [f"far{i}" for i in range(200)], np.vstack([near, far]))
    query = np.array([1.0, 0.0], dtype=np.float32)
    boost = {"b": 1.0}

    shown = [r.track_id for r in engine.recommend(query, limit=2, like_boost=boost)]
    hidden = {f"far{i}" for i in range(200)}
    without_far = [
        r.track_id for r in engine.recommend(query, limit=2, like_boost=boost, hidden_ids=hidden)
    ]

    assert shown == ["b", "a"]  # буст лайков поднял b
    assert without_far == shown
