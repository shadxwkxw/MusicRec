"""FAISS-реализация порта Recommender.

Поддерживает:
- cosine-сходство (IndexFlatIP на L2-нормированных векторах)
- евклидову дистанцию (IndexFlatL2)
- коллаборативный бустинг через внешние like-скоры
"""

from pathlib import Path

import faiss
import numpy as np
from joblib import dump, load

from recommender.config import Metric, settings
from recommender.domain.models import Recommendation
from recommender.domain.recommender import Recommender


class FaissRecommender(Recommender):
    """FAISS-индекс ближайших соседей."""

    def __init__(
        self,
        dimension: int | None = None,
        metric: Metric | None = None,
        boost_weight: float | None = None,
        source: str | None = None,
        user_boost_weight: float | None = None,
    ):
        self.dimension = dimension or settings.feature_dim
        # Из чего собраны векторы индекса (settings.feature_source_id): индекс
        # из эмбеддингов нельзя использовать с librosa-запросами и наоборот
        self.source = source or settings.feature_source_id
        self.version: str | None = None  # id версии на диске (storage/artifacts.py)
        self.metric = metric or settings.default_metric
        self.boost_weight = settings.default_boost_weight if boost_weight is None else boost_weight
        # Персональным рекомендациям лайки помогают, а похожим на трек — нет (на плейлистах
        # буст там вредил при любой нормировке), поэтому вес у путей свой. Вес из
        # конфига (recommendation.user_boost_weight) главнее сохранённого и подобранного
        if settings.fixed_user_boost_weight is not None:
            self.user_boost_weight = settings.fixed_user_boost_weight
        else:
            self.user_boost_weight = (
                self.boost_weight if user_boost_weight is None else user_boost_weight
            )
        self.index = self._new_index()
        self.track_ids: list[str] = []

    def _new_index(self) -> faiss.Index:
        if self.metric == "cosine":
            # IndexFlatIP на L2-нормированных векторах == cosine similarity
            return faiss.IndexFlatIP(self.dimension)
        return faiss.IndexFlatL2(self.dimension)

    def add_tracks(self, track_ids: list[str], features: np.ndarray) -> None:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.metric == "cosine":
            faiss.normalize_L2(features)

        self.index.add(features.astype(np.float32))
        self.track_ids.extend(track_ids)

    def recommend(
        self,
        query_features: np.ndarray,
        limit: int = settings.default_rec_limit,
        exclude_ids: set[str] | None = None,
        like_boost: dict[str, float] | None = None,
        hidden_ids: set[str] | None = None,
        boost_weight: float | None = None,
    ) -> list[Recommendation]:
        if self.index.ntotal == 0:
            return []
        weight = self.boost_weight if boost_weight is None else boost_weight
        if not weight:
            like_boost = None  # без веса буст не нужен — и полный проход индекса тоже

        query = query_features.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(query)

        # Берём с запасом: исключённые треки (сам запрос, лайки) тоже могут
        # оказаться ближайшими. При like_boost сканируем весь индекс — иначе
        # сильный буст не поднимет трек, который не попал в топ-K поиска.
        exclude_ids = exclude_ids or set()
        hidden_ids = hidden_ids or set()
        skipped = len(exclude_ids) + len(hidden_ids)
        search_k = (
            self.index.ntotal
            if like_boost
            else min(limit * settings.candidate_multiplier + skipped, self.index.ntotal)
        )
        distances, indices = self.index.search(query, search_k)

        # Дальше — массивами, без цикла по трекам: при like_boost это весь индекс
        # на каждый запрос, и поштучный перебор на Python занимал почти всё время
        # тюнинга. Позиции в индексе = позиции в track_ids.
        positions = indices[0]
        scores = distances[0].astype(np.float64)
        keep = (positions >= 0) & (positions < len(self.track_ids))
        if exclude_ids:
            keep &= ~np.isin(positions, self._positions_of(exclude_ids))
        positions, scores = positions[keep], scores[keep]  # found: всё, кроме исключённых

        candidate = (
            ~np.isin(positions, self._positions_of(hidden_ids))
            if hidden_ids
            else np.ones(len(positions), dtype=bool)
        )
        found_scores = scores
        positions, scores = positions[candidate], scores[candidate]

        if like_boost and len(positions):
            scale = self._boost_scale(scores, found_scores)
            boost = np.zeros(len(self.track_ids))
            boosted = [
                (i, v) for t, v in like_boost.items() if (i := self._position(t)) is not None
            ]
            if boosted:
                at, values = zip(*boosted, strict=True)
                boost[list(at)] = values
            bonus = boost[positions] * weight * scale
            # cosine: больше = ближе, L2: меньше = ближе
            scores = scores + bonus if self.metric == "cosine" else scores - bonus

        # Для cosine больше = лучше, для L2 меньше = лучше; stable — равные скоры
        # остаются в порядке поиска, как при сортировке списка
        order = np.argsort(-scores if self.metric == "cosine" else scores, kind="stable")[:limit]
        return [
            Recommendation(track_id=self.track_ids[positions[i]], score=float(scores[i]))
            for i in order
        ]

    def _position_map(self) -> dict[str, int]:
        """{track_id: позиция}; пересчитывается, когда track_ids заменили или дополнили."""
        key = (id(self.track_ids), len(self.track_ids))
        if getattr(self, "_position_key", None) != key:
            self._positions = {tid: i for i, tid in enumerate(self.track_ids)}
            self._position_key = key
        return self._positions

    def _position(self, track_id: str) -> int | None:
        return self._position_map().get(track_id)

    def _positions_of(self, track_ids: set[str]) -> np.ndarray:
        known = self._position_map()
        return np.fromiter((known[t] for t in track_ids if t in known), dtype=np.int64)

    def _boost_scale(self, candidates: np.ndarray, everything: np.ndarray) -> float:
        """Разрыв между ближайшим кандидатом и типичным треком: единица измерения буста.

        Масштаб скоров зависит от метрики, нормализации и весов признаков
        (для L2 это квадраты расстояний, десятки и сотни), поэтому вес буста
        задаётся в долях этого разрыва, а не в абсолютных единицах.

        Ближайший — среди кандидатов, типичный — медиана с учётом скрытых треков
        (hidden_ids): иначе скрытые источники (тысячи далёких треков FMA)
        поднимали медиану и ослабляли буст в разы.
        """
        if not len(candidates):
            return 0.0
        best = float(np.max(candidates)) if self.metric == "cosine" else float(np.min(candidates))
        gap = abs(float(np.median(everything)) - best)
        return gap if gap > 0 else 1.0

    def remove_tracks(self, track_ids: set[str]) -> int:
        positions = [i for i, tid in enumerate(self.track_ids) if tid in track_ids]
        if not positions:
            return 0
        # IndexFlat.remove_ids сдвигает оставшиеся векторы с сохранением порядка,
        # поэтому track_ids фильтруется тем же способом, чтобы позиции совпадали.
        # Python-обёртка faiss принимает массив id, стабы описывают только IDSelector
        self.index.remove_ids(np.array(positions, dtype=np.int64))  # type: ignore[arg-type, unused-ignore]
        removed = set(positions)
        self.track_ids = [tid for i, tid in enumerate(self.track_ids) if i not in removed]
        return len(positions)

    def rebuild(self, track_ids: list[str], features: np.ndarray) -> None:
        self.index = self._new_index()
        self.track_ids = []
        self.add_tracks(track_ids, features)

    def save(self, path: Path | None = None) -> None:
        path = path or settings.index_dir
        faiss.write_index(self.index, str(path / "faiss.index"))
        dump(
            {
                "track_ids": self.track_ids,
                "metric": self.metric,
                "dim": self.dimension,
                "boost_weight": self.boost_weight,
                "user_boost_weight": self.user_boost_weight,
                "source": self.source,
            },
            path / "meta.joblib",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "FaissRecommender":
        path = path or settings.index_dir
        meta = load(path / "meta.joblib")
        engine = cls(
            dimension=meta["dim"],
            metric=meta["metric"],
            boost_weight=meta.get("boost_weight", settings.default_boost_weight),
            user_boost_weight=meta.get("user_boost_weight"),
            source=meta.get("source", "librosa"),
        )
        engine.index = faiss.read_index(str(path / "faiss.index"))
        engine.track_ids = meta["track_ids"]
        return engine
