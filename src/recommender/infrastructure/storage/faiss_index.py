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
    ):
        self.dimension = dimension or settings.feature_dim
        # Из чего собраны векторы индекса (settings.feature_source_id): индекс
        # из эмбеддингов нельзя использовать с librosa-запросами и наоборот
        self.source = source or settings.feature_source_id
        self.version: str | None = None  # id версии на диске (storage/artifacts.py)
        self.metric = metric or settings.default_metric
        self.boost_weight = settings.default_boost_weight if boost_weight is None else boost_weight
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
    ) -> list[Recommendation]:
        if self.index.ntotal == 0:
            return []

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

        found = [
            (self.track_ids[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0], strict=True)
            if 0 <= idx < len(self.track_ids) and self.track_ids[idx] not in exclude_ids
        ]
        candidates = [(tid, score) for tid, score in found if tid not in hidden_ids]
        scale = (
            self._boost_scale([s for _, s in candidates], [s for _, s in found])
            if like_boost
            else 0.0
        )

        results: list[Recommendation] = []
        for track_id, score in candidates:
            if like_boost and track_id in like_boost:
                bonus = like_boost[track_id] * self.boost_weight * scale
                # cosine: больше = ближе, L2: меньше = ближе
                score = score + bonus if self.metric == "cosine" else score - bonus
            results.append(Recommendation(track_id=track_id, score=score))

        # Для cosine больше = лучше, для L2 меньше = лучше
        results.sort(key=lambda r: r.score, reverse=(self.metric == "cosine"))
        return results[:limit]

    def _boost_scale(self, candidates: list[float], everything: list[float]) -> float:
        """Разрыв между ближайшим кандидатом и типичным треком: единица измерения буста.

        Масштаб скоров зависит от метрики, нормализации и весов признаков
        (для L2 это квадраты расстояний, десятки и сотни), поэтому вес буста
        задаётся в долях этого разрыва, а не в абсолютных единицах.

        Ближайший — среди кандидатов, типичный — медиана с учётом скрытых треков
        (hidden_ids): иначе скрытые источники (тысячи далёких треков FMA)
        поднимали медиану и ослабляли буст в разы.
        """
        if not candidates:
            return 0.0
        best = max(candidates) if self.metric == "cosine" else min(candidates)
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
            source=meta.get("source", "librosa"),
        )
        engine.index = faiss.read_index(str(path / "faiss.index"))
        engine.track_ids = meta["track_ids"]
        return engine
