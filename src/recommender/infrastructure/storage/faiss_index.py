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

from recommender.config import settings
from recommender.domain.models import Recommendation
from recommender.domain.recommender import Recommender


class FaissRecommender(Recommender):
    """FAISS-индекс ближайших соседей."""

    def __init__(
        self,
        dimension: int | None = None,
        metric: str = "cosine",
        boost_weight: float = 0.3,
    ):
        self.dimension = dimension or settings.feature_dim
        self.metric = metric
        self.boost_weight = boost_weight
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
        limit: int = 10,
        exclude_ids: set[str] | None = None,
        like_boost: dict[str, float] | None = None,
    ) -> list[Recommendation]:
        if self.index.ntotal == 0:
            return []

        query = query_features.reshape(1, -1).astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(query)

        # Берём с запасом, чтобы хватило после фильтрации.
        # При наличии like_boost сканируем весь индекс — иначе сильный буст
        # не сможет поднять трек, который не попал в топ-K поиска.
        search_k = self.index.ntotal if like_boost else min(limit * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        exclude_ids = exclude_ids or set()
        results: list[Recommendation] = []

        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self.track_ids):
                continue
            track_id = self.track_ids[idx]
            if track_id in exclude_ids:
                continue

            score = float(dist)
            if like_boost and track_id in like_boost:
                bonus = like_boost[track_id] * self.boost_weight
                # cosine: больше = ближе, L2: меньше = ближе
                score = score + bonus if self.metric == "cosine" else score - bonus

            results.append(Recommendation(track_id=track_id, score=score))

        # Для cosine больше = лучше, для L2 меньше = лучше
        results.sort(key=lambda r: r.score, reverse=(self.metric == "cosine"))
        return results[:limit]

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
            boost_weight=meta.get("boost_weight", 0.3),
        )
        engine.index = faiss.read_index(str(path / "faiss.index"))
        engine.track_ids = meta["track_ids"]
        return engine
