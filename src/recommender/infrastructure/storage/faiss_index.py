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

    def __init__(self, dimension: int | None = None, metric: str = "cosine"):
        self.dimension = dimension or settings.feature_dim
        self.metric = metric
        self.index: faiss.IndexFlat | None = None
        self.track_ids: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        if self.metric == "cosine":
            # IndexFlatIP на L2-нормированных векторах == cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

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
        if like_boost:
            search_k = self.index.ntotal
        else:
            search_k = min(limit * 3, self.index.ntotal)
        distances, indices = self.index.search(query, search_k)

        exclude_ids = exclude_ids or set()
        results: list[Recommendation] = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.track_ids):
                continue
            track_id = self.track_ids[idx]
            if track_id in exclude_ids:
                continue

            score = float(dist)
            if like_boost and track_id in like_boost:
                score += like_boost[track_id]

            results.append(Recommendation(track_id=track_id, score=score))

        # Для cosine больше = лучше, для L2 меньше = лучше
        results.sort(key=lambda r: r.score, reverse=(self.metric == "cosine"))
        return results[:limit]

    def rebuild(self, track_ids: list[str], features: np.ndarray) -> None:
        self._build_index()
        self.track_ids = []
        self.add_tracks(track_ids, features)

    def save(self, path: Path | None = None) -> None:
        path = path or settings.index_dir
        faiss.write_index(self.index, str(path / "faiss.index"))
        dump(
            {"track_ids": self.track_ids, "metric": self.metric, "dim": self.dimension},
            path / "meta.joblib",
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "FaissRecommender":
        path = path or settings.index_dir
        meta = load(path / "meta.joblib")
        engine = cls(dimension=meta["dim"], metric=meta["metric"])
        engine.index = faiss.read_index(str(path / "faiss.index"))
        engine.track_ids = meta["track_ids"]
        return engine
