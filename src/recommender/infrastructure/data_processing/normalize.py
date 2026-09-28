"""Нормализация признаков — обёртка над sklearn-скейлерами с save/load.

Опционально хранит per-dimension веса признаков (подобранные тюнингом) и
сохраняет их вместе со скейлером, поэтому любой transform() даёт вектор в
том же пространстве, что и индекс. Веса применяются ПОСЛЕ скейлера: все
скейлеры работают по каждому измерению отдельно, и вес, применённый до
них, сокращается ((w·x − w·μ) / (w·σ) = (x − μ) / σ).
"""

from pathlib import Path

import numpy as np
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from recommender.config import settings


SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


class FeatureNormalizer:
    """Обёртка над sklearn-скейлерами с сохранением/загрузкой."""

    def __init__(self, method: str = "standard", weights: np.ndarray | None = None):
        if method not in SCALER_CLASSES:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(SCALER_CLASSES)}"
            )
        self.method = method
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        self.scaler = SCALER_CLASSES[method]()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Обучен ли нормализатор."""
        return self._fitted

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """Обучить на матрице сырых признаков (n_tracks, n_features)."""
        self.scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Нормализовать сырые признаки. Принимает 1D (один трек) или 2D."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted yet. Call fit() first.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        scaled = self.scaler.transform(features)
        return scaled if self.weights is None else scaled * self.weights

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def save(self, path: Path | None = None) -> None:
        path = path or (settings.models_dir / "normalizer.joblib")
        dump({"method": self.method, "scaler": self.scaler, "weights": self.weights}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "FeatureNormalizer":
        path = path or (settings.models_dir / "normalizer.joblib")
        data = load(path)
        normalizer = cls(method=data["method"], weights=data.get("weights"))
        normalizer.scaler = data["scaler"]
        normalizer._fitted = True
        return normalizer
