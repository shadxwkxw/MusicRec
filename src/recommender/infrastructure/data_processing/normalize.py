"""Нормализация признаков — обёртка над sklearn-скейлерами с save/load."""

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

    def __init__(self, method: str = "standard"):
        if method not in SCALER_CLASSES:
            raise ValueError(
                f"Unknown method: {method}. Choose from {list(SCALER_CLASSES)}"
            )
        self.method = method
        self.scaler = SCALER_CLASSES[method]()
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Обучен ли нормализатор."""
        return self._fitted

    def fit(self, features: np.ndarray) -> "FeatureNormalizer":
        """Обучить на матрице (n_tracks, n_features)."""
        self.scaler.fit(features)
        self._fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Нормализовать признаки. Принимает 1D (один трек) или 2D."""
        if not self._fitted:
            raise RuntimeError("Normalizer not fitted yet. Call fit() first.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.scaler.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def save(self, path: Path | None = None) -> None:
        path = path or (settings.models_dir / "normalizer.joblib")
        dump({"method": self.method, "scaler": self.scaler}, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "FeatureNormalizer":
        path = path or (settings.models_dir / "normalizer.joblib")
        data = load(path)
        normalizer = cls(method=data["method"])
        normalizer.scaler = data["scaler"]
        normalizer._fitted = True
        return normalizer
