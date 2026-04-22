"""Доменный порт рекомендера.

Абстрактный интерфейс, который реализуют адаптеры в infrastructure/storage
(например, FAISS-бэкенд).
"""

from abc import ABC, abstractmethod

import numpy as np

from recommender.domain.models import Recommendation


class Recommender(ABC):
    """Поиск похожих треков по вектору признаков."""

    @abstractmethod
    def add_tracks(self, track_ids: list[str], features: np.ndarray) -> None:
        """Добавить треки в индекс.

        Args:
            track_ids: идентификаторы треков (порядок совпадает со строками матрицы)
            features: матрица (n_tracks, dimension) нормализованных векторов
        """

    @abstractmethod
    def recommend(
        self,
        query_features: np.ndarray,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
        like_boost: dict[str, float] | None = None,
    ) -> list[Recommendation]:
        """Вернуть топ-N похожих треков.

        Args:
            query_features: вектор признаков запроса (dimension,)
            limit: сколько рекомендаций вернуть
            exclude_ids: треки, которые нужно исключить (обычно сам query)
            like_boost: {track_id: boost_score} — коллаборативная надбавка
        """

    @abstractmethod
    def rebuild(self, track_ids: list[str], features: np.ndarray) -> None:
        """Полностью пересобрать индекс из новых треков."""
