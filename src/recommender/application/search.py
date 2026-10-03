"""Use case: поиск треков по текстовому описанию.

Текст переводится текстовой частью той же модели, что посчитала эмбеддинги
аудио, проходит нормализацию индекса и ищется в нём, как обычный запрос.
Отдельный индекс не нужен: на FMA поиск по нормализованному индексу и по
исходным эмбеддингам аудио дал одинаковую точность.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from recommender.config import settings
from recommender.domain.models import Recommendation
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender


class TextEncoder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class TextSearchNotSupportedError(RuntimeError):
    """Индекс собран не из эмбеддингов модели, текстовая часть которой нужна поиску."""


def query_vector(encoder: TextEncoder, query: str) -> np.ndarray:
    """Средний эмбеддинг запроса по шаблонам формулировок (CLAP чувствительна к ним)."""
    texts = [template.format(query) for template in settings.search_prompt_templates]
    mean = encoder.encode(texts).mean(axis=0)
    return (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float32)


def search_tracks(
    query: str,
    engine: FaissRecommender,
    normalizer: FeatureNormalizer,
    encoder: TextEncoder,
    limit: int = settings.default_rec_limit,
) -> list[Recommendation]:
    expected = f"embedding:{encoder.model_name}"
    if engine.source != expected:
        raise TextSearchNotSupportedError(
            f"Text search needs an index built from {expected}, current index: {engine.source}"
        )
    vector = query_vector(encoder, query)
    if normalizer.is_fitted:
        vector = normalizer.transform(vector).flatten()
    return engine.recommend(vector, limit=limit)
