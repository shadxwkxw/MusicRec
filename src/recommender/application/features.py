"""Векторы треков для поиска похожих — из источника, выбранного в конфиге.

features.source:
- librosa   — 82 признака из tracks.feature_vector (считаются при загрузке);
- embedding — эмбеддинги модели features.embedding_model из track_embeddings
  (считает batch embed).

Все use cases (rebuild, рекомендации, тюнинг, batch, оценка) берут векторы
отсюда, поэтому смена источника — это смена конфига и rebuild.
"""

from collections.abc import Collection

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.config import settings
from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import TrackEmbeddingORM, TrackORM


def uses_embeddings() -> bool:
    return settings.feature_source == "embedding"


async def load_vectors(
    db: AsyncSession, track_ids: Collection[str] | None = None
) -> dict[str, np.ndarray]:
    """{track_id: вектор} для треков, у которых он есть в текущем источнике."""
    if uses_embeddings():
        query = select(TrackEmbeddingORM.track_id, TrackEmbeddingORM.vector).where(
            TrackEmbeddingORM.model == settings.embedding_model
        )
        id_column = TrackEmbeddingORM.track_id
    else:
        query = select(TrackORM.id, TrackORM.feature_vector).where(
            TrackORM.feature_vector.isnot(None)
        )
        id_column = TrackORM.id
    if track_ids is not None:
        query = query.where(id_column.in_(list(track_ids)))
    rows = (await db.execute(query)).all()
    return {track_id: bytes_to_features(blob) for track_id, blob in rows}


class IndexSourceMismatchError(RuntimeError):
    """Сохранённый индекс собран из другого источника признаков, чем в конфиге."""


def check_index_source(engine: FaissRecommender) -> None:
    if engine.source != settings.feature_source_id:
        raise IndexSourceMismatchError(
            f"Index was built from {engine.source} features, but config uses "
            f"{settings.feature_source_id}: run rebuild"
        )


def audio_path_of(track: TrackORM) -> str:
    """Где лежит аудио трека: сохранённый путь или ссылка s3://, иначе папка загрузок."""
    return track.audio_path or str(settings.audio_dir / track.filename)
