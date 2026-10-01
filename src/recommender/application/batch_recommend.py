"""Use case: precompute top-N рекомендаций для всех треков.

Проходит по всем трекам в БД, для каждого вычисляет top-N похожих через
загруженный FAISS-индекс, выгружает результат в CSV/parquet. Коллаборативный
бустинг опционален (use_likes): лайки загружаются один раз на весь прогон.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import co_like_strength, load_user_likes
from recommender.application.features import check_index_source, load_vectors
from recommender.config import settings
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender


@dataclass
class BatchRecommendResult:
    tracks_scored: int
    output_path: Path


async def run_batch_recommend(
    db: AsyncSession,
    output_path: Path,
    top_n: int = settings.default_rec_limit,
    use_likes: bool = False,
) -> BatchRecommendResult:
    """Посчитать top-N похожих для каждого трека и выгрузить в файл.

    Формат выхода: CSV или Parquet (по расширению output_path). Колонки:
    source_track_id, rank, target_track_id, score.

    Args:
        db: сессия БД
        output_path: куда писать результат (.csv или .parquet)
        top_n: сколько рекомендаций на трек
        use_likes: применять co-like бустинг, как онлайн-рекомендации по треку

    Returns:
        Статистика и путь до результата.
    """
    engine = FaissRecommender.load()
    check_index_source(engine)
    try:
        normalizer = FeatureNormalizer.load()
    except FileNotFoundError:
        normalizer = None

    vectors = await load_vectors(db)
    if not vectors:
        raise RuntimeError(f"No tracks with {settings.feature_source} features in database")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    user_likes = await load_user_likes(db) if use_likes else {}

    rows: list[tuple[str, int, str, float]] = []
    for track_id, features in vectors.items():
        if normalizer is not None and normalizer.is_fitted:
            features = normalizer.transform(features).flatten()

        boost = co_like_strength(track_id, user_likes) if use_likes else None
        recs = engine.recommend(
            features, limit=top_n, exclude_ids={track_id}, like_boost=boost or None
        )
        for rank, rec in enumerate(recs, start=1):
            rows.append((track_id, rank, rec.track_id, round(rec.score, 6)))

    _write_output(rows, output_path)

    return BatchRecommendResult(
        tracks_scored=len(vectors),
        output_path=output_path,
    )


def _write_output(rows: list[tuple[str, int, str, float]], output_path: Path) -> None:
    if output_path.suffix.lower() == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "pandas required for parquet output; install pandas + pyarrow"
            ) from e

        pd.DataFrame(
            rows, columns=["source_track_id", "rank", "target_track_id", "score"]
        ).to_parquet(output_path, index=False)
    else:
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_track_id", "rank", "target_track_id", "score"])
            writer.writerows(rows)
