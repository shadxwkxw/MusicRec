"""Use case: precompute top-N рекомендаций для всех треков.

Проходит по всем трекам в БД, для каждого вычисляет top-N похожих через
загруженный FAISS-индекс, выгружает результат в CSV/parquet — в файл или в
S3 (output_path вида s3://bucket/key). Коллаборативный бустинг опционален
(use_likes): лайки загружаются один раз на весь прогон.
"""

import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import co_like_strength, load_user_likes
from recommender.application.features import check_index_source, load_vectors
from recommender.application.visibility import hidden_track_ids
from recommender.config import settings
from recommender.infrastructure.storage.artifacts import load_current
from recommender.infrastructure.storage.audio_store import is_s3, upload_file


@dataclass
class BatchRecommendResult:
    tracks_scored: int
    output_path: Path | str


async def run_batch_recommend(
    db: AsyncSession,
    output_path: Path | str,
    top_n: int = settings.default_rec_limit,
    use_likes: bool = False,
) -> BatchRecommendResult:
    """Посчитать top-N похожих для каждого трека и выгрузить в файл.

    Формат выхода: CSV или Parquet (по расширению output_path). Колонки:
    source_track_id, rank, target_track_id, score.

    Args:
        db: сессия БД
        output_path: куда писать результат (.csv или .parquet), путь или s3://bucket/key
        top_n: сколько рекомендаций на трек
        use_likes: применять co-like бустинг, как онлайн-рекомендации по треку

    Returns:
        Статистика и путь до результата.
    """
    artifacts = load_current()
    engine, normalizer = artifacts.engine, artifacts.normalizer
    check_index_source(engine)

    vectors = await load_vectors(db)
    if not vectors:
        raise RuntimeError(f"No tracks with {settings.feature_source} features in database")

    user_likes = await load_user_likes(db) if use_likes else {}
    hidden = await hidden_track_ids(db, engine)  # скрытые — ни в строках, ни в рекомендациях

    rows: list[tuple[str, int, str, float]] = []
    for track_id, features in vectors.items():
        if track_id in hidden:
            continue
        if normalizer.is_fitted:
            features = normalizer.transform(features).flatten()

        boost = co_like_strength(track_id, user_likes) if use_likes else None
        recs = engine.recommend(
            features,
            limit=top_n,
            exclude_ids={track_id},
            like_boost=boost or None,
            hidden_ids=set(hidden),
        )
        for rank, rec in enumerate(recs, start=1):
            rows.append((track_id, rank, rec.track_id, round(rec.score, 6)))

    if isinstance(output_path, str) and is_s3(output_path):
        with tempfile.TemporaryDirectory(prefix="recommender-recs-") as folder:
            local = Path(folder) / Path(output_path).name
            _write_output(rows, local)
            upload_file(local, output_path)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_output(rows, output_path)

    return BatchRecommendResult(
        tracks_scored=len(vectors) - len(hidden & vectors.keys()),
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
