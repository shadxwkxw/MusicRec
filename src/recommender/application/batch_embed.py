"""Use case: досчитать эмбеддинги треков, у которых их ещё нет для данной модели.

Модель передаётся снаружи (Embedder), поэтому use case не зависит от torch:
в проде это ClapEmbedder, в тестах — поддельная модель.
"""

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.batch_extract import BatchExtractResult
from recommender.application.features import audio_path_of
from recommender.infrastructure.storage.audio_store import local_copies
from recommender.infrastructure.storage.postgres import TrackEmbeddingORM, TrackORM, utcnow

DOWNLOAD_CHUNK = 64


class Embedder(Protocol):
    model_name: str

    def embed_files(
        self, paths: Sequence[str | Path]
    ) -> Iterator[tuple[int, np.ndarray | str]]: ...


async def count_missing_embeddings(db: AsyncSession, model_name: str) -> int:
    """Сколько треков ещё без эмбеддинга модели — чтобы не грузить модель зря."""
    done = select(TrackEmbeddingORM.track_id).where(TrackEmbeddingORM.model == model_name)
    query = select(func.count()).select_from(TrackORM).where(TrackORM.id.not_in(done))
    return int(await db.scalar(query) or 0)


async def run_batch_embed(
    db: AsyncSession,
    embedder: Embedder,
    commit_every: int = 100,
    progress: Callable[[int, int], None] | None = None,
) -> BatchExtractResult:
    """Посчитать и сохранить эмбеддинги для треков без них. skipped — уже были."""
    tracks: Sequence[TrackORM] = (await db.execute(select(TrackORM))).scalars().all()
    done_ids: set[str] = set(
        (
            await db.execute(
                select(TrackEmbeddingORM.track_id).where(
                    TrackEmbeddingORM.model == embedder.model_name
                )
            )
        )
        .scalars()
        .all()
    )
    stats = BatchExtractResult(skipped=len(done_ids & {t.id for t in tracks}))

    todo = [(t.id, audio_path_of(t)) for t in tracks if t.id not in done_ids]

    # Пачками: файлы из S3 скачиваются во временную папку на время пачки
    pending = done = 0
    for start in range(0, len(todo), DOWNLOAD_CHUNK):
        chunk = todo[start : start + DOWNLOAD_CHUNK]
        with local_copies([location for _, location in chunk]) as copies:
            ready = []
            for (track_id, location), copy in zip(chunk, copies, strict=True):
                if isinstance(copy, Exception):
                    stats.failed.append((track_id, f"audio not found: {location}"))
                    done += 1
                else:
                    ready.append((track_id, copy))
            for i, result in embedder.embed_files([path for _, path in ready]):
                track_id = ready[i][0]
                done += 1
                if isinstance(result, str):
                    stats.failed.append((track_id, result))
                else:
                    db.add(
                        TrackEmbeddingORM(
                            track_id=track_id,
                            model=embedder.model_name,
                            vector=np.asarray(result, dtype=np.float32).tobytes(),
                            created_at=utcnow(),
                        )
                    )
                    stats.processed += 1
                    pending += 1
                if pending >= commit_every:
                    await db.commit()
                    pending = 0
                if progress:
                    progress(done, len(todo))
    await db.commit()
    return stats
