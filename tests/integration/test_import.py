"""Импорт аудио с метаданными: загрузчик FMA, параллельное извлечение, жанровая оценка."""

import shutil

import pandas as pd
import pytest
from sqlalchemy import select

from recommender.application.batch_extract import ImportItem, run_batch_import
from recommender.infrastructure.datasets.fma import load_fma_items
from recommender.infrastructure.storage.postgres import TrackORM


def _write_fma(root, rows: dict[int, tuple], with_audio: set[int]) -> None:
    """Мини-FMA в формате настоящего tracks.csv (двухуровневые заголовки)."""
    columns = pd.MultiIndex.from_tuples(
        [("track", "title"), ("artist", "name"), ("track", "genre_top"), ("set", "subset")]
    )
    df = pd.DataFrame(list(rows.values()), index=list(rows), columns=columns)
    df.index.name = "track_id"
    (root / "fma_metadata").mkdir(parents=True)
    df.to_csv(root / "fma_metadata" / "tracks.csv")
    for track_id in with_audio:
        name = f"{track_id:06d}"
        path = root / "fma_small" / name[:3] / f"{name}.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_fma_loader_reads_metadata_and_skips_missing_audio(tmp_path):
    _write_fma(
        tmp_path,
        {
            2: ("Food", "AWOL", "Hip-Hop", "small"),
            5: (None, None, "Pop", "small"),  # пустые название и артист
            10: ("Missing", "X", "Rock", "small"),  # нет mp3
            140: ("Medium only", "Y", "Folk", "medium"),  # не входит в small
        },
        with_audio={2, 5, 140},
    )

    items = {item.filename: item for item in load_fma_items(tmp_path, "small")}

    assert set(items) == {"fma_000002.mp3", "fma_000005.mp3"}
    food = items["fma_000002.mp3"]
    assert (food.title, food.artist, food.genre) == ("Food", "AWOL", "Hip-Hop")
    assert food.path == tmp_path / "fma_small" / "000" / "000002.mp3"
    assert (items["fma_000005.mp3"].title, items["fma_000005.mp3"].artist) == (
        "FMA 000005",
        "Unknown",
    )


def test_fma_loader_explains_missing_archives(tmp_path):
    with pytest.raises(FileNotFoundError, match="fma_metadata.zip"):
        load_fma_items(tmp_path)


async def test_parallel_import_saves_metadata_reports_failures_and_skips_repeats(
    api, audio_files, tmp_path
):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")
    items = [
        ImportItem(path, f"imp_{i}.wav", f"Song {i}", f"Artist {i % 2}", ["Rock", "Jazz"][i % 2])
        for i, path in enumerate(audio_files[:4])
    ]
    items.append(ImportItem(broken, "broken.mp3", "Broken", "Nobody", "Rock"))

    async with api.sessions() as db:
        stats = await run_batch_import(items, db, workers=2, commit_every=2)
        tracks = (await db.execute(select(TrackORM))).scalars().all()

    assert (stats.processed, stats.skipped) == (4, 0)
    assert [name for name, _ in stats.failed] == ["broken.mp3"]
    assert {(t.title, t.artist, t.genre) for t in tracks} == {
        ("Song 0", "Artist 0", "Rock"),
        ("Song 1", "Artist 1", "Jazz"),
        ("Song 2", "Artist 0", "Rock"),
        ("Song 3", "Artist 1", "Jazz"),
    }

    async with api.sessions() as db:
        again = await run_batch_import(items[:4], db)
    assert (again.processed, again.skipped) == (0, 4)


async def test_evaluate_saved_index_reports_genres(api, audio_files, tmp_path):
    from recommender.application.index.build_index import rebuild_index
    from recommender.application.training.evaluation import evaluate_saved_index

    copies = []
    for i, path in enumerate(audio_files[:6]):
        copy = tmp_path / f"g{i}.wav"
        shutil.copy(path, copy)
        copies.append(ImportItem(copy, copy.name, copy.stem, f"A{i}", ["Rock", "Jazz"][i % 2]))

    async with api.sessions() as db:
        await run_batch_import(copies, db)
        await rebuild_index(db)
        report = await evaluate_saved_index(db)

    genre = report["genre"]
    assert genre["all"]["tracks"] == 6
    assert set(genre) == {"all", "Rock", "Jazz"}
    # 2 других трека того же жанра из 5 других
    assert genre["Rock"]["random@10"] == pytest.approx(2 / 5)
    assert report["holdout"] == {}  # лайков нет
