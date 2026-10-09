"""Free Music Archive (https://github.com/mdeff/fma): треки с метаданными для импорта.

Ожидаемая раскладка после распаковки архивов в root:
    root/fma_metadata/tracks.csv
    root/fma_<subset>/<000>/<000002>.mp3

Подмножества вложены: small ⊂ medium ⊂ large.
"""

from pathlib import Path

import pandas as pd

from recommender.application.batch_extract import ImportItem

SUBSETS = ("small", "medium", "large")


def _text(value: object, default: str) -> str:
    return default if pd.isna(value) or not str(value).strip() else str(value).strip()


def load_fma_items(root: Path, subset: str = "small") -> list[ImportItem]:
    """Треки подмножества, для которых есть аудиофайл: название, артист, жанр."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown FMA subset {subset!r}, choose from {SUBSETS}")
    tracks_csv = root / "fma_metadata" / "tracks.csv"
    audio_dir = root / f"fma_{subset}"
    if not tracks_csv.exists():
        raise FileNotFoundError(f"{tracks_csv} not found: unpack fma_metadata.zip into {root}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"{audio_dir} not found: unpack fma_{subset}.zip into {root}")

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1], low_memory=False)
    allowed = set(SUBSETS[: SUBSETS.index(subset) + 1])
    tracks = tracks[tracks[("set", "subset")].isin(allowed)]

    items = []
    for track_id, row in tracks.iterrows():
        name = f"{int(track_id):06d}"
        path = audio_dir / name[:3] / f"{name}.mp3"
        if not path.exists():
            continue
        genre = row[("track", "genre_top")]
        items.append(
            ImportItem(
                path=path,
                filename=f"fma_{name}.mp3",
                source="fma",
                title=_text(row[("track", "title")], f"FMA {name}"),
                artist=_text(row[("artist", "name")], "Unknown"),
                genre=None if pd.isna(genre) else str(genre),
            )
        )
    return items
