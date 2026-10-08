"""Теги аудиофайла (ID3, Vorbis, MP4) и приведение жанра к общим названиям.

Жанры в тегах свободным текстом и на разных языках: «Рэп», «Хип-хоп»,
«Hip Hop», «Хип-хоп;Рэп». Для оценки и тюнинга нужен общий словарь, поэтому
известные названия приводятся к жанрам FMA (Hip-Hop, Pop, Rock, Electronic…),
а незнакомые остаются как есть.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_GENRES = {
    "Hip-Hop": (
        "рэп",
        "хип-хоп",
        "хип хоп",
        "hip-hop",
        "hip hop",
        "hiphop",
        "rap",
        "трэп",
        "trap",
        "русский рэп",
        "russian rap",
        "grime",
        "drill",
    ),
    "Pop": (
        "поп",
        "pop",
        "поп-музыка",
        "эстрада",
        "russian pop",
        "поп-рок",
        "pop/rock",
        "pop rock",
        "k-pop",
        "dance pop",
    ),
    "Rock": (
        "рок",
        "rock",
        "альтернатива",
        "alternative",
        "альтернативный рок",
        "alternative rock",
        "инди",
        "indie",
        "инди-рок",
        "indie rock",
        "панк",
        "punk",
        "метал",
        "metal",
        "post-punk",
        "пост-панк",
    ),
    "Electronic": (
        "электронная",
        "электроника",
        "electronic",
        "electronica",
        "dance",
        "house",
        "techno",
        "edm",
        "drum & bass",
        "drum and bass",
        "dubstep",
        "phonk",
        "фонк",
    ),
    "Soul-RnB": ("r&b", "rnb", "r'n'b", "соул", "soul", "soul-rnb"),
    "Folk": ("фолк", "folk", "акустика", "acoustic", "singer-songwriter"),
    "Jazz": ("джаз", "jazz"),
    "Classical": ("классика", "классическая", "classical"),
    "Instrumental": ("инструментал", "instrumental", "soundtrack", "саундтрек"),
}
GENRE_ALIASES = {alias: genre for genre, aliases in _GENRES.items() for alias in aliases}
_SPLIT = re.compile(r"\s*[;,/|]\s*")


def canonical_genre(raw: str | None) -> str | None:
    """Первый узнаваемый жанр из тега (их бывает несколько через ; , /), иначе тег как есть."""
    if not raw or not raw.strip():
        return None
    parts = [p for p in _SPLIT.split(raw.strip()) if p]
    for part in parts:
        genre = GENRE_ALIASES.get(" ".join(part.split()).casefold())
        if genre:
            return genre
    whole = GENRE_ALIASES.get(" ".join(raw.split()).casefold())  # «pop/rock» целиком
    return whole or raw.strip()


@dataclass(frozen=True)
class AudioTags:
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    genre: str | None = None  # уже приведён canonical_genre


# Один и тот же тег в разных форматах: ID3 (mp3, wav), Vorbis (flac, ogg), MP4 (m4a)
_KEYS = {
    "artist": ("TPE1", "artist", "\xa9ART"),
    "title": ("TIT2", "title", "\xa9nam"),
    "album": ("TALB", "album", "\xa9alb"),
    "genre": ("TCON", "genre", "\xa9gen"),
}


def read_tags(path: str | Path) -> AudioTags:
    """Теги файла; пустые, если тегов нет или формат не читается."""
    import mutagen

    try:
        audio = mutagen.File(str(path))
    except Exception:
        return AudioTags()
    if audio is None or not audio.tags:
        return AudioTags()

    def first(field: str) -> str | None:
        for key in _KEYS[field]:
            value = audio.tags.get(key)
            if value is None:
                continue
            values = getattr(value, "text", value)  # кадр ID3 или список строк
            if isinstance(values, str):
                values = [values]
            for item in values or []:
                if str(item).strip():
                    return str(item).strip()
        return None

    return AudioTags(
        artist=first("artist"),
        title=first("title"),
        album=first("album"),
        genre=canonical_genre(first("genre")),
    )
