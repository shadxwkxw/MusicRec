"""Use case: плейлисты как сигнал «слушают вместе».

Каждый плейлист становится пользователем playlist:<имя файла>, а его треки —
лайками этого пользователя. Так работают co-like буст, популярность для
холодного старта и оценка по отложенным лайкам; рекомендации для
playlist:<имя> — продолжение плейлиста.

Форматы файлов (по строке на трек):
- .txt — «Артист - Название» (разделитель -, — или –);
- .m3u / .m3u8 — #EXTINF:<сек>,Артист - Название или путь к файлу «…/Артист - Название.mp3»;
- .csv — колонки исполнителя и названия (artist/title, Exportify: «Artist Name(s)» и
  «Track Name», соавторы через ;), иначе первая колонка «Артист - Название».

Трек из плейлиста сопоставляется с каталогом, если совпадает название (без
регистра, ё/е, знаков и хвостов вроде «(prod. by …)», «(feat. …)») и есть
хоть один общий артист с учётом соавторов. Совпадения только по названию не
засчитываются. Повторный импорт заменяет лайки плейлиста содержимым файла.
"""

import csv
import io
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePath

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.domain.artists import artist_names
from recommender.infrastructure.storage.postgres import LikeORM, TrackORM

PLAYLIST_USER_PREFIX = "playlist:"
PLAYLIST_SUFFIXES = {".txt", ".m3u", ".m3u8", ".csv"}

_ARTIST_COLUMNS = ("artist", "artist name(s)", "artist name", "artists", "исполнитель")
_TITLE_COLUMNS = ("title", "track name", "track", "name", "название", "трек")
_DASH = re.compile(r"\s+[-—–]\s+")
_TAIL = re.compile(
    r"[(\[][^)\]]*(?:prod|feat|ft\.|remix|version|edit|official|live)[^)\]]*[)\]]", re.I
)
_COPY_MARK = re.compile(r"\s*\(\d{1,2}\)\s*$")  # «Apathy (1)» — копия файла
_FEAT_TAIL = re.compile(r"\s+(?:feat\.?|ft\.?)\s+.*$", re.I)
_NON_WORD = re.compile(r"[\W_]+")  # «Syringe_» из имени файла = «Syringe?»


@dataclass(frozen=True)
class PlaylistEntry:
    artist: str
    title: str


def normalize_title(title: str) -> str:
    """Название для сравнения: без регистра, ё/е, знаков и хвостов (prod., feat., remix)."""
    text = unicodedata.normalize("NFC", title).casefold().replace("ё", "е")
    text = _COPY_MARK.sub("", _TAIL.sub(" ", text))
    text = _FEAT_TAIL.sub("", text)
    return " ".join(_NON_WORD.sub(" ", text).split())


def _entry(text: str) -> PlaylistEntry | None:
    parts = _DASH.split(text.strip(), maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return PlaylistEntry(parts[0].strip(), parts[1].strip())


def parse_playlist(name: str, content: str) -> tuple[list[PlaylistEntry], list[str]]:
    """Треки плейлиста и строки, которые не удалось разобрать."""
    suffix = PurePath(name).suffix.lower()
    entries: list[PlaylistEntry] = []
    bad: list[str] = []
    if suffix == ".csv":
        rows = list(csv.reader(io.StringIO(content)))
        header = [c.strip().lstrip("\ufeff").casefold() for c in rows[0]] if rows else []
        a = next((header.index(c) for c in _ARTIST_COLUMNS if c in header), None)
        t = next((header.index(c) for c in _TITLE_COLUMNS if c in header), None)
        if a is not None and t is not None:
            for row in rows[1:]:
                if len(row) > max(a, t) and row[a].strip() and row[t].strip():
                    entries.append(PlaylistEntry(row[a].strip(), row[t].strip()))
                elif any(cell.strip() for cell in row):
                    bad.append(",".join(row))
        else:
            for row in rows:
                if row and row[0].strip():
                    item = _entry(row[0])
                    if item:
                        entries.append(item)
                    else:
                        bad.append(row[0])
        return entries, bad
    pending: PlaylistEntry | None = None  # из #EXTINF; путь на следующей строке — тот же трек
    for raw in content.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line:
            continue
        if suffix in (".m3u", ".m3u8"):
            if line.upper().startswith("#EXTINF"):
                pending = _entry(line.split(",", 1)[1]) if "," in line else None
                continue
            if line.startswith("#"):
                continue
            item = pending or _entry(PurePath(line.replace("\\", "/")).stem)
            pending = None
        else:
            item = _entry(line)
        if item:
            entries.append(item)
        else:
            bad.append(line)
    return entries, bad


@dataclass
class PlaylistImportResult:
    name: str
    user_id: str
    matched: int = 0
    total: int = 0
    unmatched: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)


class CatalogMatcher:
    """Поиск трека каталога по артисту и названию."""

    def __init__(self, tracks: Iterable[tuple[str, str | None, str]]) -> None:
        self.by_title: dict[str, list[tuple[str, frozenset[str]]]] = {}
        for track_id, artist, title in tracks:
            self.by_title.setdefault(normalize_title(title), []).append(
                (track_id, artist_names(artist))
            )

    def find(self, entry: PlaylistEntry) -> str | None:
        names = artist_names(entry.artist)
        for track_id, track_names in self.by_title.get(normalize_title(entry.title), []):
            if names & track_names:
                return track_id
        return None


async def import_playlist(
    db: AsyncSession, name: str, content: str, matcher: CatalogMatcher
) -> PlaylistImportResult:
    """Заменить лайки пользователя playlist:<name> треками плейлиста."""
    entries, unparsed = parse_playlist(name, content)
    result = PlaylistImportResult(
        name=name, user_id=PLAYLIST_USER_PREFIX + PurePath(name).stem, unparsed=unparsed
    )
    found: list[str] = []
    for entry in entries:
        track_id = matcher.find(entry)
        if track_id is None:
            result.unmatched.append(f"{entry.artist} - {entry.title}")
        elif track_id not in found:
            found.append(track_id)
    result.total = len(entries)
    result.matched = len(found)

    await db.execute(delete(LikeORM).where(LikeORM.user_id == result.user_id))
    db.add_all(LikeORM(user_id=result.user_id, track_id=t) for t in found)
    await db.commit()
    return result


async def import_playlists(db: AsyncSession, directory: Path) -> list[PlaylistImportResult]:
    """Импортировать все плейлисты из папки (не рекурсивно)."""
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    rows = (await db.execute(select(TrackORM.id, TrackORM.artist, TrackORM.title))).all()
    matcher = CatalogMatcher((r.id, r.artist, r.title) for r in rows)
    results = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in PLAYLIST_SUFFIXES:
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            results.append(await import_playlist(db, path.name, content, matcher))
    return results
