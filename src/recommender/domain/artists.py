"""Артисты трека: соавторы и «тот же артист».

Поле artist — строка как в теге или имени файла: «LIZER, FLESH», «Whole Lotta
Swag feat. VERi RERi», «Kai Angel x 9mice». Целиком такие строки не
сравнить: «Lizer», «LIZER, FLESH» и «FLESH, LIZER» оказались бы тремя
разными артистами. Поэтому строка делится на имена (без учёта регистра), и
«тот же артист» значит «есть общее имя».

Разделители: запятая, &, ;, feat./ft./featuring, x и vs. между словами.
& может быть частью названия группы (Simon & Garfunkel) — тогда группа
даёт два имени, но её треки по-прежнему совпадают друг с другом.
"""

import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping

_SEPARATORS = re.compile(r"\s*(?:[,&;]|\s(?:feat\.?|ft\.?|featuring|x|vs\.?)\s)\s*", re.IGNORECASE)
_PLACEHOLDERS = {"", "unknown", "unknown artist", "various artists"}


def artist_names(artist: str | None) -> frozenset[str]:
    """Имена артистов из строки, в нижнем регистре; заглушки вроде Unknown не считаются."""
    if not artist:
        return frozenset()
    names = (" ".join(part.split()).casefold() for part in _SEPARATORS.split(artist))
    return frozenset(name for name in names if name not in _PLACEHOLDERS)


def names_of(artists: Mapping[str, str | None], track_ids: Iterable[str]) -> frozenset[str]:
    """Все имена артистов этих треков (например, лайкнутых)."""
    return frozenset(name for t in track_ids for name in artist_names(artists.get(t)))


class ArtistCap:
    """Не больше max_per_artist треков на имя (0 — без лимита).

    Трек, где есть хоть один артист из liked, проходит всегда: новые треки
    любимого артиста, в том числе его коллаборации, — то, чего пользователь ждёт.
    """

    def __init__(self, max_per_artist: int, liked: Collection[str] = ()) -> None:
        self.max_per_artist = max_per_artist
        self.liked = frozenset(liked)
        self.counts: Counter[str] = Counter()

    def allows(self, names: Collection[str]) -> bool:
        if not self.max_per_artist or not self.liked.isdisjoint(names):
            return True
        return all(self.counts[n] < self.max_per_artist for n in names)

    def add(self, names: Iterable[str]) -> None:
        self.counts.update(names)


def artist_groups(artists: Mapping[str, str | None]) -> dict[str, str]:
    """{track_id: группа}: треки, связанные общими именами (в т.ч. через коллаборации),
    в одной группе. Трек без имён — сам себе группа."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    track_names = {t: sorted(artist_names(a)) for t, a in artists.items()}
    for names in track_names.values():
        for other in names[1:]:
            parent[find(other)] = find(names[0])
    return {t: find(names[0]) if names else f"track:{t}" for t, names in track_names.items()}
