"""Профиль пользователя для персональных рекомендаций: несколько интересов вместо одного среднего.

Среднее векторов лайков у человека с разными вкусами (скажем, рэп и фолк)
оказывается между ними, и выдача не попадает ни в один. Поэтому лайки
группируются в интересы (агломеративно, по косинусу центров):
- группы объединяются, пока их больше max_interests или пока центры похожи
  сильнее merge_similarity;
- группа меньше min_likes_per_interest присоединяется к ближайшей — один
  случайный лайк не отдельный интерес.

Выдача: доля mean_share — ближайшие к общему среднему (у однородного вкуса это
лучший запрос), остальные места делятся между интересами пропорционально
числу их лайков. Не больше max_per_artist треков одного артиста (соавторы —
по отдельности, см. domain/artists.py) — кроме артистов, которых пользователь уже лайкал: их новые треки и есть то, чего он
ждёт (на реальных лайках лимит для них прятал целевой трек).

Порог и доли подобраны на синтетических пользователях из жанров FMA (лайки
из 1–3 жанров): при однородном вкусе точность почти та же, что у среднего,
а все интересы попадают в топ-10 в 1.5–2 раза чаще.

Скор у рекомендации — близость к запросу её интереса, поэтому порядок в выдаче
задают слоты, а не скоры.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np

from recommender.config import settings
from recommender.domain.artists import ArtistCap, artist_names
from recommender.domain.models import Recommendation
from recommender.domain.recommender import Recommender


def interest_groups(
    vectors: np.ndarray,
    max_interests: int | None = None,
    min_likes: int | None = None,
    merge_similarity: float | None = None,
) -> list[list[int]]:
    """Индексы строк vectors, сгруппированные по интересам, от крупных к мелким."""
    max_interests = max_interests or settings.user_max_interests
    min_likes = settings.user_min_likes_per_interest if min_likes is None else min_likes
    threshold = settings.user_merge_similarity if merge_similarity is None else merge_similarity

    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    groups: list[list[int]] = [[i] for i in range(len(unit))]
    sums = [unit[i].copy() for i in range(len(unit))]

    def centre(k: int) -> np.ndarray:
        return sums[k] / max(float(np.linalg.norm(sums[k])), 1e-12)

    centres = np.stack([centre(k) for k in range(len(groups))]) if groups else unit
    sim = centres @ centres.T
    np.fill_diagonal(sim, -np.inf)

    while len(groups) > 1:
        small = [k for k, g in enumerate(groups) if len(g) < min_likes]
        if small:  # мелкая группа — к ближайшей
            i = small[0]
            j = int(np.argmax(sim[i]))
        else:
            i, j = (int(x) for x in np.unravel_index(np.argmax(sim), sim.shape))
            if sim[i, j] < threshold and len(groups) <= max_interests:
                break
        i, j = min(i, j), max(i, j)
        groups[i] += groups.pop(j)
        sums[i] = sums[i] + sums.pop(j)
        sim = np.delete(np.delete(sim, j, axis=0), j, axis=1)
        merged = centre(i)
        row = np.stack([centre(k) for k in range(len(groups))]) @ merged
        row[i] = -np.inf
        sim[i, :] = row
        sim[:, i] = row
    return sorted(groups, key=len, reverse=True)


@dataclass
class InterestCandidates:
    """Кандидаты по общему среднему и по каждому интересу (если интересов больше одного)."""

    overall: list[Recommendation]
    by_interest: list[list[Recommendation]]
    interest_sizes: list[int]

    @property
    def track_ids(self) -> set[str]:
        return {r.track_id for lst in (self.overall, *self.by_interest) for r in lst}


def interest_candidates(
    engine: Recommender,
    vectors: np.ndarray,
    limit: int,
    exclude_ids: set[str],
    like_boost: dict[str, float] | None = None,
) -> InterestCandidates:
    """Поиск по нормализованным векторам лайков (строки vectors)."""
    pool = limit * settings.candidate_multiplier * 2  # с запасом на лимит по артистам

    def search(query: np.ndarray) -> list[Recommendation]:
        return engine.recommend(query, limit=pool, exclude_ids=exclude_ids, like_boost=like_boost)

    groups = interest_groups(vectors)
    overall = search(vectors.mean(axis=0))
    if len(groups) == 1:
        return InterestCandidates(overall, [], [len(vectors)])
    return InterestCandidates(
        overall, [search(vectors[g].mean(axis=0)) for g in groups], [len(g) for g in groups]
    )


def blend(
    candidates: InterestCandidates,
    limit: int,
    artists: Mapping[str, str | None] | None = None,
    max_per_artist: int | None = None,
    mean_share: float | None = None,
    liked_artists: Collection[str] = (),
) -> list[Recommendation]:
    """Собрать выдачу: общее среднее, потом интересы по квотам, не больше max_per_artist
    треков на имя артиста (0 — без ограничения; артисты — {track_id: строка artist},
    соавторы считаются каждый), кроме имён из liked_artists."""
    max_per_artist = settings.user_max_per_artist if max_per_artist is None else max_per_artist
    mean_share = settings.user_mean_share if mean_share is None else mean_share
    artists = artists or {}
    limiter = ArtistCap(max_per_artist, liked_artists)
    taken: list[Recommendation] = []
    seen: set[str] = set()

    def take(rec: Recommendation, cap: bool = True) -> bool:
        names = artist_names(artists.get(rec.track_id))
        if rec.track_id in seen or (cap and not limiter.allows(names)):
            return False
        seen.add(rec.track_id)
        taken.append(rec)
        limiter.add(names)
        return True

    lists = candidates.by_interest
    head = limit if not lists else round(limit * mean_share)
    for rec in candidates.overall:
        if len(taken) >= head:
            break
        take(rec)

    # Остальные места — интересам по квотам: следующий слот тому, у кого больший недобор
    total = sum(candidates.interest_sizes)
    quotas = [size / total * (limit - len(taken)) for size in candidates.interest_sizes]
    got = [0] * len(lists)
    queues = [list(lst) for lst in lists]
    while len(taken) < limit and any(queues):
        k = max((k for k, q in enumerate(queues) if q), key=lambda k: quotas[k] - got[k])
        if take(queues[k].pop(0)):
            got[k] += 1

    # Если лимит по артистам выбрал не всех — добираем без него
    for cap in (True, False):
        for rec in candidates.overall:
            if len(taken) >= limit:
                return taken
            take(rec, cap=cap)
    return taken
