"""Офлайн-оценка рекомендаций по лайкам: разбиение, бейзлайны, метрики.

Запрос — это «спрятанный» лайк (target) и то, что система о пользователе знает:
- путь track: рекомендации от одного известного лайка (как /recommendations/{id});
- путь user: рекомендации по всем известным лайкам (как /recommendations/user/{id}).
Для каждого пути считаются hit@k и MRR@k.

Два протокола:
- leave_one_out_queries — для тюнинга, внутри обучающих лайков;
- holdout_queries — итоговая оценка на отложенных лайках, которых тюнинг не видел.

Отдельно genre_report: для треков с жанром (например из FMA) — доля того же
жанра среди k ближайших. Лайки для неё не нужны.
"""

import random
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import (
    co_like_strength,
    load_user_likes,
    user_co_like_strength,
)
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import TrackORM

PATHS = ("track", "user")
UserLikes = Mapping[str, set[str]]


@dataclass(frozen=True)
class Query:
    path: str
    query_ids: tuple[str, ...]
    exclude: frozenset[str]
    target: str
    visible_likes: UserLikes  # что видит co-like буст: спрятанные лайки сюда не входят


Ranker = Callable[[Query, int], list[str]]


# ── Разбиение и запросы ──────────────────────────────────────────


def split_likes(
    user_likes: UserLikes, test_fraction: float, seed: int
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Отложить долю лайков каждого пользователя в тест.

    У пользователя с >=2 лайками в тест уходит хотя бы один и в train остаётся
    хотя бы один. Пользователи с одним лайком целиком остаются в train.
    """
    rng = random.Random(seed)
    train: dict[str, set[str]] = {}
    test: dict[str, set[str]] = {}
    for user_id in sorted(user_likes):
        liked = sorted(user_likes[user_id])
        rng.shuffle(liked)
        n_test = 0
        if len(liked) >= 2 and test_fraction > 0:
            n_test = min(len(liked) - 1, max(1, round(len(liked) * test_fraction)))
        train[user_id] = set(liked[n_test:])
        if n_test:
            test[user_id] = set(liked[:n_test])
    return train, test


def _queries_for(seen: tuple[str, ...], target: str, visible_likes: UserLikes) -> list[Query]:
    track_queries = [Query("track", (q,), frozenset({q}), target, visible_likes) for q in seen]
    return [*track_queries, Query("user", seen, frozenset(seen), target, visible_likes)]


def leave_one_out_queries(user_likes: UserLikes, known: set[str]) -> list[Query]:
    """Каждый лайк по очереди прячется, остальные лайки пользователя известны."""
    queries: list[Query] = []
    for user_id, liked in sorted(user_likes.items()):
        if len(liked) < 2 or not liked <= known:
            continue
        for held_out in sorted(liked):
            visible = {u: (t - {held_out} if u == user_id else t) for u, t in user_likes.items()}
            queries += _queries_for(tuple(sorted(liked - {held_out})), held_out, visible)
    return queries


def holdout_queries(train: UserLikes, test: UserLikes, known: set[str]) -> list[Query]:
    """Каждый отложенный лайк — цель, известны только обучающие лайки (всех пользователей)."""
    queries: list[Query] = []
    for user_id, targets in sorted(test.items()):
        seen = train.get(user_id, set())
        if not seen or not (seen | targets) <= known:
            continue
        for target in sorted(targets):
            queries += _queries_for(tuple(sorted(seen)), target, train)
    return queries


def split_by_artist(
    track_ids: Collection[str], artists: Mapping[str, str], test_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Отложить долю артистов целиком: треки одного артиста не попадут по обе стороны."""
    names = sorted({artists.get(t, "") for t in track_ids})
    random.Random(seed).shuffle(names)
    n_test = min(len(names) - 1, round(len(names) * test_fraction)) if len(names) > 1 else 0
    test_artists = set(names[:n_test])
    tune = sorted(t for t in track_ids if artists.get(t, "") not in test_artists)
    test = sorted(t for t in track_ids if artists.get(t, "") in test_artists)
    return tune, test


# ── Ранжировщики ─────────────────────────────────────────────────


def system_ranker(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    use_boost: bool = True,
) -> Ranker:
    """Наша система: те же запросы, что делают API-эндпоинты."""

    def rank(query: Query, k: int) -> list[str]:
        if query.path == "track":
            (query_id,) = query.query_ids
            vector = normalized[id_to_idx[query_id]]
            boost = co_like_strength(query_id, dict(query.visible_likes)) if use_boost else {}
        else:
            vector = normalized[[id_to_idx[t] for t in query.query_ids]].mean(axis=0)
            boost = (
                user_co_like_strength(set(query.query_ids), dict(query.visible_likes))
                if use_boost
                else {}
            )
        recs = engine.recommend(
            vector, limit=k, exclude_ids=set(query.exclude), like_boost=boost or None
        )
        return [r.track_id for r in recs]

    return rank


def popularity_ranker(track_ids: list[str], train: UserLikes) -> Ranker:
    """Самые лайкаемые треки, одинаковые для всех."""
    counts = Counter(t for liked in train.values() for t in liked)
    order = sorted(track_ids, key=lambda t: (-counts[t], t))

    def rank(query: Query, k: int) -> list[str]:
        return [t for t in order if t not in query.exclude][:k]

    return rank


def same_artist_ranker(track_ids: list[str], artists: dict[str, str], train: UserLikes) -> Ranker:
    """Сначала треки артистов из известных лайков, внутри — по популярности."""
    counts = Counter(t for liked in train.values() for t in liked)

    def rank(query: Query, k: int) -> list[str]:
        affinity = Counter(artists[t] for t in query.query_ids)
        candidates = [t for t in track_ids if t not in query.exclude]
        candidates.sort(key=lambda t: (-affinity[artists[t]], -counts[t], t))
        return candidates[:k]

    return rank


# ── Метрики ──────────────────────────────────────────────────────


def _summarize(scores: dict[str, list[tuple[float, float]]], k: int) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path in PATHS:
        if scores[path]:
            n = len(scores[path])
            metrics[f"{path}_hit@{k}"] = sum(hit for hit, _ in scores[path]) / n
            metrics[f"{path}_mrr@{k}"] = sum(rr for _, rr in scores[path]) / n
    return metrics


def score(queries: list[Query], ranker: Ranker, k: int) -> dict[str, float]:
    """hit@k и MRR@k по путям."""
    scores: dict[str, list[tuple[float, float]]] = {p: [] for p in PATHS}
    for query in queries:
        top = ranker(query, k)
        rank = top.index(query.target) + 1 if query.target in top else None
        scores[query.path].append((1.0, 1 / rank) if rank else (0.0, 0.0))
    return _summarize(scores, k)


def expected_random(queries: list[Query], n_tracks: int, k: int) -> dict[str, float]:
    """Матожидание метрик для случайной выдачи — точно, без сэмплирования."""
    scores: dict[str, list[tuple[float, float]]] = {p: [] for p in PATHS}
    for query in queries:
        n = n_tracks - len(query.exclude)
        top = min(k, n)
        scores[query.path].append((top / n, sum(1 / r for r in range(1, top + 1)) / n))
    return _summarize(scores, k)


def evaluate(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    user_likes: UserLikes,
    k: int | None = None,
) -> dict[str, float]:
    """Leave-one-out оценка системы — целевая функция тюнинга."""
    k = k or settings.tuning_eval_k
    queries = leave_one_out_queries(user_likes, set(id_to_idx))
    return score(queries, system_ranker(engine, normalized, id_to_idx), k)


def objective_score(metrics: dict[str, float]) -> float:
    """Среднее MRR@k по путям."""
    mrr = [v for name, v in metrics.items() if "_mrr@" in name]
    return sum(mrr) / len(mrr)


def holdout_report(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    artists: dict[str, str],
    train: UserLikes,
    test: UserLikes,
    k: int | None = None,
) -> dict[str, dict[str, float]]:
    """Система и бейзлайны на одних и тех же отложенных лайках. Пусто, если теста нет."""
    k = k or settings.tuning_eval_k
    track_ids = list(id_to_idx)
    queries = holdout_queries(train, test, set(track_ids))
    if not queries:
        return {}
    return {
        "system": score(queries, system_ranker(engine, normalized, id_to_idx), k),
        "content_only": score(
            queries, system_ranker(engine, normalized, id_to_idx, use_boost=False), k
        ),
        "same_artist": score(queries, same_artist_ranker(track_ids, artists, train), k),
        "popularity": score(queries, popularity_ranker(track_ids, train), k),
        "random": expected_random(queries, len(track_ids), k),
    }


def genre_report(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    genres: Mapping[str, str | None],
    artists: Mapping[str, str],
    k: int | None = None,
    query_ids: Collection[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Доля того же жанра среди k ближайших (без буста) против случайного уровня.

    filtered — то же, но треки того же артиста исключены из соседей (artist
    filter): иначе метрика отчасти меряет «нашёл других треков артиста».
    Случайный уровень считается точно для каждого трека. query_ids ограничивает
    запросы (кандидаты — весь индекс). Строки: all и каждый жанр. Пусто, если
    меток нет.
    """
    k = k or settings.tuning_eval_k
    queries = id_to_idx if query_ids is None else [t for t in query_ids if t in id_to_idx]
    labeled = [t for t in queries if genres.get(t)]
    if not labeled:
        return {}
    n_tracks = len(id_to_idx)
    genre_size = Counter(genres[t] for t in id_to_idx if genres.get(t))
    by_artist: dict[str, set[str]] = {}
    for track_id in id_to_idx:
        by_artist.setdefault(artists.get(track_id, ""), set()).add(track_id)

    def same_share(track_id: str, exclude: set[str]) -> float:
        recs = engine.recommend(normalized[id_to_idx[track_id]], limit=k, exclude_ids=exclude)
        return sum(genres.get(r.track_id) == genres[track_id] for r in recs) / k

    by_genre: dict[str, list[tuple[float, float, float, float]]] = {}
    for track_id in labeled:
        genre = genres[track_id]
        assert genre is not None
        artist_tracks = by_artist[artists.get(track_id, "")]
        artist_same_genre = sum(genres.get(t) == genre for t in artist_tracks)
        by_genre.setdefault(genre, []).append(
            (
                same_share(track_id, {track_id}),
                (genre_size[genre] - 1) / (n_tracks - 1),
                same_share(track_id, artist_tracks),
                (genre_size[genre] - artist_same_genre) / max(n_tracks - len(artist_tracks), 1),
            )
        )

    def row(values: list[tuple[float, float, float, float]]) -> dict[str, float]:
        means = [sum(column) / len(values) for column in zip(*values, strict=True)]
        return {
            f"system@{k}": means[0],
            f"random@{k}": means[1],
            f"filtered@{k}": means[2],
            f"filt_random@{k}": means[3],
            "tracks": float(len(values)),
        }

    report = {"all": row([v for values in by_genre.values() for v in values])}
    for genre in sorted(by_genre, key=lambda g: -len(by_genre[g])):
        report[genre] = row(by_genre[genre])
    return report


async def evaluate_saved_index(db: AsyncSession) -> dict[str, dict[str, dict[str, float]]]:
    """Оценить сохранённые индекс и нормализатор: отложенные лайки и жанры."""
    engine = FaissRecommender.load()
    normalizer = FeatureNormalizer.load()
    result = await db.execute(select(TrackORM).where(TrackORM.feature_vector.isnot(None)))
    rows: Sequence[TrackORM] = result.scalars().all()
    tracks = {t.id: t for t in rows}
    track_ids = [t for t in engine.track_ids if t in tracks]
    raw = np.array([bytes_to_features(tracks[t].feature_vector) for t in track_ids])
    normalized = normalizer.transform(raw)

    id_to_idx = {t: i for i, t in enumerate(track_ids)}
    train, test = split_likes(
        await load_user_likes(db), settings.tuning_test_fraction, settings.tuning_seed
    )
    return {
        "holdout": holdout_report(
            engine,
            normalized,
            id_to_idx,
            {t: tracks[t].artist or "" for t in track_ids},
            train,
            test,
        ),
        "genre": genre_report(
            engine,
            normalized,
            id_to_idx,
            {t: tracks[t].genre for t in track_ids},
            {t: tracks[t].artist or "" for t in track_ids},
        ),
    }
