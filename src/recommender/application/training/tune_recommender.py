"""Use case: гиперпараметрическая оптимизация рекомендера через Optuna.

Оптимизируемое:
- метод нормализации (standard / minmax / robust)
- метрика расстояния (cosine / euclidean)
- групповые веса признаков (mfcc / chroma / contrast / ...)
- вес коллаборативного бустинга

Метрика: leave-one-out на лайках (см. evaluate) — для каждого пользователя
прячем один лайкнутый трек и проверяем, где система ставит его в выдаче,
построенной по остальным лайкам. Оптимизируется среднее MRR@10 по двум
прод-путям: рекомендации по треку (с бустом) и по пользователю.

Итог: индекс и нормализатор пересобраны с лучшими параметрами и
сохранены на диск.
"""

import json
from collections.abc import Sequence

import numpy as np
import optuna
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import co_like_strength
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import AutoMLRunORM, LikeORM, TrackORM, utcnow

EVAL_K = 10

# Индексы групп признаков в 82-мерном векторе
FEATURE_GROUPS = {
    "mfcc": (0, 26),
    "chroma": (26, 50),
    "contrast": (50, 64),
    "tonnetz": (64, 76),
    "tempo": (76, 77),
    "rms": (77, 78),
    "zcr": (78, 79),
    "spectral_stats": (79, 82),
}


def feature_weight_vector(weights: dict[str, float]) -> np.ndarray:
    """Развернуть групповые веса в per-dimension вектор."""
    dim = max(end for _, end in FEATURE_GROUPS.values())
    vec = np.ones(dim, dtype=np.float32)
    for group_name, (start, end) in FEATURE_GROUPS.items():
        vec[start:end] = weights.get(group_name, 1.0)
    return vec


def apply_feature_weights(features: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Применить групповые веса к матрице признаков."""
    return features * feature_weight_vector(weights)


def _rank(recs: list, track_id: str) -> int | None:
    """1-based позиция трека в выдаче или None."""
    for pos, rec in enumerate(recs, start=1):
        if rec.track_id == track_id:
            return pos
    return None


def evaluate(
    engine: FaissRecommender,
    normalized: np.ndarray,
    id_to_idx: dict[str, int],
    user_likes: dict[str, set[str]],
    k: int = EVAL_K,
) -> dict[str, float]:
    """Leave-one-out по лайкам для обоих прод-путей рекомендаций.

    track: запрос от каждого другого лайка пользователя, с co-like бустом;
           спрятанный лайк убран и из буста, иначе он подсказывает ответ.
    user:  запрос — среднее остальных лайков, они же исключены, без буста.
    Для каждого пути считаются hit@k и MRR@k.
    """
    ranks: dict[str, list[int | None]] = {"track": [], "user": []}
    for uid, liked in user_likes.items():
        if len(liked) < 2 or any(t not in id_to_idx for t in liked):
            continue
        for held_out in sorted(liked):
            others = sorted(liked - {held_out})
            visible = {u: (t - {held_out} if u == uid else t) for u, t in user_likes.items()}

            for query_id in others:
                recs = engine.recommend(
                    normalized[id_to_idx[query_id]],
                    limit=k,
                    exclude_ids={query_id},
                    like_boost=co_like_strength(query_id, visible) or None,
                )
                ranks["track"].append(_rank(recs, held_out))

            query = normalized[[id_to_idx[t] for t in others]].mean(axis=0)
            recs = engine.recommend(query, limit=k, exclude_ids=set(others))
            ranks["user"].append(_rank(recs, held_out))

    metrics: dict[str, float] = {}
    for path, path_ranks in ranks.items():
        n = len(path_ranks) or 1
        metrics[f"{path}_hit@{k}"] = sum(r is not None for r in path_ranks) / n
        metrics[f"{path}_mrr@{k}"] = sum(1 / r for r in path_ranks if r is not None) / n
    return metrics


def objective_score(metrics: dict[str, float], k: int = EVAL_K) -> float:
    return (metrics[f"track_mrr@{k}"] + metrics[f"user_mrr@{k}"]) / 2


async def run_tuning(db: AsyncSession, run_id: int) -> dict:
    """Запустить Optuna-оптимизацию. Возвращает best params и score."""
    # Все треки с фичами
    result = await db.execute(select(TrackORM).where(TrackORM.feature_vector.isnot(None)))
    tracks: Sequence[TrackORM] = result.scalars().all()

    if len(tracks) < 5:
        raise ValueError("Need at least 5 tracks with features to run tuning")

    track_ids = [t.id for t in tracks]
    raw_features = np.array([bytes_to_features(t.feature_vector) for t in tracks])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    # Лайки для evaluation
    result = await db.execute(select(LikeORM))
    likes: Sequence[LikeORM] = result.scalars().all()

    user_tracks: dict[str, list[str]] = {}
    for like in likes:
        user_tracks.setdefault(like.user_id, []).append(like.track_id)
    user_likes = {uid: set(tids) for uid, tids in user_tracks.items()}

    # Только пользователи с >=2 лайками (для leave-one-out)
    eval_users = {
        uid: tids
        for uid, tids in user_tracks.items()
        if len(tids) >= 2 and all(t in id_to_idx for t in tids)
    }

    if not eval_users:
        raise ValueError("Need users with >=2 liked tracks for evaluation")

    run = await db.get(AutoMLRunORM, run_id)
    if run is None:
        raise ValueError(f"AutoML run {run_id} not found")
    run.status = "running"
    run.started_at = utcnow()
    await db.commit()

    def objective(trial: optuna.Trial) -> float:
        norm_method = trial.suggest_categorical("norm_method", ["standard", "minmax", "robust"])
        metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])
        boost_weight = trial.suggest_float("boost_weight", 0.0, 3.0)

        weights = {
            group_name: trial.suggest_float(f"w_{group_name}", 0.0, 3.0)
            for group_name in FEATURE_GROUPS
        }

        normalizer = FeatureNormalizer(method=norm_method, weights=feature_weight_vector(weights))
        normalized = normalizer.fit_transform(raw_features)

        engine = FaissRecommender(
            dimension=normalized.shape[1], metric=metric, boost_weight=boost_weight
        )
        engine.add_tracks(track_ids, normalized.copy())

        metrics = evaluate(engine, normalized, id_to_idx, user_likes)
        trial.set_user_attr("metrics", metrics)
        return objective_score(metrics)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=settings.automl_n_trials,
        timeout=settings.automl_timeout,
    )

    best = study.best_params
    run.status = "completed"
    run.best_score = study.best_value
    run.best_params = json.dumps(best)
    run.n_trials = len(study.trials)
    run.completed_at = utcnow()
    await db.commit()

    # Пересобрать индекс с лучшими параметрами
    weights = {g: best.get(f"w_{g}", 1.0) for g in FEATURE_GROUPS}
    normalizer = FeatureNormalizer(
        method=best["norm_method"], weights=feature_weight_vector(weights)
    )
    normalized = normalizer.fit_transform(raw_features)
    normalizer.save()

    engine = FaissRecommender(
        dimension=normalized.shape[1],
        metric=best["metric"],
        boost_weight=best["boost_weight"],
    )
    engine.add_tracks(track_ids, normalized.copy())
    engine.save()

    return {
        "best_score": study.best_value,
        "best_params": best,
        "n_trials": len(study.trials),
        "metrics": study.best_trial.user_attrs["metrics"],
    }
