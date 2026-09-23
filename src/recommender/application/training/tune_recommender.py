"""Use case: гиперпараметрическая оптимизация рекомендера через Optuna.

Оптимизируемое:
- метод нормализации (standard / minmax / robust)
- метрика расстояния (cosine / euclidean)
- групповые веса признаков (mfcc / chroma / contrast / ...)
- вес коллаборативного бустинга

Метрика: leave-one-out hit-rate на лайках — для каждого пользователя
прячем один лайкнутый трек и проверяем, рекомендует ли система его
по другим лайкнутым.

Итог: индекс и нормализатор пересобраны с лучшими параметрами и
сохранены на диск.
"""

import datetime
import json

import numpy as np
import optuna
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from recommender.application.collaborative import co_like_strength
from recommender.config import settings
from recommender.infrastructure.data_processing.extract import bytes_to_features
from recommender.infrastructure.data_processing.normalize import FeatureNormalizer
from recommender.infrastructure.storage.faiss_index import FaissRecommender
from recommender.infrastructure.storage.postgres import AutoMLRunORM, LikeORM, TrackORM


# Индексы групп признаков в 58-мерном векторе
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


def apply_feature_weights(
    features: np.ndarray, weights: dict[str, float]
) -> np.ndarray:
    """Применить групповые веса к матрице признаков."""
    return features * feature_weight_vector(weights)


async def run_tuning(db: AsyncSession, run_id: int) -> dict:
    """Запустить Optuna-оптимизацию. Возвращает best params и score."""
    # Все треки с фичами
    result = await db.execute(
        select(TrackORM).where(TrackORM.feature_vector.isnot(None))
    )
    tracks = result.scalars().all()

    if len(tracks) < 5:
        raise ValueError("Need at least 5 tracks with features to run tuning")

    track_ids = [t.id for t in tracks]
    raw_features = np.array([bytes_to_features(t.feature_vector) for t in tracks])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}

    # Лайки для evaluation
    result = await db.execute(select(LikeORM))
    likes = result.scalars().all()

    user_tracks: dict[str, list[str]] = {}
    for like in likes:
        user_tracks.setdefault(like.user_id, []).append(like.track_id)
    user_likes = {uid: set(tids) for uid, tids in user_tracks.items()}

    # Только пользователи с >=2 лайками (для leave-one-out)
    eval_users = {
        uid: tids for uid, tids in user_tracks.items()
        if len(tids) >= 2 and all(t in id_to_idx for t in tids)
    }

    if not eval_users:
        raise ValueError("Need users with >=2 liked tracks for evaluation")

    run = await db.get(AutoMLRunORM, run_id)
    run.status = "running"
    run.started_at = datetime.datetime.utcnow()
    await db.commit()

    def objective(trial: optuna.Trial) -> float:
        norm_method = trial.suggest_categorical(
            "norm_method", ["standard", "minmax", "robust"]
        )
        metric = trial.suggest_categorical("metric", ["cosine", "euclidean"])
        boost_weight = trial.suggest_float("boost_weight", 0.0, 3.0)

        weights = {
            group_name: trial.suggest_float(f"w_{group_name}", 0.0, 3.0)
            for group_name in FEATURE_GROUPS
        }

        normalizer = FeatureNormalizer(
            method=norm_method, weights=feature_weight_vector(weights)
        )
        normalized = normalizer.fit_transform(raw_features)

        engine = FaissRecommender(
            dimension=normalized.shape[1], metric=metric, boost_weight=boost_weight
        )
        engine.add_tracks(track_ids, normalized.copy())

        # leave-one-out hit rate
        hits = 0
        total = 0
        for uid, liked_tids in eval_users.items():
            for i, held_out in enumerate(liked_tids):
                query_tids = [t for j, t in enumerate(liked_tids) if j != i]
                if not query_tids:
                    continue
                query_idx = id_to_idx[query_tids[0]]
                query_vec = normalized[query_idx]

                # Спрятанный лайк убираем и из co-like сигнала, иначе буст
                # подсказывает ответ и hit-rate завышается.
                visible_likes = {
                    u: (tids - {held_out} if u == uid else tids)
                    for u, tids in user_likes.items()
                }
                boost = co_like_strength(query_tids[0], visible_likes)

                recs = engine.recommend(
                    query_vec,
                    limit=20,
                    exclude_ids={query_tids[0]},
                    like_boost=boost or None,
                )
                rec_ids = {r.track_id for r in recs}
                if held_out in rec_ids:
                    hits += 1
                total += 1

        return hits / total if total > 0 else 0.0

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
    run.completed_at = datetime.datetime.utcnow()
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
    }
